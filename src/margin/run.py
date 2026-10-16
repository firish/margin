"""Experiment config -> planned calls -> cost estimate -> predictions on disk.

Several experiments can run in one process: every (task, coder, variant) unit runs concurrently, with
one concurrency limit per model shared across units. Jev writes `jev__<variant>.parquet`; an LLM coder
writes one file per signal, `<coder>-<signal>__<variant>.parquet` (e.g. `luna-logprob__canonical`).
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from margin import sampling
from margin.budget import ModelSpec, Registry
from margin.cache import Cache
from margin.coders import decisions, jev, llm, systemone
from margin.datasets.base import Task
from margin.perturb import STANDARD_ARMS, Variant, load_paraphrases
from margin.schema import Codebook

RUNS = Path("data/runs")
CODEBOOKS = Path("codebooks")


@dataclass(frozen=True)
class Experiment:
    name: str
    sample: str
    split: str
    tasks: list[str]
    coders: list[str]
    variants: list[str]
    limit: int | None = None
    cache_tag: str | None = None
    """Fresh, separately cached copies of identical requests (repeatability checks)."""

    @classmethod
    def load(cls, name_or_path: str) -> Experiment:
        path = Path(name_or_path)
        if not path.exists():
            path = Path("configs/experiments") / f"{name_or_path}.yaml"
        raw: dict[str, Any] = yaml.safe_load(path.read_text())
        return cls(**raw)


@dataclass
class Unit:
    """One (task, coder, variant) cell of an experiment, with its planned calls."""

    experiment: str
    task: str
    coder: str
    variant: Variant
    spec: ModelSpec
    calls: list
    n_segments: int
    combine_nouls: bool = False
    """exclusive codebook asked as one Noul per option: combine with jev.exactly_one."""
    cache_tag: str | None = None

    @property
    def llm(self) -> llm.LLMCoder | None:
        return llm.CODERS.get(self.coder)

    def out_path(self, signal: str | None = None) -> Path:
        name = self.coder if signal is None else f"{self.coder}-{signal}"
        return RUNS / self.experiment / self.task / f"{name}__{self.variant.name}.parquet"


def selected_ids(exp: Experiment, task: str, limit: int | None) -> list[str]:
    sample = sampling.load(task, exp.sample)
    ids = (sample if exp.split == "all" else sample[sample.split == exp.split]).segment_id.tolist()
    # Deterministic pseudo-random order so a limit takes a representative slice, not one stratum.
    ids.sort(key=lambda s: hashlib.sha1(s.encode()).hexdigest())
    n = limit if limit is not None else exp.limit
    return ids[:n] if n is not None else ids


def plan(exp: Experiment, registry: Registry, limit: int | None = None) -> list[Unit]:
    units = []
    for task_name in exp.tasks:
        codebook = Codebook.load(CODEBOOKS / f"{task_name}.yaml")
        paraphrases = load_paraphrases(CODEBOOKS / f"{task_name}.yaml")
        task = Task.load(task_name)
        segments = task.to_segments(selected_ids(exp, task_name, limit))
        for coder in exp.coders:
            decision_model = coder in ("jev", "oaid") or (coder in registry.models and registry[coder].provider in systemone.PROVIDERS)
            if not decision_model and coder not in llm.CODERS:
                raise KeyError(f"unknown coder {coder!r}; known: decision models in configs/models.yaml, {sorted(llm.CODERS)}")
            for vname in exp.variants:
                variant = STANDARD_ARMS[vname]
                if variant.isolated and codebook.mode != "multi":
                    continue  # exclusive codebooks are already one question per call
                if not variant.context and not any(s.context for s in segments):
                    continue  # nothing to remove
                if variant.paraphrase is not None and not paraphrases:
                    raise FileNotFoundError(f"{vname} needs codebooks/paraphrases/{task_name}.yaml")
                if decision_model and variant.elicitation != "answer_prob":
                    continue  # elicitation prompts are LLM-only (v1.4)
                if decision_model:  # decision models: all get Jev's requests (oaid remapped to its API's types)
                    module = decisions if coder == "oaid" else jev
                    calls = module.plan_calls(segments, codebook, variant, paraphrases)
                    combine = codebook.mode == "exclusive" and variant.question_format == "noul"
                    units.append(Unit(exp.name, task_name, coder, variant, registry[coder], calls, len(segments), combine, exp.cache_tag))
                else:
                    if variant.question_format != "native" or variant.isolated:
                        continue  # format/isolation arms are Jev-specific
                    calls = llm.plan_calls(segments, codebook, variant, paraphrases)
                    units.append(Unit(exp.name, task_name, coder, variant, registry[llm.CODERS[coder].model], calls, len(segments),
                                      cache_tag=exp.cache_tag))
    return units


def estimate(units: list[Unit], cache: Cache) -> pd.DataFrame:
    rows = []
    for u in units:
        if u.coder == "oaid":
            n_todo, usd = decisions.estimate_usd(u.calls, u.spec, cache, u.cache_tag)
        elif u.spec.provider in systemone.PROVIDERS:
            n_todo, usd = systemone.estimate_usd(u.calls, u.spec, cache, u.cache_tag)
        elif u.llm is None:
            n_todo, usd = jev.estimate_usd(u.calls, u.spec, cache, u.cache_tag)
        else:
            n_todo, usd = llm.estimate_usd(u.calls, u.llm, u.spec, cache, u.cache_tag)
        rows.append(
            {"experiment": u.experiment, "task": u.task, "coder": u.coder, "variant": u.variant.name,
             "segments": u.n_segments, "calls": len(u.calls), "uncached": n_todo, "est_usd": usd}
        )
    return pd.DataFrame(rows)


def _write(df: pd.DataFrame, path: Path, coder: str, variant: str) -> None:
    df = df.copy()
    df.insert(0, "variant", variant)
    df.insert(0, "coder", coder)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


async def _execute_unit(u: Unit, cache: Cache, clients: llm.Clients) -> dict[str, Any]:
    if u.coder == "oaid":
        preds, failures, diags = await decisions.run(u.calls, u.spec, cache, client=clients.openai,
                                                     semaphore=clients.semaphore(u.spec), inflight=clients.inflight, tag=u.cache_tag)
        if u.combine_nouls:
            preds = jev.exactly_one(preds)
        _write(preds, u.out_path(), u.coder, u.variant.name)
        base = u.out_path()
        d = pd.DataFrame(diags)
        d.to_parquet(base.with_suffix(".diag.parquet"), index=False)
        info = {"rows": len(preds), "refused_questions": int(d.refused.sum()) if not d.empty else 0}
    elif u.spec.provider in systemone.PROVIDERS:
        preds, failures = await systemone.run(u.calls, u.spec, cache, client=clients.http_client(), semaphore=clients.semaphore(u.spec),
                                              inflight=clients.inflight, tag=u.cache_tag)
        if u.combine_nouls:
            preds = jev.exactly_one(preds)
        _write(preds, u.out_path(), u.coder, u.variant.name)
        info = {"rows": len(preds)}
        base = u.out_path()
    elif u.llm is None:
        preds, failures = await jev.run(u.calls, u.spec, cache, client=clients.typesafe_client(u.spec), semaphore=clients.semaphore(u.spec), inflight=clients.inflight, tag=u.cache_tag)
        if u.combine_nouls:
            preds = jev.exactly_one(preds)
        _write(preds, u.out_path(), u.coder, u.variant.name)
        info: dict[str, Any] = {"rows": len(preds)}
        base = u.out_path()
    else:
        by_signal, failures, diags = await llm.run(u.calls, u.llm, u.spec, cache, clients, tag=u.cache_tag)
        for signal, preds in by_signal.items():
            _write(preds, u.out_path(signal), f"{u.coder}-{signal}", u.variant.name)
        base = u.out_path(u.llm.signals[0])
        d = pd.DataFrame(diags)
        d.to_parquet(base.with_suffix(".diag.parquet"), index=False)
        info = {"rows": {s: len(p) for s, p in by_signal.items()}}
        if not d.empty:
            info["parse_ok"] = f"{int((d.parse_ok == d.samples).sum())}/{len(d)}"
            if "logprob_ok" in d:
                info["logprob_ok"] = f"{int(d.logprob_ok.fillna(False).sum())}/{len(d)}"
    failures_path = base.with_suffix(".failures.csv")
    if failures:
        pd.DataFrame(failures).to_csv(failures_path, index=False)
    else:
        failures_path.unlink(missing_ok=True)  # a rerun that filled the gaps clears the old record
    return {"experiment": u.experiment, "task": u.task, "coder": u.coder, "variant": u.variant.name,
            "failures": len(failures)} | info


async def execute_async(units: list[Unit], cache: Cache) -> list[dict[str, Any]]:
    clients = llm.Clients.from_env() if any(u.llm or u.coder == "oaid" for u in units) else llm.Clients()
    # Jev units share one client, created inside the running loop by clients.typesafe_client().
    try:
        return await asyncio.gather(*(_execute_unit(u, cache, clients) for u in units))
    finally:
        await clients.aclose()


def execute(units: list[Unit], cache: Cache) -> list[dict[str, Any]]:
    return asyncio.run(execute_async(units, cache))
