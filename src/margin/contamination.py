"""Memorization probe (plan amendment v1.3): did the LLM coders see our public datasets in training?

Completion probe (Golchin & Surdeanu's guided vs general instruction): the model gets the first half
of an item and continues it, once told the source dataset and asked for the exact original text
(guided), once just asked to continue (general). It is scored by word-level ROUGE-L and the longest
common run of words against the true second half. Label probe: the model names the label(s) the
dataset's annotators gave an item, with only the bare label names, with and without the dataset named.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from margin import sampling
from margin.analysis import RESULTS, TaskData
from margin.budget import Registry, estimate_tokens
from margin.cache import Cache, request_key
from margin.coders import llm
from margin.datasets.base import Task

SEED, N_ITEMS, PROBE = 0, 100, "memorization-v1"
MODELS = ["luna", "haiku", "sonnet", "oss", "qwen", "qwen_big"]
COMPLETION_SETS = {  # dataset: (tasks pooled, minimum words)
    "mfrc": (["mfrc"], 30), "annomi": (["annomi_client", "annomi_therapist"], 30),
    "talkmoves": (["talkmoves_teacher", "talkmoves_student"], 20), "dunivin": (["dunivin"], 30),
}
LABEL_TASKS = ["mfrc", "annomi_client", "annomi_therapist", "talkmoves_teacher", "talkmoves_student"]
DESC = {
    "mfrc": "the Moral Foundations Reddit Corpus (MFRC; Trager et al., 2022), a public dataset of Reddit comments annotated for moral foundations",
    "annomi": "AnnoMI (Wu et al., 2022), a public dataset of annotated motivational-interviewing counselling transcripts",
    "talkmoves": "the TalkMoves dataset (Suresh et al., 2022), public annotated transcripts of K-12 mathematics classrooms",
    "dunivin": "the public dataset from Dunivin (2025) on LLM-assisted qualitative coding, passages from news and opinion pieces about W. E. B. Du Bois",
}
TASK_DATASET = {"mfrc": "mfrc", "annomi_client": "annomi", "annomi_therapist": "annomi", "talkmoves_teacher": "talkmoves",
                "talkmoves_student": "talkmoves", "dunivin": "dunivin"}
VERBATIM_RUN, VERBATIM_SHARE = 10, 0.5


# --- items ----------------------------------------------------------------------------------------------

def words(text: str) -> list[str]:
    return text.split()


def norm(ws: list[str]) -> list[str]:
    return [w for w in (re.sub(r"[^\w']", "", x.lower()) for x in ws) if w]


def completion_items() -> pd.DataFrame:
    rows = []
    for ds, (tasks, min_words) in COMPLETION_SETS.items():
        pool = []
        for t in tasks:
            task = Task.load(t)
            for seg in task.to_segments(sampling.load(t, "main").segment_id.tolist()):
                w = words(seg.text)
                if len(w) >= min_words:
                    pool.append((t, seg, w))
        pick = np.random.default_rng(SEED).choice(len(pool), size=min(N_ITEMS, len(pool)), replace=False)
        for i in sorted(pick):
            t, seg, w = pool[i]
            half = len(w) // 2
            rows.append({"dataset": ds, "task": t, "segment_id": seg.segment_id, "context": seg.context or "",
                         "prefix": " ".join(w[:half]), "target": " ".join(w[half:])})
    return pd.DataFrame(rows)


def label_items() -> pd.DataFrame:
    rows = []
    for t in LABEL_TASKS:
        td = TaskData.load(t)
        task = Task.load(t)
        ids = sorted(set(td.ref.segment_id))
        pick = np.random.default_rng(SEED).choice(len(ids), size=min(N_ITEMS, len(ids)), replace=False)
        chosen = [ids[i] for i in sorted(pick)]
        labels = {c.id: c.label for c in td.codebook.codes} | ({td.codebook.none_option.id: td.codebook.none_option.label} if td.codebook.none_option else {})
        for seg in task.to_segments(chosen):
            if td.codebook.mode == "multi":
                ref = td.ref[(td.ref.segment_id == seg.segment_id) & (td.ref.y == 1)].code_id.tolist()
            else:
                ref = td.ref[td.ref.segment_id == seg.segment_id].ref_code.tolist()
            rows.append({"task": t, "dataset": TASK_DATASET[t], "segment_id": seg.segment_id, "text": seg.text, "context": seg.context or "",
                         "mode": td.codebook.mode, "labels": labels, "reference": ref})
    return pd.DataFrame(rows)


# --- prompts --------------------------------------------------------------------------------------------

def completion_prompt(item: dict, guided: bool) -> str:
    body = (f"Previous turn: {item['context']}\nUtterance: {item['prefix']}" if item["context"] else item["prefix"])
    if guided:
        return (f"The text below is the beginning of an item from {DESC[item['dataset']]}. Continue it with the exact original text of "
                f"that item as it appears in the dataset, word for word. Output only the continuation.\n\n{body}")
    return f"Continue the text below. Output only the continuation.\n\n{body}"


def label_prompt(item: dict, guided: bool) -> str:
    options = "; ".join(item["labels"].values())
    body = f"Previous turn: {item['context']}\nItem: {item['text']}" if item["context"] else f"Item: {item['text']}"
    ask = ("Which of these labels apply? Answer with the label names only, separated by semicolons, or 'none'." if item["mode"] == "multi"
           else "Which one label applies? Answer with the label name only.")
    if guided:
        return (f"This item is from {DESC[item['dataset']]}. In that dataset, which label(s) did the human annotators assign to this exact item? "
                f"Options: {options}. {ask}\n\n{body}")
    return f"Options: {options}. {ask}\n\n{body}"


# --- calls ----------------------------------------------------------------------------------------------

@dataclass
class Call:
    kind: str
    condition: str
    item: dict
    prompt: str
    max_tokens: int


def request(call: Call, coder: llm.LLMCoder, spec) -> dict[str, Any]:
    allowance = spec.reasoning_allowance * 4
    return {"probe": PROBE, "user": call.prompt, "max_tokens": call.max_tokens + allowance, "temperature": coder.temperature, "params": spec.params}


async def complete_text(clients: llm.Clients, spec, req: dict) -> dict:
    """Plain-text completion with the model's coding settings (no JSON schema)."""
    if spec.provider == "anthropic":
        kw: dict[str, Any] = {}
        if "thinking" in req["params"]:
            kw["thinking"] = {"type": req["params"]["thinking"]}
        if "effort" in req["params"]:
            kw["output_config"] = {"effort": req["params"]["effort"]}
        if req["temperature"] is not None:
            kw["extra_body"] = {"temperature": req["temperature"]}
        m = await clients.anthropic.messages.create(model=spec.model, max_tokens=req["max_tokens"],
                                                    messages=[{"role": "user", "content": req["user"]}], **kw)
        text = next((b.text for b in m.content if b.type == "text"), "")
        u = m.usage
        return {"text": text, "model": m.model, "usage": {"input": u.input_tokens, "cache_read": u.cache_read_input_tokens or 0,
                                                          "cache_write": u.cache_creation_input_tokens or 0, "output": u.output_tokens}}
    client = clients.openai if spec.provider == "openai" else clients.together
    kw = {"model": spec.model, "messages": [{"role": "user", "content": req["user"]}]}
    kw["max_completion_tokens" if spec.provider == "openai" else "max_tokens"] = req["max_tokens"]
    if req["temperature"] is not None:
        kw["temperature"] = req["temperature"]
    kw.update(req["params"])
    r = await client.chat.completions.create(**kw)
    u = r.usage
    cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    return {"text": r.choices[0].message.content or "", "model": r.model,
            "usage": {"input": u.prompt_tokens - cached, "cache_read": cached, "cache_write": 0, "output": u.completion_tokens}}


def plan() -> list[Call]:
    calls = []
    for item in completion_items().to_dict("records"):
        n = len(words(item["target"]))
        for cond in ("guided", "general"):
            calls.append(Call("completion", cond, item, completion_prompt(item, cond == "guided"), min(400, 2 * n + 20)))
    for item in label_items().to_dict("records"):
        for cond in ("guided", "general"):
            calls.append(Call("label", cond, item, label_prompt(item, cond == "guided"), 40))
    return calls


def estimate(models: list[str] = MODELS) -> pd.DataFrame:
    reg, cache, calls = Registry.load(), Cache(), plan()
    rows = []
    for name in models:
        coder = llm.CODERS[name]
        spec = reg[coder.model]
        todo = [c for c in calls if cache.get(request_key(spec.provider, spec.model, request(c, coder, spec))) is None]
        tin = sum(estimate_tokens(c.prompt) for c in todo)
        tout = sum(min(c.max_tokens, 60 if c.kind == "completion" else 8) + spec.reasoning_allowance for c in todo)
        rows.append({"model": name, "calls": len(calls), "uncached": len(todo), "est_usd": spec.cost(tin, tout)})
    return pd.DataFrame(rows)


async def run_async(models: list[str] = MODELS) -> pd.DataFrame:
    reg, cache, calls = Registry.load(), Cache(), plan()
    clients = llm.Clients.from_env()
    rows: list[dict] = []

    async def one(name: str, call: Call) -> None:
        coder = llm.CODERS[name]
        spec = reg[coder.model]
        req = request(call, coder, spec)
        key = request_key(spec.provider, spec.model, req)
        resp = cache.get(key)
        if resp is None:
            try:
                async with clients.semaphore(spec):
                    resp = await complete_text(clients, spec, req)
            except Exception as err:
                rows.append({"model": name, "kind": call.kind, "condition": call.condition, "error": f"{type(err).__name__}: {str(err)[:200]}"})
                return
            u = resp["usage"]
            cache.put(key, provider=spec.provider, model=spec.model, request=req, response=resp,
                      input_tokens=u["input"] + u["cache_read"] + u["cache_write"], output_tokens=u["output"],
                      cost_usd=spec.cost(u["input"], u["output"], u["cache_read"], u["cache_write"]))
        rows.append({"model": name, "kind": call.kind, "condition": call.condition, **score(call, resp["text"])})

    try:
        await asyncio.gather(*(one(m, c) for m in models for c in calls))
    finally:
        await clients.aclose()
    return pd.DataFrame(rows)


# --- scoring --------------------------------------------------------------------------------------------

def lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def longest_run(a: list[str], b: list[str]) -> int:
    best, prev = 0, [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else 0)
        best = max(best, max(cur))
        prev = cur
    return best


def score(call: Call, text: str) -> dict[str, Any]:
    item = call.item
    base = {"dataset": item["dataset"], "task": item["task"], "segment_id": item["segment_id"]}
    if call.kind == "completion":
        pred, ref = norm(words(text)), norm(words(item["target"]))
        m = lcs(pred, ref)
        p, r = (m / len(pred) if pred else 0.0), (m / len(ref) if ref else 0.0)
        run = longest_run(pred, ref)
        return base | {"rouge_l": 2 * p * r / (p + r) if p + r else 0.0, "longest_run": run, "target_words": len(ref),
                       "verbatim": bool(run >= VERBATIM_RUN and run >= VERBATIM_SHARE * len(ref))}
    answer = text.lower()
    chosen = {cid for cid, lab in item["labels"].items() if lab.lower() in answer}
    if item["mode"] == "multi":
        ref = set(item["reference"])
        acc = 1.0 if not chosen and not ref else len(chosen & ref) / len(chosen | ref)
    else:
        # exclusive: the first label mentioned counts (models sometimes restate options)
        hits = sorted((answer.find(lab.lower()), cid) for cid, lab in item["labels"].items() if lab.lower() in answer)
        acc = float(bool(hits) and hits[0][1] == item["reference"][0])
    return base | {"label_accuracy": acc}


DECOY = {
    "mfrc": "the Hartwell Online Morality Corpus (HOMC; Hartwell et al., 2021), a public dataset of Reddit comments annotated for moral foundations",
    "annomi": "the Brennan Counselling Dialogue Set (BCDS; Brennan et al., 2021), a public dataset of annotated motivational-interviewing counselling transcripts",
}
"""Deviation 14: fictitious names with the real description, to separate memory from task framing."""
FLAGGED = [("haiku", "mfrc"), ("luna", "mfrc"), ("oss", "annomi_therapist")]


def decoy_prompt(item: dict) -> str:
    real = DESC[item["dataset"]]
    try:
        DESC[item["dataset"]] = DECOY[item["dataset"]]
        return label_prompt(item, guided=True)
    finally:
        DESC[item["dataset"]] = real


async def decoy_async(cells: list[tuple[str, str]] = FLAGGED) -> pd.DataFrame:
    reg, cache = Registry.load(), Cache()
    items = label_items()
    clients = llm.Clients.from_env()
    rows: list[dict] = []

    async def one(name: str, item: dict) -> None:
        coder = llm.CODERS[name]
        spec = reg[coder.model]
        call = Call("label", "decoy", item, decoy_prompt(item), 40)
        req = request(call, coder, spec)
        key = request_key(spec.provider, spec.model, req)
        resp = cache.get(key)
        if resp is None:
            async with clients.semaphore(spec):
                resp = await complete_text(clients, spec, req)
            u = resp["usage"]
            cache.put(key, provider=spec.provider, model=spec.model, request=req, response=resp,
                      input_tokens=u["input"] + u["cache_read"] + u["cache_write"], output_tokens=u["output"],
                      cost_usd=spec.cost(u["input"], u["output"], u["cache_read"], u["cache_write"]))
        rows.append({"model": name, "kind": "label", "condition": "decoy", **score(call, resp["text"])})

    try:
        await asyncio.gather(*(one(m, it) for m, t in cells for it in items[items.task == t].to_dict("records")))
    finally:
        await clients.aclose()
    return pd.DataFrame(rows)


def decoy_items_from_cache(cells: list[tuple[str, str]] = FLAGGED) -> pd.DataFrame:
    """The decoy control's item-level scores, rebuilt from the local response cache only (no API calls; a cache miss
    is an error). The output is text-free: model, task, segment id, condition and label accuracy."""
    reg, cache = Registry.load(), Cache()
    items = label_items()
    rows = []
    for name, task in cells:
        coder = llm.CODERS[name]
        spec = reg[coder.model]
        for item in items[items.task == task].to_dict("records"):
            call = Call("label", "decoy", item, decoy_prompt(item), 40)
            resp = cache.get(request_key(spec.provider, spec.model, request(call, coder, spec)))
            if resp is None:
                raise RuntimeError(f"decoy response for {name}/{item['segment_id']} is not in the cache; this function makes no calls")
            rows.append({"model": name, "kind": "label", "condition": "decoy", **score(call, resp["text"])})
    return pd.DataFrame(rows)


def decoy_from_items() -> pd.DataFrame:
    """Recompute results/contamination_decoy.csv from the released item-level scores (no cache needed)."""
    items = pd.read_csv(RESULTS / "contamination_items.csv")
    decoy = pd.read_csv(RESULTS / "contamination_decoy_items.csv")
    out = decoy_summary(items, decoy)
    out.to_csv(RESULTS / "contamination_decoy.csv", index=False)
    return out


def decoy_summary(items_scores: pd.DataFrame, decoy: pd.DataFrame) -> pd.DataFrame:
    lab = pd.concat([items_scores[items_scores.kind == "label"], decoy])
    rows = []
    for m, t in FLAGGED:
        w = lab[(lab.model == m) & (lab.task == t)].pivot_table(index="segment_id", columns="condition", values="label_accuracy").dropna()
        if t == "mfrc":
            d = w.guided - w.decoy
            p = wilcoxon(w.guided, w.decoy, alternative="greater").pvalue if (d != 0).any() else 1.0
        else:
            b, c = int(((w.guided == 1) & (w.decoy == 0)).sum()), int(((w.guided == 0) & (w.decoy == 1)).sum())
            p = binomtest(b, b + c, 0.5, alternative="greater").pvalue if b + c else 1.0
        rows.append({"model": m, "task": t, "items": len(w), "real name": w.guided.mean(), "decoy name": w.decoy.mean(),
                     "no name": w.general.mean(), "real − decoy": (w.guided - w.decoy).mean(), "p (real > decoy)": p,
                     "p (Bonferroni, 3 cells)": min(1.0, 3 * p)})
    return pd.DataFrame(rows)


# --- analysis -------------------------------------------------------------------------------------------

def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    ok = scores[scores.get("error").isna()] if "error" in scores else scores
    comp, lab = ok[ok.kind == "completion"], ok[ok.kind == "label"]
    n_comp = comp.groupby(["model", "dataset"]).ngroups
    n_lab = lab.groupby(["model", "task"]).ngroups
    rows = []
    for (m, ds), g in comp.groupby(["model", "dataset"]):
        w = g.pivot_table(index="segment_id", columns="condition", values="rouge_l").dropna()
        d = w.guided - w.general
        p = wilcoxon(w.guided, w.general, alternative="greater").pvalue if (d != 0).any() else 1.0
        v = g[g.condition == "guided"].verbatim.sum()
        rows.append({"probe": "completion", "model": m, "set": ds, "items": len(w), "guided": w.guided.mean(), "general": w.general.mean(),
                     "difference": d.mean(), "p": p, "p_bonferroni": min(1.0, p * n_comp),
                     "verbatim_guided": int(v), "verbatim_general": int(g[g.condition == "general"].verbatim.sum()),
                     "max_run_guided": int(g[g.condition == "guided"].longest_run.max()),
                     "flag": bool(min(1.0, p * n_comp) < 0.05 or v >= 2)})
    for (m, t), g in lab.groupby(["model", "task"]):
        w = g.pivot_table(index="segment_id", columns="condition", values="label_accuracy").dropna()
        if g.task.iloc[0] == "mfrc":
            d = w.guided - w.general
            p = wilcoxon(w.guided, w.general, alternative="greater").pvalue if (d != 0).any() else 1.0
        else:
            b, c = int(((w.guided == 1) & (w.general == 0)).sum()), int(((w.guided == 0) & (w.general == 1)).sum())
            p = binomtest(b, b + c, 0.5, alternative="greater").pvalue if b + c else 1.0
        rows.append({"probe": "label", "model": m, "set": t, "items": len(w), "guided": w.guided.mean(), "general": w.general.mean(),
                     "difference": (w.guided - w.general).mean(), "p": p, "p_bonferroni": min(1.0, p * n_lab),
                     "flag": bool(min(1.0, p * n_lab) < 0.05)})
    return pd.DataFrame(rows)


def run(models: list[str] = MODELS) -> dict[str, pd.DataFrame]:
    scores = asyncio.run(run_async(models))
    summary = summarize(scores)
    scores.to_csv(RESULTS / "contamination_items.csv", index=False)
    summary.to_csv(RESULTS / "contamination_summary.csv", index=False)
    return {"contamination_items": scores, "contamination_summary": summary}
