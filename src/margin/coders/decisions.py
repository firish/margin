"""OpenAI Decisions API coder, `oaid` (plan amendment v1.1, A1).

Every request is Jev's rendered request mapped to /v1/decisions, so the two decision models see the
same content, order, labels, and wording in every arm:
- Jev Choice -> `choice`; each option's description is Jev's code spec serialized as JSON text.
- Jev Noul -> `predicate`; Jev's true/false criteria are appended to the instructions as JSON text.
- Jev's state object -> `input`, as a JSON string.
A question the API refuses is recorded and left missing for that unit.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from margin.budget import ModelSpec, estimate_tokens
from margin.cache import Cache, request_key, tagged
from margin.coders import jev
from margin.perturb import Variant
from margin.render import JevCall
from margin.schema import Codebook, Segment

PROVIDER = "openai_decisions"


@dataclass
class DecisionCall:
    request: dict[str, Any]
    keymap: dict[str, str]
    """predicate question name -> code id; choice value (label) -> code id."""


def from_jev(call: JevCall) -> DecisionCall:
    questions = []
    for name, q in call.questions.items():
        if q["type"] == "choice":
            questions.append({
                "type": "choice", "name": name, "instructions": q["instructions"],
                "choices": [{"value": label, "description": json.dumps(spec, ensure_ascii=False)} for label, spec in q["criteria"].items()],
            })
        else:
            crit = q["criteria"]
            questions.append({
                "type": "predicate", "name": name,
                "instructions": q["instructions"]
                + "\nTrue when: " + json.dumps(crit["true"], ensure_ascii=False)
                + "\nFalse when: " + json.dumps(crit["false"], ensure_ascii=False),
            })
    return DecisionCall({"input": json.dumps(call.state, ensure_ascii=False), "questions": questions}, dict(call.keymap))


def plan_calls(
    segments: Sequence[Segment], codebook: Codebook, variant: Variant, paraphrases: dict[str, list[str]] | None = None
) -> list[tuple[Segment, DecisionCall]]:
    return [(seg, from_jev(c)) for seg, c in jev.plan_calls(segments, codebook, variant, paraphrases)]


def estimate_usd(calls: Sequence[tuple[Segment, DecisionCall]], spec: ModelSpec, cache: Cache, tag: str | None = None) -> tuple[int, float]:
    todo = [c for _, c in calls if cache.get(request_key(tagged(PROVIDER, tag), spec.model, c.request)) is None]
    return len(todo), spec.cost(sum(estimate_tokens(json.dumps(c.request, ensure_ascii=False)) for c in todo), 0)


def parse(call: DecisionCall, response: dict) -> tuple[list[dict[str, Any]], int]:
    """-> (rows of code_id/p, number of refused questions)."""
    rows, refused = [], 0
    for ans in response["answers"]:
        if ans["type"] == "refusal":
            refused += 1
        elif ans["type"] == "choice":
            rows += [{"code_id": call.keymap[pr["value"]], "p": float(pr["probability"])} for pr in ans["probabilities"]]
        elif ans["type"] == "predicate":
            rows.append({"code_id": call.keymap[ans["name"]], "p": float(ans["probability"])})
    return rows, refused


async def run(
    calls: Sequence[tuple[Segment, DecisionCall]],
    spec: ModelSpec,
    cache: Cache,
    *,
    client: Any,
    semaphore: asyncio.Semaphore | None = None,
    inflight: dict[str, asyncio.Future] | None = None,
    tag: str | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]], list[dict[str, Any]]]:
    """Answer every call (cache first). -> (predictions, failures, per-call diagnostics)."""
    sem = semaphore or asyncio.Semaphore(spec.concurrency)
    inflight = {} if inflight is None else inflight
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    diags: list[dict[str, Any]] = []

    async def one(seg: Segment, call: DecisionCall) -> None:
        key = request_key(tagged(PROVIDER, tag), spec.model, call.request)
        response = cache.get(key)
        if response is None and key in inflight:
            response = await inflight[key]
            if response is None:
                failures.append({"segment_id": seg.segment_id, "error": "shared in-flight request failed"})
                return
        elif response is None:
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            inflight[key] = future
            try:
                async with sem:
                    result = await client.decisions.create(model=spec.model, **call.request)
            except Exception as err:  # SDK errors after retries and transport errors; never cached
                failures.append({"segment_id": seg.segment_id, "error": f"{type(err).__name__}: {str(err)[:300]}"})
                future.set_result(None)
                return
            finally:
                inflight.pop(key, None)
            response = result.model_dump(mode="json")
            u = response.get("usage") or {}
            cache.put(key, provider=tagged(PROVIDER, tag), model=spec.model, request=call.request, response=response,
                      input_tokens=u.get("input_tokens"), output_tokens=u.get("output_tokens"),
                      cost_usd=spec.cost(u.get("input_tokens"), u.get("output_tokens")))
            future.set_result(response)
        parsed, refused = parse(call, response)
        diags.append({"segment_id": seg.segment_id, "refused": refused, "questions": len(call.request["questions"]),
                      "model_version": response.get("model")})
        rows.extend({"segment_id": seg.segment_id, **r, "model_version": response.get("model")} for r in parsed)

    await asyncio.gather(*(one(seg, call) for seg, call in calls))
    return pd.DataFrame(rows, columns=["segment_id", "code_id", "p", "model_version"]), failures, diags
