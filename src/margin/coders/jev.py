"""Jev coder: one System One request per segment (or per code, in the isolated arm)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import pandas as pd
from typesafe_sdk import AsyncTypeSafeClient

from margin.budget import ModelSpec, estimate_tokens
from margin.cache import Cache, request_key, tagged
from margin.perturb import Variant, apply_labels_and_wording
from margin.render import JevCall, jev_calls
from margin.schema import Codebook, Segment

PROVIDER = "typesafe"


def plan_calls(
    segments: Sequence[Segment], codebook: Codebook, variant: Variant, paraphrases: dict[str, list[str]] | None = None
) -> list[tuple[Segment, JevCall]]:
    rendered = apply_labels_and_wording(codebook, variant, paraphrases)
    return [(seg, call) for seg in segments for call in jev_calls(rendered, seg, variant)]


def estimate_usd(calls: Sequence[tuple[Segment, JevCall]], spec: ModelSpec, cache: Cache, tag: str | None = None) -> tuple[int, float]:
    """(uncached call count, estimated USD for them)."""
    todo = [c for _, c in calls if cache.get(request_key(tagged(PROVIDER, tag), spec.model, c.request)) is None]
    tokens = sum(estimate_tokens(json.dumps(c.request, ensure_ascii=False)) for c in todo)
    return len(todo), spec.cost(tokens, 0)


def parse(call: JevCall, response: dict[str, Any]) -> list[dict[str, Any]]:
    answers = response["answers"]
    if "code" in answers and answers["code"]["type"] == "choice":
        probs = answers["code"]["probabilities"]
        return [{"code_id": call.keymap[label], "p": float(p)} for label, p in probs.items()]
    return [{"code_id": call.keymap[key], "p": float(ans["noul"])} for key, ans in answers.items()]


CLIP = 0.005
"""Jev rounds to 0.01, so 0 and 1 are clipped to the rounding interval's midpoint to keep odds finite."""


def exactly_one(preds: pd.DataFrame, clip: float = CLIP) -> pd.DataFrame:
    """Combine independent per-option Noul probabilities into one distribution for an exclusive
    codebook: P(option k | exactly one option applies) = odds_k / sum_j odds_j, assuming the Nouls
    are independent. The raw Noul probability is kept as p_raw."""
    out = preds.rename(columns={"p": "p_raw"})
    q = out.p_raw.clip(clip, 1 - clip)
    odds = q / (1 - q)
    out["p"] = odds / odds.groupby(out.segment_id).transform("sum")
    return out


async def run(
    calls: Sequence[tuple[Segment, JevCall]],
    spec: ModelSpec,
    cache: Cache,
    *,
    client: AsyncTypeSafeClient | None = None,
    semaphore: asyncio.Semaphore | None = None,
    inflight: dict[str, asyncio.Future] | None = None,
    tag: str | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Answer every call (cache first). Returns (predictions, failures); failures are not cached.
    When several units run at once, pass a shared `semaphore` (one global limit) and `inflight`
    (identical concurrent requests share one call)."""
    sem = semaphore or asyncio.Semaphore(spec.concurrency)
    inflight = {} if inflight is None else inflight
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    own_client = client is None
    client = client or AsyncTypeSafeClient(model=spec.model)

    async def one(seg: Segment, call: JevCall) -> None:
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
                    result = await client.system_one(call.state, call.questions, model=spec.model)
            except Exception as err:  # SDK errors after retries, and raw transport errors (e.g. ssl.SSLError)
                failures.append({"segment_id": seg.segment_id, "error": f"{type(err).__name__}: {str(err)[:300]}"})
                future.set_result(None)
                return
            finally:
                inflight.pop(key, None)
            response = result.model_dump(mode="json")
            usage = response.get("usage") or {}
            cache.put(
                key,
                provider=tagged(PROVIDER, tag),
                model=spec.model,
                request=call.request,
                response=response,
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                cost_usd=spec.cost(usage.get("input_tokens"), usage.get("output_tokens")),
            )
            future.set_result(response)
        for row in parse(call, response):
            rows.append({"segment_id": seg.segment_id, **row, "model_version": response.get("model")})

    try:
        await asyncio.gather(*(one(seg, call) for seg, call in calls))
    finally:
        if own_client:
            await client.aclose()
    return pd.DataFrame(rows, columns=["segment_id", "code_id", "p", "model_version"]), failures
