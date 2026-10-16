"""Other Jev-compatible ("System One") decision models over REST (plan amendment v1.2, A5).

Cloudflare Clef / Clef-flash (Workers AI), Perplexity pplx-decider, and Kev self-hosted on Modal all
take Jev's request shape (state + named noul/choice questions) and answer in Jev's response shape,
so every coder gets Jev's exact requests and `jev.parse` reads the answers. A provider that rejects
structured option specs gets them as JSON text (`stringify_criteria`, the oaid rule).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from collections.abc import Sequence
from typing import Any

import httpx
import pandas as pd

from margin.budget import ModelSpec, estimate_tokens
from margin.cache import Cache, request_key, tagged
from margin.coders import jev
from margin.render import JevCall
from margin.schema import Segment

PROVIDERS = ("cloudflare", "perplexity", "kev")
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}


def body(call: JevCall, spec: ModelSpec) -> dict[str, Any]:
    """The request as sent (and cached): Jev's state and questions, plus the provider's model field."""
    questions = call.questions
    if spec.params.get("stringify_criteria"):
        questions = {}
        for name, q in call.questions.items():
            q = dict(q)
            if q["type"] == "choice":
                q["criteria"] = {k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for k, v in q["criteria"].items()}
            elif isinstance(q.get("criteria"), dict):
                q["criteria"] = {k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for k, v in q["criteria"].items()}
            questions[name] = q
    return {"model": spec.params.get("request_model", spec.model), "state": call.state, "questions": questions}


def endpoint(spec: ModelSpec) -> tuple[str, dict[str, str]]:
    if spec.provider == "cloudflare":
        url = f"https://api.cloudflare.com/client/v4/accounts/{os.environ['CLOUDFLARE_ACCOUNT_ID']}/ai/run/{spec.model}"
        return url, {"Authorization": f"Bearer {os.environ['CLOUDFLARE_API_KEY']}"}
    if spec.provider == "perplexity":
        return "https://api.perplexity.ai/v1/decisions", {"Authorization": f"Bearer {os.environ['PERPLEXITY_API_KEY']}"}
    if spec.provider == "kev":
        base = os.environ[spec.params["url_env"]].rstrip("/")
        return f"{base}/v1/systemone", {"Authorization": f"Bearer {os.environ['KEV_API_KEY']}"}
    raise ValueError(f"not a System One provider: {spec.provider}")


def unwrap(spec: ModelSpec, payload: dict[str, Any]) -> dict[str, Any]:
    """Cloudflare wraps model output as {"result": ..., "success": ...}; the others return it directly."""
    if spec.provider == "cloudflare":
        if not payload.get("success", True) or "result" not in payload:
            raise RuntimeError(f"cloudflare error: {json.dumps(payload.get('errors'))[:300]}")
        return payload["result"]
    return payload


def input_tokens(response: dict[str, Any]) -> int | None:
    u = response.get("usage") or {}
    return u.get("input_tokens") or u.get("prompt_tokens")


def estimate_usd(calls: Sequence[tuple[Segment, JevCall]], spec: ModelSpec, cache: Cache, tag: str | None = None) -> tuple[int, float]:
    todo = [c for _, c in calls if cache.get(request_key(tagged(spec.provider, tag), spec.model, body(c, spec))) is None]
    return len(todo), spec.cost(sum(estimate_tokens(json.dumps(body(c, spec), ensure_ascii=False)) for c in todo), 0)


async def post(client: httpx.AsyncClient, spec: ModelSpec, request: dict[str, Any], attempts: int = 8) -> dict[str, Any]:
    url, headers = endpoint(spec)
    for attempt in range(attempts):
        try:
            r = await client.post(url, json=request, headers=headers)
        except (httpx.TransportError, httpx.TimeoutException):
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(min(60, 2**attempt) + random.random())
            continue
        if r.status_code in RETRY_STATUS and attempt < attempts - 1:
            wait = float(r.headers.get("retry-after", 0) or 0) or min(60, 2**attempt) + random.random()
            await asyncio.sleep(wait)
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        return unwrap(spec, r.json())
    raise RuntimeError("unreachable")


async def run(
    calls: Sequence[tuple[Segment, JevCall]],
    spec: ModelSpec,
    cache: Cache,
    *,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore | None = None,
    inflight: dict[str, asyncio.Future] | None = None,
    tag: str | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Answer every call (cache first). -> (predictions, failures); failures are never cached."""
    sem = semaphore or asyncio.Semaphore(spec.concurrency)
    inflight = {} if inflight is None else inflight
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    provider = tagged(spec.provider, tag)

    async def one(seg: Segment, call: JevCall) -> None:
        request = body(call, spec)
        key = request_key(provider, spec.model, request)
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
                    response = await post(client, spec, request)
                jev.parse(call, response)  # a malformed answer is a failure, not a cache entry
            except Exception as err:
                failures.append({"segment_id": seg.segment_id, "error": f"{type(err).__name__}: {str(err)[:300]}"})
                future.set_result(None)
                return
            finally:
                inflight.pop(key, None)
            tokens = input_tokens(response)
            cache.put(key, provider=provider, model=spec.model, request=request, response=response,
                      input_tokens=tokens, output_tokens=0, cost_usd=spec.cost(tokens, 0))
            future.set_result(response)
        for row in jev.parse(call, response):
            rows.append({"segment_id": seg.segment_id, **row, "model_version": response.get("model")})

    await asyncio.gather(*(one(seg, call) for seg, call in calls))
    return pd.DataFrame(rows, columns=["segment_id", "code_id", "p", "model_version"]), failures
