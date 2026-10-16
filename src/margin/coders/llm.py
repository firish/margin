"""LLM baseline coders (RQ1 comparison signals).

One schema-constrained JSON request per segment returns, for every decision, a one-sentence rationale,
the answer, and a verbalized probability (0-100). Where the provider exposes token logprobs (OpenAI with
reasoning off, Together), the answer token's top alternatives give a second signal from the same call.
Self-consistency samples n answers at temperature 1 and uses vote shares.

Codebook content is identical to Jev's (`render.code_spec`); options are lettered (exclusive) or keyed
c1..cN (multi) so keys never leak labels and logprob positions are easy to locate.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import string
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from margin.budget import ModelSpec, estimate_tokens
from margin.cache import Cache, request_key, tagged
from margin.perturb import Variant, apply_labels_and_wording, order_options
from margin.render import code_spec, scope_note, state_for
from margin.schema import Code, Codebook, Segment

LETTERS = string.ascii_uppercase


@dataclass(frozen=True)
class LLMCoder:
    name: str
    model: str
    """Registry key in configs/models.yaml."""
    samples: int = 1
    temperature: float | None = 0.0
    """None leaves the provider default (Sonnet 5.5 rejects sampling parameters)."""
    logprobs: bool = False

    @property
    def signals(self) -> list[str]:
        if self.samples > 1:
            return ["vote"]
        return ["verbal"] + (["logprob"] if self.logprobs else [])


CODERS = {
    c.name: c
    for c in [
        LLMCoder("luna", "luna", logprobs=True),
        LLMCoder("luna_sc", "luna", samples=5, temperature=1.0),
        LLMCoder("haiku", "haiku"),
        LLMCoder("sonnet", "sonnet", temperature=None),
        LLMCoder("oss", "gpt_oss", logprobs=True),
        LLMCoder("qwen", "qwen", logprobs=True),  # amendment v1.1 (A2)
        LLMCoder("qwen_big", "qwen_big", logprobs=True),
    ]
}


# --- prompt ---------------------------------------------------------------------------------------

def describe(code: Code, indent: str = "   ") -> str:
    spec = code_spec(code)
    lines = [f"{indent}Definition: {spec['definition']}"]
    for key, title in (("applies_when", "Applies when"), ("does_not_apply_when", "Does not apply when"),
                       ("examples", "Examples"), ("non_examples", "Non-examples")):
        if key in spec:
            lines.append(f"{indent}{title}:")
            lines += [f"{indent}- {item}" for item in spec[key]]
    return "\n".join(lines)


@dataclass
class LLMCall:
    system: str
    user: str
    schema: dict[str, Any]
    keymap: dict[str, str]
    """exclusive: option letter -> code id. multi: code key (c1..) -> code id."""
    mode: str

    @property
    def request(self) -> dict[str, Any]:
        return {"system": self.system, "user": self.user, "schema": self.schema}


def render(codebook: Codebook, segment: Segment, variant: Variant) -> LLMCall:
    """`codebook` must already have the variant's labels/wording applied."""
    options = order_options(codebook.all_options, variant, segment.segment_id)
    head = (
        "You are an experienced qualitative researcher applying a fixed codebook (deductive coding). "
        "Apply the codebook exactly as written.\n\n"
        f"Task: {codebook.instructions}\nUnit of analysis: one {codebook.unit}.\n\n"
    )
    dist = variant.elicitation == "distribution"
    if codebook.mode == "exclusive" and dist:
        keys = [LETTERS[i] for i in range(len(options))]
        book = "Options (exactly one applies):\n" + "\n".join(
            f"{k}) {o.label}\n{describe(o)}" for k, o in zip(keys, options)
        )
        fmt = (
            "Respond with JSON only. \"rationale\": one sentence explaining your decision. \"probabilities\": for every "
            "option letter, an integer from 0 to 100, your probability that this option is the one that applies; the "
            "probabilities must sum to 100."
        )
        schema = {
            "type": "object",
            "properties": {
                "rationale": {"type": "string"},
                "probabilities": {"type": "object", "properties": {k: {"type": "integer"} for k in keys}, "required": keys,
                                  "additionalProperties": False},
            },
            "required": ["rationale", "probabilities"],
            "additionalProperties": False,
        }
    elif codebook.mode == "exclusive":
        keys = [LETTERS[i] for i in range(len(options))]
        book = "Options (exactly one applies):\n" + "\n".join(
            f"{k}) {o.label}\n{describe(o)}" for k, o in zip(keys, options)
        )
        fmt = (
            "Respond with JSON only. \"rationale\": one sentence explaining your decision. \"answer\": the "
            "letter of the option that applies. \"probability\": an integer from 0 to 100, your probability "
            "that this answer is correct."
        )
        schema = {
            "type": "object",
            "properties": {
                "rationale": {"type": "string"},
                "answer": {"type": "string", "enum": keys},
                "probability": {"type": "integer"},
            },
            "required": ["rationale", "answer", "probability"],
            "additionalProperties": False,
        }
    else:
        keys = [f"c{i + 1}" for i in range(len(options))]
        book = "Codes (any number may apply, including none):\n" + "\n".join(
            f"[{k}] {o.label}\n{describe(o)}" for k, o in zip(keys, options)
        )
        if dist:
            fmt = (
                "Respond with JSON only, with one entry per code key (" + ", ".join(keys) + "). For each code: "
                "\"rationale\": one sentence. \"probability\": an integer from 0 to 100, your probability that "
                "the code applies (50 or more means it applies)."
            )
            entry = {
                "type": "object",
                "properties": {"rationale": {"type": "string"}, "probability": {"type": "integer"}},
                "required": ["rationale", "probability"],
                "additionalProperties": False,
            }
        else:
            fmt = (
                "Respond with JSON only, with one entry per code key (" + ", ".join(keys) + "). For each code: "
                "\"rationale\": one sentence. \"applies\": \"yes\" or \"no\". \"probability\": an integer from 0 "
                "to 100, your probability that the code applies."
            )
            entry = {
                "type": "object",
                "properties": {
                    "rationale": {"type": "string"},
                    "applies": {"type": "string", "enum": ["yes", "no"]},
                    "probability": {"type": "integer"},
                },
                "required": ["rationale", "applies", "probability"],
                "additionalProperties": False,
            }
        schema = {"type": "object", "properties": {k: entry for k in keys}, "required": keys, "additionalProperties": False}
    state = state_for(segment, variant)
    parts = []
    if "setting" in state:
        parts.append(f"setting: {state['setting']}")
    if "previous_turn" in state:
        parts.append(f"previous_turn (context only, not coded):\n{state['previous_turn']}")
    parts.append(f"segment:\n{state['segment']}")
    parts.append(scope_note(codebook, variant, segment))
    return LLMCall(head + book + "\n\n" + fmt, "\n\n".join(parts), schema, dict(zip(keys, (o.id for o in options))), codebook.mode)


def plan_calls(
    segments: Sequence[Segment], codebook: Codebook, variant: Variant, paraphrases: dict[str, list[str]] | None = None
) -> list[tuple[Segment, LLMCall]]:
    rendered = apply_labels_and_wording(codebook, variant, paraphrases)
    return [(seg, render(rendered, seg, variant)) for seg in segments]


# --- requests ------------------------------------------------------------------------------------

def payload(call: LLMCall, coder: LLMCoder, spec: ModelSpec) -> dict[str, Any]:
    """Everything that determines the response; this is what the cache key hashes."""
    max_tokens = 400 if call.mode == "exclusive" else 120 * len(call.keymap) + 100
    return call.request | {
        "samples": coder.samples,
        "temperature": coder.temperature,
        "logprobs": coder.logprobs,
        "max_tokens": max_tokens + spec.reasoning_allowance * 4,
        "params": spec.params,
    }


def expected_output_tokens(call: LLMCall) -> int:
    return 80 if call.mode == "exclusive" else 40 * len(call.keymap) + 20


def estimate_usd(calls: Sequence[tuple[Segment, LLMCall]], coder: LLMCoder, spec: ModelSpec, cache: Cache,
                 tag: str | None = None) -> tuple[int, float]:
    todo = [c for _, c in calls if cache.get(request_key(tagged(spec.provider, tag), spec.model, payload(c, coder, spec))) is None]
    tin = sum(estimate_tokens(c.system + c.user) for c in todo)
    tout = sum((expected_output_tokens(c) + spec.reasoning_allowance) * coder.samples for c in todo)
    return len(todo), spec.cost(tin, tout)


# --- parsing -------------------------------------------------------------------------------------

def clamp01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def verbal(data: dict, call: LLMCall) -> list[tuple[str, float]]:
    """(code_id, p) rows. exclusive: the chosen option gets probability/100 (kept the argmax), the rest
    share the remainder. multi: the decision is the stated "applies" and the confidence is the stated
    probability's distance from 50, so p(applies) = c if yes else 1 - c with c = max(q, 1 - q).
    Models differ in what they report for a yes/no code even under the same instruction: most give
    P(code applies), Luna gives P(its answer is right) (86% of its pairs disagree under the first
    reading). This mapping is identical to the first reading for coherent answers and reading-agnostic
    otherwise; it uses no labels."""
    if call.mode == "exclusive" and "probabilities" in data:  # v1.4 distribution prompt
        raw = {call.keymap[k]: max(float(data["probabilities"].get(k, 0)), 0.0) for k in call.keymap}
        total = sum(raw.values())
        return [(cid, v / total if total > 0 else 1 / len(raw)) for cid, v in raw.items()]
    if call.mode == "multi" and all("applies" not in data[k] for k in call.keymap):  # v1.4: P(applies) directly
        return [(cid, clamp01(data[key]["probability"] / 100)) for key, cid in call.keymap.items()]
    if call.mode == "exclusive":
        k = len(call.keymap)
        chosen = call.keymap[data["answer"]]
        p = max(clamp01(data["probability"] / 100), 1 / k + 1e-6)
        rest = (1 - p) / (k - 1)
        return [(cid, p if cid == chosen else rest) for cid in call.keymap.values()]
    out = []
    for key, cid in call.keymap.items():
        q = clamp01(data[key]["probability"] / 100)
        c = max(q, 1 - q)
        out.append((cid, c if data[key]["applies"] == "yes" else 1 - c))
    return out


def token_spans(tokens: list[dict]) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for t in tokens:
        spans.append((pos, pos + len(t["t"])))
        pos += len(t["t"])
    return spans


def alternatives_at(tokens: list[dict], text_pos: int) -> list[tuple[str, float]] | None:
    """Top alternatives (token, prob) of the token that covers character `text_pos`. A null logprob
    (Together marks tokens masked by the JSON grammar this way) is probability 0."""
    for (start, end), t in zip(token_spans(tokens), tokens):
        if start <= text_pos < end:
            top = t.get("top") or [[t["t"], t["lp"]]]
            return [(tok, 0.0 if lp is None else math.exp(lp)) for tok, lp in top]
    return None


def clean(tok: str) -> str:
    return tok.strip().strip('"').strip()


def logprob_signal(text: str, tokens: list[dict], call: LLMCall) -> list[tuple[str, float]] | None:
    """Answer-token probabilities. exclusive: a distribution over options (mass of options missing from
    the top list is split evenly). multi: P(yes) per code from the "applies" token. None if not found.
    Some providers (Together + gpt-oss) prepend reasoning-channel tokens; the answer text is then located
    as the last occurrence inside the token stream and positions are shifted accordingly."""
    joined = "".join(t["t"] for t in tokens)
    offset = 0 if joined == text else joined.rfind(text)
    if offset < 0:
        return None
    if call.mode == "exclusive":
        m = re.search(r'"answer"\s*:\s*"', text)
        alts = alternatives_at(tokens, offset + m.end()) if m else None
        if not alts:
            return None
        mass: dict[str, float] = {}
        for tok, p in alts:
            if clean(tok) in call.keymap:
                mass[clean(tok)] = mass.get(clean(tok), 0.0) + p
        missing = [k for k in call.keymap if k not in mass]
        # Options absent from the top list have at most what the whole list leaves over (non-option
        # tokens such as a stray quote keep their own mass and are then normalized away).
        residual = max(0.0, 1.0 - sum(p for _, p in alts))
        for k in missing:
            mass[k] = residual / len(missing)
        total = sum(mass.values())
        return [(call.keymap[k], mass[k] / total) for k in call.keymap]
    out, cursor = [], 0
    for key in call.keymap:
        block = re.compile(rf'"{key}"\s*:\s*\{{').search(text, cursor)
        if not block:
            return None
        m = re.compile(r'"applies"\s*:\s*"').search(text, block.end())
        alts = alternatives_at(tokens, offset + m.end()) if m else None
        if not alts:
            return None
        yes = sum(p for tok, p in alts if clean(tok).lower().startswith("y"))
        no = sum(p for tok, p in alts if clean(tok).lower().startswith("n"))
        residual = max(1e-12, 1.0 - sum(p for _, p in alts))
        yes, no = (yes or residual), (no or residual)
        out.append((call.keymap[key], yes / (yes + no)))
        cursor = m.end()
    return out


def votes(datas: list[dict], call: LLMCall) -> list[tuple[str, float]]:
    """Self-consistency: share of samples choosing each option (exclusive) / saying the code applies."""
    n = len(datas)
    if call.mode == "exclusive":
        chosen = [call.keymap[d["answer"]] for d in datas]
        return [(cid, chosen.count(cid) / n) for cid in call.keymap.values()]
    return [(call.keymap[k], sum(d[k]["applies"] == "yes" for d in datas) / n) for k in call.keymap]


def parse(response: dict, call: LLMCall, coder: LLMCoder) -> tuple[dict[str, list[tuple[str, float]]], dict]:
    """-> ({signal: [(code_id, p)]}, diagnostics). Signals that cannot be read are omitted."""
    datas = []
    for text in response["texts"]:
        try:
            datas.append(json.loads(text))
        except (json.JSONDecodeError, TypeError):
            datas.append(None)
    diag = {"parse_ok": sum(d is not None for d in datas), "samples": len(datas)}
    out: dict[str, list[tuple[str, float]]] = {}
    good = [d for d in datas if d is not None]
    try:
        if coder.samples > 1:
            if good:
                out["vote"] = votes(good, call)
        elif good:
            out["verbal"] = verbal(good[0], call)
            if call.mode == "multi" and all("applies" in good[0][k] for k in call.keymap):
                diag["incoherent"] = sum((good[0][k]["applies"] == "yes") != (good[0][k]["probability"] >= 50) for k in call.keymap)
            if coder.logprobs and response.get("logprobs") and response["logprobs"][0]:
                lp = logprob_signal(response["texts"][0], response["logprobs"][0], call)
                if lp is not None:
                    out["logprob"] = lp
                diag["logprob_ok"] = lp is not None
    except (KeyError, TypeError) as err:
        diag["error"] = f"{type(err).__name__}: {err}"
        out = {}
    return out, diag


# --- providers -----------------------------------------------------------------------------------

@dataclass
class Clients:
    openai: Any = None
    together: Any = None
    anthropic: Any = None
    semaphores: dict[str, asyncio.Semaphore] = field(default_factory=dict)
    inflight: dict[str, asyncio.Future] = field(default_factory=dict)
    """Identical requests issued concurrently by different units share one API call."""
    typesafe: Any = None
    """One Jev client for all units (one connection pool and TLS context)."""
    http: Any = None
    """One httpx client for the REST System One providers (amendment v1.2)."""

    def http_client(self) -> Any:
        if self.http is None:
            import httpx

            # Kev on Modal: a cold start can take minutes and answers with a 303 to a result URL after 150 s.
            self.http = httpx.AsyncClient(timeout=httpx.Timeout(900, connect=30), follow_redirects=True,
                                          limits=httpx.Limits(max_connections=128, max_keepalive_connections=64))
        return self.http

    def typesafe_client(self, spec: ModelSpec) -> Any:
        if self.typesafe is None:
            from typesafe_sdk import AsyncTypeSafeClient

            self.typesafe = AsyncTypeSafeClient(model=spec.model)
        return self.typesafe

    def semaphore(self, spec: ModelSpec) -> asyncio.Semaphore:
        if spec.name not in self.semaphores:
            self.semaphores[spec.name] = asyncio.Semaphore(spec.concurrency)
        return self.semaphores[spec.name]

    @classmethod
    def from_env(cls) -> Clients:
        import os

        import anthropic
        import openai

        return cls(
            # 10 retries: the SDK honours retry-after, so rate-limited calls wait out the per-minute window
            # (the Decisions API beta's 2M tokens/min cap needed this on 2026-10-10).
            openai=openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), max_retries=10),
            together=openai.AsyncOpenAI(api_key=os.environ.get("TOGETHER_API_KEY"), base_url="https://api.together.xyz/v1", max_retries=4),
            anthropic=anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), max_retries=4),
        )

    async def aclose(self) -> None:
        for c in (self.openai, self.together, self.anthropic):
            if c is not None:
                await c.close()
        if self.typesafe is not None:
            await self.typesafe.aclose()
        if self.http is not None:
            await self.http.aclose()


async def call_openai_compatible(client: Any, spec: ModelSpec, body: dict) -> dict:
    kw: dict[str, Any] = {
        "model": spec.model,
        "messages": [{"role": "system", "content": body["system"]}, {"role": "user", "content": body["user"]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "coding", "strict": True, "schema": body["schema"]}},
    }
    kw["max_completion_tokens" if spec.provider == "openai" else "max_tokens"] = body["max_tokens"]
    if body["temperature"] is not None:
        kw["temperature"] = body["temperature"]
    if body["samples"] > 1:
        kw["n"] = body["samples"]
    if body["logprobs"]:
        kw["logprobs"], kw["top_logprobs"] = True, 5
    kw.update(body["params"])
    r = await client.chat.completions.create(**kw)
    u = r.usage
    cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    reasoning = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
    logprobs = []
    for ch in r.choices:
        lp = getattr(ch, "logprobs", None)
        logprobs.append(
            [{"t": t.token, "lp": t.logprob, "top": [[a.token, a.logprob] for a in (t.top_logprobs or [])]} for t in lp.content]
            if lp is not None and lp.content else None
        )
    return {
        "texts": [ch.message.content for ch in r.choices],
        "logprobs": logprobs,
        "stop": [ch.finish_reason for ch in r.choices],
        "model": r.model,
        "usage": {"input": u.prompt_tokens - cached, "cache_read": cached, "cache_write": 0, "output": u.completion_tokens, "reasoning": reasoning},
    }


async def call_anthropic(client: Any, spec: ModelSpec, body: dict) -> dict:
    if body["samples"] > 1:
        raise NotImplementedError("self-consistency is run on an OpenAI-compatible model")
    output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": body["schema"]}}
    kw: dict[str, Any] = {}
    if "effort" in body["params"]:
        output_config["effort"] = body["params"]["effort"]
    if "thinking" in body["params"]:
        kw["thinking"] = {"type": body["params"]["thinking"]}
    if body["temperature"] is not None:
        kw["extra_body"] = {"temperature": body["temperature"]}  # SDK 1.13 has no temperature kwarg
    m = await client.messages.create(
        model=spec.model,
        max_tokens=body["max_tokens"],
        system=[{"type": "text", "text": body["system"], "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": body["user"]}],
        output_config=output_config,
        **kw,
    )
    text = next((b.text for b in m.content if b.type == "text"), None)
    u = m.usage
    return {
        "texts": [text],
        "logprobs": [None],
        "stop": [m.stop_reason],
        "model": m.model,
        "usage": {"input": u.input_tokens, "cache_read": u.cache_read_input_tokens or 0,
                  "cache_write": u.cache_creation_input_tokens or 0, "output": u.output_tokens, "reasoning": 0},
    }


async def complete(clients: Clients, spec: ModelSpec, body: dict) -> dict:
    if spec.provider == "openai":
        return await call_openai_compatible(clients.openai, spec, body)
    if spec.provider == "together":
        return await call_openai_compatible(clients.together, spec, body)
    if spec.provider == "anthropic":
        return await call_anthropic(clients.anthropic, spec, body)
    raise ValueError(f"unknown provider {spec.provider!r}")


# --- run -----------------------------------------------------------------------------------------

async def run(
    calls: Sequence[tuple[Segment, LLMCall]], coder: LLMCoder, spec: ModelSpec, cache: Cache, clients: Clients,
    tag: str | None = None,
) -> tuple[dict[str, pd.DataFrame], list[dict], list[dict]]:
    """Answer every call (cache first). -> ({signal: predictions}, failures, diagnostics)."""
    rows: dict[str, list[dict]] = {s: [] for s in coder.signals}
    failures: list[dict] = []
    diags: list[dict] = []
    sem = clients.semaphore(spec)

    async def one(seg: Segment, call: LLMCall) -> None:
        body = payload(call, coder, spec)
        key = request_key(tagged(spec.provider, tag), spec.model, body)
        response = cache.get(key)
        if response is None and key in clients.inflight:
            response = await clients.inflight[key]
            if response is None:
                failures.append({"segment_id": seg.segment_id, "error": "shared in-flight request failed"})
                return
        elif response is None:
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            clients.inflight[key] = future
            try:
                async with sem:
                    response = await complete(clients, spec, body)
            except Exception as err:  # provider SDK errors after retries; recorded, never cached
                failures.append({"segment_id": seg.segment_id, "error": f"{type(err).__name__}: {str(err)[:300]}"})
                future.set_result(None)
                return
            finally:
                clients.inflight.pop(key, None)
            u = response["usage"]
            cache.put(key, provider=tagged(spec.provider, tag), model=spec.model, request=body, response=response,
                      input_tokens=u["input"] + u["cache_read"] + u["cache_write"], output_tokens=u["output"],
                      cost_usd=spec.cost(u["input"], u["output"], u["cache_read"], u["cache_write"]))
            future.set_result(response)
        signals, diag = parse(response, call, coder)
        diags.append({"segment_id": seg.segment_id, "stop": ",".join(map(str, response.get("stop", []))),
                      "model_version": response.get("model"), **diag})
        for signal, pairs in signals.items():
            rows[signal] += [{"segment_id": seg.segment_id, "code_id": cid, "p": p, "model_version": response.get("model")} for cid, p in pairs]

    await asyncio.gather(*(one(seg, call) for seg, call in calls))
    cols = ["segment_id", "code_id", "p", "model_version"]
    return {s: pd.DataFrame(r, columns=cols) for s, r in rows.items()}, failures, diags
