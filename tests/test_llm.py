import asyncio
import json
import math
import re

import pytest

from margin.budget import ModelSpec
from margin.cache import Cache
from margin.coders import llm
from margin.perturb import STANDARD_ARMS, apply_labels_and_wording
from margin.schema import Code, Codebook, Segment

EXCL = Codebook(
    name="talk", dataset="t", unit="client utterance", mode="exclusive", instructions="Which talk type?",
    codes=[Code(id="change", label="Change talk", definition="Favours change.", include=["Commitment."]),
           Code(id="sustain", label="Sustain talk", definition="Favours status quo.")],
    none_option=Code(id="none", label="Neutral", definition="Neither."),
)
MULTI = Codebook(
    name="topics", dataset="t", unit="comment", mode="multi", instructions="Which topics?",
    codes=[Code(id="care", label="Care", definition="About care."), Code(id="fair", label="Fairness", definition="About fairness.")],
)
SEG = Segment(segment_id="s1", text="I want to quit.", context="Therapist: How's it going?", setting="Target: smoking.", group="g")
SPEC = ModelSpec(name="m", provider="openai", model="m-1", price_in=1.0, price_out=2.0, concurrency=4)


def tokens(text: str, alts: dict[int, list[tuple[str, float]]] | None = None) -> list[dict]:
    """Split like a tokenizer (words, single punctuation, spaces) and attach top alternatives by index."""
    toks = re.findall(r"\w+|\s+|[^\w\s]", text)
    out = [{"t": t, "lp": 0.0, "top": []} for t in toks]
    for i, top in (alts or {}).items():
        out[i]["top"] = [[tok, math.log(p)] for tok, p in top]
    return out


def index_of(toks: list[dict], value: str, nth: int = 0) -> int:
    hits = [i for i, t in enumerate(toks) if t["t"] == value]
    return hits[nth]


def test_render_exclusive_and_multi():
    call = llm.render(EXCL, SEG, STANDARD_ARMS["canonical"])
    assert call.keymap == {"A": "change", "B": "sustain", "C": "none"}
    assert call.schema["properties"]["answer"]["enum"] == ["A", "B", "C"]
    assert "A) Change talk" in call.system and "- Commitment." in call.system and "Favours status quo." in call.system
    assert "previous_turn (context only, not coded):\nTherapist: How's it going?" in call.user
    assert call.user.index("setting:") < call.user.index("segment:")
    assert SEG.text not in call.system  # segment-specific text stays out of the cacheable prefix

    m = llm.render(MULTI, SEG, STANDARD_ARMS["canonical"])
    assert m.keymap == {"c1": "care", "c2": "fair"} and m.schema["required"] == ["c1", "c2"]

    rev = llm.render(EXCL, SEG, STANDARD_ARMS["reversed"])
    assert rev.keymap == {"A": "none", "B": "sustain", "C": "change"}
    v = STANDARD_ARMS["neutral_labels"]
    neutral = llm.render(apply_labels_and_wording(EXCL, v), SEG, v)
    assert "Change talk" not in neutral.system and "option_A" in neutral.system


def test_verbal_exclusive_keeps_chosen_as_argmax():
    call = llm.render(EXCL, SEG, STANDARD_ARMS["canonical"])
    p = dict(llm.verbal({"rationale": "r", "answer": "B", "probability": 80}, call))
    assert p == pytest.approx({"change": 0.1, "sustain": 0.8, "none": 0.1})
    low = dict(llm.verbal({"rationale": "r", "answer": "B", "probability": 10}, call))
    assert max(low, key=low.get) == "sustain"


def test_logprob_exclusive_distribution():
    call = llm.render(EXCL, SEG, STANDARD_ARMS["canonical"])
    text = json.dumps({"rationale": "Wants to quit.", "answer": "A", "probability": 95})
    toks = tokens(text)
    i = index_of(toks, "A")
    toks[i]["top"] = [["A", math.log(0.7)], ["B", math.log(0.2)], ["\"", math.log(0.05)]]
    p = dict(llm.logprob_signal(text, toks, call))
    # C was missing from the top list: it gets the residual 1 - (0.7 + 0.2 + 0.05).
    assert p["change"] == pytest.approx(0.7 / 0.95) and p["none"] == pytest.approx(0.05 / 0.95)


def test_logprob_null_alternatives_are_zero():
    call = llm.render(EXCL, SEG, STANDARD_ARMS["canonical"])
    text = json.dumps({"rationale": "Wants to quit.", "answer": "A", "probability": 95})
    toks = tokens(text)
    toks[index_of(toks, "A")]["top"] = [["A", math.log(0.75)], ["B", math.log(0.25)], ["#", None]]
    p = dict(llm.logprob_signal(text, toks, call))
    assert p["change"] == pytest.approx(0.75) and p["sustain"] == pytest.approx(0.25) and p["none"] == 0


def test_logprob_multi_yes_probability():
    call = llm.render(MULTI, SEG, STANDARD_ARMS["canonical"])
    text = json.dumps({"c1": {"rationale": "says yes and no", "applies": "yes", "probability": 70},
                       "c2": {"rationale": "no", "applies": "no", "probability": 5}})
    toks = tokens(text)
    toks[index_of(toks, "yes", 1)]["top"] = [["yes", math.log(0.6)], ["no", math.log(0.4)]]  # 1st "yes" is in the rationale
    toks[index_of(toks, "no", 2)]["top"] = [["no", math.log(0.9)]]
    p = dict(llm.logprob_signal(text, toks, call))
    assert p["care"] == pytest.approx(0.6) and p["fair"] == pytest.approx(0.1 / (0.1 + 0.9))


def test_logprob_rejects_misaligned_tokens():
    call = llm.render(EXCL, SEG, STANDARD_ARMS["canonical"])
    assert llm.logprob_signal('{"answer": "A"}', [{"t": "x", "lp": 0}], call) is None


def test_votes():
    call = llm.render(EXCL, SEG, STANDARD_ARMS["canonical"])
    datas = [{"answer": a} for a in "AABAC"]
    assert dict(llm.votes(datas, call)) == {"change": 0.6, "sustain": 0.2, "none": 0.2}
    m = llm.render(MULTI, SEG, STANDARD_ARMS["canonical"])
    d = [{"c1": {"applies": "yes"}, "c2": {"applies": "no"}}, {"c1": {"applies": "no"}, "c2": {"applies": "no"}}]
    assert dict(llm.votes(d, m)) == {"care": 0.5, "fair": 0.0}


def test_parse_handles_bad_json():
    call = llm.render(EXCL, SEG, STANDARD_ARMS["canonical"])
    out, diag = llm.parse({"texts": ["not json"], "logprobs": [None]}, call, llm.CODERS["haiku"])
    assert out == {} and diag["parse_ok"] == 0


def test_cost_with_cache():
    spec = ModelSpec(name="x", provider="anthropic", model="m", price_in=1.0, price_out=5.0, price_cache_read=0.1, price_cache_write=1.25)
    assert spec.cost(1_000_000, 0, cache_read=1_000_000, cache_write=1_000_000) == pytest.approx(2.35)


def test_run_caches_and_writes_signals(tmp_path, monkeypatch):
    calls = llm.plan_calls([SEG, Segment(segment_id="s2", text="No.", group="g")], EXCL, STANDARD_ARMS["canonical"])
    n_api = 0

    async def fake_complete(clients, spec, body):
        nonlocal n_api
        n_api += 1
        text = json.dumps({"rationale": "r", "answer": "A", "probability": 90})
        toks = tokens(text)
        toks[index_of(toks, "A")]["top"] = [["A", math.log(0.9)], ["C", math.log(0.1)]]
        return {"texts": [text], "logprobs": [toks], "stop": ["stop"], "model": "m-1-2026",
                "usage": {"input": 100, "cache_read": 900, "cache_write": 0, "output": 30, "reasoning": 0}}

    monkeypatch.setattr(llm, "complete", fake_complete)
    cache = Cache(tmp_path / "c.sqlite")
    coder = llm.CODERS["luna"]
    out, failures, diags = asyncio.run(llm.run(calls, coder, SPEC, cache, llm.Clients()))
    assert n_api == 2 and not failures and set(out) == {"verbal", "logprob"}
    lp = out["logprob"].set_index(["segment_id", "code_id"]).p
    assert lp[("s1", "change")] == pytest.approx(0.9) and lp[("s1", "sustain")] == pytest.approx(0.0)
    assert all(d["logprob_ok"] for d in diags)
    assert cache.spent() == pytest.approx(2 * (100 * 1.0 + 900 * 1.0 + 30 * 2.0) / 1e6)
    asyncio.run(llm.run(calls, coder, SPEC, cache, llm.Clients()))
    assert n_api == 2  # replay is free
    assert llm.estimate_usd(calls, coder, SPEC, cache) == (0, 0.0)


def test_logprob_with_reasoning_prefix_tokens():
    call = llm.render(EXCL, SEG, STANDARD_ARMS["canonical"])
    text = json.dumps({"rationale": "Neutral.", "answer": "C", "probability": 95})
    prefix = [{"t": "<|channel|>analysis<|message|>", "lp": 0.0, "top": []}, {"t": 'answer "C" maybe', "lp": 0.0, "top": []}]
    toks = tokens(text)
    toks[index_of(toks, "C")]["top"] = [["C", math.log(0.8)], ["A", math.log(0.2)]]
    p = dict(llm.logprob_signal(text, prefix + toks, call))
    assert p["none"] == pytest.approx(0.8) and p["change"] == pytest.approx(0.2)


def test_concurrent_identical_requests_share_one_call(tmp_path, monkeypatch):
    calls = llm.plan_calls([SEG], EXCL, STANDARD_ARMS["canonical"])
    n_api = 0

    async def slow_complete(clients, spec, body):
        nonlocal n_api
        n_api += 1
        await asyncio.sleep(0.05)
        return {"texts": [json.dumps({"rationale": "r", "answer": "A", "probability": 90})], "logprobs": [None],
                "stop": ["stop"], "model": "m", "usage": {"input": 10, "cache_read": 0, "cache_write": 0, "output": 5, "reasoning": 0}}

    monkeypatch.setattr(llm, "complete", slow_complete)
    cache, clients = Cache(tmp_path / "c.sqlite"), llm.Clients()

    async def both():
        return await asyncio.gather(*(llm.run(calls, llm.CODERS["haiku"], SPEC, cache, clients) for _ in range(3)))

    results = asyncio.run(both())
    assert n_api == 1 and all(len(r[0]["verbal"]) == 3 for r in results)


def test_agreement_signal():
    import pandas as pd

    from margin.ensemble import agreement

    def excl(rows):
        return pd.DataFrame([(s, c, p) for s, ps in rows.items() for c, p in zip(["change", "sustain", "none"], ps)],
                            columns=["segment_id", "code_id", "p"])

    a = excl({"s1": (0.9, 0.05, 0.05), "s2": (0.2, 0.7, 0.1)})
    b = excl({"s1": (0.6, 0.3, 0.1), "s2": (0.1, 0.1, 0.8)})
    c = excl({"s1": (0.1, 0.8, 0.1)})  # s2 missing: dropped rather than counted
    out = agreement({"a": a, "b": b, "c": c}, EXCL).set_index(["segment_id", "code_id"]).p
    assert out[("s1", "change")] == pytest.approx(2 / 3) and out[("s1", "sustain")] == pytest.approx(1 / 3)
    assert "s2" not in out.index.get_level_values(0)

    m = lambda ps: pd.DataFrame({"segment_id": ["s1", "s1"], "code_id": ["care", "fair"], "p": ps})  # noqa: E731
    out = agreement({"a": m([0.9, 0.2]), "b": m([0.6, 0.5]), "c": m([0.1, 0.4])}, MULTI).set_index("code_id").p
    assert out["care"] == pytest.approx(2 / 3) and out["fair"] == pytest.approx(1 / 3)


def test_verbal_multi_is_reading_agnostic():
    call = llm.render(MULTI, SEG, STANDARD_ARMS["canonical"])
    coherent = {"c1": {"applies": "yes", "probability": 80}, "c2": {"applies": "no", "probability": 10}}
    answer_conf = {"c1": {"applies": "yes", "probability": 80}, "c2": {"applies": "no", "probability": 90}}
    assert dict(llm.verbal(coherent, call)) == pytest.approx({"care": 0.8, "fair": 0.1})
    assert dict(llm.verbal(answer_conf, call)) == pytest.approx({"care": 0.8, "fair": 0.1})


def test_distribution_prompt_parses_full_distribution_and_direct_probability():
    v = STANDARD_ARMS["dist_prompt"]
    call = llm.render(EXCL, SEG, v)
    assert "probabilities" in call.schema["properties"] and "answer" not in call.schema["properties"]
    p = dict(llm.verbal({"rationale": "r", "probabilities": {"A": 60, "B": 30, "C": 10}}, call))
    assert sum(p.values()) == pytest.approx(1.0) and max(p, key=p.get) == call.keymap["A"]
    mcall = llm.render(MULTI, SEG, v)
    out = dict(llm.verbal({k: {"rationale": "r", "probability": 70} for k in mcall.keymap}, mcall))
    assert all(x == pytest.approx(0.7) for x in out.values())
