import asyncio
import json

import pytest

from margin.budget import ModelSpec
from margin.cache import Cache
from margin.coders import decisions, jev
from margin.perturb import STANDARD_ARMS
from margin.schema import Code, Codebook, Segment

EXCL = Codebook(
    name="talk", dataset="t", unit="client utterance", mode="exclusive", instructions="Which talk type?",
    codes=[Code(id="change", label="Change talk", definition="Favours change."), Code(id="sustain", label="Sustain talk", definition="Favours status quo.")],
    none_option=Code(id="none", label="Neutral", definition="Neither."),
)
MULTI = Codebook(
    name="topics", dataset="t", unit="comment", mode="multi", instructions="Which topics?",
    codes=[Code(id="care", label="Care", definition="About care.", exclude=["Not care."]), Code(id="fair", label="Fairness", definition="About fairness.")],
)
SEG = Segment(segment_id="s1", text="I want to quit.", context="Therapist: Hi.", group="g")
SPEC = ModelSpec(name="oaid", provider="openai_decisions", model="gpt-6-luna", price_in=0.10, price_out=0.0)


def test_choice_mapping_keeps_jev_content_and_order():
    for arm in ("canonical", "reversed", "neutral_labels"):
        (_, j), = jev.plan_calls([SEG], EXCL, STANDARD_ARMS[arm])
        (_, d), = decisions.plan_calls([SEG], EXCL, STANDARD_ARMS[arm])
        (q,) = d.request["questions"]
        assert q["type"] == "choice" and q["instructions"] == j.questions["code"]["instructions"]
        assert [c["value"] for c in q["choices"]] == list(j.questions["code"]["criteria"])
        assert [json.loads(c["description"]) for c in q["choices"]] == list(j.questions["code"]["criteria"].values())
        assert json.loads(d.request["input"]) == j.state and d.keymap == j.keymap


def test_predicate_mapping_carries_criteria():
    (_, j), = jev.plan_calls([SEG], MULTI, STANDARD_ARMS["canonical"])
    (_, d), = decisions.plan_calls([SEG], MULTI, STANDARD_ARMS["canonical"])
    assert [q["name"] for q in d.request["questions"]] == list(j.questions)
    q0 = d.request["questions"][0]
    assert q0["type"] == "predicate" and q0["instructions"].startswith(j.questions["q00"]["instructions"])
    assert "Not care." in q0["instructions"] and "About care." in q0["instructions"]


def test_parse_with_refusal():
    (_, d), = decisions.plan_calls([SEG], MULTI, STANDARD_ARMS["canonical"])
    resp = {"answers": [{"type": "predicate", "name": "q00", "probability": 0.8}, {"type": "refusal", "name": "q01"}]}
    rows, refused = decisions.parse(d, resp)
    assert rows == [{"code_id": "care", "p": 0.8}] and refused == 1


class FakeDecisions:
    def __init__(self):
        self.calls = 0

    async def create(self, model, input, questions):
        self.calls += 1
        from openai.types.decision import Decision
        answers = [{"type": "choice", "name": "code", "choice": "Change talk", "confidence": 0.7,
                    "probabilities": [{"value": c["value"], "probability": p} for c, p in zip(questions[0]["choices"], [0.8, 0.15, 0.05])]}]
        return Decision.model_validate({"answers": answers, "model": model, "usage": {"input_tokens": 500, "output_tokens": 3, "total_tokens": 503,
                                        "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}})


class FakeClient:
    def __init__(self):
        self.decisions = FakeDecisions()


def test_run_caches_and_maps(tmp_path):
    calls = decisions.plan_calls([SEG], EXCL, STANDARD_ARMS["canonical"])
    cache, client = Cache(tmp_path / "c.sqlite"), FakeClient()
    preds, failures, diags = asyncio.run(decisions.run(calls, SPEC, cache, client=client))
    assert not failures and client.decisions.calls == 1
    assert preds.set_index("code_id").p.to_dict() == pytest.approx({"change": 0.8, "sustain": 0.15, "none": 0.05})
    assert cache.spent() == pytest.approx(500 * 0.10 / 1e6)
    asyncio.run(decisions.run(calls, SPEC, cache, client=client))
    assert client.decisions.calls == 1


def test_cache_tag_makes_a_fresh_separately_cached_call(tmp_path):
    calls = decisions.plan_calls([SEG], EXCL, STANDARD_ARMS["canonical"])
    cache, client = Cache(tmp_path / "c.sqlite"), FakeClient()
    asyncio.run(decisions.run(calls, SPEC, cache, client=client))
    asyncio.run(decisions.run(calls, SPEC, cache, client=client, tag="repeat1"))
    assert client.decisions.calls == 2  # the tagged run did not reuse the untagged response
    asyncio.run(decisions.run(calls, SPEC, cache, client=client, tag="repeat1"))
    assert client.decisions.calls == 2  # but its own copy is cached
    assert cache.rekey() == 0  # tagged rows keep their keys under the migration
