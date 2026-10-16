import asyncio

import pytest
from typesafe_sdk import SystemOneResponse

from margin.budget import BudgetError, ModelSpec, Registry, check_cap
from margin.cache import Cache, request_key
from margin.coders import jev
from margin.perturb import CANONICAL
from margin.schema import Code, Codebook, Segment

SPEC = ModelSpec(name="jev", provider="typesafe", model="jev-1.13.0", price_in=0.042, price_out=0.0)


@pytest.fixture
def cache(tmp_path):
    c = Cache(tmp_path / "c.sqlite")
    yield c
    c.close()


def test_request_key_is_order_sensitive():
    # Option order is part of the request (RQ3); a Choice with reordered criteria is a new request.
    assert request_key("p", "m", {"a": 1, "b": 2}) != request_key("p", "m", {"b": 2, "a": 1})
    assert request_key("p", "m", {"a": [1, 2]}) != request_key("p", "m", {"a": [2, 1]})
    assert request_key("p", "m", {"a": 1, "b": 2}) == request_key("p", "m", {"a": 1, "b": 2})


def test_rekey_matches_runtime_keys(tmp_path):
    import json as _json
    c = Cache(tmp_path / "c.sqlite")
    req = {"state": {"segment": "x"}, "questions": {"code": {"criteria": {"Z": None, "A": None}}}}
    c.put("old-key", provider="typesafe", model="m", request=req, response={"ok": 1}, input_tokens=1, output_tokens=0, cost_usd=0.0)
    assert c.rekey() == 1
    assert c.get(request_key("typesafe", "m", req)) == {"ok": 1}
    reordered = _json.loads(_json.dumps(req))
    reordered["questions"]["code"]["criteria"] = {"A": None, "Z": None}
    assert c.get(request_key("typesafe", "m", reordered)) is None


def test_cap(cache):
    reg = Registry(models={}, cap_usd=1.0, per_run_cap_usd=0.5)
    check_cap(reg, cache, 0.4)
    with pytest.raises(BudgetError):
        check_cap(reg, cache, 0.6)
    cache.put("k", provider="x", model="m", request={}, response={}, input_tokens=1, output_tokens=0, cost_usd=0.8)
    with pytest.raises(BudgetError):
        check_cap(reg, cache, 0.3)


def test_unpriced_model_refuses():
    with pytest.raises(BudgetError):
        ModelSpec(name="x", provider="openai", model="m", price_in=None, price_out=None).cost(10, 10)


class FakeClient:
    """Answers Nouls with p = 0.1 * (question index + 1); records calls."""

    def __init__(self):
        self.calls = 0

    async def system_one(self, state, questions, model=None):
        self.calls += 1
        answers = {k: {"type": "noul", "noul": 0.1 * (i + 1)} for i, k in enumerate(questions)}
        return SystemOneResponse.model_validate(
            {"model": model, "usage": {"input_tokens": 1000, "output_tokens": 4}, "answers": answers}
        )

    async def aclose(self):
        pass


def test_jev_run_caches_and_maps_codes(cache):
    cb = Codebook(
        name="m",
        dataset="t",
        unit="comment",
        mode="multi",
        instructions="?",
        codes=[Code(id=c, label=c, definition=c) for c in ["a", "b", "c"]],
    )
    segs = [Segment(segment_id=f"s{i}", text=f"t{i}", group="g") for i in range(3)]
    calls = jev.plan_calls(segs, cb, CANONICAL)
    client = FakeClient()
    preds, failures = asyncio.run(jev.run(calls, SPEC, cache, client=client))
    assert not failures and client.calls == 3
    assert preds.set_index(["segment_id", "code_id"])["p"].round(2).to_dict()[("s0", "c")] == 0.3
    assert cache.spent() == pytest.approx(3 * 1000 * 0.042 / 1e6)
    # Rerun: everything from cache, no new spend.
    preds2, _ = asyncio.run(jev.run(calls, SPEC, cache, client=client))
    assert client.calls == 3 and len(preds2) == 9
    assert jev.estimate_usd(calls, SPEC, cache) == (0, 0.0)


def test_exactly_one_odds_normalization():
    import pandas as pd

    preds = pd.DataFrame({"segment_id": ["s"] * 3 + ["t"] * 2, "code_id": ["a", "b", "c", "a", "b"],
                          "p": [0.9, 0.1, 0.0, 0.5, 0.5]})
    out = jev.exactly_one(preds).set_index(["segment_id", "code_id"])
    odds = [9, 0.1 / 0.9, 0.005 / 0.995]
    for code, o in zip("abc", odds):
        assert out.p[("s", code)] == pytest.approx(o / sum(odds))
    assert out.p[("t", "a")] == pytest.approx(0.5) and out.p_raw[("s", "c")] == 0.0
    assert out.groupby(level=0).p.sum().round(9).eq(1).all()


def test_cli_has_all_commands():
    from typer.main import get_command

    from margin.cli import app

    cmd = get_command(app)
    expected = {"smoke", "prepare", "sample", "estimate", "run", "evaluate", "ensemble", "paraphrase", "analyze", "adjudicate", "spend", "rq5", "replicate", "a3"}
    assert expected <= set(cmd.commands)
    assert {"export", "import", "panel", "resolve"} <= set(cmd.commands["adjudicate"].commands)
