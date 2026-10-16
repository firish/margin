import numpy as np
import pandas as pd
import pytest

from margin.adjudicate import ExportConfig, parse, select, summary
from margin.metrics.core import apply_adjudication
from margin.schema import Code, Codebook

EXCL = Codebook(
    name="e", dataset="d", unit="u", mode="exclusive", instructions="?",
    codes=[Code(id="a", label="A", definition="a"), Code(id="b", label="B", definition="b")],
    none_option=Code(id="none", label="None", definition="n"),
)
MULTI = Codebook(
    name="m", dataset="d", unit="u", mode="multi", instructions="?",
    codes=[Code(id="x", label="X", definition="x")],
)


def test_parse_answers():
    letters = {"A": "b", "B": "a"}
    assert parse("A", letters, multi=False) == (["b"], False)
    assert parse(" a + b ", letters, multi=False) == (["b", "a"], False)
    assert parse("none", letters, multi=False) == ([], False)
    assert parse("y", {}, multi=True) == (["1"], False)
    assert parse("N", {}, multi=True) == (["0"], False)
    assert parse("?", {}, multi=True) == ([], True)
    for bad, multi in (("C", False), ("", False), ("maybe", True)):
        with pytest.raises(ValueError):
            parse(bad, letters, multi=multi)


def test_select_takes_all_high_conf_then_rest_then_controls():
    u = pd.DataFrame({
        "segment_id": [f"s{i}" for i in range(100)],
        "n_disagree": [1] * 30 + [0] * 70,
        "max_conf_disagree": [0.99] * 5 + [0.6] * 25 + [-np.inf] * 70,
    })
    sel = select(u, ExportConfig("x", ("jev",), n_per_task=20, control_share=0.1), np.random.default_rng(0))
    counts = sel.stratum.value_counts().to_dict()
    assert counts == {"high_conf_disagreement": 5, "other_disagreement": 13, "control": 2}
    assert sel.segment_id.is_unique
    assert sel[sel.stratum == "high_conf_disagreement"].inclusion.eq(1.0).all()


def test_apply_adjudication_exclusive():
    dec = pd.DataFrame({"segment_id": ["s1", "s2", "s3", "s4"], "pred": ["a", "b", "a", "a"],
                        "ref_code": ["b", "a", "b", "a"], "correct": [0, 0, 0, 1], "conf": 0.9})
    v = pd.DataFrame({"segment_id": ["s1", "s2", "s3"], "acceptable": [["a"], ["a"], []], "ambiguous": False})
    out = apply_adjudication(dec, v, EXCL).set_index("segment_id")
    assert "s3" not in out.index  # judged "none": dropped
    assert out.correct.to_dict() == {"s1": 1, "s2": 0, "s4": 1}  # s4 untouched
    assert out.ref_code.to_dict() == {"s1": "a", "s2": "a", "s4": "a"}


def test_apply_adjudication_multi():
    dec = pd.DataFrame({"segment_id": ["s1", "s2", "s3"], "code_id": "x", "pred": [1, 0, 1], "y": [0, 0, 1], "correct": [0, 1, 1]})
    v = pd.DataFrame({"segment_id": ["s1", "s2"], "code_id": "x", "acceptable": [["1"], []], "ambiguous": [False, True]})
    out = apply_adjudication(dec, v, MULTI).set_index("segment_id")
    assert "s2" not in out.index  # ambiguous: dropped
    assert out.y.to_dict() == {"s1": 1, "s3": 1} and out.correct.to_dict() == {"s1": 1, "s3": 1}


def test_summary_upheld_rates():
    v = pd.DataFrame({
        "task": "t", "stratum": ["hi", "hi", "control"], "human": ["a", "a", "b"], "jev_label": ["b", "b", "b"],
        "acceptable": [["b"], ["a", "b"], ["b"]], "ambiguous": False,
    })
    s = summary(v, ["jev"]).set_index("stratum")
    assert s.loc["hi", "human_ok"] == 0.5 and s.loc["hi", "jev_ok"] == 1.0 and s.loc["control", "human_ok"] == 1.0


def test_summary_treats_missing_labels_as_not_coded():
    v = pd.DataFrame({"task": "t", "stratum": "hi", "human": ["a", "a"], "jev_label": ["b", np.nan],
                      "acceptable": [["b"], ["a"]], "ambiguous": False})
    s = summary(v, ["jev"]).set_index("stratum")
    assert s.loc["hi", "jev_ok"] == 1.0  # only the item jev coded counts
