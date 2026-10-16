import numpy as np
import pandas as pd
import pytest

from margin.metrics.agreement import kappa, krippendorff_nominal, macro_f1
from margin.metrics.calibration import brier, ece
from margin.metrics.core import bootstrap, decisions, reference, soft_labels
from margin.metrics.disagreement import leave_one_coder_out, uncertainty_vs_humans
from margin.metrics.robustness import compare
from margin.metrics.selective import aurc, coverage_at_risk, error_auroc, risk_coverage
from margin.schema import Code, Codebook

EXCL = Codebook(
    name="e", dataset="d", unit="u", mode="exclusive", instructions="?",
    codes=[Code(id="a", label="A", definition="a"), Code(id="b", label="B", definition="b")],
    none_option=Code(id="none", label="None", definition="n"),
)
MULTI = Codebook(
    name="m", dataset="d", unit="u", mode="multi", instructions="?",
    codes=[Code(id="x", label="X", definition="x"), Code(id="y", label="Y", definition="y")],
)


def lab(rows):
    return pd.DataFrame(rows, columns=["segment_id", "annotator", "codes"])


def test_soft_and_majority_reference_multi():
    labels = lab([("s1", "h1", ["x"]), ("s1", "h2", ["x", "y"]), ("s1", "h3", []),
                  ("s2", "h1", ["y"]), ("s2", "h2", [])])
    soft = soft_labels(labels, MULTI).set_index(["segment_id", "code_id"]).soft
    assert soft[("s1", "x")] == pytest.approx(2 / 3) and soft[("s1", "y")] == pytest.approx(1 / 3)
    ref = reference(labels, MULTI).set_index(["segment_id", "code_id"]).y
    assert ref[("s1", "x")] == 1 and ref[("s1", "y")] == 0
    assert ("s2", "y") not in ref.index  # 1 of 2: exact tie dropped
    assert ref[("s2", "x")] == 0


def test_reference_exclusive_drops_tied_plurality():
    labels = lab([("s1", "h1", ["a"]), ("s1", "h2", ["a"]), ("s1", "h3", ["b"]),
                  ("s2", "h1", ["a"]), ("s2", "h2", ["b"])])
    ref = reference(labels, EXCL).set_index("segment_id").ref_code
    assert ref.to_dict() == {"s1": "a"}


def preds_excl(rows):
    """rows: segment -> (p_a, p_b, p_none)"""
    return pd.DataFrame(
        [(s, c, p) for s, ps in rows.items() for c, p in zip(["a", "b", "none"], ps)],
        columns=["segment_id", "code_id", "p"],
    )


def test_decisions_exclusive_and_multi():
    ref = pd.DataFrame({"segment_id": ["s1", "s2"], "ref_code": ["a", "b"], "n_annotators": 1})
    d = decisions(preds_excl({"s1": (0.7, 0.2, 0.1), "s2": (0.5, 0.4, 0.1)}), ref, EXCL).set_index("segment_id")
    assert d.pred.to_dict() == {"s1": "a", "s2": "a"} and d.correct.to_dict() == {"s1": 1, "s2": 0}
    assert d.conf["s1"] == pytest.approx(0.7)

    mref = pd.DataFrame({"segment_id": ["s1", "s1"], "code_id": ["x", "y"], "y": [1, 0], "soft": [1.0, 0.0]})
    mp = pd.DataFrame({"segment_id": ["s1", "s1"], "code_id": ["x", "y"], "p": [0.8, 0.3]})
    d = decisions(mp, mref, MULTI).set_index("code_id")
    assert d.pred.to_dict() == {"x": 1, "y": 0} and d.conf["y"] == pytest.approx(0.7)
    d = decisions(mp, mref, MULTI, threshold={"x": 0.9, "y": 0.2}).set_index("code_id")
    assert d.pred.to_dict() == {"x": 0, "y": 1}


def test_kappa_hand_value_and_macro():
    dec = pd.DataFrame({"code_id": "x", "y": [1, 1, 0, 0], "pred": [1, 0, 0, 0], "weight": 1.0})
    assert kappa(dec, MULTI) == pytest.approx(0.5)  # po=.75, pe=.5
    both = pd.concat([dec, dec.assign(code_id="y", pred=[1, 1, 0, 0])])
    assert kappa(both, MULTI) == pytest.approx(0.75)  # mean of 0.5 and 1.0
    assert macro_f1(both, MULTI) == pytest.approx((2 / 3 + 1) / 2)


def test_krippendorff_perfect_and_chance():
    labels = lab([(f"s{i}", h, [c]) for i, c in enumerate("aabba") for h in ("h1", "h2")])
    assert krippendorff_nominal(labels, EXCL) == pytest.approx(1.0)


def test_ece_and_brier_hand_values():
    assert ece([0.9, 0.9, 0.6, 0.6], [1, 0, 1, 1]) == pytest.approx(0.4)
    ref = pd.DataFrame({"segment_id": ["s1"], "ref_code": ["a"]})
    assert brier(preds_excl({"s1": (1.0, 0.0, 0.0)}), ref, EXCL) == pytest.approx(0.0)
    assert brier(preds_excl({"s1": (0.5, 0.5, 0.0)}), ref, EXCL) == pytest.approx(0.5)


def test_risk_coverage_aurc_auroc():
    conf, correct = [0.9, 0.8, 0.7, 0.6], [1, 1, 0, 1]
    rc = risk_coverage(conf, correct)
    assert rc.coverage.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert rc.risk.tolist() == pytest.approx([0, 0, 1 / 3, 1 / 4])
    assert aurc(conf, correct) == pytest.approx(0.25 * (1 / 3 + 1 / 4))
    assert error_auroc([0.9, 0.8, 0.2], [1, 1, 0]) == 1.0
    # Tied confidences are accepted together.
    assert len(risk_coverage([0.9, 0.9, 0.5], [1, 0, 1])) == 2


def test_coverage_at_risk_picks_on_calib_and_reports_test():
    calib = pd.DataFrame({"conf": [0.9, 0.8, 0.7, 0.6], "correct": [1, 1, 0, 1]})
    test = pd.DataFrame({"conf": [0.95, 0.85, 0.5], "correct": [1, 0, 1]})
    t = coverage_at_risk(calib, test, eps=0.05)
    assert t.threshold == 0.8 and t.calib_coverage == 0.5
    assert t.test_accepted == 2 and t.test_coverage == pytest.approx(2 / 3) and t.test_risk == pytest.approx(0.5)
    # The Clopper-Pearson version cannot certify 5% error from 2 items: accepts nothing.
    assert coverage_at_risk(calib, test, eps=0.05, method="ucb").test_coverage == 0.0


def test_ucb_certifies_with_enough_clean_items():
    calib = pd.DataFrame({"conf": np.linspace(1, 0.5, 200), "correct": [1] * 150 + [0] * 50})
    ucb = coverage_at_risk(calib, calib, eps=0.05, method="ucb")
    emp = coverage_at_risk(calib, calib, eps=0.05)
    assert emp.calib_coverage == pytest.approx(157 / 200)  # 150 clean + 7 errors: 7/157 <= 5%
    assert 0.75 <= ucb.calib_coverage < emp.calib_coverage and ucb.calib_risk <= 0.05


def test_leave_one_coder_out_exclusive():
    # h1 and h2 always agree; h3 disagrees on half; model copies h1.
    truth = ["a", "b", "a", "b", "a", "b", "a", "b"]
    noisy = ["a", "a", "a", "a", "a", "b", "a", "b"]
    rows = [(f"s{i}", "h1", [t]) for i, t in enumerate(truth)]
    rows += [(f"s{i}", "h2", [t]) for i, t in enumerate(truth)]
    rows += [(f"s{i}", "h3", [n]) for i, n in enumerate(noisy)]
    labels = lab(rows)
    preds = preds_excl({f"s{i}": ((0.9, 0.05, 0.05) if t == "a" else (0.05, 0.9, 0.05)) for i, t in enumerate(truth)})
    loo = leave_one_coder_out(preds, labels, EXCL).set_index("annotator")
    assert loo.kappa_model["h3"] == pytest.approx(1.0)  # h1+h2 majority == model
    assert loo.kappa_human["h3"] == pytest.approx(0.5)
    assert loo.model_minus_human["h3"] == pytest.approx(0.5)


def test_uncertainty_tracks_human_entropy():
    labels = lab(
        [(f"s{i}", f"h{j}", ["a" if j < k else "b"]) for i, k in enumerate([3, 2, 3, 2]) for j in range(3)]
    )
    preds = preds_excl({"s0": (0.98, 0.01, 0.01), "s1": (0.5, 0.45, 0.05), "s2": (0.97, 0.02, 0.01), "s3": (0.6, 0.35, 0.05)})
    r = uncertainty_vs_humans(preds, labels, EXCL)
    assert r["auroc_split"] == 1.0 and r["share_split"] == 0.5 and r["spearman"] > 0.8


def test_bootstrap_and_robustness_identity():
    df = pd.DataFrame({"group": list("aabbcc"), "v": [1, 2, 3, 4, 5, 6]})
    est, lo, hi = bootstrap(df, lambda d: d.v.mean(), reps=200)
    assert est == 3.5 and lo <= est <= hi
    ref = pd.DataFrame({"segment_id": ["s1", "s2", "s3"], "ref_code": ["a", "b", "a"], "n_annotators": 1})
    d = decisions(preds_excl({"s1": (0.7, 0.2, 0.1), "s2": (0.2, 0.7, 0.1), "s3": (0.4, 0.5, 0.1)}), ref, EXCL)
    r = compare(d, d.copy(), EXCL)
    assert r["flip_rate"] == 0 and r["d_kappa"] == 0 and r["d_ece"] == 0
