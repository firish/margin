import numpy as np
import pandas as pd
import pytest

from margin.analysis import fast_aurc, fast_auroc, fast_coverage, paired
from margin.metrics.selective import aurc, coverage_at_risk, error_auroc


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 400
    conf = np.round(rng.uniform(0.3, 1.0, n), 2)  # rounded so ties occur, like Jev
    correct = (rng.uniform(size=n) < conf).astype(float)
    w = rng.choice([1.0, 2.5], n)
    return conf, correct, w


def test_fast_metrics_match_reference(data):
    conf, correct, w = data
    assert fast_aurc(conf, correct, w) == pytest.approx(aurc(conf, correct, w))
    assert fast_auroc(conf, correct, w) == pytest.approx(error_auroc(conf, correct, w))
    half = len(conf) // 2
    cal = pd.DataFrame({"conf": conf[:half], "correct": correct[:half], "weight": w[:half]})
    tst = pd.DataFrame({"conf": conf[half:], "correct": correct[half:], "weight": w[half:]})
    for eps in (0.05, 0.10, 0.2):
        ref = coverage_at_risk(cal, tst, eps)
        cov, risk = fast_coverage(cal.conf.values, cal.correct.values, cal.weight.values, tst.conf.values, tst.correct.values, tst.weight.values, eps)
        assert cov == pytest.approx(ref.test_coverage)
        assert (np.isnan(risk) and np.isnan(ref.test_risk)) or risk == pytest.approx(ref.test_risk)


def test_paired_identity_is_zero(data):
    conf, correct, w = data
    n = len(conf)
    d = pd.DataFrame({"segment_id": [f"s{i}" for i in range(n)], "conf": conf, "correct": correct, "weight": w,
                      "group": [f"g{i // 4}" for i in range(n)], "split": np.where(np.arange(n) < n * 0.3, "calib", "test")})
    r = paired(d, d.copy(), ["segment_id"], reps=50)
    for k in ("error_auroc", "cov@5%", "aurc", "cov@10%"):
        assert r[k][2] == 0
        assert np.nanmax(np.abs(r[f"{k}_reps"])) == 0
    assert r["n_test"] + r["n_calib"] == n
