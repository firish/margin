"""RQ1 triage: does confidence rank errors, and how much can be auto-accepted at a fixed error rate?"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_auc_score


def risk_coverage(conf, correct, weight=None) -> pd.DataFrame:
    """Accept decisions from most to least confident; risk = weighted error rate among accepted.
    Tied confidences are accepted together (one point per distinct confidence)."""
    conf, err = np.asarray(conf, float), 1 - np.asarray(correct, float)
    w = np.ones_like(conf) if weight is None else np.asarray(weight, float)
    order = np.argsort(-conf, kind="stable")
    c, e, ww = conf[order], err[order], w[order]
    cum_w, cum_e = np.cumsum(ww), np.cumsum(ww * e)
    last = np.r_[c[1:] != c[:-1], True]  # last index of each tie block
    return pd.DataFrame({"threshold": c[last], "coverage": cum_w[last] / ww.sum(), "risk": cum_e[last] / cum_w[last]})


def aurc(conf, correct, weight=None) -> float:
    """Area under the risk-coverage curve (step function from coverage 0); lower is better."""
    rc = risk_coverage(conf, correct, weight)
    widths = np.diff(np.r_[0.0, rc.coverage.to_numpy()])
    return float((widths * rc.risk.to_numpy()).sum())


def error_auroc(conf, correct, weight=None) -> float:
    """AUROC of confidence for separating correct from incorrect decisions (0.5 = no signal; ties
    count half). With `weight`, pairs are weighted by sampling weights (population-representative)."""
    correct = np.asarray(correct)
    if len(correct) == 0 or correct.min() == correct.max():
        return float("nan")
    return float(roc_auc_score(correct, conf, sample_weight=None if weight is None else np.asarray(weight, float)))


def _upper_bound(errors: float, n: float, alpha: float) -> float:
    """One-sided Clopper-Pearson upper bound on the error rate."""
    if n <= 0 or errors >= n:
        return 1.0
    return float(beta.ppf(1 - alpha, errors + 1, n - errors))


@dataclass(frozen=True)
class Triage:
    threshold: float
    calib_coverage: float
    calib_risk: float
    test_coverage: float
    test_risk: float
    test_accepted: int


def coverage_at_risk(
    calib: pd.DataFrame, test: pd.DataFrame, eps: float = 0.05, *, method: str = "empirical", alpha: float = 0.05
) -> Triage:
    """Pick the lowest confidence threshold on `calib` whose accepted-set error is <= eps (method
    "empirical"), or whose Clopper-Pearson upper bound is <= eps (method "ucb"), then apply it to `test`.
    Both frames need conf and correct; an optional weight column weights coverage and risk."""
    rc = risk_coverage(calib.conf, calib.correct, calib.get("weight"))
    if method == "ucb":
        conf, err = calib.conf.to_numpy(), 1 - calib.correct.to_numpy()
        ok = [_upper_bound(err[conf >= t].sum(), (conf >= t).sum(), alpha) <= eps for t in rc.threshold]
    else:
        ok = (rc.risk <= eps).to_numpy()
    if not np.any(ok):
        return Triage(np.inf, 0.0, np.nan, 0.0, np.nan, 0)
    best = rc[ok].iloc[-1]  # largest coverage that satisfies the constraint
    t = float(best.threshold)
    w = test.get("weight", pd.Series(1.0, index=test.index))
    acc = test.conf >= t
    cov = float(w[acc].sum() / w.sum())
    risk = float((w[acc] * (1 - test.correct[acc])).sum() / w[acc].sum()) if acc.any() else np.nan
    return Triage(t, float(best.coverage), float(best.risk), cov, risk, int(acc.sum()))
