"""Amendment v1.1, A3: what it takes to use confidence for triage in practice.

1. Recalibration, fitted on calib: temperature scaling of the probabilities, and Platt and isotonic
   maps from decision confidence to P(correct). Each is scored on test with ECE, Brier, error AUROC,
   and the realized error of the fixed rule "accept if confidence >= 0.95".
2. Thresholds with no calibration data from the task: the raw fixed rule, and a threshold fitted
   on the other tasks' calib data (leave one task out).
3. Per-code thresholds fitted on calib. Codes with too few calib decisions fall back to the task
   threshold. A cross-fitted per-code Platt ranking is included too.
4. Risk control:
   - the naive calib threshold (the primary analysis's rule);
   - the v1.0 pointwise Clopper-Pearson rule;
   - decision-level LTT ("ltt_dec"): fixed-sequence binomial tests on decisions, alpha = delta = 0.05.
     It treats decisions as independent, which clustered data violate (kept for comparison);
   - corpus LTT ("ltt"; deviations 17-18): for each threshold on a fixed grid, a without-replacement
     betting test of the uncoded remainder's selective risk, with clusters (comments, transcripts,
     lessons) as sampling units, final capital averaged over random visiting orders, and Bonferroni over
     the grid. Valid when the hand-coded clusters are a uniformly random subset of the corpus and are
     visited in random orders, whatever the dependence among errors;
   - LTT on i.i.d. clusters ("ltt_iid"): the same test with a constant null mean, for the population
     target; it additionally needs independent clusters and a true bound on cluster weight.
   Each is run on the frozen split and on repeated cluster re-splits. For corpus LTT each test split is
   the uncoded remainder of its re-split, so held-out exceedance is a direct check of its guarantee;
   for the other rules it is a diagnostic.
5. Calibration budget: the same rules when only n hand-coded calib segments are available.
6. Cascades: a decision model, then an LLM, then a human, with thresholds fitted per stage on calib.
   We report automation, error, realized API cost per 1,000 segments, and human workload, against
   both the human and the adjudicated reference.

Labelling (amendment v1.1, A3): exploratory for Jev and the v1.0 LLM signals, whose test results were
seen first; pre-specified for oaid and qwen. The qwen_big arm is exploratory (deviation 6).
Weighting: design weights throughout. The decision-level binomial rules use Kish's effective n, with
weighted errors scaled to it; cluster-level LTT works with each cluster's weighted loss directly.
"""

from __future__ import annotations

import itertools
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import betainc, expit, logit
from scipy.stats import beta as beta_dist
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from margin.analysis import CONFIRMATORY, RESULTS, TaskData, fast_auroc, joint_draws, resample
from margin.metrics.calibration import ece

EPS, DELTA, SEED, CALIB_SHARE = 0.05, 0.05, 0, 0.3
FULL = ["jev", "oaid", "luna-verbal", "haiku-verbal", "oss-verbal", "qwen-verbal",
        "luna-logprob", "oss-logprob", "qwen-logprob", "agree3", "sonnet-verbal",
        "clef", "clef_flash", "pplx", "kev9b", "kev27b", "dm_committee"]  # v1.2 decision models; committee (deviation 13)
SUBSET = ["luna_sc-vote", "qwen_big-verbal", "qwen_big-logprob"]
PRESPECIFIED = {"oaid", "qwen-verbal", "qwen-logprob", "clef", "clef_flash", "pplx", "kev9b", "kev27b"}
N_MIN = math.ceil(math.log(DELTA) / math.log(1 - EPS))
"""59: an accepted set smaller than this cannot pass the binomial test even with zero errors."""
CLIP = 1e-12
"""Probability floor for log/logit transforms. 1e-4 erased the distinctions among near-certain token probabilities."""


def status(signal: str) -> str:
    """All A3 analyses are exploratory (deviation 8); this records whether an amendment listed the coder for A3."""
    return "exploratory; coder listed in an amendment" if signal in PRESPECIFIED else "exploratory"


# --- data ------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def task_data(task: str, adjudication: str | None = None) -> TaskData:
    return TaskData.load(task, adjudication)


def load(task: str, signal: str, adjudication: str | None = None) -> pd.DataFrame | None:
    td = task_data(task, adjudication)
    p = td.preds(signal)
    return None if p is None else td.decisions(p)


def arrays(d: pd.DataFrame, conf: str = "conf") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (confidence, error indicator, weight)."""
    return d[conf].to_numpy(float), 1 - d.correct.to_numpy(float), d.weight.to_numpy(float)


# --- threshold rules ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class Curve:
    thr: np.ndarray
    cov: np.ndarray
    risk: np.ndarray
    n: np.ndarray
    neff: np.ndarray


def curve(conf: np.ndarray, err: np.ndarray, w: np.ndarray) -> Curve:
    """Risk-coverage at the end of each confidence tie block, most confident first."""
    order = np.argsort(-conf, kind="stable")
    c, e, ww = conf[order], err[order], w[order]
    cw, ce, cw2 = np.cumsum(ww), np.cumsum(ww * e), np.cumsum(ww**2)
    last = np.r_[c[1:] != c[:-1], True]
    return Curve(c[last], cw[last] / ww.sum(), ce[last] / cw[last], np.arange(1, len(c) + 1)[last], cw[last] ** 2 / cw2[last])


def binom_cdf(k: np.ndarray, n: np.ndarray, p: float) -> np.ndarray:
    """P(Binomial(n, p) <= k), continuous in n and k (regularized incomplete beta)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    out = np.ones_like(n)
    ok = n - k > 1e-12
    out[ok] = betainc(n[ok] - k[ok], k[ok] + 1, 1 - p)
    return out


@lru_cache(maxsize=None)
def ltt_start(eps: float = EPS, delta: float = DELTA, power: float = 0.8) -> int:
    """Smallest accepted-set size at which a set with true error eps/5 passes the test with the given
    power (124 for eps = delta = 0.05). It depends only on (eps, delta), never on the data."""
    from scipy.stats import binom

    for n in range(1, 100_000):
        passing = np.flatnonzero(binom.cdf(np.arange(n), n, eps) <= delta)
        if passing.size and binom.cdf(passing[-1], n, eps / 5) >= power:
            return n
    raise ValueError("no feasible start")


GRID = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0])
"""Candidate thresholds for the cluster-level rules (Bonferroni over all 16). Fixed when these rules were added
after the external review (deviation 16), i.e. after the first results had been seen."""
GRID_CASCADE = np.array([0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0])
"""Per-stage candidates for multi-stage cascades, tested jointly (8 per stage, plus "stage off" after the first)."""


def _bets(y: np.ndarray, delta: float) -> np.ndarray:
    """Predictable bets of the WSR plug-in rule: nu_i = min(1, sqrt(2 log(1/delta) / (n s2_{i-1})))."""
    m, n = y.shape
    i = np.arange(1, n + 1)
    mu = (0.5 + np.cumsum(y, axis=1)) / (1 + i)
    s2 = (0.25 + np.cumsum((y - mu) ** 2, axis=1)) / (1 + i)
    s2_prev = np.concatenate([np.full((m, 1), 0.25), s2[:, :-1]], axis=1)
    return np.minimum(1.0, np.sqrt(2 * np.log(1 / delta) / (n * s2_prev)))


ORDERS = 100  # random visiting orders averaged in the betting tests
ORDER_SEED = 20261015  # recorded before the corpus-LTT rerun (deviation 18); not chosen by any outcome


def _orders(n: int, orders: int = ORDERS) -> np.ndarray:
    """Uniformly random visiting orders of n calibration clusters, from a recorded seed and independent of
    every label: the without-replacement argument needs the order, not only the set, to be random."""
    rng = np.random.default_rng(ORDER_SEED)
    return np.stack([rng.permutation(n) for _ in range(orders)])


def _log_wealth(y: np.ndarray, null_mean, n_pool: int | None, delta: float) -> np.ndarray:
    """Final log capital of the betting process for each row of y (observations in visiting order).
    n_pool None: constant null mean; otherwise the without-replacement conditional null mean
    mu_i = (n_pool m0 - S_{i-1}) / (n_pool - i + 1), clipped to [0, 1]."""
    m, n = y.shape
    m0 = np.broadcast_to(np.asarray(null_mean, float).reshape(-1, 1), (m, 1))
    if n_pool is None:
        mu = m0
    else:
        i = np.arange(1, n + 1)
        s_prev = np.concatenate([np.zeros((m, 1)), np.cumsum(y, axis=1)[:, :-1]], axis=1)
        mu = np.clip((n_pool * m0 - s_prev) / (n_pool - i + 1), 0.0, 1.0)
    return np.log1p(-_bets(y, delta) * (y - mu)).sum(axis=1)


def _order_averaged_pvalues(y: np.ndarray, null_mean, n_pool: int | None, delta: float) -> np.ndarray:
    """p = min(1, 1 / E), with E the final capital averaged over ORDERS random visiting orders. For each
    order the capital is a non-negative supermartingale under H0, so its final value has expectation at most
    1 (an e-value); an average of e-values is an e-value, and Markov's inequality gives P(p <= delta) <= delta.
    Averaging removes the dependence of the result on any one order."""
    y = np.atleast_2d(np.asarray(y, float))
    m, n = y.shape
    if n == 0:
        return np.ones(m)
    null = np.broadcast_to(np.asarray(null_mean, float).reshape(-1), (m,))
    perms = _orders(n)
    out = np.empty(m)
    chunk = max(1, 4_000_000 // (len(perms) * n))
    for a in range(0, m, chunk):
        yy = y[a:a + chunk][:, perms].reshape(-1, n)  # (rows x orders, n)
        lw = _log_wealth(yy, np.repeat(null[a:a + chunk], len(perms)), n_pool, delta).reshape(-1, len(perms))
        log_e = np.logaddexp.reduce(lw, axis=1) - np.log(len(perms))
        out[a:a + chunk] = np.exp(-np.maximum(log_e, 0.0))
    return np.minimum(1.0, out)


def wsr_pvalues(y: np.ndarray, alpha: float, delta: float) -> np.ndarray:
    """One-sided p-values for H0: E[y] >= alpha, one row per hypothesis, columns = observations in [0, 1].
    Valid when the columns are independent draws with E[y_i | past] = E[y] (i.i.d. clusters): the capital
    K_k = prod_{i<=k} (1 - nu_i (y_i - alpha)) is then a non-negative supermartingale under H0 (Waudby-Smith &
    Ramdas 2024). Final capital averaged over random visiting orders (_order_averaged_pvalues)."""
    return _order_averaged_pvalues(y, alpha, None, delta)


def wor_pvalues(y: np.ndarray, m0: np.ndarray, n_pool: int, delta: float) -> np.ndarray:
    """One-sided p-values for H0: the mean of a finite population of n_pool values in [0, 1] is at least m0
    (one value per row), from a uniformly random sample of n units drawn without replacement and visited in a
    uniformly random order. Under H0 the next unit visited has conditional mean at least
    mu_i = (n_pool m0 - S_{i-1}) / (n_pool - i + 1), so K_k = prod_{i<=k} (1 - nu_i (y_i - clip(mu_i, 0, 1)))
    is a non-negative supermartingale (the without-replacement construction of Waudby-Smith & Ramdas 2024).
    No independence is assumed: the randomness is which units are observed and the order they are visited
    in. The final capital rises, and the p-value falls, as m0 rises."""
    return _order_averaged_pvalues(y, m0, n_pool, delta)


def cluster_losses(accept: np.ndarray, err: np.ndarray, w: np.ndarray, codes: np.ndarray, n_clusters: int,
                   bound: float, eps: float = EPS) -> np.ndarray:
    """Per-cluster transformed loss in [0, 1] for each candidate (rows of `accept`):
    y_i = (Z_i + eps B) / B with Z_i = sum_j w_ij 1{accepted} (err_ij - eps). B must bound every cluster's
    total weight. A set of clusters has selective risk <= eps exactly when its Z sum is <= 0."""
    z = np.vstack([np.bincount(codes, weights=w * a * (err - eps), minlength=n_clusters) for a in np.atleast_2d(accept)])
    return np.clip((z + eps * bound) / bound, 0.0, 1.0)


def cluster_codes(groups: np.ndarray) -> tuple[np.ndarray, int]:
    """Cluster index per decision, in sorted order of the cluster id. The betting tests visit clusters in
    random orders of their own (_orders), so this order carries no meaning."""
    codes, uniq = pd.factorize(np.asarray(groups).astype(str), sort=True)
    return codes, len(uniq)


def certify(y: np.ndarray, eps: float, delta: float, n_pool: int | None) -> np.ndarray:
    """Which candidates (rows of y, calib clusters as columns) are certified, testing each at delta.
    n_pool given: corpus version. The remainder of the corpus (n_pool - n unobserved clusters) has selective
    risk <= eps exactly when the corpus mean of y is <= m* = ((n_pool - n) eps + n mean(y_calib)) / n_pool;
    because the p-value falls as m0 rises, rejecting H0 at m* implies rejecting it at the true mean whenever
    the remainder's risk exceeds eps, so the error probability is at most delta.
    n_pool None: i.i.d.-cluster version, H0: E[y] >= eps."""
    n = y.shape[1]
    if n_pool is None:
        return wsr_pvalues(y, eps, delta) <= delta
    if n_pool <= n:
        raise ValueError("the corpus must contain clusters beyond the calibration sample")
    m_star = ((n_pool - n) * eps + n * y.mean(axis=1)) / n_pool
    return wor_pvalues(y, m_star, n_pool, delta) <= delta


def cluster_ltt(conf: np.ndarray, err: np.ndarray, w: np.ndarray, groups: np.ndarray, bound: float,
                n_pool: int | None = None, eps: float = EPS, delta: float = DELTA, grid: np.ndarray = GRID) -> float:
    """Lowest grid threshold certified at delta / len(grid) (inf if none); all certified thresholds then
    hold simultaneously with probability >= 1 - delta. n_pool: number of clusters in the corpus (calib plus
    remainder) for the corpus version; None for the i.i.d.-cluster version."""
    if len(conf) == 0:
        return np.inf
    codes, n = cluster_codes(groups)
    y = cluster_losses(conf[None, :] >= grid[:, None], err, w, codes, n, bound, eps)
    ok = certify(y, eps, delta / len(grid), n_pool)
    return float(grid[ok].min()) if ok.any() else np.inf


def task_bound(task: str) -> float:
    """B for a task: the largest total design weight of one cluster's decisions in the task sample (all
    sampled segments, before any reference-label tie is dropped). A known constant of the corpus; for MFRC
    every comment has 8 code decisions at one weight, so B is also a bound for any future comment."""
    td = task_data(task)
    per_segment = len(td.codebook.code_ids) if td.codebook.mode == "multi" else 1
    s = td.sample
    return float((s.weight * per_segment).groupby(s.group).sum().max())


def fit_threshold(conf: np.ndarray, err: np.ndarray, w: np.ndarray, rule: str = "empirical",
                  eps: float = EPS, delta: float = DELTA, groups: np.ndarray | None = None, bound: float | None = None,
                  n_pool: int | None = None) -> float:
    """Lowest confidence threshold that the rule accepts on calib; inf when none does.
    - empirical: calib risk <= eps (the primary analysis's rule).
    - cp: one-sided Clopper-Pearson bound <= eps at each threshold, no multiplicity correction (v1.0).
    - ltt_dec: decision-level LTT, fixed-sequence binomial tests of H0 "risk > eps" at level delta on
      Kish-effective counts, from the most confident block down, stopping at the first failure; starts
      at the first block with ltt_start(eps, delta) decisions. Assumes independent decisions.
    - ltt: cluster-level LTT for the rest of the corpus (cluster_ltt with n_pool); needs `groups`, the
      design bound `bound` and the corpus size `n_pool` in clusters.
    - ltt_iid: cluster-level LTT for the population, assuming i.i.d. clusters (cluster_ltt, n_pool None).
    """
    if len(conf) == 0:
        return np.inf
    if rule in ("ltt", "ltt_iid"):
        if groups is None or bound is None or (rule == "ltt" and n_pool is None):
            raise ValueError("cluster-level LTT needs cluster ids, the design bound and (corpus version) the corpus size")
        return cluster_ltt(conf, err, w, groups, bound, n_pool if rule == "ltt" else None, eps, delta)
    cv = curve(conf, err, w)
    k = cv.risk * cv.neff
    if rule == "empirical":
        ok = cv.risk <= eps
    elif rule == "cp":
        ub = np.where(cv.neff - k > 1e-12, beta_dist.ppf(1 - delta, k + 1, np.maximum(cv.neff - k, 1e-12)), 1.0)
        ok = ub <= eps
    elif rule == "ltt_dec":
        start = np.flatnonzero(cv.n >= ltt_start(eps, delta))
        if start.size == 0:
            return np.inf
        s = start[0]
        fail = np.flatnonzero(binom_cdf(k[s:], cv.neff[s:], eps) > delta)
        stop = s + (fail[0] if fail.size else len(k) - s)
        return float(cv.thr[stop - 1]) if stop > s else np.inf
    else:
        raise ValueError(rule)
    idx = np.flatnonzero(ok)
    return float(cv.thr[idx[-1]]) if idx.size else np.inf


def apply_threshold(conf: np.ndarray, err: np.ndarray, w: np.ndarray, thr: float) -> tuple[float, float]:
    """-> (coverage, risk among accepted); risk is nan when nothing is accepted."""
    acc = conf >= thr
    if not acc.any():
        return 0.0, float("nan")
    return float(w[acc].sum() / w.sum()), float((w[acc] * err[acc]).sum() / w[acc].sum())


def split_masks(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (d.split == "calib").to_numpy(), (d.split == "test").to_numpy()


# --- 1. recalibration ---------------------------------------------------------------------------------

def temperature(td: TaskData, preds: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """One temperature per task, fitted on calib by weighted NLL of the reference.
    exclusive: softmax(log p / T) over options; multi: sigmoid(logit p / T) per code. The multi-label
    decision at p = 0.5 and the exclusive argmax are unchanged, so only confidence moves."""
    calib = set(td.sample[td.sample.split == "calib"].segment_id)
    w = td.sample.set_index("segment_id").weight
    if td.codebook.mode == "exclusive":
        wide = preds.pivot_table(index="segment_id", columns="code_id", values="p").reindex(columns=td.codebook.code_ids).dropna()
        z = np.log(np.clip(wide.to_numpy(float), CLIP, 1.0))
        ref = td.ref.set_index("segment_id").ref_code
        ids = [s for s in wide.index if s in calib and s in ref.index]
        zc, wc = z[wide.index.get_indexer(ids)], w[ids].to_numpy(float)
        yc = np.array([td.codebook.code_ids.index(ref[s]) for s in ids])

        def softmax(a: np.ndarray) -> np.ndarray:
            a = a - a.max(axis=1, keepdims=True)
            e = np.exp(a)
            return e / e.sum(axis=1, keepdims=True)

        def nll(log_t: float) -> float:
            q = softmax(zc / math.exp(log_t))[np.arange(len(yc)), yc]
            return float(-(wc * np.log(np.clip(q, 1e-12, 1))).sum() / wc.sum())

        t = math.exp(minimize_scalar(nll, bounds=(math.log(0.05), math.log(20)), method="bounded").x)
        q = pd.DataFrame(softmax(z / t), index=wide.index, columns=wide.columns)
        return t, q.stack().rename("p").reset_index()
    d = preds[["segment_id", "code_id", "p"]].merge(td.ref[["segment_id", "code_id", "y"]], on=["segment_id", "code_id"])
    d = d[d.segment_id.isin(calib)]
    zc, yc, wc = logit(np.clip(d.p.to_numpy(float), CLIP, 1 - CLIP)), d.y.to_numpy(float), w[d.segment_id].to_numpy(float)

    def bce(log_t: float) -> float:
        q = np.clip(expit(zc / math.exp(log_t)), 1e-12, 1 - 1e-12)
        return float(-(wc * (yc * np.log(q) + (1 - yc) * np.log(1 - q))).sum() / wc.sum())

    t = math.exp(minimize_scalar(bce, bounds=(math.log(0.05), math.log(20)), method="bounded").x)
    out = preds[["segment_id", "code_id", "p"]].copy()
    out["p"] = expit(logit(np.clip(out.p.to_numpy(float), CLIP, 1 - CLIP)) / t)
    return t, out


def platt(cc: np.ndarray, kc: np.ndarray, wc: np.ndarray):
    """Weighted logistic map from logit(confidence) to P(correct)."""
    if kc.min() == kc.max():
        const = float(kc.mean())
        return lambda c: np.full(len(c), const)
    x = logit(np.clip(cc, CLIP, 1 - CLIP)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=2000).fit(x, kc.astype(int), sample_weight=wc)
    return lambda c: m.predict_proba(logit(np.clip(c, CLIP, 1 - CLIP)).reshape(-1, 1))[:, 1]


def isotonic(cc: np.ndarray, kc: np.ndarray, wc: np.ndarray):
    m = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(cc, kc, sample_weight=wc)
    return m.predict


def conf_metrics(conf: np.ndarray, correct: np.ndarray, w: np.ndarray) -> dict[str, float]:
    err = 1 - correct
    out = {"ece": ece(conf, correct, w), "brier_conf": float(np.average((conf - correct) ** 2, weights=w)),
           "error_auroc": fast_auroc(conf, correct, w)}
    for r in (0.95, 0.90):
        out[f"rule{r:.2f}_cov"], out[f"rule{r:.2f}_risk"] = apply_threshold(conf, err, w, r)
    return out


def recalibration_rows(task: str, signal: str) -> list[dict]:
    td = task_data(task)
    preds = td.preds(signal)
    if preds is None:
        return []
    d = td.decisions(preds)
    cal, tst = split_masks(d)
    if not cal.any() or not tst.any():
        return []
    cc, kc, wc = d.conf.to_numpy(float)[cal], d.correct.to_numpy(float)[cal], d.weight.to_numpy(float)[cal]
    ct, kt, wt = d.conf.to_numpy(float)[tst], d.correct.to_numpy(float)[tst], d.weight.to_numpy(float)[tst]
    t, scaled = temperature(td, preds)
    ds = td.decisions(scaled)
    key = ["segment_id"] + (["code_id"] if td.codebook.mode == "multi" else [])
    ds = d[key].merge(ds[key + ["conf"]], on=key, how="left")  # same rows, same order as d
    tconf = ds.conf.fillna(d.conf).to_numpy(float)  # a segment with a missing option keeps its raw confidence
    pl, iso = platt(cc, kc, wc), isotonic(cc, kc, wc)
    versions = {
        "raw": (cc, ct),
        "temperature": (tconf[cal], tconf[tst]),
        "platt": (pl(cc), pl(ct)),
        "isotonic": (iso(cc), iso(ct)),
    }
    rows = []
    for method, (c_cal, c_tst) in versions.items():
        cov5, risk5 = apply_threshold(c_tst, 1 - kt, wt, fit_threshold(c_cal, 1 - kc, wc))
        rows.append({"task": task, "signal": signal, "status": status(signal), "method": method,
                     "temperature": t if method == "temperature" else np.nan, "n_test": int(tst.sum()),
                     **conf_metrics(c_tst, kt, wt), "cov@5%": cov5, "risk@5%": risk5})
    return rows


# --- 2. thresholds without task calibration data -----------------------------------------------------------

def transfer_rows(signal: str) -> list[dict]:
    decs = {t: load(t, signal) for t in CONFIRMATORY}
    rows = []
    for held, d in decs.items():
        if d is None:
            continue
        others = []
        for t, o in decs.items():
            if t == held or o is None:
                continue
            o = o[o.split == "calib"].copy()
            o["weight"] = o.weight / o.weight.mean()  # each task counts by its decisions, not its design weights
            o["group"] = t + "|" + o.group.astype(str)
            others.append(o)
        pool = pd.concat(others)
        tst = d[d.split == "test"]
        ct, et, wt = arrays(tst)
        own = d[d.split == "calib"]
        fits = {"own calib, empirical": fit_threshold(*arrays(own)),
                "other tasks, empirical": fit_threshold(*arrays(pool)),
                "other tasks, ltt": fit_threshold(*arrays(pool), rule="ltt_iid", groups=pool.group.to_numpy(),
                                                  bound=float(pool.groupby("group").weight.sum().max())),
                "fixed 0.95": 0.95, "fixed 0.99": 0.99}
        for source, thr in fits.items():
            cov, risk = apply_threshold(ct, et, wt, thr)
            rows.append({"task": held, "signal": signal, "status": status(signal), "threshold_from": source,
                         "threshold": thr, "test_coverage": cov, "test_risk": risk,
                         "violates_5%": float(cov > 0 and risk > EPS)})
    return rows


# --- 3. per-code thresholds --------------------------------------------------------------------------

def cluster_folds(groups: np.ndarray, k: int = 5, seed: int = SEED) -> np.ndarray:
    uniq = np.unique(groups)
    fold_of = dict(zip(uniq, np.random.default_rng(seed).permutation(len(uniq)) % k))
    return np.array([fold_of[g] for g in groups])


def per_code_platt(cal: pd.DataFrame, apply_to: pd.DataFrame, key: str, min_n: int) -> np.ndarray:
    """Per-code map from confidence to P(correct), with the task-level map for thin codes."""
    base = platt(*[cal.conf.to_numpy(float), cal.correct.to_numpy(float), cal.weight.to_numpy(float)])
    out = base(apply_to.conf.to_numpy(float))
    for g, sub in cal.groupby(key):
        if len(sub) < min_n or sub.correct.nunique() < 2:
            continue
        m = (apply_to[key] == g).to_numpy()
        if m.any():
            out[m] = platt(sub.conf.to_numpy(float), sub.correct.to_numpy(float), sub.weight.to_numpy(float))(apply_to.conf.to_numpy(float)[m])
    return out


def oof_per_code(cal: pd.DataFrame, key: str, min_n: int, k: int = 5) -> np.ndarray:
    """Out-of-fold per-code calibrated confidence for calib (cluster folds), for fitting thresholds."""
    folds = cluster_folds(cal.group.to_numpy(), k)
    out = np.empty(len(cal))
    for f in range(folds.max() + 1):
        m = folds == f
        out[m] = per_code_platt(cal[~m], cal[m], key, min_n)
    return out


def rule_rows(c_cal: np.ndarray, e_cal: np.ndarray, w_cal: np.ndarray,
              c_tst: np.ndarray, e_tst: np.ndarray, w_tst: np.ndarray, ltt: dict, bound: float) -> dict[str, float]:
    """Empirical threshold fitted on calib and the oracle (fitted on test), at 5%, plus cluster-level LTT:
    certified on ltt["conf"], ltt["err"], ltt["w"], ltt["groups"] (corpus of ltt["n_pool"] clusters) and
    applied to ltt["test_conf"], the test scores of the same, unchanged scorer."""
    out = {}
    out["empirical_coverage"], out["empirical_risk"] = apply_threshold(c_tst, e_tst, w_tst, fit_threshold(c_cal, e_cal, w_cal))
    thr = fit_threshold(ltt["conf"], ltt["err"], ltt["w"], "ltt", groups=ltt["groups"], bound=bound, n_pool=ltt["n_pool"])
    out["ltt_coverage"], out["ltt_risk"] = apply_threshold(ltt["test_conf"], e_tst, w_tst, thr)
    out["oracle_coverage"] = apply_threshold(c_tst, e_tst, w_tst, fit_threshold(c_tst, e_tst, w_tst))[0]
    return out


def split_per_code(cal: pd.DataFrame, tst: pd.DataFrame, key: str, min_n: int) -> dict:
    """For LTT with learned maps: per-code maps fitted on half of the calib clusters (by a fixed seed) and
    frozen; LTT certifies them on the other half, and the same frozen maps score the test decisions. The
    fitting half is coded but neither tested nor deployed on, so the corpus is the other half plus test."""
    codes, uniq = pd.factorize(cal.group)
    half = (np.random.default_rng(SEED + 1).random(len(uniq)) < 0.5)[codes]
    cert = cal[~half]
    _, ec, wc = arrays(cert)
    return {"conf": per_code_platt(cal[half], cert, key, min_n), "err": ec, "w": wc, "groups": cert.group.to_numpy(),
            "test_conf": per_code_platt(cal[half], tst, key, min_n), "n_pool": cert.group.nunique() + tst.group.nunique()}


def per_code_rows(task: str, signal: str, min_n: int = 50) -> list[dict]:
    d = load(task, signal)
    if d is None:
        return []
    key = "code_id" if task_data(task).codebook.mode == "multi" else "pred"
    cal, tst = d[d.split == "calib"], d[d.split == "test"]
    cc, ec, wc = arrays(cal)
    ct, et, wt = arrays(tst)
    bound, gc = task_bound(task), cal.group.to_numpy()
    g_thr = fit_threshold(cc, ec, wc)
    thr = {g: fit_threshold(*arrays(sub)) if len(sub) >= min_n else g_thr for g, sub in cal.groupby(key)}
    t_vec = tst[key].map(thr).fillna(g_thr).to_numpy(float)
    acc = ct >= t_vec
    q_cal, q_tst = oof_per_code(cal, key, min_n), per_code_platt(cal, tst, key, min_n)
    raw = {"conf": cc, "err": ec, "w": wc, "groups": gc, "test_conf": ct, "n_pool": d.group.nunique()}
    rows = [
        {"method": "task threshold", "error_auroc": fast_auroc(ct, 1 - et, wt), **rule_rows(cc, ec, wc, ct, et, wt, raw, bound)},
        {"method": "per-code thresholds", "empirical_coverage": float(wt[acc].sum() / wt.sum()),
         "empirical_risk": float((wt[acc] * et[acc]).sum() / wt[acc].sum()) if acc.any() else np.nan,
         "codes_with_own_threshold": int(sum(len(s) >= min_n for _, s in cal.groupby(key)))},
        # Per-code Platt ranking; calib confidence is cross-fitted so thresholds are not fitted on in-sample maps.
        # LTT uses maps fitted on half the calib clusters and tests on the other half (split_per_code).
        {"method": "per-code calibrated ranking", "error_auroc": fast_auroc(q_tst, 1 - et, wt),
         **rule_rows(q_cal, ec, wc, q_tst, et, wt, split_per_code(cal, tst, key, min_n), bound)},
    ]
    return [{"task": task, "signal": signal, "status": status(signal), **r} for r in rows]


def per_code_resplits(task: str, signal: str, reps: int, min_n: int = 50) -> list[dict]:
    """Validity check for the per-code calibrated ranking: maps and thresholds refitted on each
    random cluster re-split; LTT should break 5% on test in at most ~5% of re-splits."""
    d = load(task, signal)
    if d is None:
        return []
    key = "code_id" if task_data(task).codebook.mode == "multi" else "pred"
    codes, uniq = pd.factorize(d.group)
    bound = task_bound(task)
    rng = np.random.default_rng(SEED)
    res = {r: [] for r in ("empirical", "ltt")}
    for _ in range(reps):
        c = (rng.random(len(uniq)) < CALIB_SHARE)[codes]
        cal, tst = d[c], d[~c]
        q_cal, q_tst = oof_per_code(cal, key, min_n), per_code_platt(cal, tst, key, min_n)
        _, ec, wc = arrays(cal)
        _, et, wt = arrays(tst)
        res["empirical"].append(apply_threshold(q_tst, et, wt, fit_threshold(q_cal, ec, wc)))
        sp = split_per_code(cal, tst, key, min_n)
        thr = fit_threshold(sp["conf"], sp["err"], sp["w"], "ltt", groups=sp["groups"], bound=bound, n_pool=sp["n_pool"])
        res["ltt"].append(apply_threshold(sp["test_conf"], et, wt, thr))
    out = []
    for rule, vals in res.items():
        cov = np.array([v[0] for v in vals])
        risk = np.array([v[1] for v in vals])
        viol = np.where(cov > 0, np.nan_to_num(risk, nan=0.0) > EPS, False)
        out.append({"task": task, "signal": signal, "status": status(signal), "rule": rule, "reps": reps,
                    "mean_coverage": float(cov.mean()), "violation_rate": float(viol.mean()),
                    "share_zero_coverage": float((cov == 0).mean())})
    return out


def _per_code_resplit_job(args) -> list[dict]:
    return per_code_resplits(*args)


def boot_auroc_diff(task: str, groups: np.ndarray, correct: np.ndarray, w: np.ndarray, a: np.ndarray, b: np.ndarray,
                    reps: int, correct_b: np.ndarray | None = None) -> np.ndarray:
    """AUROC(a) - AUROC(b) on test, resampling the corpus's test clusters jointly with related tasks
    (calibration maps held fixed, so intervals are conditional on them)."""
    members = pd.Series(np.arange(len(groups))).groupby(np.asarray(groups).astype(str)).apply(np.array).to_dict()
    shared = joint_draws(task, reps)["t"]
    kb = correct if correct_b is None else correct_b
    out = np.empty(reps)
    for r in range(reps):
        idx = resample(members, shared[r])
        out[r] = fast_auroc(a[idx], correct[idx], w[idx]) - fast_auroc(b[idx], kb[idx], w[idx])
    return out


def per_code_gain(task: str, reps: int, min_n: int = 50) -> list[dict]:
    """Per-code Platt ranking vs raw confidence, and Jev with per-code maps vs the frontier coders on
    the frontier subset, with every map fitted on the same calib items."""
    key = "code_id" if task_data(task).codebook.mode == "multi" else "pred"
    ukey = ["segment_id"] + (["code_id"] if key == "code_id" else [])
    rows = []
    decs = {s: load(task, s) for s in ["jev", "oaid", "clef", "clef_flash", "pplx", "kev9b", "kev27b", "luna-verbal", "haiku-verbal",
                                       "oss-verbal", "qwen-verbal", "sonnet-verbal", "qwen_big-verbal"]}
    for s, d in decs.items():
        if d is None:
            continue
        cal, tst = d[d.split == "calib"], d[d.split == "test"]
        q = per_code_platt(cal, tst, key, min_n)
        k, w, g = tst.correct.to_numpy(float), tst.weight.to_numpy(float), tst.group.to_numpy()
        diff = boot_auroc_diff(task, g, k, w, q, tst.conf.to_numpy(float), reps)
        rows.append({"task": task, "comparison": f"{s}: per-code calibrated - raw", "a": f"{s} per-code", "b": f"{s} raw",
                     "n_test": len(tst), "auroc_a": fast_auroc(q, k, w), "auroc_b": fast_auroc(tst.conf.to_numpy(float), k, w),
                     "diff_reps": diff})
    # Head-to-head on the frontier subset's test items. Jev's per-code maps come from its full calib
    # set: the frontier coders were run only on the subset, whose ~80-segment calib sets are too small
    # for per-code maps on the exclusive tasks.
    jev = decs["jev"]
    jcal = jev[jev.split == "calib"]
    for frontier in ["sonnet-verbal", "qwen_big-verbal"]:
        f = decs[frontier]
        if f is None:
            continue
        ft = f[f.split == "test"]
        jt = jev[jev.split == "test"].merge(ft[ukey], on=ukey)
        ft = jt[ukey].merge(ft, on=ukey)  # same items, same order
        q_j = per_code_platt(jcal, jt, key, min_n)
        w, g = jt.weight.to_numpy(float), jt.group.to_numpy()
        kj, kf = jt.correct.to_numpy(float), ft.correct.to_numpy(float)
        q_f = per_code_platt(f[f.split == "calib"], ft, key, min_n)  # frontier maps from its own calib
        for name, a, ka, b in [(f"jev per-code (full calib) - {frontier} raw", q_j, kj, ft.conf.to_numpy(float)),
                               (f"jev raw - {frontier} raw", jt.conf.to_numpy(float), kj, ft.conf.to_numpy(float)),
                               (f"jev per-code - {frontier} per-code (own calib)", q_j, kj, q_f)]:
            diff = boot_auroc_diff(task, g, ka, w, a, b, reps, correct_b=kf)
            rows.append({"task": task, "comparison": name, "a": name.split(" - ")[0], "b": name.split(" - ")[1],
                         "n_test": len(jt), "auroc_a": fast_auroc(a, ka, w), "auroc_b": fast_auroc(b, kf, w), "diff_reps": diff})
    return rows


def per_code_gains(reps: int) -> pd.DataFrame:
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        parts = list(ex.map(per_code_gain, CONFIRMATORY, [reps] * len(CONFIRMATORY)))
    rows = [r for p in parts for r in p]
    out = []
    for r in rows:
        x = r["diff_reps"][np.isfinite(r["diff_reps"])]
        out.append({k: v for k, v in r.items() if k != "diff_reps"} | {"diff": r["auroc_a"] - r["auroc_b"],
                   "lo": np.percentile(x, 2.5), "hi": np.percentile(x, 97.5)})
    for comp in dict.fromkeys(r["comparison"] for r in rows):
        parts_c = [r for r in rows if r["comparison"] == comp]
        if len(parts_c) != len(CONFIRMATORY):
            continue
        x = np.nanmean(np.vstack([r["diff_reps"] for r in parts_c]), axis=0)
        out.append({"task": "POOLED", "comparison": comp, "a": parts_c[0]["a"], "b": parts_c[0]["b"],
                    "auroc_a": float(np.mean([r["auroc_a"] for r in parts_c])), "auroc_b": float(np.mean([r["auroc_b"] for r in parts_c])),
                    "diff": float(np.mean([r["auroc_a"] - r["auroc_b"] for r in parts_c])),
                    "lo": np.percentile(x, 2.5), "hi": np.percentile(x, 97.5)})
    return pd.DataFrame(out)


# --- 4. risk control: frozen split and cluster re-splits ---------------------------------------------------

RULES = ["empirical", "cp", "ltt_dec", "ltt_iid", "ltt"]


def risk_control_rows(task: str, signal: str, reps: int) -> tuple[list[dict], list[dict]]:
    d = load(task, signal)
    if d is None:
        return [], []
    conf, err, w = arrays(d)
    groups, bound, n_pool = d.group.to_numpy(), task_bound(task), d.group.nunique()
    cal, tst = split_masks(d)
    frozen = []
    for eps in (0.02, 0.05, 0.10):
        for rule in RULES + ["oracle"]:
            fit = tst if rule == "oracle" else cal  # oracle: fitted on test, the risk-matched upper bound
            thr = fit_threshold(conf[fit], err[fit], w[fit], "empirical" if rule == "oracle" else rule, eps, groups=groups[fit], bound=bound,
                                n_pool=n_pool)
            cov, risk = apply_threshold(conf[tst], err[tst], w[tst], thr)
            frozen.append({"task": task, "signal": signal, "status": status(signal), "eps": eps, "rule": rule,
                           "threshold": thr, "test_coverage": cov, "test_risk": risk})
    # Re-splits: whole clusters to calib with probability CALIB_SHARE, the rest to test.
    codes, uniq = pd.factorize(d.group)
    rng = np.random.default_rng(SEED)
    draws = {r: [] for r in RULES}
    for _ in range(reps):
        to_cal = rng.random(len(uniq)) < CALIB_SHARE
        c = to_cal[codes]
        for rule in RULES:
            thr = fit_threshold(conf[c], err[c], w[c], rule, groups=groups[c], bound=bound, n_pool=n_pool)
            draws[rule].append(apply_threshold(conf[~c], err[~c], w[~c], thr))
    resplit = []
    for rule, vals in draws.items():
        cov = np.array([v[0] for v in vals])
        risk = np.array([v[1] for v in vals])
        viol = np.where(cov > 0, np.nan_to_num(risk, nan=0.0) > EPS, False)
        resplit.append({"task": task, "signal": signal, "status": status(signal), "rule": rule, "reps": reps,
                        "clusters": len(uniq), "mean_coverage": float(cov.mean()), "share_zero_coverage": float((cov == 0).mean()),
                        "violation_rate": float(viol.mean()), "median_risk": float(np.nanmedian(risk)) if np.isfinite(risk).any() else np.nan,
                        "p95_risk": float(np.nanpercentile(risk, 95)) if np.isfinite(risk).any() else np.nan})
    return frozen, resplit


# --- 5. calibration budget ------------------------------------------------------------------------------

BUDGETS = [50, 100, 200, 400]


def budget_rows(task: str, signal: str, reps: int) -> list[dict]:
    """Calib reduced to whole clusters until at least n segments; the frozen test set is unchanged. A budget
    sensitivity analysis conditional on the frozen split; unused calib clusters are outside the corpus."""
    d = load(task, signal)
    if d is None:
        return []
    cal, tst = d[d.split == "calib"], d[d.split == "test"]
    ct, et, wt = arrays(tst)
    seg_per_group = cal.groupby("group").segment_id.nunique()
    groups = seg_per_group.index.to_numpy()
    rows_by_group = cal.groupby("group").indices
    conf, err, w = arrays(cal)
    gc, bound, n_test = cal.group.to_numpy(), task_bound(task), tst.group.nunique()
    rng = np.random.default_rng(SEED)
    out = []
    for n in BUDGETS + [int(seg_per_group.sum())]:
        if n > seg_per_group.sum():
            continue
        res = {r: [] for r in ("empirical", "ltt")}
        realized = []
        for _ in range(reps):
            order = rng.permutation(len(groups))
            take = order[: np.searchsorted(seg_per_group.to_numpy()[order].cumsum(), n) + 1]
            idx = np.concatenate([rows_by_group[g] for g in groups[take]])
            realized.append((int(seg_per_group.to_numpy()[take].sum()), len(take)))
            for rule in res:
                # The corpus is the drawn calib clusters plus the test clusters: given that frame, the drawn
                # clusters are a uniformly random subset of it, and its remainder is exactly the evaluated test set.
                res[rule].append(apply_threshold(ct, et, wt, fit_threshold(conf[idx], err[idx], w[idx], rule, groups=gc[idx], bound=bound,
                                                                           n_pool=len(take) + n_test)))
            if n == int(seg_per_group.sum()):
                break  # the full calib set has one draw
        for rule, vals in res.items():
            cov = np.array([v[0] for v in vals])
            risk = np.array([v[1] for v in vals])
            viol = np.where(cov > 0, np.nan_to_num(risk, nan=0.0) > EPS, False)
            out.append({"task": task, "signal": signal, "status": status(signal), "calib_segments": n, "rule": rule,
                        "draws": len(vals), "realized_segments": float(np.mean([r[0] for r in realized])),
                        "realized_clusters": float(np.mean([r[1] for r in realized])),
                        "mean_coverage": float(cov.mean()), "share_zero_coverage": float((cov == 0).mean()),
                        "violation_rate": float(viol.mean())})
    return out


# --- 6. cascades ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Cascade:
    name: str
    stages: tuple[str, ...]
    """Signals in order; "agree:a+b" accepts where a and b give the same label (confidence = the lower)."""
    pool: str = "full"


CASCADES = [
    Cascade("human only", ()),
    *[Cascade(f"{s} -> human", (s,)) for s in ["jev", "oaid", "luna-verbal", "haiku-verbal", "oss-verbal", "qwen-verbal", "agree3"]],
    *[Cascade(f"{a} -> {b} -> human", (a, b)) for a in ["jev", "oaid"] for b in ["luna-verbal", "haiku-verbal", "oss-verbal", "qwen-verbal"]],
    Cascade("haiku-verbal -> jev -> human", ("haiku-verbal", "jev")),
    Cascade("luna-verbal -> jev -> human", ("luna-verbal", "jev")),
    Cascade("agree:jev+haiku-verbal -> human", ("agree:jev+haiku-verbal",)),
    Cascade("agree:jev+luna-verbal -> human", ("agree:jev+luna-verbal",)),
    Cascade("jev -> agree:luna-verbal+haiku-verbal -> human", ("jev", "agree:luna-verbal+haiku-verbal")),
    # Sonnet on the full sample (A4).
    Cascade("sonnet-verbal -> human", ("sonnet-verbal",)),
    *[Cascade(f"{a} -> sonnet-verbal -> human", (a, "sonnet-verbal")) for a in ["jev", "oaid", "haiku-verbal"]],
    Cascade("agree:jev+sonnet-verbal -> human", ("agree:jev+sonnet-verbal",)),
    Cascade("agree:haiku-verbal+sonnet-verbal -> human", ("agree:haiku-verbal+sonnet-verbal",)),
    # Other decision models (v1.2) and the committee (deviation 13).
    *[Cascade(f"{s} -> human", (s,)) for s in ["clef", "clef_flash", "pplx", "kev9b", "kev27b", "dm_committee"]],
    Cascade("agree:jev+kev27b -> human", ("agree:jev+kev27b",)),
    Cascade("agree:pplx+kev27b -> human", ("agree:pplx+kev27b",)),
    Cascade("kev27b -> sonnet-verbal -> human", ("kev27b", "sonnet-verbal")),
    # Frontier subset (llm_frontier items only).
    Cascade("human only", (), "subset"),
    *[Cascade(f"{s} -> human", (s,), "subset") for s in ["jev", "oaid", "haiku-verbal", "sonnet-verbal", "qwen_big-verbal"]],
    *[Cascade(f"{a} -> {b} -> human", (a, b), "subset") for a in ["jev", "oaid"] for b in ["sonnet-verbal", "qwen_big-verbal"]],
    Cascade("jev -> haiku-verbal -> sonnet-verbal -> human", ("jev", "haiku-verbal", "sonnet-verbal"), "subset"),
    Cascade("agree:jev+sonnet-verbal -> human", ("agree:jev+sonnet-verbal",), "subset"),
]
POOLS = {"full": ["jev", "oaid", "luna-verbal", "haiku-verbal", "oss-verbal", "qwen-verbal", "agree3", "sonnet-verbal",
                  "clef", "clef_flash", "pplx", "kev9b", "kev27b", "dm_committee"],
         "subset": ["jev", "oaid", "haiku-verbal", "sonnet-verbal", "qwen_big-verbal"]}
CONTRASTS = [  # (a, b, pool): a minus b on automation, auto-accepted risk, and API cost
    ("jev -> luna-verbal -> human", "luna-verbal -> human", "full"),
    ("jev -> haiku-verbal -> human", "haiku-verbal -> human", "full"),
    ("jev -> haiku-verbal -> human", "jev -> human", "full"),
    ("jev -> haiku-verbal -> human", "oaid -> haiku-verbal -> human", "full"),
    ("jev -> human", "oaid -> human", "full"),
    ("agree:jev+haiku-verbal -> human", "jev -> haiku-verbal -> human", "full"),
    ("jev -> sonnet-verbal -> human", "sonnet-verbal -> human", "subset"),
    ("jev -> qwen_big-verbal -> human", "qwen_big-verbal -> human", "subset"),
    ("jev -> sonnet-verbal -> human", "jev -> human", "subset"),
    ("agree:jev+haiku-verbal -> human", "jev -> human", "full"),
    ("agree:jev+haiku-verbal -> human", "haiku-verbal -> human", "full"),
    ("agree:jev+luna-verbal -> human", "jev -> human", "full"),
    ("jev -> sonnet-verbal -> human", "sonnet-verbal -> human", "full"),
    ("jev -> human", "sonnet-verbal -> human", "full"),
    ("agree:jev+haiku-verbal -> human", "sonnet-verbal -> human", "full"),
    ("agree:jev+sonnet-verbal -> human", "sonnet-verbal -> human", "full"),
    ("kev27b -> human", "jev -> human", "full"),
    ("dm_committee -> human", "jev -> human", "full"),
    ("dm_committee -> human", "sonnet-verbal -> human", "full"),
    ("agree:pplx+kev27b -> human", "sonnet-verbal -> human", "full"),
]
CONTRAST_RULES = ("empirical", "ltt", "oracle")
COST_SOURCES = {
    "jev": [("jev_main", "jev")], "oaid": [("oaid_main", "oaid")],
    "luna-verbal": [("llm_main", "luna")], "luna-logprob": [("llm_main", "luna")],
    "haiku-verbal": [("llm_main", "haiku")], "oss-verbal": [("llm_main", "oss")], "oss-logprob": [("llm_main", "oss")],
    "agree3": [("llm_main", "luna"), ("llm_main", "haiku"), ("llm_main", "oss")],
    "luna_sc-vote": [("llm_sc", "luna_sc")], "sonnet-verbal": [("sonnet_main", "sonnet")],
    "qwen-verbal": [("qwen_main", "qwen")], "qwen-logprob": [("qwen_main", "qwen")],
    "qwen_big-verbal": [("qwen_big_frontier", "qwen_big")], "qwen_big-logprob": [("qwen_big_frontier", "qwen_big")],
    "clef": [("dm_main", "clef")], "clef_flash": [("dm_main", "clef_flash")], "pplx": [("dm_main", "pplx")],
    "kev9b": [("dm_main", "kev9b")], "kev27b": [("dm_main", "kev27b")],  # ledger cost 0: Kev's GPU time is added in report
    "dm_committee": [("jev_main", "jev"), ("oaid_main", "oaid"), ("dm_main", "clef"), ("dm_main", "clef_flash"), ("dm_main", "pplx"),
                     ("dm_main", "kev9b"), ("dm_main", "kev27b")],
}


KEV_USD_PER_CALL = {"kev9b": 0.073 / 1000, "kev27b": 0.315 / 1000}
"""Estimated Modal serving cost: observed throughput (Kev-9B 30.1 calls/s on 2 x H100 at $3.95/h; Kev-27B 11.0 calls/s on
2 x B200 at $6.25/h, 2026-10-11). Excludes cold starts and idle scale-down time."""


LEDGER = Path("data/ledger.csv.gz")


def export_ledger() -> int:
    """Write the text-free spend ledger (request key, provider, model, tokens, cost, time) from the local
    response cache, so costs can be recomputed without the cache, which contains dataset text."""
    import sqlite3

    from margin.cache import DEFAULT_PATH

    db = sqlite3.connect(DEFAULT_PATH, timeout=60)
    df = pd.read_sql("SELECT key, provider, model, input_tokens, output_tokens, cost_usd, created_at FROM responses", db)
    db.close()
    df.to_csv(LEDGER, index=False, compression="gzip")
    return len(df)


def segment_costs() -> pd.DataFrame:
    """Realized API cost per segment (ledger cost of the canonical calls / segments) by task and signal."""
    import sqlite3

    from margin import run as runner
    from margin.budget import Registry
    from margin.cache import DEFAULT_PATH, request_key
    from margin.coders import decisions as oaid_mod
    from margin.coders import jev as jev_mod
    from margin.coders import llm, systemone

    registry = Registry.load()
    if Path(DEFAULT_PATH).exists():
        db = sqlite3.connect(DEFAULT_PATH, timeout=60)
        lookup = lambda chunk: db.execute(  # noqa: E731
            f"SELECT COALESCE(SUM(cost_usd), 0) FROM responses WHERE key IN ({','.join('?' * len(chunk))})", chunk).fetchone()[0]
    else:  # a fresh checkout: the released, text-free ledger has every request's cost by request key
        db = None
        led = pd.read_csv(LEDGER, usecols=["key", "cost_usd"]).set_index("key").cost_usd

        def lookup(chunk: list[str]) -> float:
            missing = [k for k in chunk if k not in led.index]
            if missing:
                raise KeyError(f"{len(missing)} request keys are not in {LEDGER}")
            return float(led.loc[chunk].sum())
    unit_cost: dict[tuple[str, str, str], float] = {}
    for exp in sorted({e for v in COST_SOURCES.values() for e, _ in v}):
        for u in runner.plan(runner.Experiment.load(exp), registry):
            if u.variant.name != "canonical":
                continue
            if u.coder == "jev":
                keys = [request_key(jev_mod.PROVIDER, u.spec.model, c.request) for _, c in u.calls]
            elif u.coder == "oaid":
                keys = [request_key(oaid_mod.PROVIDER, u.spec.model, c.request) for _, c in u.calls]
            elif u.spec.provider in systemone.PROVIDERS:
                keys = [request_key(u.spec.provider, u.spec.model, systemone.body(c, u.spec)) for _, c in u.calls]
            else:
                keys = [request_key(u.spec.provider, u.spec.model, llm.payload(c, u.llm, u.spec)) for _, c in u.calls]
            keys = list(dict.fromkeys(keys))  # a repeated request was served once (the cache keys on the request)
            total = sum(lookup(keys[i:i + 500]) for i in range(0, len(keys), 500))
            unit_cost[(exp, u.task, u.coder)] = total / max(u.n_segments, 1)
    if db is not None:
        db.close()
    for t in CONFIRMATORY + ["dunivin"]:  # Kev's ledger cost is 0: use the estimated GPU serving cost per call (one call per segment)
        for coder, usd in KEV_USD_PER_CALL.items():
            unit_cost[("dm_main", t, coder)] = usd
    rows = [{"task": t, "signal": s, "usd_per_segment": sum(unit_cost.get((e, t, c), np.nan) for e, c in src)}
            for s, src in COST_SOURCES.items() for t in CONFIRMATORY + ["dunivin"]]
    return pd.DataFrame(rows)


def human_preds(td: TaskData) -> pd.DataFrame:
    """The human majority label as a pseudo-coder, so adjudication can score it like any coder."""
    if td.codebook.mode == "multi":
        return td.ref[["segment_id", "code_id", "y"]].rename(columns={"y": "p"}).astype({"p": float})
    ids = td.ref.segment_id.unique()
    grid = pd.MultiIndex.from_product([ids, td.codebook.code_ids], names=["segment_id", "code_id"]).to_frame(index=False)
    grid = grid.merge(td.ref[["segment_id", "ref_code"]], on="segment_id")
    grid["p"] = (grid.code_id == grid.ref_code).astype(float)
    return grid[["segment_id", "code_id", "p"]]


def stage_columns(task: str, signal: str, key: list[str], adjudication: str | None) -> pd.DataFrame | None:
    """Per decision: conf, correct (human reference), pred, and correct_adj (adjudicated reference)."""
    if signal.startswith("agree:"):
        a, b = signal[6:].split("+")
        da, db = stage_columns(task, a, key, adjudication), stage_columns(task, b, key, adjudication)
        if da is None or db is None:
            return None
        m = da.merge(db, on=key, suffixes=("_a", "_b"))
        same = (m.pred_a == m.pred_b).to_numpy()
        return pd.DataFrame({**{k: m[k] for k in key}, "pred": m.pred_a,
                             "conf": np.where(same, np.minimum(m.conf_a, m.conf_b), -1.0),
                             "correct": m.correct_a, "correct_adj": m.correct_adj_a})
    d = load(task, signal)
    if d is None:
        return None
    out = d[key + ["pred", "conf", "correct"]].copy()
    if adjudication:
        da = load(task, signal, adjudication)
        out = out.merge(da[key + ["correct"]].rename(columns={"correct": "correct_adj"}), on=key, how="left")
    else:
        out["correct_adj"] = out.correct
    return out


def cascade_frame(task: str, pool: str, adjudication: str | None) -> pd.DataFrame:
    """One row per decision on the items every pool signal answered; columns <name>|conf etc."""
    td = task_data(task)
    key = ["segment_id"] + (["code_id"] if td.codebook.mode == "multi" else [])
    signals = sorted({s for c in CASCADES if c.pool == pool for s in c.stages} | set(POOLS[pool]))
    base = td.decisions(td.preds("jev"))[key + ["weight", "group", "split"]]
    for s in signals:
        cols = stage_columns(task, s, key, adjudication)
        if cols is None:
            raise FileNotFoundError(f"{task}: no predictions for {s}")
        base = base.merge(cols.rename(columns={c: f"{s}|{c}" for c in cols.columns if c not in key}), on=key, how="inner")
    if adjudication:
        tda = task_data(task, adjudication)
        h = tda.decisions(human_preds(tda))[key + ["correct"]].rename(columns={"correct": "human|correct_adj"})
        base = base.merge(h, on=key, how="inner")  # items adjudicated as "none"/ambiguous are dropped, as for coders
    else:
        base["human|correct_adj"] = 1.0
    return base.reset_index(drop=True)


def cascade_ltt(m: pd.DataFrame, stages: tuple[str, ...], cal: np.ndarray, bound: float,
                eps: float = EPS, delta: float = DELTA) -> tuple[float, ...]:
    """Thresholds for a whole pipeline from one joint cluster-level LTT. Every combination of per-stage
    candidates (GRID for one stage; GRID_CASCADE for the first stage and GRID_CASCADE or "off" for later
    ones) is tested on the calib clusters, Bonferroni over all combinations, so the pipeline's selective
    risk is controlled, not each stage's. The certified combination with the largest calib coverage is
    used; all inf when none is certified."""
    w = m.weight.to_numpy(float)[cal]
    confs = [m[f"{s}|conf"].to_numpy(float)[cal] for s in stages]
    errs = [1 - m[f"{s}|correct"].to_numpy(float)[cal] for s in stages]
    grids = [GRID] if len(stages) == 1 else [GRID_CASCADE] + [np.r_[GRID_CASCADE, np.inf]] * (len(stages) - 1)
    combos = list(itertools.product(*grids))
    codes, n = cluster_codes(m.group.to_numpy()[cal])
    n_pool = m.group.nunique()  # the corpus: every cluster in the frame, calib and remainder
    z, cov = np.empty((len(combos), n)), np.empty(len(combos))
    for k, ts in enumerate(combos):
        acc, err, open_ = np.zeros(len(w), bool), np.zeros(len(w)), np.ones(len(w), bool)
        for c, e, t in zip(confs, errs, ts):
            take = open_ & (c >= t)
            err[take], acc, open_ = e[take], acc | take, open_ & ~take
        z[k] = np.bincount(codes, weights=w * acc * (err - eps), minlength=n)
        cov[k] = w[acc].sum() / w.sum()
    y = np.clip((z + eps * bound) / bound, 0.0, 1.0)
    ok = certify(y, eps, delta / len(combos), n_pool)
    if not ok.any():
        return tuple(np.inf for _ in stages)
    # Largest calib coverage; among ties, call as few later stages as possible (an unused stage costs money).
    calls = np.array([sum(np.isfinite(t) for t in ts) for ts in combos])
    best = np.flatnonzero(ok)[np.lexsort((calls[ok], -cov[ok]))[0]]
    return combos[int(best)]


def simulate(m: pd.DataFrame, stages: tuple[str, ...], cost: dict[str, float], cal: np.ndarray, tst: np.ndarray,
             rule: str = "empirical", eps: float = EPS, bound: float | None = None) -> dict[str, float]:
    """Thresholds on calib, metrics on test. "empirical" fits each stage on the calib decisions still
    open at that stage; "ltt" fits the whole pipeline jointly (cascade_ltt); "oracle" fits each stage on
    the open *test* decisions (not deployable: the risk-matched upper bound). A stage with no threshold
    (inf) is not called. Cost counts a called stage once per segment that still has an open decision."""
    w = m.weight.to_numpy(float)
    fixed = cascade_ltt(m, stages, cal, bound, eps) if rule == "ltt" and stages else None
    seg = pd.factorize(m.segment_id)[0]
    open_c, open_t = cal.copy(), tst.copy()
    auto_err = np.zeros(len(m))
    auto_err_adj = np.zeros(len(m))
    auto = np.zeros(len(m), bool)
    usd = 0.0
    n_seg_t = len(np.unique(seg[tst]))
    out: dict[str, float] = {}
    for i, s in enumerate(stages):
        conf = m[f"{s}|conf"].to_numpy(float)
        err = 1 - m[f"{s}|correct"].to_numpy(float)
        if fixed is not None:
            thr = fixed[i]
        else:
            fit = open_t if rule == "oracle" else open_c
            thr = fit_threshold(conf[fit], err[fit], w[fit], "empirical" if rule == "oracle" else rule, eps)
        reach = len(np.unique(seg[open_t])) / n_seg_t if n_seg_t else 0.0
        if np.isfinite(thr):
            usd += reach * sum(cost.get(x, 0.0) for x in (s[6:].split("+") if s.startswith("agree:") else [s]))
        take = open_t & (conf >= thr)
        auto_err[take] = err[take]
        auto_err_adj[take] = 1 - m[f"{s}|correct_adj"].to_numpy(float)[take]
        auto |= take
        out[f"stage{i + 1}_share"] = float(w[take].sum() / w[tst].sum())
        open_c &= ~(conf >= thr)
        open_t &= ~take
    wt = w[tst]
    human = open_t
    human_err_adj = 1 - m["human|correct_adj"].to_numpy(float)
    final_err_adj = np.where(auto, auto_err_adj, human_err_adj)
    out |= {
        "automation": float(w[auto].sum() / wt.sum()),
        "auto_risk": float((w[auto] * auto_err[auto]).sum() / w[auto].sum()) if auto.any() else np.nan,
        "final_error_vs_human": float((w[auto] * auto_err[auto]).sum() / wt.sum()),
        "final_error_vs_adjudicated": float((w[tst] * final_err_adj[tst]).sum() / wt.sum()),
        "human_only_error_vs_adjudicated": float((w[tst] * human_err_adj[tst]).sum() / wt.sum()),
        "human_decisions": float(w[human].sum() / wt.sum()),
        "human_segments": len(np.unique(seg[human])) / n_seg_t if n_seg_t else np.nan,
        "api_usd_per_1k_segments": 1000 * usd,
    }
    return out


def _cascade_job(args) -> tuple[list[dict], list[dict]]:
    task, pool, reps, costs, adjudication = args
    m = cascade_frame(task, pool, adjudication)
    cal, tst = split_masks(m)
    cost, bound = costs.get(task, {}), task_bound(task)
    configs = [c for c in CASCADES if c.pool == pool]
    rows = []
    for c in configs:
        for rule in ("empirical", "ltt", "oracle"):
            rows.append({"task": task, "pool": pool, "reference_for_eval": "adjudicated" if adjudication else "human",
                         "cascade": c.name, "rule": rule, "n_test_decisions": int(tst.sum()),
                         **simulate(m, c.stages, cost, cal, tst, rule, bound=bound)})
    if adjudication or reps == 0:
        return rows, []
    # Paired cluster bootstrap for the contrasts (empirical rule), resampling within calib and test.
    by = {c.name: c for c in configs}
    pairs = [(a, b) for a, b, p in CONTRASTS if p == pool]
    groups = m.group.astype(str).to_numpy()
    members = {s: pd.Series(np.flatnonzero(mask)).groupby(groups[mask]).apply(np.array).to_dict() for s, mask in (("c", cal), ("t", tst))}
    shared = joint_draws(task, reps)  # one draw of the corpus's clusters per replicate, shared by related tasks
    draws = {(p, r): [] for p in pairs for r in CONTRAST_RULES}
    for rep in range(reps):
        idx, lab = {}, {}
        for s, mem in members.items():
            parts = [(mem[g], k) for k, g in enumerate(shared[s][rep]) if g in mem]
            idx[s] = np.concatenate([q for q, _ in parts]) if parts else np.array([], int)
            lab[s] = np.concatenate([np.full(len(q), k) for q, k in parts]) if parts else np.array([], int)
        sub = m.iloc[np.r_[idx["c"], idx["t"]]].reset_index(drop=True)
        c_mask = np.r_[np.ones(len(idx["c"]), bool), np.zeros(len(idx["t"]), bool)]
        sub["group"] = np.r_[lab["c"], lab["t"] + len(shared["c"][rep])]  # each drawn copy of a cluster is its own cluster
        sub["segment_id"] = (sub.segment_id + np.where(c_mask, "|c", "|t") + "|" + sub.group.astype(str))  # copies stay distinct
        for rule in CONTRAST_RULES:
            res = {name: simulate(sub, by[name].stages, cost, c_mask, ~c_mask, rule, bound=bound) for name in {x for p in pairs for x in p}}
            for a, b in pairs:
                draws[((a, b), rule)].append([res[a][k] - res[b][k] for k in ("automation", "auto_risk", "api_usd_per_1k_segments")])
    point = {(r["cascade"], r["rule"]): r for r in rows}
    boot = []
    for ((a, b), rule), vals in draws.items():
        v = np.array(vals, float)
        boot.append({"task": task, "pool": pool, "a": a, "b": b, "rule": rule,
                     **{f"{k}_diff": point[(a, rule)][k] - point[(b, rule)][k] for k in ("automation", "auto_risk", "api_usd_per_1k_segments")},
                     **{f"{k}_reps": v[:, j] for j, k in enumerate(("automation", "auto_risk", "api_usd_per_1k_segments"))}})
    return rows, boot


def cascades(reps: int, costs: pd.DataFrame, adjudication: str | None = "plan_v1") -> tuple[pd.DataFrame, pd.DataFrame]:
    cost = {t: g.set_index("signal").usd_per_segment.to_dict() for t, g in costs.groupby("task")}
    jobs = [(t, p, reps, cost, None) for t in CONFIRMATORY for p in POOLS]
    if adjudication:
        jobs += [(t, p, 0, cost, adjudication) for t in CONFIRMATORY for p in POOLS]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        done = list(ex.map(_cascade_job, jobs))
    table = pd.DataFrame([r for rows, _ in done for r in rows])
    boot = [b for _, bs in done for b in bs]
    out = []
    for b in boot:
        row = {k: v for k, v in b.items() if not k.endswith("_reps")}
        for k in ("automation", "auto_risk", "api_usd_per_1k_segments"):
            x = b[f"{k}_reps"]
            x = x[np.isfinite(x)]
            row[f"{k}_lo"], row[f"{k}_hi"] = (np.percentile(x, [2.5, 97.5]) if x.size else (np.nan, np.nan))
        out.append(row)
    for (a, bname, pool), rule in [(c, r) for c in CONTRASTS for r in CONTRAST_RULES]:  # equal-weight mean over tasks
        parts = [b for b in boot if b["a"] == a and b["b"] == bname and b["pool"] == pool and b["rule"] == rule]
        if not parts:
            continue
        row = {"task": "POOLED", "pool": pool, "a": a, "b": bname, "rule": rule}
        for k in ("automation", "auto_risk", "api_usd_per_1k_segments"):
            row[f"{k}_diff"] = float(np.nanmean([p[f"{k}_diff"] for p in parts]))
            x = np.nanmean(np.vstack([p[f"{k}_reps"] for p in parts]), axis=0)
            x = x[np.isfinite(x)]
            row[f"{k}_lo"], row[f"{k}_hi"] = (np.percentile(x, [2.5, 97.5]) if x.size else (np.nan, np.nan))
        out.append(row)
    pooled = (table.groupby(["pool", "reference_for_eval", "cascade", "rule"], sort=False)
              .mean(numeric_only=True).reset_index().assign(task="POOLED"))
    return pd.concat([table, pooled], ignore_index=True), pd.DataFrame(out)


# --- driver ---------------------------------------------------------------------------------------------

def _signal_job(args) -> dict[str, list[dict]]:
    task, signal, reps = args
    frozen, resplit = risk_control_rows(task, signal, reps)
    return {"recalibration": recalibration_rows(task, signal), "per_code": per_code_rows(task, signal),
            "risk_control": frozen, "resplits": resplit, "budget": budget_rows(task, signal, max(1, reps // 2))}


def pooled(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Append equal-weight means over tasks (only where every confirmatory task is present)."""
    g = df.groupby(by, sort=False)
    full = g.filter(lambda x: x.task.nunique() == len(CONFIRMATORY))
    if full.empty:
        return df
    p = full.groupby(by, sort=False).mean(numeric_only=True).reset_index().assign(task="POOLED")
    return pd.concat([df, p], ignore_index=True)


def run(reps: int = 500, boot_reps: int = 1000) -> dict[str, pd.DataFrame]:
    RESULTS.mkdir(exist_ok=True)
    jobs = [(t, s, reps) for t in CONFIRMATORY for s in FULL + SUBSET]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        parts = list(ex.map(_signal_job, jobs))
    tables = {k: pd.DataFrame([r for p in parts for r in p[k]]) for k in parts[0]}
    out = {
        "a3_recalibration": pooled(tables["recalibration"], ["signal", "method"]),
        "a3_per_code": pooled(tables["per_code"], ["signal", "method"]),
        "a3_risk_control": pooled(tables["risk_control"], ["signal", "eps", "rule"]),
        "a3_resplits": pooled(tables["resplits"], ["signal", "rule"]),
        "a3_budget": pooled(tables["budget"], ["signal", "calib_segments", "rule"]),
        "a3_transfer": pooled(pd.DataFrame([r for s in FULL for r in transfer_rows(s)]), ["signal", "threshold_from"]),
    }
    out["a3_per_code_gain"] = per_code_gains(boot_reps)
    jobs_pc = [(t, sig, max(1, reps // 2)) for t in CONFIRMATORY for sig in ["jev", "oaid", "luna-verbal", "haiku-verbal", "oss-verbal", "qwen-verbal"]]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        pc = [r for rows in ex.map(_per_code_resplit_job, jobs_pc) for r in rows]
    out["a3_per_code_resplits"] = pooled(pd.DataFrame(pc), ["signal", "rule"])
    costs = segment_costs()
    out["a3_costs"] = costs
    out["a3_cascades"], out["a3_cascade_contrasts"] = cascades(boot_reps, costs)
    for name, df in out.items():
        df.to_csv(RESULTS / f"{name}.csv", index=False)
    return out
