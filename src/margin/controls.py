"""Checks added after the review of the first draft (deviation 16; exploratory).

1. category_only: error AUROC when each decision is scored only by the calib accuracy of its predicted code
   (multi-label: of its code). Per-code recalibration uses this information too, so it separates "which
   code" from "how confident within a code".
2. ltt_feasibility / single_threshold: how many hand-coded clusters our corpus LTT needs before it certifies
   even a perfect coder, and how much a single threshold fixed in advance changes that. These describe this
   procedure (capped bets, normalization by the largest cluster, Bonferroni over the grid), not every possible
   guarantee.
3. ltt_simulation: decision-level, i.i.d.-cluster and corpus LTT on synthetic clustered corpora with calibrated
   confidence conf = 1 - 0.5 * Beta(1, b), where the true selective risk is known in closed form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from margin import triage
from margin.analysis import CONFIRMATORY, RESULTS, fast_auroc

SIGNALS = ["jev", "oaid", "clef", "clef_flash", "pplx", "kev9b", "kev27b", "luna-verbal", "haiku-verbal", "oss-verbal", "sonnet-verbal"]


def category_only() -> pd.DataFrame:
    rows = []
    for s in SIGNALS:
        for t in CONFIRMATORY:
            d = triage.load(t, s)
            key = "code_id" if triage.task_data(t).codebook.mode == "multi" else "pred"
            cal, tst = d[d.split == "calib"], d[d.split == "test"]
            acc = cal.groupby(key).apply(lambda g: np.average(g.correct, weights=g.weight))
            q = tst[key].map(acc).fillna(np.average(cal.correct, weights=cal.weight)).to_numpy(float)
            k, w = tst.correct.to_numpy(float), tst.weight.to_numpy(float)
            rows.append({"task": t, "signal": s, "auroc_category_only": fast_auroc(q, k, w), "auroc_raw": fast_auroc(tst.conf.to_numpy(float), k, w)})
    df = pd.DataFrame(rows)
    pooled = df.groupby("signal", as_index=False).mean(numeric_only=True).assign(task="POOLED")
    return pd.concat([df, pooled], ignore_index=True)


def cluster_weights(task: str, unit: str = "cluster") -> np.ndarray:
    """Total design weight of each sampling unit's decisions in the task sample (what the bound B is the maximum
    of). unit "cluster": comment, transcript or lesson; "segment": one comment, utterance or sentence."""
    td = triage.task_data(task)
    per_segment = len(td.codebook.code_ids) if td.codebook.mode == "multi" else 1
    w = td.sample.weight * per_segment
    return (w.groupby(td.sample.group).sum() if unit == "cluster" else w).to_numpy(float)


def _perfect_coder_certified(r: np.ndarray, n: int, n_pool: int, delta: float) -> bool:
    """Corpus LTT for a coder that accepts every decision and is never wrong, from the first n units."""
    y = (triage.EPS * (1 - r[:n]))[None, :]  # Z_i = -eps W_i, so y_i = eps (1 - W_i / B)
    return bool(triage.certify(y, triage.EPS, delta, n_pool)[0])


def _fewest_possible(r: np.ndarray, delta: float) -> float:
    """A bound that holds for every calibration set and visiting order: for a perfect coder each unit's bet
    multiplies the capital by at most 1 + eps r_i (bets are capped at 1, and the null mean is at most eps),
    so certification needs sum_{coded} log(1 + eps r_i) >= log(1 / delta). Returns the fewest units for which
    that is possible (the largest units first), or nan if even all units but one fall short."""
    gain = np.sort(np.log1p(triage.EPS * r))[::-1]
    reach = np.cumsum(gain[:-1])  # at least one unit must remain uncoded
    hit = np.flatnonzero(reach >= np.log(1 / delta))
    return float(hit[0] + 1) if hit.size else np.nan


def ltt_feasibility(sequences: int = 20) -> pd.DataFrame:
    """Procedure-specific: how many hand-coded units corpus LTT needs before it certifies a perfect coder, with
    the 16-threshold grid (Bonferroni) and with one threshold fixed in advance, under two sampling designs:
    whole clusters (as in our splits) and single segments drawn uniformly from the corpus. 'Fewest possible' is
    an ordering-independent bound; 'procedure' is the median first certification along random nested
    sequences of units (one sequence is one possible hand-coding order, so this is not a bound)."""
    rows = []
    for t in CONFIRMATORY:
        td = triage.task_data(t)
        d = triage.load(t, "jev")
        cal_groups = d[d.split == "calib"].group.unique()
        for unit in ("cluster", "segment"):
            if unit == "segment" and td.codebook.mode == "multi":
                continue  # one MFRC comment is both the cluster and the segment
            wts = cluster_weights(t, unit)
            r = wts / wts.max()
            n_pool = len(r)
            row = {"task": t, "unit": "transcript or lesson" if unit == "cluster" and t != "mfrc" else ("comment" if t == "mfrc" else "utterance or sentence"),
                   "units_in_corpus": n_pool, "mean_unit_share_of_bound": float(r.mean())}
            if unit == "cluster":
                per_segment = len(td.codebook.code_ids) if td.codebook.mode == "multi" else 1
                w_by = (td.sample.weight * per_segment).groupby(td.sample.group).sum()
                r_cal = (w_by.reindex(cal_groups) / wts.max()).to_numpy(float)
                row["calib_units"] = len(r_cal)
            for label, delta in (("grid of 16", triage.DELTA / len(triage.GRID)), ("one threshold", triage.DELTA)):
                if unit == "cluster":
                    row[f"frozen split certifies a perfect coder ({label})"] = _perfect_coder_certified(r_cal, len(r_cal), n_pool, delta)
                row[f"fewest possible ({label})"] = _fewest_possible(r, delta)
                needs = []
                for k in range(sequences):
                    rr = r[np.random.default_rng(k).permutation(n_pool)]
                    lo = int(row[f"fewest possible ({label})"]) if np.isfinite(row[f"fewest possible ({label})"]) else None
                    needs.append(np.nan if lo is None else next((n for n in range(lo, n_pool) if _perfect_coder_certified(rr, n, n_pool, delta)), np.nan))
                row[f"procedure, median of {sequences} sequences ({label})"] = float(np.nanmedian(needs)) if np.isfinite(needs).any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def utterance_sampling(reps: int = 200, share: float = triage.CALIB_SHARE) -> pd.DataFrame:
    """Exploratory: the dialogue tasks under a different hand-coding design. Each re-split hand-codes a uniformly
    random share of the task's utterances (or sentences), with their context as the coder saw it, and certifies
    the rest of the corpus with corpus LTT, the sampling unit being the segment (B = largest segment weight).
    The guarantee needs no independence between segments of one transcript, only the random choice of segments."""
    rows = []
    for t in [x for x in CONFIRMATORY if x != "mfrc"]:
        bound = float(cluster_weights(t, "segment").max())
        for s in SIGNALS:
            d = triage.load(t, s)
            if d is None:
                continue
            seg = d.segment_id.astype(str).to_numpy()
            codes, uniq = pd.factorize(seg, sort=True)
            conf, err, w = triage.arrays(d)
            rng = np.random.default_rng(triage.SEED)
            res = {r: [] for r in ("empirical", "ltt")}
            for _ in range(reps):
                cal = (rng.random(len(uniq)) < share)[codes]
                for rule in res:
                    thr = triage.fit_threshold(conf[cal], err[cal], w[cal], rule, groups=seg[cal], bound=bound, n_pool=len(uniq))
                    res[rule].append(triage.apply_threshold(conf[~cal], err[~cal], w[~cal], thr))
            for rule, vals in res.items():
                cov = np.array([v[0] for v in vals])
                risk = np.array([v[1] for v in vals])
                rows.append({"task": t, "signal": s, "rule": rule, "reps": reps, "mean_coverage": float(cov.mean()),
                             "share_zero_coverage": float((cov == 0).mean()),
                             "violation_rate": float(np.where(cov > 0, np.nan_to_num(risk, nan=0.0) > triage.EPS, False).mean())})
    return pd.DataFrame(rows)


def single_threshold(t0: float = 0.95) -> pd.DataFrame:
    """Sensitivity: corpus LTT testing one threshold fixed in advance (t0, no multiplicity) vs the 16-point grid."""
    rows = []
    for t in CONFIRMATORY:
        bound = triage.task_bound(t)
        for s in SIGNALS:
            d = triage.load(t, s)
            cal, tst = d[d.split == "calib"], d[d.split == "test"]
            cc, ec, wc = triage.arrays(cal)
            ct, et, wt = triage.arrays(tst)
            row = {"task": t, "signal": s}
            for label, grid in (("grid", triage.GRID), ("single", np.array([t0]))):
                thr = triage.cluster_ltt(cc, ec, wc, cal.group.to_numpy(), bound, d.group.nunique(), grid=grid)
                row[f"{label}_coverage"], row[f"{label}_risk"] = triage.apply_threshold(ct, et, wt, thr)
            rows.append(row)
    df = pd.DataFrame(rows)
    pooled = df.groupby("signal", as_index=False).mean(numeric_only=True).assign(task="POOLED")
    return pd.concat([df, pooled], ignore_index=True)


def _true_risk(t: np.ndarray, b: float) -> np.ndarray:
    """Selective risk E[1 - conf | conf >= t] for conf = 1 - 0.5 X, X ~ Beta(1, b), P(error | conf) = 1 - conf."""
    u = np.clip(2 * (1 - np.asarray(t, float)), 0, 1)  # conf >= t  <=>  X <= u
    mass = 1 - (1 - u) ** b
    first = -u * (1 - u) ** b + (1 - (1 - u) ** (b + 1)) / (b + 1)  # integral of x * b (1 - x)^(b-1) on [0, u]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(mass > 0, 0.5 * first / mass, 0.0)


def _corpus(rng, n_clusters: int, size: int, loading: float, b: float):
    g = np.repeat(np.arange(n_clusters), size)
    conf = 1 - 0.5 * rng.beta(1, b, n_clusters * size)
    z = loading * np.repeat(rng.standard_normal(n_clusters), size) + np.sqrt(1 - loading**2) * rng.standard_normal(n_clusters * size)
    return g, conf, (norm.cdf(z) > conf).astype(float)  # calibrated marginally; errors correlated within a cluster


def ltt_simulation(sims: int = 300, b: float = 6.0, seed: int = 1) -> pd.DataFrame:
    """Each simulation draws a corpus of independent clusters, hand-codes a random 30% of them, and certifies a
    threshold with each rule. Two targets: the selective risk of the uncoded remainder (corpus guarantee) and of
    the population (closed form). The copula loading is not the error correlation, which is reported separately."""
    rows = []
    for n_clusters, size, loading in [(2000, 8, 0.0), (2000, 8, 0.5), (2000, 8, 0.9), (400, 10, 0.9), (170, 30, 0.9)]:
        rng = np.random.default_rng(seed)
        g, _, e = _corpus(np.random.default_rng(seed + 99), 4000, size, loading, b)
        e = e.reshape(-1, size)
        pair = np.mean([np.corrcoef(e[:, 0], e[:, k])[0, 1] for k in range(1, size)])
        res = {r: {"remainder": [], "population": [], "coverage": []} for r in ("ltt_dec", "ltt_iid", "ltt")}
        for _ in range(sims):
            g, conf, err = _corpus(rng, n_clusters, size, loading, b)
            cal = (rng.random(n_clusters) < triage.CALIB_SHARE)[g]
            w = np.ones_like(conf)
            for rule in res:
                thr = triage.fit_threshold(conf[cal], err[cal], w[cal], rule, groups=g[cal], bound=float(size), n_pool=n_clusters)
                cov, risk = triage.apply_threshold(conf[~cal], err[~cal], w[~cal], thr)
                res[rule]["remainder"].append(cov > 0 and risk > triage.EPS)
                res[rule]["population"].append(np.isfinite(thr) and _true_risk(np.array([thr]), b)[0] > triage.EPS + 1e-12)
                res[rule]["coverage"].append(cov)
        for rule, v in res.items():
            rows.append({"clusters": n_clusters, "cluster_size": size, "copula_loading": loading, "error_correlation": float(pair),
                         "rule": rule, "sims": sims, "p_remainder_risk_above_eps": float(np.mean(v["remainder"])),
                         "p_population_risk_above_eps": float(np.mean(v["population"])), "mean_coverage": float(np.mean(v["coverage"]))})
    best = np.linspace(0.5, 1, 100_001)
    safe = best[_true_risk(best, b) <= triage.EPS].min()
    rows.append({"rule": "best safe threshold", "sims": 0, "mean_coverage": float(1 - (1 - 2 * (1 - safe)) ** b)})
    return pd.DataFrame(rows)


def temperature_floor(floors=(1e-4, 1e-12)) -> pd.DataFrame:
    """Temperature scaling and per-code Platt maps with different probability floors before log/logit (the review's
    sensitivity check)."""
    rows = []
    old = triage.CLIP
    try:
        for floor in floors:
            triage.CLIP = floor
            for s in ["jev", "oaid", "kev27b", "sonnet-verbal", "luna-logprob", "oss-logprob", "qwen-logprob"]:
                rr = [r for t in CONFIRMATORY for r in triage.recalibration_rows(t, s) if r["method"] in ("raw", "temperature")]
                df = pd.DataFrame(rr)
                for method, g in df.groupby("method"):
                    rows.append({"floor": floor, "signal": s, "method": method, "ece": g.ece.mean(), "error_auroc": g.error_auroc.mean(),
                                 "temperature_range": f"{g.temperature.min():.2f}-{g.temperature.max():.2f}" if method == "temperature" else ""})
                pc = []  # per-code Platt maps (EQ3) are not monotone across codes, so the floor can change their ranking
                for t in CONFIRMATORY:
                    d = triage.load(t, s)
                    key = "code_id" if triage.task_data(t).codebook.mode == "multi" else "pred"
                    cal, tst = d[d.split == "calib"], d[d.split == "test"]
                    q = triage.per_code_platt(cal, tst, key, 50)
                    pc.append(fast_auroc(q, tst.correct.to_numpy(float), tst.weight.to_numpy(float)))
                rows.append({"floor": floor, "signal": s, "method": "per-code Platt", "ece": np.nan, "error_auroc": float(np.mean(pc)),
                             "temperature_range": ""})
    finally:
        triage.CLIP = old
    return pd.DataFrame(rows)


def utterance_evidence(share: float = triage.CALIB_SHARE) -> pd.DataFrame:
    """Illustration for the segment-sampling design (first re-split of utterance_sampling): on the hand-coded segments,
    the largest grid-threshold accepted set with empirical error <= 5%, its size and error. Shows whether a coder fails
    to certify because its confident sets are too error-prone or too small."""
    rows = []
    for t in [x for x in CONFIRMATORY if x != "mfrc"]:
        for s in SIGNALS:
            d = triage.load(t, s)
            if d is None:
                continue
            codes, uniq = pd.factorize(d.segment_id.astype(str).to_numpy(), sort=True)
            conf, err, w = triage.arrays(d)
            cal = (np.random.default_rng(triage.SEED).random(len(uniq)) < share)[codes]
            c, e, ww = conf[cal], err[cal], w[cal]
            best = {"threshold": np.nan, "accepted": 0, "error": np.nan}
            for g in triage.GRID:
                a = c >= g
                if a.any():
                    r = float((ww[a] * e[a]).sum() / ww[a].sum())
                    if r <= triage.EPS and a.sum() > best["accepted"]:
                        best = {"threshold": float(g), "accepted": int(a.sum()), "error": r}
            rows.append({"task": t, "signal": s, "hand_coded_segments": int(cal.sum()), **best})
    return pd.DataFrame(rows)


def run() -> dict[str, pd.DataFrame]:
    out = {"category_only": category_only(), "ltt_feasibility": ltt_feasibility(), "ltt_single_threshold": single_threshold(),
           "ltt_simulation": ltt_simulation(), "temperature_floor": temperature_floor(), "utterance_sampling": utterance_sampling(),
           "utterance_evidence": utterance_evidence()}
    for name, df in out.items():
        df.to_csv(RESULTS / f"{name}.csv", index=False)
    return out
