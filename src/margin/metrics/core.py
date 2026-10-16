"""Human reference labels, coder decisions, and the clustered bootstrap shared by all metrics.

A *decision* is the unit RQ1 triages:
- exclusive codebooks: one decision per segment (which code), confidence = top probability;
- multi codebooks: one decision per (segment, code) (applies or not), confidence = max(p, 1 - p).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from margin.schema import Codebook


def soft_labels(labels: pd.DataFrame, codebook: Codebook, exclude: str | None = None) -> pd.DataFrame:
    """segment_id x code_id -> share of annotators who applied the code, plus annotator count."""
    if exclude is not None:
        labels = labels[labels.annotator != exclude]
    n = labels.groupby("segment_id").size().rename("n_annotators")
    applied = labels.explode("codes").groupby(["segment_id", "codes"]).size().rename("k")
    grid = pd.MultiIndex.from_product([n.index, codebook.code_ids], names=["segment_id", "code_id"])
    k = applied.rename_axis(["segment_id", "code_id"]).reindex(grid, fill_value=0)
    out = k.reset_index().merge(n.reset_index(), on="segment_id")
    out["soft"] = out.k / out.n_annotators
    return out


def reference(labels: pd.DataFrame, codebook: Codebook, exclude: str | None = None) -> pd.DataFrame:
    """Majority reference. multi: y = 1 if more than half applied the code (exact ties dropped).
    exclusive: ref_code = plurality code (segments with a tied plurality dropped)."""
    soft = soft_labels(labels, codebook, exclude)
    if codebook.mode == "multi":
        soft = soft[soft.soft != 0.5].copy()
        soft["y"] = (soft.soft > 0.5).astype(int)
        return soft
    top = soft.sort_values(["segment_id", "k"], ascending=[True, False])
    best = top.groupby("segment_id").head(2).groupby("segment_id").k.agg(list)
    clear = best[best.map(lambda ks: len(ks) == 1 or ks[0] > ks[1])].index
    ref = top[top.segment_id.isin(clear)].drop_duplicates("segment_id")[["segment_id", "code_id", "n_annotators"]]
    return ref.rename(columns={"code_id": "ref_code"})


def decisions(
    preds: pd.DataFrame,
    ref: pd.DataFrame,
    codebook: Codebook,
    sample: pd.DataFrame | None = None,
    threshold: float | dict[str, float] = 0.5,
) -> pd.DataFrame:
    """Join coder probabilities to the reference. Columns: segment_id, [code_id], pred, conf, correct,
    p, y, weight, group. `threshold` (multi only) may be per code."""
    if codebook.mode == "exclusive":
        wide = preds.pivot_table(index="segment_id", columns="code_id", values="p").reindex(columns=codebook.code_ids)
        out = pd.DataFrame({"pred": wide.idxmax(axis=1), "conf": wide.max(axis=1)}).reset_index()
        out = out.merge(ref[["segment_id", "ref_code"]], on="segment_id")
        out["correct"] = (out.pred == out.ref_code).astype(int)
    else:
        out = preds[["segment_id", "code_id", "p"]].merge(ref[["segment_id", "code_id", "y", "soft"]], on=["segment_id", "code_id"])
        t = out.code_id.map(threshold) if isinstance(threshold, dict) else threshold
        out["pred"] = (out.p >= t).astype(int)
        out["conf"] = np.maximum(out.p, 1 - out.p)
        out["correct"] = (out.pred == out.y).astype(int)
    if sample is not None:
        out = out.merge(sample[["segment_id", "weight", "group"]], on="segment_id", how="left")
    else:
        out["weight"], out["group"] = 1.0, out.segment_id
    return out


def apply_adjudication(dec: pd.DataFrame, verdicts: pd.DataFrame, codebook: Codebook) -> pd.DataFrame:
    """Swap in adjudicated labels where an item was adjudicated; everything else keeps the human
    reference (conservative: unadjudicated disagreements still count as coder errors).
    exclusive: correct = pred is among the acceptable codes; items judged "none" are dropped.
    multi: y = adjudicated value; items judged ambiguous are dropped."""
    key = ["segment_id"] + (["code_id"] if codebook.mode == "multi" else [])
    v = verdicts[key + ["acceptable", "ambiguous"]]
    m = dec.merge(v, on=key, how="left")
    has = m.acceptable.notna()
    if codebook.mode == "exclusive":
        drop = has & (m.acceptable.map(lambda a: len(a) if isinstance(a, list) else 1) == 0)
        m = m[~drop].copy()
        has = m.acceptable.notna()
        ok = [p in a for p, a in zip(m.pred[has], m.acceptable[has])]
        m.loc[has, "correct"] = np.array(ok, dtype=int)
        m.loc[has, "ref_code"] = [p if good else a[0] for p, a, good in zip(m.pred[has], m.acceptable[has], ok)]
    else:
        m = m[~(has & m.ambiguous.fillna(False).astype(bool))].copy()
        has = m.acceptable.notna()
        m.loc[has, "y"] = [int(a[0]) for a in m.acceptable[has]]
        m["correct"] = (m.pred == m.y).astype(int)
    return m.drop(columns=["acceptable", "ambiguous"])


def bootstrap(
    df: pd.DataFrame, stat: Callable[[pd.DataFrame], float], *, reps: int = 1000, seed: int = 0, cluster: str = "group"
) -> tuple[float, float, float]:
    """(estimate, lo, hi) with a 95% percentile interval, resampling whole clusters."""
    est = stat(df)
    groups = df[cluster].unique()
    by_group = {g: idx for g, idx in df.groupby(cluster).indices.items()}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pick = rng.choice(groups, size=len(groups), replace=True)
        rows = np.concatenate([by_group[g] for g in pick])
        v = stat(df.iloc[rows])
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return float(est), float(lo), float(hi)
