"""Calibration of decision confidence (top-label ECE) and of probabilities (Brier)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from margin.schema import Codebook


def reliability(conf: np.ndarray, correct: np.ndarray, weight: np.ndarray | None = None, bins: int = 10) -> pd.DataFrame:
    """Equal-width bins over [0, 1]: mean confidence, accuracy, and weight per bin."""
    w = np.ones_like(conf, dtype=float) if weight is None else np.asarray(weight, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        rows.append(
            {"bin": b, "lo": edges[b], "hi": edges[b + 1], "conf": np.average(conf[m], weights=w[m]),
             "acc": np.average(correct[m], weights=w[m]), "weight": w[m].sum(), "n": int(m.sum())}
        )
    return pd.DataFrame(rows)


def ece(conf, correct, weight=None, bins: int = 10) -> float:
    r = reliability(np.asarray(conf, float), np.asarray(correct, float), weight, bins)
    return float((r.weight * (r.conf - r.acc).abs()).sum() / r.weight.sum())


def brier(preds: pd.DataFrame, ref: pd.DataFrame, codebook: Codebook) -> float:
    """multi: mean (p - y)^2 over (segment, code). exclusive: mean over segments of sum_k (p_k - 1[k=ref])^2."""
    if codebook.mode == "multi":
        d = preds.merge(ref[["segment_id", "code_id", "y"]], on=["segment_id", "code_id"])
        return float(((d.p - d.y) ** 2).mean())
    d = preds.merge(ref[["segment_id", "ref_code"]], on="segment_id")
    d["sq"] = (d.p - (d.code_id == d.ref_code).astype(float)) ** 2
    return float(d.groupby("segment_id").sq.sum().mean())
