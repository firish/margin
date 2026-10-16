"""RQ3: how a surface perturbation changes decisions, accuracy, calibration, and triage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from margin.metrics.agreement import kappa
from margin.metrics.calibration import ece
from margin.metrics.selective import aurc, error_auroc
from margin.schema import Codebook


def compare(base: pd.DataFrame, variant: pd.DataFrame, codebook: Codebook) -> dict:
    """`base` and `variant` are decision frames (metrics.core.decisions) for the same coder.
    Returns flip rate, deltas (variant - base), and whether flips concentrate where base was unsure."""
    key = ["segment_id"] + (["code_id"] if codebook.mode == "multi" else [])
    d = base.merge(variant, on=key, suffixes=("_b", "_v"))
    flipped = (d.pred_b != d.pred_v).to_numpy()
    b = base[base.set_index(key).index.isin(d.set_index(key).index)]
    v = variant[variant.set_index(key).index.isin(d.set_index(key).index)]
    w = d.weight_b.to_numpy(float) if "weight_b" in d else np.ones(len(d))  # design weights, as in the bootstrap CIs
    wb = b.weight if "weight" in b else None
    wv = v.weight if "weight" in v else None
    out = {
        "n": len(d),
        "flip_rate": float(np.average(flipped, weights=w)),
        "mean_abs_dp": float(np.average(np.abs(d.conf_b - d.conf_v), weights=w)),
        "d_kappa": kappa(v, codebook, weighted=wv is not None) - kappa(b, codebook, weighted=wb is not None),
        "d_ece": ece(v.conf, v.correct, wv) - ece(b.conf, b.correct, wb),
        "d_error_auroc": error_auroc(v.conf, v.correct, wv) - error_auroc(b.conf, b.correct, wb),
        "d_aurc": aurc(v.conf, v.correct, wv) - aurc(b.conf, b.correct, wb),
    }
    # Do flips happen where the canonical run was already unsure? AUROC of (1 - conf) for flipping.
    out["flips_track_uncertainty_auroc"] = error_auroc(d.conf_b, (~flipped).astype(int)) if 0 < flipped.sum() < len(d) else np.nan
    return out
