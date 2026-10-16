"""Agreement with the human reference (supporting result): Cohen's kappa, F1, Krippendorff's alpha."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from margin.schema import Codebook


def kappa(dec: pd.DataFrame, codebook: Codebook, weighted: bool = False) -> float:
    """exclusive: multi-class kappa on (ref_code, pred). multi: macro mean of per-code binary kappa."""
    w = dec.weight if weighted else None
    if codebook.mode == "exclusive":
        return float(cohen_kappa_score(dec.ref_code, dec.pred, labels=codebook.code_ids, sample_weight=w))
    per = per_code_kappa(dec, weighted)
    return float(np.nanmean(list(per.values())))


def per_code_kappa(dec: pd.DataFrame, weighted: bool = False) -> dict[str, float]:
    out = {}
    for code, d in dec.groupby("code_id"):
        if d.y.nunique() < 2 and d.pred.nunique() < 2:
            out[code] = np.nan  # kappa undefined when both raters are constant
            continue
        out[code] = float(cohen_kappa_score(d.y, d.pred, labels=[0, 1], sample_weight=d.weight if weighted else None))
    return out


def macro_f1(dec: pd.DataFrame, codebook: Codebook) -> float:
    if codebook.mode == "exclusive":
        return float(f1_score(dec.ref_code, dec.pred, labels=codebook.code_ids, average="macro", zero_division=0))
    return float(np.mean([f1_score(d.y, d.pred, zero_division=0) for _, d in dec.groupby("code_id")]))


def krippendorff_nominal(labels: pd.DataFrame, codebook: Codebook, code: str | None = None) -> float:
    """Nominal alpha across all annotators. exclusive: the chosen code. multi: applied/not for `code`."""
    import krippendorff

    if codebook.mode == "exclusive":
        values = labels.assign(v=labels.codes.str[0].map({c: i for i, c in enumerate(codebook.code_ids)}))
    else:
        values = labels.assign(v=labels.codes.map(lambda cs: int(code in cs)))
    matrix = values.pivot_table(index="annotator", columns="segment_id", values="v", aggfunc="first")
    if matrix.shape[0] < 2:
        return float("nan")
    return float(krippendorff.alpha(reliability_data=matrix.to_numpy(dtype=float), level_of_measurement="nominal"))
