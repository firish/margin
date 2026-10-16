"""RQ2: does model uncertainty sit where human coders disagree, and is the model inside the human range?"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from margin.metrics.core import reference, soft_labels
from margin.schema import Codebook


def _entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(p, 1e-12, 1)
    return -(p * np.log(p)).sum(axis=axis)


def uncertainty_vs_humans(preds: pd.DataFrame, labels: pd.DataFrame, codebook: Codebook, min_annotators: int = 3) -> dict:
    """Spearman(model entropy, human label entropy) and AUROC(model entropy -> humans not unanimous).
    Unit: segment (exclusive) or (segment, code) (multi). Entropies are normalized to [0, 1]."""
    soft = soft_labels(labels, codebook)
    soft = soft[soft.n_annotators >= min_annotators]
    if codebook.mode == "exclusive":
        k = len(codebook.code_ids)
        h = soft.pivot(index="segment_id", columns="code_id", values="soft").reindex(columns=codebook.code_ids)
        m = preds.pivot_table(index="segment_id", columns="code_id", values="p").reindex(columns=codebook.code_ids)
        ids = h.index.intersection(m.index)
        human = _entropy(h.loc[ids].to_numpy()) / np.log(k)
        model = _entropy(m.loc[ids].to_numpy()) / np.log(k)
        split = (h.loc[ids].max(axis=1) < 1).to_numpy()
    else:
        d = preds.merge(soft, on=["segment_id", "code_id"])
        human = _entropy(np.c_[d.soft, 1 - d.soft]) / np.log(2)
        model = _entropy(np.c_[d.p, 1 - d.p]) / np.log(2)
        split = ((d.soft > 0) & (d.soft < 1)).to_numpy()
    rho = spearmanr(model, human).statistic if len(human) > 2 else np.nan
    auroc = roc_auc_score(split, model) if 0 < split.sum() < len(split) else np.nan
    return {"n": int(len(human)), "spearman": float(rho), "auroc_split": float(auroc), "share_split": float(split.mean())}


def leave_one_coder_out(preds: pd.DataFrame, labels: pd.DataFrame, codebook: Codebook, min_others: int = 2) -> pd.DataFrame:
    """For each annotator a: kappa(a, majority of the others) and kappa(model, same majority), on the
    same items. exclusive: multi-class kappa. multi: macro mean of per-code binary kappa."""
    model_wide = None
    if codebook.mode == "exclusive":
        model_wide = preds.pivot_table(index="segment_id", columns="code_id", values="p").idxmax(axis=1)
    rows = []
    for a in sorted(labels.annotator.unique()):
        mine = labels[labels.annotator == a].set_index("segment_id").codes
        others = labels[labels.segment_id.isin(mine.index) & (labels.annotator != a)]
        enough = others.groupby("segment_id").size()
        others = others[others.segment_id.isin(enough[enough >= min_others].index)]
        ref = reference(others, codebook)
        if codebook.mode == "exclusive":
            ids = ref.segment_id[ref.segment_id.isin(model_wide.index)]
            r = ref.set_index("segment_id").ref_code.loc[ids]
            human_k = cohen_kappa_score(r, mine.loc[ids].str[0], labels=codebook.code_ids)
            model_k = cohen_kappa_score(r, model_wide.loc[ids], labels=codebook.code_ids)
            n = len(ids)
        else:
            mp = preds.assign(pred=(preds.p >= 0.5).astype(int))[["segment_id", "code_id", "pred"]]
            d = ref.merge(mp, on=["segment_id", "code_id"])
            d["human"] = [int(c in mine[s]) for s, c in zip(d.segment_id, d.code_id)]
            hk, mk = [], []
            for _, g in d.groupby("code_id"):
                if g.y.nunique() > 1:
                    hk.append(cohen_kappa_score(g.y, g.human, labels=[0, 1]))
                    mk.append(cohen_kappa_score(g.y, g.pred, labels=[0, 1]))
            human_k, model_k, n = np.mean(hk), np.mean(mk), d.segment_id.nunique()
        rows.append({"annotator": a, "n": n, "kappa_human": float(human_k), "kappa_model": float(model_k)})
    out = pd.DataFrame(rows)
    out["model_minus_human"] = out.kappa_model - out.kappa_human
    return out
