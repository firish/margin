"""MFRC: how much of the accepted share is "code absent", and what happens to the codes that are present.

Most MFRC decisions (89% on test) are "this code does not apply", so a policy can accept many decisions at
low error while sending almost every present code to a human, or by never finding them. For each coder
and threshold rule we report, on the frozen test split:
- the share of accepted decisions that are "absent" calls;
- recall of present codes among accepted decisions (reference-positive decisions accepted as positive,
  over all reference-positive decisions) and the share of present codes sent to a human;
- precision of accepted "present" calls.
A prevalence baseline predicts "absent" for every code, with confidence equal to the share of "absent" labels
for that code: the best a coder can do without reading the text. Its scores are learned from labels, so for the
naive rule they are fitted on all calib comments (exploratory, like the rule itself), and for cluster-level LTT
they are fitted on half of the calib comments, frozen, certified on the other half and deployed unchanged.
Exploratory (added after the reviews of the first draft); design-weighted throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from margin import triage
from margin.analysis import RESULTS

SIGNALS = ["jev", "oaid", "clef", "clef_flash", "pplx", "kev9b", "kev27b", "luna-verbal", "haiku-verbal", "oss-verbal", "sonnet-verbal"]


def prevalence_scores(fit: pd.DataFrame, apply_to: pd.DataFrame) -> pd.DataFrame:
    """Predict "absent" for every code, with confidence = the code's weighted share of absent labels in `fit`."""
    absent = fit.assign(neg=1 - fit.y).groupby("code_id").apply(lambda g: np.average(g.neg, weights=g.weight))
    out = apply_to.copy()
    out["pred"], out["conf"] = 0, out.code_id.map(absent).astype(float)
    out["correct"] = (out.y == 0).astype(float)
    return out


def prevalence_baseline() -> pd.DataFrame:
    d = triage.load("mfrc", "jev")
    return prevalence_scores(d[d.split == "calib"], d)


def baseline_ltt(d: pd.DataFrame) -> tuple[float, np.ndarray]:
    """Cluster-level LTT for the prevalence baseline with honest fitting: scores fitted on half of the calib
    comments, certified on the other half (corpus = that half plus test), deployed unchanged on test."""
    cal, tst = d[d.split == "calib"], d[d.split == "test"]
    codes, uniq = pd.factorize(cal.group)
    half = (np.random.default_rng(triage.SEED + 1).random(len(uniq)) < 0.5)[codes]
    cert = prevalence_scores(cal[half], cal[~half])
    cc, ec, wc = triage.arrays(cert)
    thr = triage.fit_threshold(cc, ec, wc, "ltt", groups=cert.group.to_numpy(), bound=triage.task_bound("mfrc"),
                               n_pool=cert.group.nunique() + tst.group.nunique())
    return thr, prevalence_scores(cal[half], tst).conf.to_numpy(float)


def rows(name: str, d: pd.DataFrame) -> list[dict]:
    cal, tst = d[d.split == "calib"], d[d.split == "test"]
    cc, ec, wc = triage.arrays(cal)
    ct_all, et, wt = triage.arrays(tst)
    bound = triage.task_bound("mfrc")
    out = []
    for rule in ["empirical", "ltt", "oracle"]:
        ct = ct_all
        if rule == "oracle":
            thr = triage.fit_threshold(ct, et, wt)
        elif rule == "ltt" and name == "prevalence baseline":
            thr, ct = baseline_ltt(triage.load("mfrc", "jev"))
        else:
            thr = triage.fit_threshold(cc, ec, wc, rule, groups=cal.group.to_numpy(), bound=bound, n_pool=d.group.nunique())
        acc = ct >= thr
        pos, pred_pos = (tst.y == 1).to_numpy(), (tst.pred == 1).to_numpy()
        w = wt
        share = w[acc].sum() / w.sum()
        out.append({"signal": name, "rule": rule, "threshold": thr, "accepted": share,
                    "error_among_accepted": (w[acc] * et[acc]).sum() / w[acc].sum() if acc.any() else np.nan,
                    "accepted_absent_share": (w[acc & ~pred_pos]).sum() / w[acc].sum() if acc.any() else np.nan,
                    "present_codes": w[pos].sum() / w.sum(),
                    "present_recall_accepted": (w[acc & pos & pred_pos]).sum() / w[pos].sum(),
                    "present_sent_to_human": (w[~acc & pos]).sum() / w[pos].sum(),
                    "present_precision_accepted": (w[acc & pred_pos & pos]).sum() / w[acc & pred_pos].sum() if (acc & pred_pos).any() else np.nan})
    return out


def run() -> pd.DataFrame:
    out = []
    for s in SIGNALS:
        d = triage.load("mfrc", s)
        if d is not None:
            out += rows(s, d)
    out += rows("prevalence baseline", prevalence_baseline())
    df = pd.DataFrame(out)
    df.to_csv(RESULTS / "mfrc_prevalence.csv", index=False)
    return df
