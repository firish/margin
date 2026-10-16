"""Amendment v1.4: LLM prompt sensitivity. The verbalized-distribution prompt vs the canonical prompt on
the same test items (the first 150 per task in the harness order), for Sonnet and Haiku."""

from __future__ import annotations

import numpy as np
import pandas as pd

from margin.analysis import CONFIRMATORY, RESULTS, SIGNALS, TaskData, fast_auroc, joint_draws, resample
from margin.metrics.agreement import kappa
from margin.metrics.calibration import ece
from margin.run import RUNS

CODERS = ["sonnet-verbal", "haiku-verbal"]


def arm_metrics(d: pd.DataFrame, codebook) -> dict[str, float]:
    k = d.correct.to_numpy(float)
    w = d.weight.to_numpy(float)
    return {"error_auroc": fast_auroc(d.conf.to_numpy(float), k, w), "ece": ece(d.conf.to_numpy(float), k, w),
            "kappa": float(kappa(d, codebook)), "accuracy": float(np.average(k, weights=w))}


def run(reps: int = 1000) -> pd.DataFrame:
    rows, reps_by = [], {}
    for task in CONFIRMATORY:
        td = TaskData.load(task)
        for sig in CODERS:
            path = RUNS / "prompt_robust" / task / f"{sig}__dist_prompt.parquet"
            if not path.exists():
                continue
            alt = td.decisions(pd.read_parquet(path))
            exp, stem = SIGNALS[sig]
            base = td.decisions(pd.read_parquet(RUNS / exp / task / f"{stem}.parquet"))
            m = base.merge(alt[td.key + ["conf", "correct", "pred"]], on=td.key, suffixes=("", "_alt"))
            m = m[m.split == "test"]
            canon = arm_metrics(m, td.codebook)
            dist = arm_metrics(m.assign(conf=m.conf_alt, correct=m.correct_alt, pred=m.pred_alt), td.codebook)
            # The two arms can differ in correctness too, so the paired AUROC difference uses each arm's own errors.
            # Test clusters drawn once per replicate for the whole corpus, shared with related tasks, as in RQ1.
            members = pd.Series(np.arange(len(m))).groupby(m.group.astype(str).to_numpy()).apply(np.array).to_dict()
            shared = joint_draws(task, reps)["t"]
            draws = np.empty(reps)
            ka, kb, w = m.correct_alt.to_numpy(float), m.correct.to_numpy(float), m.weight.to_numpy(float)
            ca, cb = m.conf_alt.to_numpy(float), m.conf.to_numpy(float)
            for r in range(reps):
                idx = resample(members, shared[r])
                draws[r] = fast_auroc(ca[idx], ka[idx], w[idx]) - fast_auroc(cb[idx], kb[idx], w[idx]) if len(idx) else np.nan
            reps_by[(task, sig)] = draws
            rows.append({"task": task, "signal": sig, "decisions": len(m),
                         **{f"{k}_canonical": v for k, v in canon.items()}, **{f"{k}_dist": v for k, v in dist.items()},
                         "error_auroc_diff": dist["error_auroc"] - canon["error_auroc"],
                         "lo": float(np.nanpercentile(draws, 2.5)), "hi": float(np.nanpercentile(draws, 97.5))})
    out = pd.DataFrame(rows)
    for sig in CODERS:
        parts = [reps_by[(t, sig)] for t in CONFIRMATORY if (t, sig) in reps_by]
        if len(parts) == len(CONFIRMATORY):
            sub = out[out.signal == sig]
            pooled = np.nanmean(np.vstack(parts), axis=0)
            rows.append({"task": "POOLED", "signal": sig, "decisions": int(sub.decisions.sum()),
                         **{c: sub[c].mean() for c in sub.columns if c.endswith(("_canonical", "_dist"))},
                         "error_auroc_diff": float(sub.error_auroc_diff.mean()),
                         "lo": float(np.nanpercentile(pooled, 2.5)), "hi": float(np.nanpercentile(pooled, 97.5))})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "prompt_sensitivity.csv", index=False)
    return out
