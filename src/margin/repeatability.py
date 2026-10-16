"""Run-to-run repeatability: the canonical requests for 100 test segments per task were sent again
(experiment `repeat1`, cached separately), and the second answers are compared with the first.

Per coder signal: share of decisions whose label flips, share of identical probabilities, mean
absolute change in probability and in decision confidence, accuracy in each run, and whether flips
sit on low-confidence decisions (AUROC of 1 - first-run confidence for predicting a flip).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from margin.analysis import CONFIRMATORY, RESULTS, TaskData
from margin.run import RUNS

PAIRS = {  # signal: (original experiment, file stem); the repeat has the same stem under repeat1/
    "jev": ("jev_main", "jev__canonical"),
    "oaid": ("oaid_main", "oaid__canonical"),
    "luna-verbal": ("llm_main", "luna-verbal__canonical"),
    "luna-logprob": ("llm_main", "luna-logprob__canonical"),
    "haiku-verbal": ("llm_main", "haiku-verbal__canonical"),
    "sonnet-verbal": ("sonnet_main", "sonnet-verbal__canonical"),
    "clef": ("dm_main", "clef__canonical"), "clef_flash": ("dm_main", "clef_flash__canonical"), "pplx": ("dm_main", "pplx__canonical"),
}


def compare(td: TaskData, first: pd.DataFrame, second: pd.DataFrame) -> dict[str, float]:
    first = first[first.segment_id.isin(second.segment_id)]
    d1, d2 = td.decisions(first), td.decisions(second)
    m = d1.merge(d2[td.key + ["pred", "conf", "correct"]], on=td.key, suffixes=("_1", "_2"))
    p = first[["segment_id", "code_id", "p"]].merge(second[["segment_id", "code_id", "p"]], on=["segment_id", "code_id"], suffixes=("_1", "_2"))
    flip = (m.pred_1 != m.pred_2).to_numpy(int)
    return {
        "segments": m.segment_id.nunique(), "decisions": len(m), "flip_rate": float(flip.mean()),
        "identical_p": float((p.p_1 == p.p_2).mean()), "mean_abs_dp": float((p.p_1 - p.p_2).abs().mean()),
        "mean_abs_dconf": float((m.conf_1 - m.conf_2).abs().mean()),
        "accuracy_1": float(m.correct_1.mean()), "accuracy_2": float(m.correct_2.mean()),
        "flips_at_low_conf_auroc": float(roc_auc_score(flip, -m.conf_1)) if 0 < flip.sum() < len(flip) else np.nan,
    }


def run() -> pd.DataFrame:
    rows = []
    for task in CONFIRMATORY:
        td = TaskData.load(task)
        for signal, (exp, stem) in PAIRS.items():
            a, b = RUNS / exp / task / f"{stem}.parquet", RUNS / "repeat1" / task / f"{stem}.parquet"
            if a.exists() and b.exists():
                rows.append({"task": task, "signal": signal} | compare(td, pd.read_parquet(a), pd.read_parquet(b)))
    out = pd.DataFrame(rows)
    pooled = out.groupby("signal", sort=False).mean(numeric_only=True).reset_index().assign(task="POOLED")
    out = pd.concat([out, pooled], ignore_index=True)
    out.to_csv(RESULTS / "repeatability.csv", index=False)
    return out
