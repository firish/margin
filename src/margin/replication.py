"""Dunivin (2025) replication section (plan section 11; exploratory).

1. Reproduce the paper's Table 2 from the published GPT predictions (also pinned by tests/test_dunivin.py).
2. Per-code kappa on the paper's 111-passage test set for the published GPT runs, Jev, and our LLM signals.
3. Triage metrics on our calib/test split of the 120 coded passages (small n; exploratory).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from margin.analysis import RESULTS, SIGNALS, TaskData, contrasts, signal_table
from margin.datasets.dunivin import TEST_IDS, published_predictions
from margin.metrics.agreement import per_code_kappa
from margin.metrics.core import decisions

PUBLISHED = {
    "GPT-4 per-code CoT (paper's best)": ("per-code-with-justification_t=0_top_p=1_model=gpt-4", 0.68),
    "GPT-4 per-code no CoT": ("per-code-without-justification_t=0_top_p=1_model=gpt-4", 0.59),
    "GPT-4 full-codebook CoT": ("full-codebook_with_justification_t=0_top_p=1_model=gpt-4", 0.60),
    "GPT-4 full-codebook no CoT": ("full-codebook_without_justification_t=0_top_p=1_model=gpt-4", 0.46),
    "GPT-3.5 per-code CoT": ("per-code-with-justification_t=0_top_p=1_model=gpt-3.5", 0.34),
    "GPT-3.5 per-code no CoT": ("per-code-without-justification_t=0_top_p=1_model=gpt-3.5", 0.29),
}
OURS = ["jev", "luna-verbal", "haiku-verbal", "oss-verbal", "luna-logprob", "oss-logprob", "agree3", "luna_sc-vote", "sonnet-verbal",
        "oaid", "clef", "clef_flash", "pplx", "kev9b", "kev27b", "dm_committee"]


def per_code_table() -> pd.DataFrame:
    td = TaskData.load("dunivin")
    test = {f"dunivin-{i}" for i in TEST_IDS}
    ref = td.ref[td.ref.segment_id.isin(test)]
    rows = []
    for name, (cfg, paper_mean) in PUBLISHED.items():
        k = per_code_kappa(decisions(published_predictions(cfg), ref, td.codebook))
        rows.append({"coder": name, "source": "published run", "paper_mean": paper_mean} | k)
    for s in OURS:
        p = td.preds(s)
        if p is None:
            continue
        p = p[p.segment_id.isin(test)]
        rows.append({"coder": s, "source": "this study", "n_passages": p.segment_id.nunique()} | per_code_kappa(decisions(p, ref, td.codebook)))
    out = pd.DataFrame(rows)
    codes = td.codebook.code_ids
    out["mean_kappa"] = out[codes].mean(axis=1)
    return out[["coder", "source", "n_passages", "paper_mean", "mean_kappa"] + codes]


def run(reps: int = 1000) -> dict[str, pd.DataFrame]:
    out = {
        "dunivin_per_code": per_code_table(),
        "dunivin_triage": signal_table(["dunivin"]),
        "dunivin_contrasts": contrasts([("jev", s) for s in OURS if s != "jev"], ["dunivin"], reps),
    }
    for name, df in out.items():
        df.to_csv(RESULTS / f"{name}.csv", index=False)
    return out
