"""Stratified samples with inclusion weights, and a calib/test split by group.

Exclusive tasks are stratified on the reference (plurality) label with a per-class floor so rare
codes have enough items; each segment carries a Horvitz-Thompson weight N_stratum / n_stratum so
prevalence-sensitive metrics can be reported for the population. Segments flagged
``multi_annotated`` (AnnoMI's 10-annotator subset) are always included, with weight 1.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from margin.datasets.base import Task
from margin.schema import Codebook

SAMPLES = Path("data/samples")
CALIB_PERCENT = 30


def reference_labels(labels: pd.DataFrame, codebook: Codebook) -> pd.Series:
    """Per-segment reference used for stratification: plurality code (exclusive) or None (multi)."""
    if codebook.mode == "multi":
        return pd.Series(None, index=labels.segment_id.unique(), dtype=object)
    order = {c: i for i, c in enumerate(codebook.code_ids)}
    first = labels.assign(code=labels.codes.str[0]).groupby(["segment_id", "code"]).size().reset_index(name="n")
    first["rank"] = first.code.map(order)
    first = first.sort_values(["segment_id", "n", "rank"], ascending=[True, False, True])
    return first.drop_duplicates("segment_id").set_index("segment_id").code


def split_of(group: str) -> str:
    return "calib" if int(hashlib.sha1(group.encode()).hexdigest(), 16) % 100 < CALIB_PERCENT else "test"


def allocate(sizes: pd.Series, n: int, floor: int) -> pd.Series:
    """Per-stratum sample sizes: at least min(floor, size), the rest proportional, never above size."""
    take = np.minimum(sizes, floor).astype(int)
    remaining = n - int(take.sum())
    if remaining < 0:
        raise ValueError(f"n={n} is below the floors' total {int(take.sum())}")
    spare = sizes - take
    while remaining > 0 and spare.sum() > 0:
        share = np.floor(remaining * spare / spare.sum()).astype(int)
        if share.sum() == 0:
            share[spare.idxmax()] = 1
        share = np.minimum(share, spare)
        take += share
        spare -= share
        remaining -= int(share.sum())
    return take


def draw(task: Task, codebook: Codebook, n: int, *, floor: int = 60, seed: int = 0) -> pd.DataFrame:
    segs = task.segments.set_index("segment_id")
    forced = segs.index[segs.get("multi_annotated", pd.Series(False, index=segs.index)).astype(bool)]
    pool = segs.drop(index=forced)
    ref = reference_labels(task.labels, codebook).reindex(pool.index)
    strata = ref.fillna("all")
    sizes = strata.value_counts()
    take = allocate(sizes, n - len(forced), floor if codebook.mode == "exclusive" else 0)

    rng = np.random.default_rng(seed)
    parts = []
    for stratum, k in take.items():
        ids = strata.index[strata == stratum].to_numpy()
        chosen = rng.choice(ids, size=int(k), replace=False)
        parts.append(pd.DataFrame({"segment_id": chosen, "stratum": stratum, "weight": sizes[stratum] / k}))
    if len(forced):
        parts.append(pd.DataFrame({"segment_id": forced, "stratum": "multi_annotated", "weight": 1.0}))
    out = pd.concat(parts, ignore_index=True)
    out["group"] = segs.loc[out.segment_id, "group"].to_numpy()
    out["split"] = out.group.map(split_of)
    return out.sort_values("segment_id").reset_index(drop=True)


def save(sample: pd.DataFrame, task: str, tag: str) -> Path:
    SAMPLES.mkdir(parents=True, exist_ok=True)
    path = SAMPLES / f"{task}-{tag}.parquet"
    sample.to_parquet(path, index=False)
    return path


def load(task: str, tag: str) -> pd.DataFrame:
    return pd.read_parquet(SAMPLES / f"{task}-{tag}.parquet")
