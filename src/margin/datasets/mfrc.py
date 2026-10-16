"""Moral Foundations Reddit Corpus (HF USC-MOLA-Lab/MFRC, CC-BY-4.0), deduplicated parquet."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from margin.datasets.base import RAW, Task, short_hash

LABELS = {
    "Care": "care",
    "Equality": "equality",
    "Proportionality": "proportionality",
    "Loyalty": "loyalty",
    "Authority": "authority",
    "Purity": "purity",
    "Thin Morality": "thin_morality",
    "Non-Moral": "non_moral",
}


def prepare(raw: Path = RAW / "mfrc") -> list[Task]:
    df = pd.read_parquet(raw / "train_dedup.parquet")
    df["segment_id"] = "mfrc-" + df.text.map(short_hash)

    def codes(annotations: pd.Series) -> list[str]:
        out: set[str] = set()
        for a in annotations:
            for label in str(a).split(","):
                label = label.strip()
                if label not in LABELS:
                    raise ValueError(f"unknown MFRC label {label!r}")
                out.add(LABELS[label])
        return sorted(out)

    # 292 (text, annotator) pairs appear twice in the dedup file; all are the same label set in a
    # different order, so the union is the annotator's labels.
    labels = df.groupby(["segment_id", "annotator"]).annotation.agg(codes).rename("codes").reset_index()
    segs = (
        df.groupby("segment_id")
        .agg(text=("text", "first"), subreddit=("subreddit", "first"), bucket=("bucket", "first"))
        .reset_index()
    )
    segs["context"] = None
    segs["setting"] = None
    segs["group"] = segs.segment_id
    report = {
        "source": "data/raw/mfrc/train_dedup.parquet",
        "rows": len(df),
        "segments": len(segs),
        "annotators_per_segment": labels.groupby("segment_id").size().value_counts().sort_index().to_dict(),
    }
    return [Task("mfrc", segs, labels, report)]
