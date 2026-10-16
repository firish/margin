"""Canonical on-disk format for a prepared coding task (one codebook over one dataset)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from margin.schema import Codebook, Segment

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

SEGMENT_COLUMNS = ["segment_id", "text", "context", "setting", "group"]


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:n]


@dataclass
class Task:
    """A prepared task: segments, every annotator's labels, and a record of what was excluded."""

    name: str
    segments: pd.DataFrame
    """SEGMENT_COLUMNS plus any dataset-specific metadata columns."""
    labels: pd.DataFrame
    """segment_id, annotator, codes (list of code ids)."""
    report: dict[str, Any] = field(default_factory=dict)

    def validate(self, codebook: Codebook) -> None:
        missing = [c for c in SEGMENT_COLUMNS if c not in self.segments.columns]
        if missing:
            raise ValueError(f"{self.name}: segments missing columns {missing}")
        if self.segments.segment_id.duplicated().any():
            raise ValueError(f"{self.name}: duplicate segment ids")
        known = set(self.segments.segment_id)
        if not set(self.labels.segment_id) <= known:
            raise ValueError(f"{self.name}: labels reference unknown segments")
        if self.labels.duplicated(["segment_id", "annotator"]).any():
            raise ValueError(f"{self.name}: more than one label row per (segment, annotator)")
        valid = set(codebook.code_ids)
        bad = {c for codes in self.labels.codes for c in codes} - valid
        if bad:
            raise ValueError(f"{self.name}: labels use codes not in the codebook: {sorted(bad)}")
        if codebook.mode == "exclusive" and (self.labels.codes.map(len) != 1).any():
            raise ValueError(f"{self.name}: exclusive codebook but some label rows have != 1 code")

    def save(self, root: Path = PROCESSED) -> Path:
        out = root / self.name
        out.mkdir(parents=True, exist_ok=True)
        self.segments.to_parquet(out / "segments.parquet", index=False)
        self.labels.to_parquet(out / "labels.parquet", index=False)
        (out / "prep.json").write_text(json.dumps(self.report, indent=2, default=str))
        return out

    @classmethod
    def load(cls, name: str, root: Path = PROCESSED) -> Task:
        d = root / name
        labels = pd.read_parquet(d / "labels.parquet")
        labels["codes"] = labels.codes.map(list)
        return cls(name, pd.read_parquet(d / "segments.parquet"), labels, json.loads((d / "prep.json").read_text()))

    def to_segments(self, ids: list[str] | None = None) -> list[Segment]:
        df = self.segments if ids is None else self.segments.set_index("segment_id").loc[ids].reset_index()
        return [
            Segment(
                segment_id=r.segment_id,
                text=r.text,
                context=r.context if isinstance(r.context, str) else None,
                setting=r.setting if isinstance(r.setting, str) else None,
                group=r.group,
            )
            for r in df.itertuples()
        ]
