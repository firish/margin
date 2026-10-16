"""Core types shared by loaders, coders, and metrics.

On disk, a prepared dataset is two parquet tables:

- ``segments.parquet``: one row per unit to code (``segment_id, text, context, group, meta``).
- ``labels.parquet``: one row per (segment, human annotator) with the list of codes they applied.

Coders write a long ``predictions`` table: one row per (segment, code) with ``p``, the coder's
probability-like score that the code applies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

NONE_CODE = "none"
"""Code id used for "none of these" in exclusive codebooks."""


class Code(BaseModel):
    id: str
    label: str
    definition: str
    include: list[str] = Field(default_factory=list)
    """Inclusion criteria and boundary notes, taken from the dataset's own guidelines."""
    exclude: list[str] = Field(default_factory=list)
    examples_pos: list[str] = Field(default_factory=list)
    examples_neg: list[str] = Field(default_factory=list)


class Codebook(BaseModel):
    name: str
    version: str = "1"
    dataset: str
    unit: str
    """What one segment is, in words the coder sees ("Reddit comment", "client utterance")."""
    mode: Literal["multi", "exclusive"]
    """multi: any subset of codes may apply. exclusive: exactly one code (or `none`) applies."""
    instructions: str
    """The framing question, e.g. "Which moral sentiments does this comment express?"."""
    codes: list[Code]
    none_option: Code | None = None
    """Exclusive codebooks only: the "none of these" option, if the scheme has one."""
    source: str = ""
    """Citation for the guidelines the definitions were taken from."""

    @model_validator(mode="after")
    def _check(self) -> Codebook:
        ids = [c.id for c in self.all_options]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate code ids in {self.name}: {ids}")
        if self.mode == "multi" and self.none_option is not None:
            raise ValueError("none_option only applies to exclusive codebooks")
        if self.none_option is not None and self.none_option.id != NONE_CODE:
            raise ValueError(f"none_option id must be {NONE_CODE!r}")
        return self

    @property
    def all_options(self) -> list[Code]:
        """Codes plus the none option (exclusive mode); the full answer space."""
        return self.codes + ([self.none_option] if self.none_option else [])

    @property
    def code_ids(self) -> list[str]:
        return [c.id for c in self.all_options]

    @classmethod
    def load(cls, path: str | Path) -> Codebook:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


class Segment(BaseModel):
    segment_id: str
    text: str
    context: str | None = None
    """Preceding turn(s) shown alongside the segment when the context arm is on."""
    setting: str | None = None
    """Always-shown framing the code depends on, e.g. the session's target behaviour in AnnoMI."""
    group: str
    """Cluster id (conversation, transcript, or the segment itself) for clustered bootstrap."""
