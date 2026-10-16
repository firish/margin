"""Robustness arms (RQ3) as deterministic transforms of a codebook and the per-segment option order.

Each variant changes only surface form: option/question order, label names, definition wording,
whether context is shown, or whether codes are asked together or one per call. Definitions and
examples are never otherwise altered, and every rendered key maps back to a canonical code id.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from margin.schema import Code, Codebook

Order = Literal["canonical", "reversed", "shuffle"]
QuestionFormat = Literal["native", "noul"]


@dataclass(frozen=True)
class Variant:
    name: str = "canonical"
    order: Order = "canonical"
    """canonical = the guideline's own order. shuffle = seeded per-segment permutation."""
    seed: int = 0
    neutral_labels: bool = False
    """Replace label names with code_A/option_A; definitions stay, so this tests label vs definition."""
    paraphrase: int | None = None
    """Index into the codebook's paraphrase file; None keeps the original definitions."""
    context: bool = True
    isolated: bool = False
    """multi codebooks only: one call per code instead of all codes in one call."""
    question_format: QuestionFormat = "native"
    """native: Choice for exclusive codebooks, Noul per code for multi. noul: exclusive codebooks are
    also asked as one Noul per option (incl. none), then combined with `coders.jev.exactly_one`."""
    elicitation: str = "answer_prob"
    """LLM confidence prompt. answer_prob: answer + probability it is right (exclusive), yes/no +
    probability per code (multi). distribution: a probability for every option / per code (v1.4)."""


CANONICAL = Variant()


def neutral_label(index: int, mode: str) -> str:
    prefix = "code" if mode == "multi" else "option"
    return f"{prefix}_{string.ascii_uppercase[index]}"


def paraphrase_path(codebook_path: str | Path) -> Path:
    p = Path(codebook_path)
    return p.parent / "paraphrases" / p.name


def load_paraphrases(codebook_path: str | Path) -> dict[str, list[str]]:
    path = paraphrase_path(codebook_path)
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def apply_labels_and_wording(
    codebook: Codebook, variant: Variant, paraphrases: dict[str, list[str]] | None = None
) -> Codebook:
    """Return a codebook with the variant's labels and definitions; ids are unchanged."""
    if variant.paraphrase is None and not variant.neutral_labels:
        return codebook
    options: list[Code] = []
    for i, code in enumerate(codebook.all_options):
        update: dict[str, str] = {}
        if variant.neutral_labels:
            update["label"] = neutral_label(i, codebook.mode)
        if variant.paraphrase is not None:
            versions = (paraphrases or {}).get(code.id)
            if not versions or variant.paraphrase >= len(versions):
                raise ValueError(f"no paraphrase #{variant.paraphrase} for code {code.id!r} in {codebook.name}")
            update["definition"] = versions[variant.paraphrase]
        options.append(code.model_copy(update=update))
    has_none = codebook.none_option is not None
    return codebook.model_copy(
        update={"codes": options[:-1] if has_none else options, "none_option": options[-1] if has_none else None}
    )


def order_options(options: list[Code], variant: Variant, segment_id: str) -> list[Code]:
    if variant.order == "canonical":
        return list(options)
    if variant.order == "reversed":
        return list(reversed(options))
    rng = random.Random(f"{variant.seed}:{segment_id}")
    shuffled = list(options)
    rng.shuffle(shuffled)
    return shuffled


STANDARD_ARMS: dict[str, Variant] = {
    v.name: v
    for v in [
        CANONICAL,
        Variant(name="reversed", order="reversed"),
        Variant(name="shuffle1", order="shuffle", seed=1),
        Variant(name="shuffle2", order="shuffle", seed=2),
        Variant(name="shuffle3", order="shuffle", seed=3),
        Variant(name="isolated", isolated=True),
        Variant(name="neutral_labels", neutral_labels=True),
        Variant(name="paraphrase0", paraphrase=0),
        Variant(name="paraphrase1", paraphrase=1),
        Variant(name="no_context", context=False),
        Variant(name="noul_format", question_format="noul"),
        Variant(name="dist_prompt", elicitation="distribution"),  # amendment v1.4, LLMs only
    ]
}
