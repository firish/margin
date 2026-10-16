"""Codebook + segment -> model requests. The single source of truth for what every coder sees."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from margin.perturb import Variant, order_options
from margin.schema import Code, Codebook, Segment


def code_spec(code: Code) -> dict[str, Any]:
    """Structured description of one code, shared by Jev criteria and LLM prompts."""
    spec: dict[str, Any] = {"definition": code.definition}
    if code.include:
        spec["applies_when"] = code.include
    if code.exclude:
        spec["does_not_apply_when"] = code.exclude
    if code.examples_pos:
        spec["examples"] = code.examples_pos
    if code.examples_neg:
        spec["non_examples"] = code.examples_neg
    return spec


def state_for(segment: Segment, variant: Variant) -> dict[str, str]:
    state: dict[str, str] = {}
    if segment.setting:
        state["setting"] = segment.setting
    if variant.context and segment.context:
        state["previous_turn"] = segment.context
    state["segment"] = segment.text
    return state


def scope_note(codebook: Codebook, variant: Variant, segment: Segment) -> str:
    if variant.context and segment.context:
        return f"Judge only the {codebook.unit} in `segment`; `previous_turn` is context and is not coded."
    return f"Judge the {codebook.unit} in `segment`."


@dataclass
class JevCall:
    state: dict[str, str]
    questions: dict[str, dict[str, Any]]
    keymap: dict[str, str] = field(default_factory=dict)
    """multi: question key -> code id. exclusive: option key (label) -> code id."""

    @property
    def request(self) -> dict[str, Any]:
        return {"state": self.state, "questions": self.questions}


def jev_calls(codebook: Codebook, segment: Segment, variant: Variant) -> list[JevCall]:
    """Requests for one segment. `codebook` must already have the variant's labels/wording applied."""
    state = state_for(segment, variant)
    scope = scope_note(codebook, variant, segment)
    options = order_options(codebook.all_options, variant, segment.segment_id)

    if codebook.mode == "exclusive" and variant.question_format == "native":
        criteria = {opt.label: code_spec(opt) for opt in options}
        question = {"type": "choice", "instructions": f"{codebook.instructions} {scope}", "criteria": criteria}
        return [JevCall(state, {"code": question}, {opt.label: opt.id for opt in options})]

    def noul(code: Code, instructions: str, false_description: str) -> dict[str, Any]:
        false_spec: dict[str, Any] = {"description": false_description}
        if code.exclude:
            false_spec["including_when"] = code.exclude
        if code.examples_neg:
            false_spec["examples"] = code.examples_neg
        true_spec = code_spec(code)
        true_spec.pop("does_not_apply_when", None)
        true_spec.pop("non_examples", None)
        return {"type": "noul", "instructions": instructions, "criteria": {"true": true_spec, "false": false_spec}}

    if codebook.mode == "exclusive":
        # Same option descriptions as the Choice arm, one yes/no per option; every question names all
        # options so it carries the Choice question's "exactly one applies" information.
        labels = ", ".join(f'"{o.label}"' for o in options)

        def question(code: Code) -> dict[str, Any]:
            return noul(
                code,
                f"{codebook.instructions} Exactly one of these options applies: {labels}. "
                f"Does \"{code.label}\" apply to the {codebook.unit}? {scope}",
                f"Another option applies to the {codebook.unit} instead of \"{code.label}\".",
            )
    else:

        def question(code: Code) -> dict[str, Any]:
            return noul(
                code,
                f"Does the {codebook.unit} express the code \"{code.label}\"? {scope}",
                f"The {codebook.unit} does not express {code.label}.",
            )

    # Question keys are positional and neutral (q00, q01, ...) so key names never leak labels.
    keyed = [(f"q{i:02d}", code) for i, code in enumerate(options)]
    if variant.isolated:
        return [JevCall(state, {key: question(code)}, {key: code.id}) for key, code in keyed]
    return [JevCall(state, {key: question(code) for key, code in keyed}, {key: code.id for key, code in keyed})]
