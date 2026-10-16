"""Meaning-preserving paraphrases of code definitions for the RQ3 paraphrase arms.

One Sonnet call per codebook writes two paraphrases of every definition (p0: reworded, same structure;
p1: restructured, new vocabulary). Only the `definition` field is paraphrased; criteria and examples
stay verbatim. Output goes to codebooks/paraphrases/<task>.yaml (read by perturb.load_paraphrases) and
a side-by-side review file for a human spot-check before the arms run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import yaml

from margin.budget import Registry
from margin.cache import Cache, request_key
from margin.coders import llm
from margin.perturb import paraphrase_path
from margin.schema import Codebook

SYSTEM = """You write meaning-preserving paraphrases of qualitative codebook definitions for a robustness test.
A coder given a paraphrase must apply the code to exactly the same cases as a coder given the original.

Rules for every paraphrase:
- Keep every condition, inclusion, exclusion, scope limit, and hedge (e.g. "explicitly", "clearly implied", "only", "unless", "even if").
- Add nothing: no new examples, criteria, clarifications, or interpretations. Remove nothing.
- Keep the labels of codes and codebook-specific terms unchanged (e.g. "change goal", "talk move", "Du Bois").
- Do not resolve ambiguities or fix wording in a way that changes meaning; preserve vagueness where the original is vague.
- Stay within roughly 25% of the original length.

Write two paraphrases per definition:
- "p0": restate it in different words, keeping roughly the original sentence order.
- "p1": restructure it (reorder or merge/split sentences, e.g. lead with the conditions), using vocabulary different from both the original and p0."""


def codebook_text(cb: Codebook) -> str:
    parts = [f"Codebook: {cb.name}. Unit: {cb.unit}. Task: {cb.instructions}", ""]
    for c in cb.all_options:
        parts.append(f"[{c.id}] {c.label}\nDefinition: {c.definition}")
        for title, items in (("Applies when", c.include), ("Does not apply when", c.exclude)):
            if items:
                parts.append(f"{title}: " + " | ".join(items))
        parts.append("")
    parts.append("Paraphrase only the Definition of each code id above. The other fields are context.")
    return "\n".join(parts)


def schema(cb: Codebook) -> dict:
    entry = {"type": "object", "properties": {"p0": {"type": "string"}, "p1": {"type": "string"}},
             "required": ["p0", "p1"], "additionalProperties": False}
    ids = cb.code_ids
    return {"type": "object", "properties": {i: entry for i in ids}, "required": ids, "additionalProperties": False}


def checks(cb: Codebook, para: dict[str, list[str]]) -> list[str]:
    """Automatic red flags; a human still reads the review file."""
    issues = []
    for c in cb.all_options:
        orig = len(c.definition.split())
        for k, text in enumerate(para[c.id]):
            ratio = len(text.split()) / max(orig, 1)
            if not 0.7 <= ratio <= 1.4:
                issues.append(f"{cb.name}/{c.id}/p{k}: length ratio {ratio:.2f}")
            if text.strip() == c.definition.strip():
                issues.append(f"{cb.name}/{c.id}/p{k}: identical to original")
        if para[c.id][0].strip() == para[c.id][1].strip():
            issues.append(f"{cb.name}/{c.id}: p0 == p1")
    return issues


async def _generate(codebooks: list[Codebook], cache: Cache) -> dict[str, dict[str, list[str]]]:
    import json

    spec = Registry.load()["sonnet"]
    clients = llm.Clients.from_env()
    out = {}
    try:
        for cb in codebooks:
            body = {"system": SYSTEM, "user": codebook_text(cb), "schema": schema(cb), "samples": 1,
                    "temperature": None, "logprobs": False, "max_tokens": 12000, "params": spec.params}
            key = request_key(spec.provider, spec.model, body)
            response = cache.get(key)
            if response is None:
                response = await llm.complete(clients, spec, body)
                u = response["usage"]
                cache.put(key, provider=spec.provider, model=spec.model, request=body, response=response,
                          input_tokens=u["input"] + u["cache_read"] + u["cache_write"], output_tokens=u["output"],
                          cost_usd=spec.cost(u["input"], u["output"], u["cache_read"], u["cache_write"]))
            data = json.loads(response["texts"][0])
            out[cb.name] = {cid: [data[cid]["p0"], data[cid]["p1"]] for cid in cb.code_ids}
    finally:
        await clients.aclose()
    return out


def generate(tasks: list[str], review_path: Path = Path("docs/paraphrase_review.md")) -> list[str]:
    codebooks = [Codebook.load(f"codebooks/{t}.yaml") for t in tasks]
    result = asyncio.run(_generate(codebooks, Cache()))
    issues, review = [], ["# Paraphrase review", "",
                          "Check that each paraphrase applies to exactly the same cases as the original. "
                          "Flag any added, dropped, or changed condition.", ""]
    for cb in codebooks:
        para = result[cb.name]
        path = paraphrase_path(f"codebooks/{cb.name}.yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(para, sort_keys=False, allow_unicode=True, width=100))
        issues += checks(cb, para)
        review.append(f"## {cb.name}\n")
        for c in cb.all_options:
            review.append(f"### {c.label} (`{c.id}`)\n\n- **Original:** {c.definition}\n"
                          f"- **p0:** {para[c.id][0]}\n- **p1:** {para[c.id][1]}\n")
    review_path.write_text("\n".join(review))
    return issues
