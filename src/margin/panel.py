"""Two-family LLM adjudication panel + author tie-break (ANALYSIS_PLAN deviation 2).

Each panel model sees exactly what a human adjudicator sees: the instructions, the task's codebook,
and the item row (setting, context, segment, question, options). Its answer is written in the human
answer format ("A+B", "none", "Y") so it goes through the same import and parsing as a person's.
The author then judges every item the panel disagrees on, plus a blind audit of agreed items.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from margin.adjudicate import (
    ADJUDICATION,
    INSTRUCTIONS,
    LETTERS,
    _verdict,
    codebooks_markdown,
    import_answers,
    intra_rater,
    read_answers,
    write_xlsx,
)
from margin.budget import Registry, check_cap, estimate_tokens
from margin.cache import Cache, request_key
from margin.coders import llm
from margin.schema import Codebook

PANEL = ("opus", "sol")
AUTHOR = "rishi"
AUDIT_N = 20
REPEAT_SHARE = 0.1
FIELDS = ["item", "task", "setting", "context", "segment", "question", "options", "answer", "note"]


def is_yes_no(question: str) -> bool:
    return question.endswith("(Y / N / ?)")


def letters_of(options: str) -> list[str]:
    return re.findall(r"^([A-Z])\) ", options, flags=re.M)


def system_prompt(codebook: Codebook) -> str:
    return (
        "You are adjudicating a qualitative coding item for a research study. Follow these instructions "
        "exactly as a human adjudicator would.\n\n" + INSTRUCTIONS + "\n\n" + codebooks_markdown([codebook])
        + "\n\nRespond with JSON only: a short rationale citing the codebook rule you relied on, your answer, "
        "and an optional note (empty string if none)."
    )


def user_prompt(row: pd.Series) -> str:
    parts = [f"setting: {row.setting}" if row.setting else "", f"context: {row.context}" if row.context else "",
             f"segment: {row.segment}", f"question: {row.question}", f"options:\n{row.options}"]
    return "\n\n".join(p for p in parts if p)


def schema(row: pd.Series) -> dict:
    if is_yes_no(row.question):
        answer = {"answer": {"type": "string", "enum": ["Y", "N", "?"]}}
    else:
        answer = {"acceptable": {"type": "array", "items": {"type": "string", "enum": letters_of(row.options)}}}
    props = {"rationale": {"type": "string"}} | answer | {"note": {"type": "string"}}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def to_answer(data: dict, row: pd.Series) -> str:
    if is_yes_no(row.question):
        return data["answer"]
    picked = sorted(set(data["acceptable"]))
    return "+".join(picked) if picked else "none"


def handout_rows(batch: str) -> pd.DataFrame:
    path = ADJUDICATION / batch / "to_adjudicator" / "adjudication.xlsx"
    if not path.exists():
        path = ADJUDICATION / batch / "superseded_full_handout" / "adjudication.xlsx"
    return pd.read_excel(path, sheet_name="items", dtype=str).fillna("")


def bodies(batch: str, spec_name: str) -> list[tuple[pd.Series, dict]]:
    spec = Registry.load()[spec_name]
    rows = handout_rows(batch)
    books = {t: Codebook.load(f"codebooks/{t}.yaml") for t in rows.task.unique()}
    return [(r, {"system": system_prompt(books[r.task]), "user": user_prompt(r), "schema": schema(r), "samples": 1,
                 "temperature": None, "logprobs": False, "max_tokens": 16000, "params": spec.params})
            for _, r in rows.iterrows()]


def estimate_usd(batch: str, cache: Cache, out_per_item: int = 1500) -> float:
    reg, usd = Registry.load(), 0.0
    for name in PANEL:
        spec = reg[name]
        todo = [b for _, b in bodies(batch, name) if cache.get(request_key(spec.provider, spec.model, b)) is None]
        usd += spec.cost(sum(estimate_tokens(b["system"] + b["user"]) for b in todo), out_per_item * len(todo))
    return usd


async def _judge(batch: str, name: str, cache: Cache, clients: llm.Clients) -> pd.DataFrame:
    spec = Registry.load()[name]
    sem = clients.semaphore(spec)
    out = []

    async def one(row: pd.Series, body: dict) -> None:
        key = request_key(spec.provider, spec.model, body)
        response = cache.get(key)
        if response is None:
            async with sem:
                try:
                    response = await llm.complete(clients, spec, body)
                except Exception as err:  # recorded as unanswered; the author then judges the item
                    out.append({"item": row["item"], "answer": "", "note": f"[{name} error] {type(err).__name__}"})
                    return
            u = response["usage"]
            cache.put(key, provider=spec.provider, model=spec.model, request=body, response=response,
                      input_tokens=u["input"] + u["cache_read"] + u["cache_write"], output_tokens=u["output"],
                      cost_usd=spec.cost(u["input"], u["output"], u["cache_read"], u["cache_write"]))
        try:
            data = json.loads(response["texts"][0])
            out.append({"item": row["item"], "answer": to_answer(data, row),
                        "note": f"{data.get('rationale', '')} | {data.get('note', '')}".strip(" |")})
        except (json.JSONDecodeError, TypeError, KeyError):
            out.append({"item": row["item"], "answer": "", "note": f"[{name} unparseable; stop={response.get('stop')}]"})

    await asyncio.gather(*(one(r, b) for r, b in bodies(batch, name)))
    return pd.DataFrame(out, columns=["item", "answer", "note"])


def run_panel(batch: str) -> dict[str, pd.DataFrame]:
    cache, registry = Cache(), Registry.load()
    check_cap(registry, cache, estimate_usd(batch, cache))

    async def both() -> dict[str, pd.DataFrame]:
        clients = llm.Clients.from_env()
        try:
            results = await asyncio.gather(*(_judge(batch, n, cache, clients) for n in PANEL))
        finally:
            await clients.aclose()
        return dict(zip(PANEL, results))

    answers = asyncio.run(both())
    for name, df in answers.items():
        path = ADJUDICATION / batch / "panel" / f"panel_{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        answered = df[df.answer != ""]
        answered.to_csv(path.with_suffix(".answered.csv"), index=False)
        import_answers(batch, [path.with_suffix(".answered.csv")], name)
    return answers


def panel_agreement(batch: str) -> pd.DataFrame:
    """One row per main (non-repeat) item: each panel verdict and whether the two agree."""
    d = ADJUDICATION / batch
    key = pd.read_parquet(d / "key.parquet")
    main = key[key.stratum != "repeat"][["item", "task", "stratum"]]
    for name in PANEL:
        v = pd.read_parquet(d / f"verdicts_{name}.parquet")
        v[name] = [_verdict(a, m) for a, m in zip(v.acceptable, v.ambiguous)]
        main = main.merge(v[["item", name]], on="item", how="left")
    main["agree"] = main[list(PANEL)].notna().all(axis=1) & (main[PANEL[0]] == main[PANEL[1]]) & (main[PANEL[0]] != "?")
    return main


def build_author_handout(batch: str, seed: int = 0) -> Path:
    """Panel disagreements + a blind audit of agreed items (+ repeats for a later sitting)."""
    d = ADJUDICATION / batch
    agree = panel_agreement(batch)
    rng = np.random.default_rng(seed)
    disputed = agree[~agree.agree].item.tolist()
    agreed = agree[agree.agree].item.to_numpy()
    audit = rng.choice(agreed, size=min(AUDIT_N, len(agreed)), replace=False).tolist()
    rows = handout_rows(batch).set_index("item")
    chosen = disputed + audit
    sheet = [rows.loc[i].to_dict() | {"item": i, "answer": "", "note": ""} for i in chosen]
    rng.shuffle(sheet)
    key = pd.read_parquet(d / "key.parquet").set_index("item")
    used = set(key.index)
    n_rep = max(1, int(round(REPEAT_SHARE * len(sheet))))
    part2, new_keys = [], []
    for i in rng.choice(len(sheet), size=n_rep, replace=False):
        new = "".join(rng.choice(list(LETTERS + "23456789"), size=6))
        while new in used:
            new = "".join(rng.choice(list(LETTERS + "23456789"), size=6))
        used.add(new)
        orig = sheet[i]["item"]
        part2.append(sheet[i] | {"item": new})
        new_keys.append(key.loc[orig].to_dict() | {"item": new, "stratum": "repeat", "repeat_of": orig})
    pd.concat([key.reset_index(), pd.DataFrame(new_keys)], ignore_index=True).to_parquet(d / "key.parquet", index=False)

    old = d / "to_adjudicator"
    if old.exists():  # the 200-item handout is no longer the author's task
        shutil.move(str(old), str(d / "superseded_full_handout"))
    out = d / "to_author"
    out.mkdir(exist_ok=True)
    write_xlsx([{k: r[k] for k in FIELDS} for r in sheet], out / "author.xlsx")
    write_xlsx([{k: r[k] for k in FIELDS} for r in part2], out / "author_part2_later.xlsx")
    tasks = sorted({r["task"] for r in sheet})
    (out / "codebooks.md").write_text(codebooks_markdown([Codebook.load(f"codebooks/{t}.yaml") for t in tasks]))
    (out / "INSTRUCTIONS.md").write_text(INSTRUCTIONS.replace("adjudication_part2_later.xlsx", "author_part2_later.xlsx"))
    (d / "panel" / "author_selection.json").write_text(json.dumps(
        {"disputed": disputed, "audit": audit, "repeats": [k["item"] for k in new_keys]}, indent=2))
    return out


def resolve(batch: str) -> dict:
    """Final verdicts: author where given; else the panel's when both agree; else unresolved."""
    d = ADJUDICATION / batch
    agree = panel_agreement(batch)
    panel_v = pd.read_parquet(d / f"verdicts_{PANEL[0]}.parquet").set_index("item")
    author_path = d / f"verdicts_{AUTHOR}.parquet"
    author = pd.read_parquet(author_path) if author_path.exists() else pd.DataFrame(columns=["item", "stratum"])
    author_main = author[author.stratum != "repeat"].set_index("item")
    final = []
    for r in agree.itertuples():
        if r.item in author_main.index:
            final.append(author_main.loc[r.item].to_dict() | {"item": r.item, "source": "author"})
        elif r.agree:
            final.append(panel_v.loc[r.item].to_dict() | {"item": r.item, "source": "panel"})
    f = pd.DataFrame(final)
    f.to_parquet(d / "verdicts_final.parquet", index=False)
    sel = json.loads((d / "panel" / "author_selection.json").read_text()) if (d / "panel" / "author_selection.json").exists() else {}
    audit_ids = set(sel.get("audit", []))
    audit_rows = agree[agree.item.isin(audit_ids) & agree.item.isin(author_main.index)]
    author_v = {i: _verdict(a, m) for i, a, m in zip(author_main.index, author_main.acceptable, author_main.ambiguous)}
    audit_match = [author_v[i] == p for i, p in zip(audit_rows.item, audit_rows[PANEL[0]])]
    return {
        "items": len(agree), "panel_agree": int(agree.agree.sum()), "panel_agree_rate": float(agree.agree.mean()),
        "final_from_author": int((f.source == "author").sum()) if len(f) else 0,
        "final_from_panel": int((f.source == "panel").sum()) if len(f) else 0,
        "unresolved": len(agree) - len(f),
        "audit_n": len(audit_match), "audit_author_panel_agreement": float(np.mean(audit_match)) if audit_match else float("nan"),
        "author_intra_rater": intra_rater(author) if len(author) else {},
    }
