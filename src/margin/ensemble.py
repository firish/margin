"""Multi-model agreement, an RQ1 confidence signal built from other coders' outputs at no extra cost.

exclusive: p(option) = share of coders whose top option it is. multi: p(code applies) = share of coders
with p >= 0.5. Only units every coder answered are kept, so a missing answer never counts as a vote.
"""

from __future__ import annotations

import pandas as pd

from margin.run import RUNS
from margin.schema import Codebook


def agreement(preds: dict[str, pd.DataFrame], codebook: Codebook) -> pd.DataFrame:
    if codebook.mode == "exclusive":
        tops = []
        for name, p in preds.items():
            wide = p.pivot_table(index="segment_id", columns="code_id", values="p").reindex(columns=codebook.code_ids)
            tops.append(wide.idxmax(axis=1).rename(name))
        votes = pd.concat(tops, axis=1, join="inner")
        rows = [
            {"segment_id": seg, "code_id": cid, "p": float((row == cid).mean())}
            for seg, row in votes.iterrows()
            for cid in codebook.code_ids
        ]
        return pd.DataFrame(rows, columns=["segment_id", "code_id", "p"])
    yes = [p.set_index(["segment_id", "code_id"]).p.ge(0.5).rename(name) for name, p in preds.items()]
    table = pd.concat(yes, axis=1, join="inner")
    return table.mean(axis=1).rename("p").reset_index()


def write(experiment: str, task: str, coders: list[str], name: str, variant: str = "canonical") -> pd.DataFrame:
    codebook = Codebook.load(f"codebooks/{task}.yaml")
    d = RUNS / experiment / task
    preds = {c: pd.read_parquet(d / f"{c}__{variant}.parquet") for c in coders}
    out = agreement(preds, codebook)
    out.insert(0, "variant", variant)
    out.insert(0, "coder", name)
    out["model_version"] = "+".join(coders)
    out.to_parquet(d / f"{name}__{variant}.parquet", index=False)
    return out


DM_COMMITTEE = {"jev": ("jev_main", "jev"), "oaid": ("oaid_main", "oaid"), "clef": ("dm_main", "clef"), "clef_flash": ("dm_main", "clef_flash"),
                "pplx": ("dm_main", "pplx"), "kev9b": ("dm_main", "kev9b"), "kev27b": ("dm_main", "kev27b")}
"""Deviation 13 (exploratory): all seven decision models, membership fixed before any committee result."""


def mean_probability(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average each (segment, code) probability over coders; only units every coder answered are kept.
    For exclusive codebooks the average of distributions is itself a distribution."""
    cols = [p.set_index(["segment_id", "code_id"]).p.rename(name) for name, p in preds.items()]
    table = pd.concat(cols, axis=1, join="inner").dropna()
    complete = table.groupby(level=0).size()
    width = max(p.groupby("segment_id").code_id.nunique().max() for p in preds.values())  # codes per segment
    table = table[table.index.get_level_values(0).isin(complete[complete == width].index)]
    return table.mean(axis=1).rename("p").reset_index()


def write_committee(task: str, name: str = "dm_committee", variant: str = "canonical") -> pd.DataFrame:
    preds = {c: pd.read_parquet(RUNS / exp / task / f"{stem}__{variant}.parquet") for c, (exp, stem) in DM_COMMITTEE.items()}
    out = mean_probability(preds)
    out.insert(0, "variant", variant)
    out.insert(0, "coder", name)
    out["model_version"] = "+".join(DM_COMMITTEE)
    path = RUNS / "dm_main" / task / f"{name}__{variant}.parquet"
    out.to_parquet(path, index=False)
    return out
