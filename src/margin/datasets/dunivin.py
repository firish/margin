"""Dunivin (2025) Du Bois news passages (OSF k4fg9): 9 multi-label codes, one gold-standard coder set.

Only rows with id 0-119 were gold-coded: 0-8 were used by the authors for prompt development and
9-119 are the paper's 111-passage test set. Rows 120-232 are blank because they were never coded
(not because no code applies), so they are excluded. The published GPT predictions in `output/`
give an exact baseline on the same passages.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from margin.datasets.base import RAW, Task

COLUMNS = {
    "Scholar": "scholar",
    "Activist": "activist",
    "Monumental Memorialization": "monumental_memorialization",
    "Mention of Scholarly Work": "scholarly_work",
    "Social/Political Advocacy": "advocacy",
    "Coalition Building": "coalition_building",
    "Out of the Mouth of Academics": "mouth_of_academics",
    "Out of the Mouth of Activists": "mouth_of_activists",
    "Collective Synecdoche": "collective_synecdoche",
}
CODED_IDS = range(0, 120)
TEST_IDS = range(9, 120)


def codes_of(row: pd.Series) -> list[str]:
    return [cid for col, cid in COLUMNS.items() if row.get(col, 0) == 1]


def prepare(raw: Path = RAW / "dunivin") -> list[Task]:
    g = pd.read_csv(raw / "gold_standard_coding.csv")
    if not (g.id == g.index).all():
        raise ValueError("expected id to equal row index in gold_standard_coding.csv")
    uncoded = g[~g.id.isin(CODED_IDS)]
    if uncoded[list(COLUMNS)].notna().any().any():
        raise ValueError("rows outside 0-119 unexpectedly carry codes; recheck the coded range")
    g = g[g.id.isin(CODED_IDS)].copy()
    g[list(COLUMNS)] = g[list(COLUMNS)].fillna(0).astype(int)
    segs = pd.DataFrame(
        {
            "segment_id": "dunivin-" + g.id.astype(str),
            "text": g.passage.str.strip(),
            "context": None,
            "setting": None,
            "group": "dunivin-" + g.id.astype(str),
            "published_split": ["test" if i in TEST_IDS else "dev" for i in g.id],
            "date": g.date,
            "article_type": g.article_type,
        }
    )
    labels = pd.DataFrame({"segment_id": segs.segment_id, "annotator": "gold", "codes": g.apply(codes_of, axis=1).to_list()})
    report = {
        "source": str(raw / "gold_standard_coding.csv"),
        "segments": len(segs),
        "excluded_uncoded_rows": len(uncoded),
        "published_test": int((segs.published_split == "test").sum()),
        "code_counts_test": {cid: int(g.loc[g.id.isin(TEST_IDS), col].sum()) for col, cid in COLUMNS.items()},
    }
    return [Task("dunivin", segs.reset_index(drop=True), labels, report)]


def published_predictions(config: str, raw: Path = RAW / "dunivin") -> pd.DataFrame:
    """A published GPT run as a predictions frame (p is 0/1: those runs output codes, not scores)."""
    d = pd.read_csv(raw / "output" / config / "processed_responses.csv")
    d = d[d.id.isin(CODED_IDS)]
    rows = [
        {"segment_id": f"dunivin-{r.id}", "code_id": cid, "p": float(r[col] == 1)}
        for _, r in d.iterrows()
        for col, cid in COLUMNS.items()
    ]
    return pd.DataFrame(rows)
