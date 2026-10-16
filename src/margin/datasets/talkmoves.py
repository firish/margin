"""TalkMoves (SumnerLab/TalkMoves, CC BY-NC-SA 4.0): per-transcript xlsx files, one row per sentence.

Tags are free text with a leading number ("8 - press for  accuracy", "8 - prestudents for accuracy"),
so the number is the label. Teacher tag 7 ("context") is not defined in the coding manual and is
excluded; any other unknown number is excluded and counted.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from margin.datasets.base import RAW, Task, short_hash

TEACHER = {1: "none", 2: "keeping_together", 3: "relate", 4: "restating", 5: "revoicing", 8: "press_accuracy", 9: "press_reasoning"}
STUDENT = {1: "none", 2: "relating", 3: "asking_info", 4: "claim", 5: "evidence"}
CONTEXT_SENTENCES = 3
TAG_NUMBER = re.compile(r"^\s*(\d+)")


def tag_number(tag: object) -> int | None:
    if not isinstance(tag, str | int | float) or pd.isna(tag):
        return None
    m = TAG_NUMBER.match(str(tag))
    return int(m.group(1)) if m else None


def prepare(raw: Path = RAW / "talkmoves" / "repo" / "data") -> list[Task]:
    files = sorted(raw.glob("Subset */*.xlsx"))
    if not files:
        raise FileNotFoundError(f"no transcripts under {raw}")
    out: dict[str, list[dict]] = {"teacher": [], "student": []}
    excluded: dict[str, Counter] = {"teacher": Counter(), "student": Counter()}
    for f in files:
        transcript = f"{f.parent.name}/{f.stem}"
        group = "tm-" + short_hash(transcript, 10)
        df = pd.read_excel(f)
        df = df[df.Sentence.notna() & (df.Sentence.astype(str).str.strip() != "")].reset_index(drop=True)
        teacher_n = df["Teacher Tag"].map(tag_number)
        student_n = df["Student Tag"].map(tag_number)
        who = ["Teacher" if pd.notna(t) else "Student" for t in df["Teacher Tag"]]
        lines = [f"{w}: {str(s).strip()}" for w, s in zip(who, df.Sentence)]
        for i in range(len(df)):
            for role, n, mapping in (("teacher", teacher_n[i], TEACHER), ("student", student_n[i], STUDENT)):
                if n is None or pd.isna(n):
                    continue
                if int(n) not in mapping:
                    excluded[role][int(n)] += 1
                    continue
                out[role].append(
                    {
                        "segment_id": f"{group}-{i}",
                        "text": str(df.Sentence[i]).strip(),
                        "context": "\n".join(lines[max(0, i - CONTEXT_SENTENCES) : i]) or None,
                        "setting": None,
                        "group": group,
                        "transcript": transcript,
                        "code": mapping[int(n)],
                    }
                )
    tasks = []
    for role in ("teacher", "student"):
        segs = pd.DataFrame(out[role])
        labels = pd.DataFrame({"segment_id": segs.segment_id, "annotator": "gold", "codes": segs.code.map(lambda c: [c])})
        report = {
            "source": str(raw),
            "transcripts": len(files),
            "segments": len(segs),
            "excluded_tag_numbers": dict(excluded[role]),
            "label_counts": segs.code.value_counts().to_dict(),
        }
        tasks.append(Task(f"talkmoves_{role}", segs.drop(columns="code"), labels, report))
    return tasks
