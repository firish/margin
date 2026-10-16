"""AnnoMI (uccollab/AnnoMI), full version: client talk type and main therapist behaviour.

Seven transcripts are coded by all ten annotators (the agreement subset); the rest by one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from margin.datasets.base import RAW, Task

THERAPIST = {"reflection": "reflection", "question": "question", "therapist_input": "therapist_input", "other": "none"}
CLIENT = {"change": "change", "sustain": "sustain", "neutral": "neutral"}


def prepare(raw: Path = RAW / "annomi") -> list[Task]:
    df = pd.read_csv(raw / "AnnoMI-full.csv")
    df["segment_id"] = "annomi-" + df.transcript_id.astype(str) + "-" + df.utterance_id.astype(str)
    df["annotator"] = "a" + df.annotator_id.astype(str)
    n_annotators = df.groupby("transcript_id").annotator_id.nunique()
    multi = set(n_annotators[n_annotators > 1].index)

    # One row per utterance, in conversation order, for text and context.
    utt = df.drop_duplicates(["transcript_id", "utterance_id"]).sort_values(["transcript_id", "utterance_id"])
    speaker = utt.interlocutor.str.capitalize()
    utt["context"] = (speaker + ": " + utt.utterance_text).groupby(utt.transcript_id).shift(1)
    utt["setting"] = "Motivational interviewing session. Target behaviour change: " + utt.topic.astype(str) + "."
    utt["group"] = "annomi-" + utt.transcript_id.astype(str)
    utt["multi_annotated"] = utt.transcript_id.isin(multi)

    tasks = []
    for name, role, column, mapping in [
        ("annomi_client", "client", "client_talk_type", CLIENT),
        ("annomi_therapist", "therapist", "main_therapist_behaviour", THERAPIST),
    ]:
        rows = df[df.interlocutor == role]
        unmapped = rows[~rows[column].isin(mapping)]
        rows = rows[rows[column].isin(mapping)]
        labels = rows[["segment_id", "annotator"]].assign(codes=rows[column].map(lambda v: [mapping[v]]))
        segs = utt[utt.segment_id.isin(set(labels.segment_id))].rename(columns={"utterance_text": "text"})
        segs = segs[["segment_id", "text", "context", "setting", "group", "transcript_id", "mi_quality", "multi_annotated"]]
        report = {
            "source": "data/raw/annomi/AnnoMI-full.csv",
            "segments": len(segs),
            "multi_annotated_transcripts": sorted(int(t) for t in multi),
            "multi_annotated_segments": int(segs.multi_annotated.sum()),
            "dropped_unmapped_label_rows": len(unmapped),
        }
        tasks.append(Task(name, segs.reset_index(drop=True), labels.reset_index(drop=True), report))
    return tasks
