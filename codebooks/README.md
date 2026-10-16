# Codebooks: sources and terms

These files quote or adapt published coding guidelines. The repository's MIT licence covers the code, not this
text; each source keeps its own terms. Every YAML file also names its source in its `source:` field.

| File | Source | Terms |
|---|---|---|
| `mfrc.yaml` | Foundation descriptions from Trager et al. (2022), "The Moral Foundations Reddit Corpus", arXiv:2208.05545 | Quoted with attribution; the MFRC data are CC BY 4.0 |
| `talkmoves_teacher.yaml`, `talkmoves_student.yaml` | TalkMoves Coding Manual (Suresh et al., [SumnerLab/TalkMoves](https://github.com/SumnerLab/TalkMoves)), sections 1.3-1.6 and 2.3-2.5 | CC BY-NC-SA 4.0, like the manual: these two files are shared under the same licence (attribution, non-commercial, share-alike) |
| `annomi_client.yaml`, `annomi_therapist.yaml` | AnnoMI labels (Wu et al., 2022; 2023), with wording quoted from the MITI 4.2.1 manual (Moyers et al.) and the AnnoMI-to-MISC mapping of Sun et al. (2024) | Quoted with attribution; MITI wording remains its authors' |
| `dunivin.yaml` | Code definitions from Dunivin (2025), EPJ Data Science, and the accompanying OSF materials ([osf.io/k4fg9](https://osf.io/k4fg9)), verbatim | Quoted with attribution for the replication; the OSF materials state no licence |
| `paraphrases/` | LLM-generated paraphrases of the definitions above, used in the RQ3 robustness arms | Same terms as the file each paraphrases |

The TalkMoves files and `annomi_therapist.yaml` include short example utterances quoted from the TalkMoves manual and the
MITI manual respectively; none of the AnnoMI-file examples occurs in the AnnoMI data. No other file contains example
utterances.
