**Table 1. Datasets, codebooks and samples**

| task | source | unit | license | codebook | codes | calib segments | test segments | clusters | annotators/item (median) | human α (multiply-coded items) |
|---|---|---|---|---|---|---|---|---|---|---|
| MFRC | MFRC (Trager et al. 2022) | Reddit comment | CC-BY-4.0 | multi-label (yes/no per code) | 8 | 608 | 1392 | 2000 | 3.00 | 0.17 |
| AnnoMI client | AnnoMI (Wu et al. 2022) | client utterance (+ previous turn) | public GitHub | mutually exclusive | 3 | 569 | 931 | 128 | 1.00 | 0.47 |
| AnnoMI therapist | AnnoMI (Wu et al. 2022) | therapist utterance (+ previous turn) | public GitHub | mutually exclusive | 4 | 602 | 898 | 132 | 1.00 | 0.74 |
| TalkMoves teacher | TalkMoves (Suresh et al. 2022) | teacher sentence (+ previous turn) | CC-BY-NC-SA | mutually exclusive | 7 | 370 | 1130 | 444 | 1.00 | – |
| TalkMoves student | TalkMoves (Suresh et al. 2022) | student sentence (+ previous turn) | CC-BY-NC-SA | mutually exclusive | 5 | 383 | 1117 | 436 | 1.00 | – |
| Dunivin | Dunivin (2025) replication | historical passage | OSF | multi-label (yes/no per code) | 9 | 34 | 86 | 120 | 1.00 | – |

_Exclusive codebooks include their 'none' option among the codes. α: Krippendorff's alpha on items coded by ≥ 2 annotators (macro over codes for multi-label)._
