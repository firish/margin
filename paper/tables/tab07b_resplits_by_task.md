**Table 7b. Naive rule and LTT on decisions by task over 500 cluster re-splits (percent)**

| signal | task | naive: accepted | naive: none accepted | naive: error > 5% | LTT on decisions: accepted | LTT on decisions: none accepted | LTT on decisions: error > 5% |
|---|---|---|---|---|---|---|---|
| Clef (27B) | AnnoMI client | 7.3 | 5.0 | 73.8 | 0.0 | 100.0 | 0.0 |
| Clef (27B) | AnnoMI therapist | 18.8 | 0.0 | 63.6 | 0.2 | 99.4 | 0.6 |
| Clef (27B) | MFRC | 90.2 | 0.0 | 47.0 | 88.3 | 0.0 | 8.4 |
| Clef (27B) | TalkMoves student | 1.4 | 0.2 | 65.4 | 0.0 | 100.0 | 0.0 |
| Clef (27B) | TalkMoves teacher | 0.6 | 0.8 | 67.8 | 0.0 | 100.0 | 0.0 |
| Clef-flash (9B) | AnnoMI client | 11.5 | 1.2 | 84.8 | 0.0 | 100.0 | 0.0 |
| Clef-flash (9B) | AnnoMI therapist | 10.8 | 8.2 | 56.4 | 0.0 | 100.0 | 0.0 |
| Clef-flash (9B) | MFRC | 89.8 | 0.0 | 51.8 | 87.1 | 0.0 | 8.4 |
| Clef-flash (9B) | TalkMoves student | 2.4 | 0.0 | 33.2 | 0.0 | 100.0 | 0.0 |
| Clef-flash (9B) | TalkMoves teacher | 0.4 | 6.6 | 53.2 | 0.0 | 100.0 | 0.0 |
| Jev | AnnoMI client | 3.1 | 91.0 | 9.0 | 0.0 | 100.0 | 0.0 |
| Jev | AnnoMI therapist | 18.0 | 62.8 | 37.2 | 0.1 | 99.8 | 0.2 |
| Jev | MFRC | 80.8 | 0.0 | 49.4 | 78.3 | 0.0 | 6.8 |
| Jev | TalkMoves student | 1.7 | 85.0 | 15.0 | 0.0 | 100.0 | 0.0 |
| Jev | TalkMoves teacher | 2.5 | 81.0 | 19.0 | 0.0 | 100.0 | 0.0 |
| Kev-27B | AnnoMI client | 15.6 | 0.2 | 70.2 | 0.0 | 100.0 | 0.0 |
| Kev-27B | AnnoMI therapist | 51.9 | 0.0 | 41.8 | 12.4 | 68.0 | 2.0 |
| Kev-27B | MFRC | 94.0 | 0.0 | 47.0 | 91.7 | 0.0 | 9.2 |
| Kev-27B | TalkMoves student | 20.3 | 0.0 | 62.8 | 0.0 | 100.0 | 0.0 |
| Kev-27B | TalkMoves teacher | 21.9 | 0.6 | 72.2 | 0.0 | 100.0 | 0.0 |
| Kev-9B | AnnoMI client | 2.8 | 0.4 | 68.8 | 0.0 | 100.0 | 0.0 |
| Kev-9B | AnnoMI therapist | 14.2 | 0.0 | 79.8 | 0.1 | 99.6 | 0.4 |
| Kev-9B | MFRC | 87.8 | 0.0 | 52.4 | 56.8 | 16.0 | 9.0 |
| Kev-9B | TalkMoves student | 0.2 | 67.0 | 33.0 | 0.0 | 100.0 | 0.0 |
| Kev-9B | TalkMoves teacher | 0.4 | 5.6 | 41.6 | 0.0 | 100.0 | 0.0 |
| OpenAI Decisions | AnnoMI client | 1.8 | 91.8 | 8.2 | 0.0 | 100.0 | 0.0 |
| OpenAI Decisions | AnnoMI therapist | 32.1 | 18.8 | 51.2 | 1.4 | 96.8 | 3.2 |
| OpenAI Decisions | MFRC | 74.3 | 0.0 | 28.0 | 69.6 | 0.0 | 5.8 |
| OpenAI Decisions | TalkMoves student | 0.9 | 30.6 | 69.4 | 0.0 | 100.0 | 0.0 |
| OpenAI Decisions | TalkMoves teacher | 0.1 | 99.6 | 0.4 | 0.0 | 100.0 | 0.0 |
| pplx-decider | AnnoMI client | 17.3 | 0.0 | 65.0 | 0.0 | 100.0 | 0.0 |
| pplx-decider | AnnoMI therapist | 58.8 | 0.0 | 53.4 | 9.1 | 80.0 | 2.6 |
| pplx-decider | MFRC | 91.5 | 0.0 | 50.0 | 89.2 | 0.0 | 11.2 |
| pplx-decider | TalkMoves student | 13.1 | 0.0 | 52.0 | 0.0 | 100.0 | 0.0 |
| pplx-decider | TalkMoves teacher | 3.4 | 44.0 | 56.0 | 0.0 | 100.0 | 0.0 |
| Sonnet verbal | AnnoMI client | 12.7 | 0.2 | 49.2 | 0.2 | 99.2 | 0.8 |
| Sonnet verbal | AnnoMI therapist | 56.0 | 0.0 | 38.6 | 7.7 | 83.8 | 3.0 |
| Sonnet verbal | MFRC | 89.8 | 0.0 | 6.0 | 88.9 | 0.0 | 0.0 |
| Sonnet verbal | TalkMoves student | 11.6 | 10.6 | 62.0 | 0.0 | 100.0 | 0.0 |
| Sonnet verbal | TalkMoves teacher | 39.0 | 0.0 | 39.2 | 1.8 | 95.4 | 4.2 |

_Accepted: mean share of test decisions accepted; none accepted: share of re-splits accepting nothing; error > 5%: share of re-splits whose held-out error exceeds 5%. A re-split that accepts nothing cannot exceed 5%; read exceedance with acceptance._
