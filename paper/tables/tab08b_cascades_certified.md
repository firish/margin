**Table 10. A3 (exploratory): configurations certified by corpus LTT on MFRC (percent; the only task where anything is certified when whole clusters are hand-coded)**

| configuration | certified: accepted | error | USD per 1k comments | remainder comments still read | all comments still read | overall error vs adjudicated |
|---|---|---|---|---|---|---|
| Jev → human | 59.8 | 1.6 | 0.042 | 93.2 | 95.2 | 1.1 |
| OpenAI Decisions → human | 0.0 | – | 0.000 | 100.0 | 100.0 | 0.2 |
| Clef (27B) → human | 82.3 | 3.1 | 0.245 | 73.3 | 81.4 | 2.6 |
| Clef-flash (9B) → human | 77.9 | 2.7 | 0.092 | 76.2 | 83.4 | 2.2 |
| pplx-decider → human | 77.1 | 2.3 | 0.029 | 73.2 | 81.3 | 1.9 |
| Kev-9B → human | 67.1 | 2.4 | 0.073 | 99.9 | 99.9 | 1.8 |
| Kev-27B → human | 76.6 | 2.1 | 0.315 | 84.6 | 89.3 | 1.7 |
| Decision-model committee (7) → human | 84.5 | 2.9 | 0.970 | 68.2 | 77.8 | 2.5 |
| pplx-decider + Kev-27B agree → human | 81.1 | 2.6 | 0.344 | 74.9 | 82.5 | 2.2 |
| Jev + Kev-27B agree → human | 75.7 | 2.2 | 0.357 | 78.1 | 84.7 | 1.8 |
| Kev-27B → Sonnet verbal → human | 82.3 | 2.5 | 4.141 | 74.0 | 81.9 | 2.1 |
| Luna verbal → human | 64.0 | 2.2 | 0.218 | 93.0 | 95.1 | 1.5 |
| Haiku verbal → human | 65.8 | 1.7 | 3.473 | 83.5 | 88.5 | 1.2 |
| Sonnet verbal → human | 82.7 | 2.5 | 4.521 | 65.7 | 76.1 | 2.2 |
| Jev → Haiku verbal → human | 66.5 | 1.7 | 3.513 | 83.1 | 88.2 | 1.3 |
| Haiku verbal → Jev → human | 66.5 | 1.8 | 3.509 | 83.1 | 88.2 | 1.3 |
| Jev + Haiku verbal agree → human | 63.3 | 1.8 | 3.516 | 90.2 | 93.2 | 1.3 |
| Jev + Luna verbal agree → human | 67.9 | 2.2 | 0.260 | 88.4 | 91.9 | 1.6 |
| Jev → Sonnet verbal → human | 77.0 | 1.9 | 4.479 | 78.4 | 85.0 | 1.5 |
| Jev + Sonnet verbal agree → human | 73.5 | 2.1 | 4.563 | 82.5 | 87.8 | 1.6 |
| Haiku verbal + Sonnet verbal agree → human | 74.2 | 2.3 | 7.995 | 75.3 | 82.8 | 1.8 |

_Multi-stage configurations are certified jointly over the pipeline. Accepted, error and remainder comments still read refer to the uncoded remainder (the test split); a comment is still read if at least one of its decisions is left for a person. All comments still read adds the 608 hand-coded calibration comments to the numerator and denominator. Overall error vs adjudicated: certified decisions scored against the adjudicated reference, the rest taken as human-coded; human-only coding of MFRC has 0.2% error against that reference._
