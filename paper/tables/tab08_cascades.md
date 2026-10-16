**Table 9. A3 (exploratory): configurations under the naive and oracle rules, pooled over the five tasks**

| configuration | naive: accepted | naive: error | naive: USD/1k | oracle: accepted | oracle: USD/1k |
|---|---|---|---|---|---|
| Jev → human | 32.9 | 6.8 | 0.021 | 16.3 | 0.008 |
| OpenAI Decisions → human | 22.7 | 4.7 | 0.046 | 17.8 | 0.065 |
| Clef (27B) → human | 24.0 | 5.8 | 0.213 | 20.3 | 0.213 |
| Clef-flash (9B) → human | 25.4 | 7.0 | 0.080 | 20.7 | 0.080 |
| pplx-decider → human | 40.8 | 7.6 | 0.013 | 32.1 | 0.013 |
| Kev-9B → human | 23.9 | 6.3 | 0.073 | 19.2 | 0.058 |
| Kev-27B → human | 44.2 | 7.0 | 0.315 | 37.2 | 0.315 |
| Decision-model committee (7) → human | 37.5 | 7.2 | 0.849 | 30.9 | 0.849 |
| pplx-decider + Kev-27B agree → human | 42.7 | 6.8 | 0.334 | 40.5 | 0.334 |
| Jev + Kev-27B agree → human | 42.9 | 7.0 | 0.362 | 36.2 | 0.362 |
| Kev-27B → Sonnet verbal → human | 49.6 | 6.9 | 1.406 | 42.8 | 1.328 |
| Luna verbal → human | 20.2 | 4.6 | 0.082 | 16.2 | 0.066 |
| Haiku verbal → human | 16.1 | 3.2 | 1.137 | 16.8 | 1.906 |
| Sonnet verbal → human | 43.4 | 6.2 | 1.700 | 40.1 | 1.964 |
| Jev → Haiku verbal → human | 33.4 | 6.7 | 0.579 | 17.9 | 1.741 |
| Haiku verbal → Jev → human | 31.6 | 6.4 | 1.155 | 16.9 | 1.913 |
| Jev + Haiku verbal agree → human | 26.0 | 5.5 | 1.158 | 28.4 | 1.954 |
| Jev + Luna verbal agree → human | 34.3 | 6.6 | 0.130 | 17.8 | 0.131 |
| Jev → Sonnet verbal → human | 40.4 | 6.2 | 1.327 | 39.2 | 1.746 |
| Jev + Sonnet verbal agree → human | 43.1 | 6.5 | 1.736 | 39.1 | 2.011 |
| Haiku verbal + Sonnet verbal agree → human | 41.5 | 6.0 | 3.870 | 38.2 | 3.870 |

_Accepted: percent of decisions auto-accepted; error: percent wrong among accepted decisions (tasks weighted equally); USD/1k: API cost per 1,000 segments under that policy. 'A + B agree' accepts where both give the same label (confidence = the lower)._
