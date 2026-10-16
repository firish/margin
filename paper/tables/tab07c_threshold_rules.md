**Table 8. EQ2 (exploratory): threshold rules at a 5% target (percent)**

| signal | naive: accepted | naive: error | corpus LTT: accepted | corpus LTT: error | oracle: accepted | naive | LTT on decisions | LTT on i.i.d. clusters | corpus LTT | per-code maps, corpus LTT: accepted |
|---|---|---|---|---|---|---|---|---|---|---|
| Jev | 32.9 | 6.8 | 12.0 | 1.6 | 16.3 | 49.4 | 6.8 | 0.0 | 0.0 | 0.0 |
| OpenAI Decisions | 22.7 | 4.7 | 0.0 | – | 17.8 | 28.0 | 5.8 | 0.0 | 0.0 | 0.0 |
| Clef (27B) | 24.0 | 5.8 | 16.5 | 3.0 | 20.3 | 47.0 | 8.4 | 0.0 | 0.0 | 75.4 |
| Clef-flash (9B) | 25.4 | 7.0 | 15.6 | 2.7 | 20.7 | 51.8 | 8.4 | 0.0 | 0.0 | 65.5 |
| pplx-decider | 40.8 | 7.5 | 15.4 | 2.3 | 32.1 | 50.0 | 11.2 | 0.0 | 0.0 | 72.0 |
| Kev-9B | 23.9 | 6.3 | 13.4 | 2.4 | 19.2 | 52.4 | 9.0 | 0.0 | 0.0 | 0.0 |
| Kev-27B | 44.2 | 7.0 | 15.3 | 2.1 | 37.2 | 47.0 | 9.2 | 0.0 | 0.0 | 0.0 |
| Luna verbal | 20.2 | 4.6 | 12.8 | 2.2 | 16.2 | 5.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| Haiku verbal | 16.1 | 3.2 | 13.2 | 1.7 | 16.8 | 19.4 | 6.0 | 0.0 | 0.0 | 0.0 |
| gpt-oss verbal | 22.1 | 5.6 | 0.0 | – | 20.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Qwen3.5-9B verbal | 3.4 | 11.4 | 0.0 | – | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Luna logprob | 0.0 | – | 0.0 | – | 11.8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| gpt-oss logprob | 1.2 | 29.7 | 0.0 | – | 0.0 | 15.2 | 0.0 | 0.0 | 0.0 | 0.0 |
| Qwen3.5-9B logprob | 9.3 | 7.9 | 0.0 | – | 8.2 | 47.2 | 5.2 | 0.0 | 0.0 | 0.0 |
| Sonnet verbal | 43.4 | 6.2 | 16.5 | 2.5 | 40.1 | 6.0 | 0.0 | 0.0 | 0.0 | 68.9 |
| Qwen3.8-2.4T verbal | 37.9 | 4.3 | 0.0 | – | 38.0 | 5.2 | 0.6 | 0.0 | 0.0 | 0.0 |

_Panel A: frozen split, equal-weight means over the five tasks; oracle: fitted on test. Panel B: 500 random cluster re-splits of MFRC; for corpus LTT each test split is the uncoded remainder of its re-split, so exceedance there is a direct check of its guarantee. Per-code maps are fitted on half of the calib clusters, frozen, certified on the other half and applied unchanged to test._
