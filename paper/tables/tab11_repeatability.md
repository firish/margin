**Table 11. Run-to-run repeatability (identical requests sent twice; 100 test segments per task; mean over tasks)**

| signal | segments | decisions | label flip rate | share identical | mean |Δp| | mean |Δ confidence| | accuracy, run 1 | accuracy, run 2 | flips at low confidence (AUROC) |
|---|---|---|---|---|---|---|---|---|---|
| Jev | 100 | 240 | 0.021 | 0.612 | 0.008 | 0.013 | 0.743 | 0.741 | 0.967 |
| OpenAI Decisions | 100 | 240 | 0 | 1.000 | 0.000 | 0.000 | 0.723 | 0.723 | – |
| Luna verbal | 100 | 240 | 0.066 | 0.567 | 0.030 | 0.020 | 0.742 | 0.742 | 0.874 |
| Luna logprob | 100 | 240 | 0.066 | 0.455 | 0.031 | 0.003 | 0.742 | 0.742 | 0.647 |
| Haiku verbal | 100 | 240 | <0.001 | 0.989 | 0.001 | 0.001 | 0.702 | 0.703 | 0.895 |
| Sonnet verbal | 100 | 240 | 0.033 | 0.514 | 0.016 | 0.023 | 0.764 | 0.765 | 0.921 |
| Clef (27B) | 100 | 240 | 0 | 1.000 | 0.000 | 0.000 | 0.693 | 0.693 | – |
| Clef-flash (9B) | 100 | 240 | 0 | 1.000 | 0.000 | 0.000 | 0.685 | 0.685 | – |
| pplx-decider | 100 | 240 | 0 | 1.000 | 0.000 | 0.000 | 0.759 | 0.759 | – |

_Flips at low confidence: AUROC of low confidence for the decisions whose label changed (– : no changes)._
