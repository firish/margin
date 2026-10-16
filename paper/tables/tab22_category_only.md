**Table 22. Error AUROC from raw confidence, from the predicted code alone, and from per-code maps (pooled)**

| signal | raw confidence | predicted code only | per-code maps |
|---|---|---|---|
| Clef (27B) | 0.721 | 0.707 | 0.829 |
| Clef-flash (9B) | 0.712 | 0.693 | 0.801 |
| Haiku verbal | 0.783 | 0.696 | 0.812 |
| Jev | 0.756 | 0.714 | 0.797 |
| Kev-27B | 0.788 | 0.657 | 0.814 |
| Kev-9B | 0.694 | 0.713 | 0.809 |
| Luna verbal | 0.741 | 0.694 | 0.783 |
| OpenAI Decisions | 0.736 | 0.659 | 0.771 |
| gpt-oss verbal | 0.727 | 0.692 | 0.787 |
| pplx-decider | 0.764 | 0.691 | 0.813 |
| Sonnet verbal | 0.801 | 0.696 | 0.826 |

_'Predicted code only' scores each decision by the calibration accuracy of its predicted code (multi-label: of its code), with no confidence._
