**Table 19. LLM prompt sensitivity: verbalized distribution vs canonical prompt (amendment v1.4)**

| task | signal | decisions | err. AUROC canonical | err. AUROC distribution | difference | lo | hi | κ canonical | κ distribution | ECE canonical | ECE distribution |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MFRC | Sonnet verbal | 1200 | 0.933 | 0.897 | -0.036 | -0.052 | -0.019 | 0.346 | 0.365 | 0.048 | 0.031 |
| MFRC | Haiku verbal | 1200 | 0.942 | 0.897 | -0.045 | -0.063 | -0.029 | 0.166 | 0.174 | 0.128 | 0.076 |
| AnnoMI client | Sonnet verbal | 150 | 0.750 | 0.726 | -0.024 | -0.074 | 0.018 | 0.413 | 0.432 | 0.065 | 0.079 |
| AnnoMI client | Haiku verbal | 150 | 0.733 | 0.776 | 0.043 | -0.041 | 0.136 | 0.479 | 0.453 | 0.118 | 0.158 |
| AnnoMI therapist | Sonnet verbal | 150 | 0.808 | 0.837 | 0.029 | -0.012 | 0.083 | 0.751 | 0.715 | 0.053 | 0.083 |
| AnnoMI therapist | Haiku verbal | 150 | 0.770 | 0.717 | -0.053 | -0.139 | 0.014 | 0.634 | 0.661 | 0.155 | 0.158 |
| TalkMoves teacher | Sonnet verbal | 150 | 0.817 | 0.747 | -0.070 | -0.135 | 0.000 | 0.590 | 0.603 | 0.089 | 0.045 |
| TalkMoves teacher | Haiku verbal | 150 | 0.747 | 0.767 | 0.021 | -0.080 | 0.120 | 0.554 | 0.528 | 0.141 | 0.215 |
| TalkMoves student | Sonnet verbal | 150 | 0.829 | 0.758 | -0.071 | -0.133 | -0.012 | 0.539 | 0.548 | 0.118 | 0.090 |
| TalkMoves student | Haiku verbal | 150 | 0.786 | 0.770 | -0.015 | -0.082 | 0.046 | 0.445 | 0.430 | 0.261 | 0.330 |
| Pooled | Sonnet verbal | 1800 | 0.827 | 0.793 | -0.034 | -0.056 | -0.011 | 0.528 | 0.533 | 0.074 | 0.066 |
| Pooled | Haiku verbal | 1800 | 0.796 | 0.786 | -0.010 | -0.045 | 0.022 | 0.456 | 0.449 | 0.161 | 0.187 |

_Same test items (first 150 per task). Difference = distribution − canonical error AUROC, paired cluster bootstrap 95% CI; related tasks share cluster draws._
