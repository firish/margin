**Table 17. Flagged label-probe cells: decoy-name control and coding accuracy on 'recalled' items**

| model | task | items | real name | decoy name | no name | real − decoy | p (real > decoy) | p (Bonferroni, 3 cells) | recalled items | other items | coding accuracy, recalled | coding accuracy, other | p (recalled > other) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Haiku | MFRC | 100 | 0.314 | 0.246 | 0.174 | 0.068 | 0.021 | 0.063 | 22 | 78 | 0.841 | 0.782 | 0.085 |
| Luna | MFRC | 100 | 0.408 | 0.280 | 0.220 | 0.128 | 0.012 | 0.035 | 35 | 65 | 0.868 | 0.883 | 0.779 |
| gpt-oss-120b | AnnoMI therapist | 100 | 0.680 | 0.650 | 0.440 | 0.030 | 0.304 | 0.911 | 33 | 67 | 0.970 | 0.672 | 0.000 |

_Decoy: the same dataset description with a fictitious name (deviation 14). Memory is supported only if the real name beats the decoy. Recalled items: guided better than no name. Coding accuracy is the model's RQ1 accuracy on those items._
