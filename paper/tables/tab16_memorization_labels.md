**Table 16. Memorization probe: recalling annotated labels without the codebook (amendment v1.3)**

| model | task | items | accuracy guided | accuracy general | difference | p (Bonferroni) | evidence of label leakage |
|---|---|---|---|---|---|---|---|
| Haiku | AnnoMI client | 100 | 0.510 | 0.620 | -0.110 | 1.000 | no |
| Haiku | AnnoMI therapist | 100 | 0.580 | 0.560 | 0.020 | 1.000 | no |
| Haiku | MFRC | 100 | 0.314 | 0.174 | 0.140 | 0.014 | yes |
| Haiku | TalkMoves student | 100 | 0.440 | 0.470 | -0.030 | 1.000 | no |
| Haiku | TalkMoves teacher | 100 | 0.160 | 0.280 | -0.120 | 1.000 | no |
| Luna | AnnoMI client | 100 | 0.580 | 0.710 | -0.130 | 1.000 | no |
| Luna | AnnoMI therapist | 100 | 0.530 | 0.560 | -0.030 | 1.000 | no |
| Luna | MFRC | 100 | 0.408 | 0.220 | 0.188 | 0.015 | yes |
| Luna | TalkMoves student | 100 | 0.530 | 0.520 | 0.010 | 1.000 | no |
| Luna | TalkMoves teacher | 100 | 0.520 | 0.470 | 0.050 | 1.000 | no |
| gpt-oss-120b | AnnoMI client | 100 | 0.560 | 0.510 | 0.050 | 1.000 | no |
| gpt-oss-120b | AnnoMI therapist | 100 | 0.680 | 0.440 | 0.240 | 0.004 | yes |
| gpt-oss-120b | MFRC | 100 | 0.260 | 0.223 | 0.037 | 1.000 | no |
| gpt-oss-120b | TalkMoves student | 100 | 0.510 | 0.410 | 0.100 | 0.621 | no |
| gpt-oss-120b | TalkMoves teacher | 100 | 0.200 | 0.270 | -0.070 | 1.000 | no |
| Qwen3.5-9B | AnnoMI client | 100 | 0.720 | 0.640 | 0.080 | 1.000 | no |
| Qwen3.5-9B | AnnoMI therapist | 100 | 0.570 | 0.560 | 0.010 | 1.000 | no |
| Qwen3.5-9B | MFRC | 100 | 0.174 | 0.170 | 0.005 | 1.000 | no |
| Qwen3.5-9B | TalkMoves student | 100 | 0.420 | 0.400 | 0.020 | 1.000 | no |
| Qwen3.5-9B | TalkMoves teacher | 100 | 0.360 | 0.300 | 0.060 | 1.000 | no |
| Qwen3.8-2.4T | AnnoMI client | 100 | 0.720 | 0.620 | 0.100 | 0.959 | no |
| Qwen3.8-2.4T | AnnoMI therapist | 100 | 0.730 | 0.700 | 0.030 | 1.000 | no |
| Qwen3.8-2.4T | MFRC | 100 | 0.185 | 0.160 | 0.026 | 1.000 | no |
| Qwen3.8-2.4T | TalkMoves student | 100 | 0.490 | 0.450 | 0.040 | 1.000 | no |
| Qwen3.8-2.4T | TalkMoves teacher | 100 | 0.280 | 0.210 | 0.070 | 1.000 | no |
| Sonnet | AnnoMI client | 100 | 0.720 | 0.740 | -0.020 | 1.000 | no |
| Sonnet | AnnoMI therapist | 100 | 0.730 | 0.770 | -0.040 | 1.000 | no |
| Sonnet | MFRC | 100 | 0.573 | 0.518 | 0.055 | 1.000 | no |
| Sonnet | TalkMoves student | 100 | 0.590 | 0.560 | 0.030 | 1.000 | no |
| Sonnet | TalkMoves teacher | 100 | 0.630 | 0.500 | 0.130 | 0.287 | no |

_Accuracy against the majority label (Jaccard for MFRC). Guided names the dataset and asks what its annotators assigned._
