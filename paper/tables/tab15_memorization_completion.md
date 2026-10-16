**Table 15. Memorization probe: completing the second half of dataset items, overlap with the true continuation (amendment v1.3)**

| model | dataset | items | ROUGE-L, named | ROUGE-L, unnamed | difference | p (Bonferroni) |
|---|---|---|---|---|---|---|
| Haiku | annomi | 100 | 0.079 | 0.146 | -0.067 | 1.000 |
| Haiku | dunivin | 100 | 0.080 | 0.144 | -0.063 | 1.000 |
| Haiku | mfrc | 100 | 0.080 | 0.105 | -0.025 | 1.000 |
| Haiku | talkmoves | 100 | 0.082 | 0.155 | -0.072 | 1.000 |
| Luna | annomi | 100 | 0.082 | 0.156 | -0.074 | 1.000 |
| Luna | dunivin | 100 | 0.095 | 0.151 | -0.055 | 1.000 |
| Luna | mfrc | 100 | 0.082 | 0.111 | -0.030 | 1.000 |
| Luna | talkmoves | 100 | 0.063 | 0.174 | -0.111 | 1.000 |
| gpt-oss-120b | annomi | 100 | 0.057 | 0.090 | -0.034 | 1.000 |
| gpt-oss-120b | dunivin | 100 | 0.016 | 0.066 | -0.050 | 1.000 |
| gpt-oss-120b | mfrc | 100 | 0.038 | 0.055 | -0.016 | 1.000 |
| gpt-oss-120b | talkmoves | 100 | 0.049 | 0.066 | -0.017 | 1.000 |
| Qwen3.5-9B | annomi | 100 | 0.142 | 0.137 | 0.006 | 1.000 |
| Qwen3.5-9B | dunivin | 100 | 0.143 | 0.137 | 0.007 | 1.000 |
| Qwen3.5-9B | mfrc | 100 | 0.098 | 0.102 | -0.004 | 1.000 |
| Qwen3.5-9B | talkmoves | 100 | 0.163 | 0.148 | 0.015 | 1.000 |
| Qwen3.8-2.4T | annomi | 100 | 0.083 | 0.150 | -0.067 | 1.000 |
| Qwen3.8-2.4T | dunivin | 100 | 0.112 | 0.148 | -0.036 | 1.000 |
| Qwen3.8-2.4T | mfrc | 100 | 0.086 | 0.101 | -0.015 | 1.000 |
| Qwen3.8-2.4T | talkmoves | 100 | 0.125 | 0.168 | -0.043 | 1.000 |
| Sonnet | annomi | 100 | 0.098 | 0.153 | -0.055 | 1.000 |
| Sonnet | dunivin | 100 | 0.099 | 0.175 | -0.076 | 1.000 |
| Sonnet | mfrc | 100 | 0.089 | 0.116 | -0.027 | 1.000 |
| Sonnet | talkmoves | 100 | 0.076 | 0.156 | -0.080 | 1.000 |

_Named: the dataset is named and the exact original text requested. Unnamed: plain continuation. Difference: named − unnamed mean ROUGE-L; p: one-sided Wilcoxon signed-rank test for named > unnamed, Bonferroni-adjusted. Near-verbatim matches are in tab15b_memorization_verbatim._
