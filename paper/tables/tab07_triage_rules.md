**Table 7. EQ1 (exploratory): recalibration and the fixed rule 'accept if confidence ≥ 0.95' (pooled)**

| signal | ECE raw | ECE Platt | ECE isotonic | ECE temperature | error AUROC raw | error AUROC Platt | share accepted by '≥ 0.95' | its error |
|---|---|---|---|---|---|---|---|---|
| Jev | 0.120 | 0.038 | 0.040 | 0.120 | 0.756 | 0.756 | 0.364 | 0.121 |
| OpenAI Decisions | 0.122 | 0.073 | 0.042 | 0.098 | 0.736 | 0.736 | 0.498 | 0.115 |
| Clef (27B) | 0.055 | 0.041 | 0.045 | 0.054 | 0.721 | 0.721 | 0.081 | 0.007 |
| Clef-flash (9B) | 0.065 | 0.043 | 0.035 | 0.051 | 0.712 | 0.712 | 0.043 | 0.001 |
| pplx-decider | 0.052 | 0.043 | 0.048 | 0.047 | 0.764 | 0.764 | 0.281 | 0.041 |
| Kev-9B | 0.100 | 0.041 | 0.053 | 0.069 | 0.694 | 0.694 | 0.041 | 0.032 |
| Kev-27B | 0.073 | 0.032 | 0.037 | 0.029 | 0.788 | 0.788 | 0.155 | 0.034 |
| Luna verbal | 0.180 | 0.047 | 0.040 | 0.059 | 0.741 | 0.741 | 0.635 | 0.139 |
| Haiku verbal | 0.154 | 0.045 | 0.035 | 0.075 | 0.783 | 0.783 | 0.310 | 0.044 |
| gpt-oss verbal | 0.204 | 0.056 | 0.038 | 0.075 | 0.727 | 0.727 | 0.523 | 0.120 |
| Qwen3.5-9B verbal | 0.303 | 0.045 | 0.037 | 0.099 | 0.646 | 0.646 | 0.966 | 0.331 |
| Luna logprob | 0.243 | 0.038 | 0.039 | 0.106 | 0.558 | 0.556 | 0.994 | 0.241 |
| gpt-oss logprob | 0.284 | 0.040 | 0.045 | 0.033 | 0.558 | 0.539 | 0.998 | 0.285 |
| Qwen3.5-9B logprob | 0.334 | 0.052 | 0.054 | 0.060 | 0.684 | 0.684 | 0.984 | 0.334 |
| Sonnet verbal | 0.051 | 0.034 | 0.039 | 0.053 | 0.801 | 0.801 | 0.225 | 0.021 |
| Qwen3.8-2.4T verbal | 0.102 | 0.060 | 0.061 | 0.091 | 0.802 | 0.802 | 0.259 | 0.020 |

_Maps fitted per task on calib. Platt and isotonic maps act on the decision's confidence; temperature scaling acts on the full distribution and is sensitive to how exact 0s and 1s are floored (tab24_temperature_floor; here 1e-12). Errors are among accepted test decisions._
