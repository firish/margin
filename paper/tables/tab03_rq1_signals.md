**Table 3. RQ1: agreement, calibration and triage on the test split (mean over tasks)**

| signal | tasks | sample | κ (weighted) | accuracy | ECE | error AUROC | AURC | cov@5% (naive) | realized error | cov@5% (oracle) | cov@5% (LTT, decision-level) | cov@5% (LTT, cluster-level) | its error | share conf ≥ 0.99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Jev | 5 | full | 0.475 | 0.736 | 0.120 | 0.756 | 0.144 | 0.329 | 0.068 | 0.163 | 0.149 | 0.120 | 0.016 | 0.220 |
| OpenAI Decisions | 5 | full | 0.449 | 0.736 | 0.122 | 0.736 | 0.155 | 0.227 | 0.047 | 0.178 | 0.131 | 0.000 | – | 0.309 |
| Jev (yes/no format) | 4 | full | 0.522 | 0.710 | 0.103 | 0.729 | 0.160 | 0.126 | 0.103 | – | – | – | – | 0.067 |
| OpenAI Decisions (yes/no) | 4 | full | 0.514 | 0.712 | 0.201 | 0.719 | 0.166 | 0.149 | 0.069 | – | – | – | – | 0.515 |
| Luna verbal | 5 | full | 0.483 | 0.756 | 0.180 | 0.741 | 0.146 | 0.202 | 0.046 | 0.162 | 0.159 | 0.128 | 0.022 | 0.226 |
| Luna logprob | 5 | full | 0.483 | 0.756 | 0.243 | 0.558 | 0.225 | 0.000 | – | 0.118 | 0.000 | 0.000 | – | 0.990 |
| Luna self-consistency | 5 | 350/task | 0.506 | 0.758 | 0.196 | 0.624 | 0.197 | 0.000 | – | 0.000 | 0.000 | 0.000 | – | 0.851 |
| Haiku verbal | 5 | full | 0.465 | 0.723 | 0.154 | 0.783 | 0.149 | 0.161 | 0.032 | 0.168 | 0.150 | 0.132 | 0.017 | 0.003 |
| gpt-oss verbal | 5 | full | 0.452 | 0.715 | 0.204 | 0.727 | 0.192 | 0.221 | 0.056 | 0.206 | 0.000 | 0.000 | – | 0.043 |
| gpt-oss logprob | 5 | full | 0.452 | 0.715 | 0.284 | 0.558 | 0.264 | 0.012 | 0.297 | 0.000 | 0.000 | 0.000 | – | 0.998 |
| Qwen3.5-9B verbal | 5 | full | 0.383 | 0.663 | 0.303 | 0.646 | 0.276 | 0.034 | 0.114 | 0.000 | 0.000 | 0.000 | – | 0.389 |
| Qwen3.5-9B logprob | 5 | full | 0.384 | 0.663 | 0.334 | 0.684 | 0.237 | 0.093 | 0.079 | 0.082 | 0.072 | 0.000 | – | 0.964 |
| 3-LLM agreement | 5 | full | 0.493 | 0.747 | 0.153 | 0.670 | 0.193 | 0.000 | – | 0.000 | 0.000 | 0.000 | – | 0.719 |
| Sonnet verbal | 5 | full | 0.529 | 0.773 | 0.051 | 0.801 | 0.108 | 0.434 | 0.062 | 0.401 | 0.224 | 0.165 | 0.025 | 0.026 |
| Qwen3.8-2.4T verbal | 5 | 270/task | 0.499 | 0.751 | 0.102 | 0.802 | 0.122 | 0.379 | 0.043 | 0.380 | 0.150 | 0.000 | – | 0.002 |
| Qwen3.8-2.4T logprob | 5 | 270/task | 0.499 | 0.751 | 0.248 | 0.703 | 0.148 | 0.189 | 0.066 | 0.217 | 0.000 | 0.000 | – | 0.961 |
| Clef (27B) | 5 | full | 0.377 | 0.668 | 0.055 | 0.721 | 0.228 | 0.240 | 0.058 | 0.203 | 0.177 | 0.165 | 0.030 | 0.000 |
| Clef-flash (9B) | 5 | full | 0.369 | 0.659 | 0.065 | 0.712 | 0.234 | 0.254 | 0.070 | 0.207 | 0.173 | 0.156 | 0.027 | 0.000 |
| pplx-decider | 5 | full | 0.482 | 0.764 | 0.052 | 0.764 | 0.122 | 0.408 | 0.075 | 0.321 | 0.177 | 0.154 | 0.023 | 0.084 |
| Kev-9B | 5 | full | 0.300 | 0.617 | 0.100 | 0.694 | 0.281 | 0.239 | 0.063 | 0.192 | 0.166 | 0.134 | 0.024 | 0.000 |
| Kev-27B | 5 | full | 0.474 | 0.768 | 0.073 | 0.788 | 0.105 | 0.442 | 0.070 | 0.372 | 0.267 | 0.153 | 0.021 | 0.003 |
| Clef (yes/no) | 4 | full | 0.458 | 0.661 | 0.065 | 0.693 | 0.217 | 0.090 | 0.110 | – | – | – | – | 0.015 |
| Clef-flash (yes/no) | 4 | full | 0.413 | 0.633 | 0.081 | 0.700 | 0.234 | 0.075 | 0.096 | – | – | – | – | 0.000 |
| pplx-decider (yes/no) | 4 | full | 0.557 | 0.742 | 0.124 | 0.758 | 0.124 | 0.312 | 0.088 | – | – | – | – | 0.334 |
| Kev-9B (yes/no) | 4 | full | 0.289 | 0.490 | 0.076 | 0.666 | 0.395 | 0.052 | 0.136 | – | – | – | – | 0.003 |
| Kev-27B (yes/no) | 4 | full | 0.487 | 0.677 | 0.078 | 0.741 | 0.176 | 0.223 | 0.076 | – | – | – | – | 0.073 |

_Means are over the tasks listed (yes/no-format arms exist only for the four exclusive tasks), so compare rows with the same task set. κ uses design weights. cov@5% (naive): threshold with calib error ≤ 5%, applied to test (primary analysis); realized error = error among all accepted test decisions (tasks weighted equally). Oracle: threshold fitted on test (risk-matched upper bound). LTT, decision-level: binomial tests that treat decisions as independent; LTT, cluster-level: valid for exchangeable clusters (comments, transcripts, lessons), δ = 0.05._
