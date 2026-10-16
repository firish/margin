**Table 4. Confirmatory contrasts with Holm adjustment (pooled over the five tasks)**

| family | comparison | metric | difference | 95% CI | p (bootstrap) | p (Holm, within family) | claim |
|---|---|---|---|---|---|---|---|
| v1.0 confirmatory: Jev − comparator | Jev − Luna verbal | error AUROC | 0.016 | [0.0004, 0.031] | 0.046 | 0.184 | no claim |
| v1.0 confirmatory: Jev − comparator | Jev − Luna verbal | coverage@5% | 0.128 | [-0.044, 0.180] | 0.170 | 0.510 | no claim |
| v1.0 confirmatory: Jev − comparator | Jev − Haiku verbal | error AUROC | -0.026 | [-0.043, -0.011] | 0.002 | 0.016 | Jev worse |
| v1.0 confirmatory: Jev − comparator | Jev − Haiku verbal | coverage@5% | 0.168 | [-0.071, 0.184] | 0.256 | 0.510 | no claim |
| v1.0 confirmatory: Jev − comparator | Jev − gpt-oss verbal | error AUROC | 0.029 | [0.011, 0.051] | <0.001 | 0.016 | Jev better |
| v1.0 confirmatory: Jev − comparator | Jev − gpt-oss verbal | coverage@5% | 0.108 | [-0.064, 0.155] | 0.244 | 0.510 | no claim |
| v1.0 confirmatory: Jev − comparator | Jev − Luna logprob | error AUROC | 0.198 | [0.177, 0.221] | <0.001 | 0.016 | Jev better |
| v1.0 confirmatory: Jev − comparator | Jev − Luna logprob | coverage@5% | 0.329 | [0.148, 0.381] | <0.001 | 0.016 | Jev better |
| v1.0 confirmatory: Jev − comparator | Jev − gpt-oss logprob | error AUROC | 0.198 | [0.174, 0.221] | <0.001 | 0.016 | Jev better |
| v1.0 confirmatory: Jev − comparator | Jev − gpt-oss logprob | coverage@5% | 0.317 | [0.148, 0.367] | <0.001 | 0.016 | Jev better |
| v1.0 confirmatory: Jev − comparator | Jev − 3-LLM agreement | error AUROC | 0.087 | [0.069, 0.103] | <0.001 | 0.016 | Jev better |
| v1.0 confirmatory: Jev − comparator | Jev − 3-LLM agreement | coverage@5% | 0.329 | [0.160, 0.382] | <0.001 | 0.016 | Jev better |
| v1.0 confirmatory: Jev − comparator | Jev − Luna self-consistency | error AUROC | 0.153 | [0.118, 0.186] | <0.001 | 0.016 | Jev better |
| v1.0 confirmatory: Jev − comparator | Jev − Luna self-consistency | coverage@5% | 0.164 | [0.160, 0.446] | <0.001 | 0.016 | Jev better |
| v1.0 confirmatory: Jev − comparator | Jev − Sonnet verbal (v1.0 subset) | error AUROC | -0.048 | [-0.082, -0.015] | 0.002 | 0.016 | Jev worse |
| v1.0 confirmatory: Jev − comparator | Jev − Sonnet verbal (v1.0 subset) | coverage@5% | -0.249 | [-0.296, -0.029] | 0.022 | 0.110 | no claim |
| v1.1 A1 confirmatory: Jev − OpenAI Decisions | Jev − OpenAI Decisions | error AUROC | 0.020 | [0.005, 0.035] | 0.012 | 0.024 | Jev better |
| v1.1 A1 confirmatory: Jev − OpenAI Decisions | Jev − OpenAI Decisions | coverage@5% | 0.103 | [-0.050, 0.187] | 0.274 | 0.274 | no claim |
| v1.2 A5 confirmatory: Jev − other decision models | Jev − Clef (27B) | error AUROC | 0.036 | [0.017, 0.054] | <0.001 | 0.010 | Jev better |
| v1.2 A5 confirmatory: Jev − other decision models | Jev − Clef (27B) | coverage@5% | 0.090 | [-0.085, 0.135] | 0.572 | 1.000 | no claim |
| v1.2 A5 confirmatory: Jev − other decision models | Jev − Clef-flash (9B) | error AUROC | 0.045 | [0.024, 0.064] | <0.001 | 0.010 | Jev better |
| v1.2 A5 confirmatory: Jev − other decision models | Jev − Clef-flash (9B) | coverage@5% | 0.075 | [-0.086, 0.128] | 0.502 | 1.000 | no claim |
| v1.2 A5 confirmatory: Jev − other decision models | Jev − pplx-decider | error AUROC | -0.008 | [-0.027, 0.010] | 0.376 | 1.000 | no claim |
| v1.2 A5 confirmatory: Jev − other decision models | Jev − pplx-decider | coverage@5% | -0.079 | [-0.245, -0.047] | <0.001 | 0.010 | Jev worse |
| v1.2 A5 confirmatory: Jev − other decision models | Jev − Kev-9B | error AUROC | 0.062 | [0.041, 0.086] | <0.001 | 0.010 | Jev better |
| v1.2 A5 confirmatory: Jev − other decision models | Jev − Kev-9B | coverage@5% | 0.090 | [-0.041, 0.158] | 0.218 | 0.872 | no claim |
| v1.2 A5 confirmatory: Jev − other decision models | Jev − Kev-27B | error AUROC | -0.032 | [-0.047, -0.017] | <0.001 | 0.010 | Jev worse |
| v1.2 A5 confirmatory: Jev − other decision models | Jev − Kev-27B | coverage@5% | -0.113 | [-0.272, -0.063] | <0.001 | 0.010 | Jev worse |
| v1.1 A2 secondary: Jev − Qwen | Jev − Qwen3.5-9B verbal | error AUROC | 0.110 | [0.093, 0.129] | <0.001 | – | Jev better |
| v1.1 A2 secondary: Jev − Qwen | Jev − Qwen3.5-9B verbal | coverage@5% | 0.295 | [0.149, 0.364] | <0.001 | – | Jev better |
| v1.1 A2 secondary: Jev − Qwen | Jev − Qwen3.5-9B logprob | error AUROC | 0.072 | [0.050, 0.093] | <0.001 | – | Jev better |
| v1.1 A2 secondary: Jev − Qwen | Jev − Qwen3.5-9B logprob | coverage@5% | 0.236 | [0.065, 0.283] | <0.001 | – | Jev better |
| v1.0 extension (exploratory) | Jev − Sonnet verbal (full sample) | error AUROC | -0.045 | [-0.060, -0.032] | <0.001 | – | Jev worse |
| v1.0 extension (exploratory) | Jev − Sonnet verbal (full sample) | coverage@5% | -0.105 | [-0.283, -0.055] | <0.001 | – | Jev worse |

_Bootstrap p: two-sided, 1,000 clustered replicates with thresholds refitted (p < 0.001 shown as 0.001 before adjustment). Intervals resample each corpus's clusters jointly for related tasks. A claim needs the pooled 95% CI to exclude 0 and Holm p < 0.05 within the family. Holm is part of the plan's rule for v1.2 and a sensitivity analysis added for v1.0 and v1.1 (deviation 10). Sonnet enters the v1.0 family as specified, on its nested subset; its full-sample extension (deviation 9) is reported separately, outside the family._
