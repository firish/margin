**Table 14. Same base model (Qwen3.5-9B): decision-model training vs the base LLM's own confidence (pooled)**

| comparison | metric | difference | 95% CI | p (bootstrap) |
|---|---|---|---|---|
| Clef-flash (9B) − Qwen3.5-9B verbal | error AUROC | 0.065 | [0.048, 0.085] | 0.000 |
| Clef-flash (9B) − Qwen3.5-9B verbal | coverage@5% | 0.221 | [0.181, 0.297] | 0.000 |
| Clef-flash (9B) − Qwen3.5-9B logprob | error AUROC | 0.027 | [0.004, 0.052] | 0.026 |
| Clef-flash (9B) − Qwen3.5-9B logprob | coverage@5% | 0.161 | [0.110, 0.211] | 0.000 |
| Kev-9B − Qwen3.5-9B verbal | error AUROC | 0.048 | [0.030, 0.066] | 0.000 |
| Kev-9B − Qwen3.5-9B verbal | coverage@5% | 0.205 | [0.152, 0.280] | 0.000 |
| Kev-9B − Qwen3.5-9B logprob | error AUROC | 0.010 | [-0.018, 0.036] | 0.512 |
| Kev-9B − Qwen3.5-9B logprob | coverage@5% | 0.146 | [0.079, 0.189] | 0.000 |
| Clef-flash (9B) − Kev-9B | error AUROC | 0.018 | [-0.007, 0.043] | 0.142 |
| Clef-flash (9B) − Kev-9B | coverage@5% | 0.015 | [-0.043, 0.097] | 0.560 |

_Pre-specified secondary (amendment v1.2). Our Qwen arm is the instruct model with thinking off; Kev-9B uses the base checkpoint; Clef-flash's exact base variant is undocumented._
