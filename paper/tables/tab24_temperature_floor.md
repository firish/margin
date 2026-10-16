**Table 24. Temperature scaling and per-code maps depend on how exact 0s and 1s are floored before taking logarithms (pooled)**

| signal | raw | temperature, floor 1e-4 | temperature, floor 1e-12 | raw  | temperature, floor 1e-4  | temperature, floor 1e-12  | per-code Platt, floor 1e-4 | per-code Platt, floor 1e-12 |
|---|---|---|---|---|---|---|---|---|
| Jev | 0.120 | 0.046 | 0.120 | 0.756 | 0.757 | 0.749 | 0.809 | 0.797 |
| Kev-27B | 0.073 | 0.029 | 0.029 | 0.788 | 0.788 | 0.788 | 0.814 | 0.814 |
| Luna logprob | 0.243 | 0.040 | 0.106 | 0.558 | 0.489 | 0.579 | 0.670 | 0.671 |
| OpenAI Decisions | 0.122 | 0.040 | 0.098 | 0.736 | 0.735 | 0.735 | 0.785 | 0.771 |
| gpt-oss logprob | 0.284 | 0.033 | 0.033 | 0.558 | 0.495 | 0.542 | 0.698 | 0.717 |
| Qwen3.5-9B logprob | 0.334 | 0.055 | 0.060 | 0.684 | 0.665 | 0.666 | 0.733 | 0.757 |
| Sonnet verbal | 0.051 | 0.053 | 0.053 | 0.801 | 0.802 | 0.802 | 0.826 | 0.826 |

_One temperature per task, fitted by weighted log-loss on calib. A task-level Platt or isotonic map is monotone and leaves error ranking unchanged at any floor; per-code Platt maps (EQ3) are not monotone across codes, so the floor changes their ranking for coders that return exact 0s and 1s._
