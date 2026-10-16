**Table 10. A3 (exploratory): per-code recalibration of confidence (error AUROC on test, pooled)**

| comparison | AUROC a | AUROC b | a − b | 95% CI low | 95% CI high |
|---|---|---|---|---|---|
| Jev: per-code calibrated - raw | 0.797 | 0.756 | 0.040 | 0.027 | 0.054 |
| OpenAI Decisions: per-code calibrated - raw | 0.771 | 0.736 | 0.034 | 0.016 | 0.049 |
| Clef (27B): per-code calibrated - raw | 0.829 | 0.721 | 0.108 | 0.097 | 0.120 |
| Clef-flash (9B): per-code calibrated - raw | 0.801 | 0.712 | 0.090 | 0.077 | 0.102 |
| pplx-decider: per-code calibrated - raw | 0.813 | 0.764 | 0.048 | 0.039 | 0.056 |
| Kev-9B: per-code calibrated - raw | 0.809 | 0.694 | 0.115 | 0.094 | 0.136 |
| Kev-27B: per-code calibrated - raw | 0.814 | 0.788 | 0.026 | 0.020 | 0.034 |
| Luna verbal: per-code calibrated - raw | 0.783 | 0.741 | 0.042 | 0.029 | 0.055 |
| Haiku verbal: per-code calibrated - raw | 0.812 | 0.783 | 0.029 | 0.021 | 0.037 |
| gpt-oss verbal: per-code calibrated - raw | 0.787 | 0.727 | 0.060 | 0.049 | 0.071 |
| Qwen3.5-9B verbal: per-code calibrated - raw | 0.764 | 0.646 | 0.118 | 0.101 | 0.135 |
| Sonnet verbal: per-code calibrated - raw | 0.826 | 0.801 | 0.025 | 0.019 | 0.032 |
| Qwen3.8-2.4T verbal: per-code calibrated - raw | 0.798 | 0.802 | -0.005 | -0.012 | 0.004 |
| Jev per-code (full calib) - Sonnet verbal raw | 0.797 | 0.801 | -0.004 | -0.018 | 0.008 |
| Jev raw - Sonnet verbal raw | 0.756 | 0.801 | -0.045 | -0.060 | -0.032 |
| Jev per-code - Sonnet verbal per-code (own calib) | 0.797 | 0.826 | -0.030 | -0.041 | -0.019 |
| Jev per-code (full calib) - Qwen3.8-2.4T verbal raw | 0.790 | 0.802 | -0.013 | -0.048 | 0.019 |
| Jev raw - Qwen3.8-2.4T verbal raw | 0.769 | 0.802 | -0.033 | -0.069 | -0.001 |
| Jev per-code - Qwen3.8-2.4T verbal per-code (own calib) | 0.790 | 0.798 | -0.008 | -0.043 | 0.024 |

_Per-code Platt maps fitted on calib (codes with ≥ 50 calib decisions; others use the task map). CIs resample test clusters with the fitted maps held fixed, so they are conditional on those maps._
