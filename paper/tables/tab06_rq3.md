**Table 6. RQ3: robustness to perturbations (design-weighted within tasks; equal-weight mean over the tasks each arm covers; vs the canonical arm on identical items)**

| signal | arm | tasks | flip rate | ΔAURC | Δκ | Δ error AUROC | ΔECE | flips at low confidence (AUROC) |
|---|---|---|---|---|---|---|---|---|
| Clef-flash (9B) | one code per call | 1 | 0.040 | -0.001 | -0.074 | -0.003 | -0.055 | 0.935 |
| Clef-flash (9B) | neutral labels | 5 | 0.094 | -0.008 | 0.006 | 0.004 | -0.004 | 0.881 |
| Clef-flash (9B) | no previous turn | 4 | 0.236 | 0.026 | -0.016 | -0.026 | 0.000 | 0.768 |
| Clef-flash (9B) | paraphrase A | 5 | 0.060 | 0.001 | -0.005 | -0.001 | 0.005 | 0.908 |
| Clef-flash (9B) | paraphrase B | 5 | 0.096 | 0.010 | -0.019 | -0.001 | -0.010 | 0.877 |
| Clef-flash (9B) | options reversed | 5 | 0.008 | -0.000 | -0.002 | 0.003 | -0.002 | 0.918 |
| Clef-flash (9B) | shuffle 1 | 5 | 0.004 | -0.000 | 0.000 | 0.002 | -0.002 | 0.953 |
| Clef-flash (9B) | shuffle 2 | 5 | 0.004 | -0.000 | 0.001 | 0.001 | -0.002 | 0.963 |
| Clef-flash (9B) | shuffle 3 | 5 | 0.004 | -0.000 | -0.001 | 0.002 | -0.002 | 0.954 |
| Jev | one code per call | 1 | 0.008 | -0.000 | -0.004 | 0.000 | -0.001 | 0.988 |
| Jev | neutral labels | 5 | 0.050 | -0.003 | -0.003 | 0.007 | 0.002 | 0.943 |
| Jev | no previous turn | 4 | 0.151 | 0.037 | -0.038 | -0.025 | 0.021 | 0.816 |
| Jev | paraphrase A | 5 | 0.032 | -0.000 | -0.006 | 0.011 | -0.001 | 0.952 |
| Jev | paraphrase B | 5 | 0.036 | 0.006 | -0.007 | -0.003 | 0.001 | 0.946 |
| Jev | options reversed | 5 | 0.042 | 0.006 | -0.005 | 0.000 | -0.005 | 0.951 |
| Jev | shuffle 1 | 5 | 0.042 | 0.001 | -0.002 | 0.003 | -0.003 | 0.949 |
| Jev | shuffle 2 | 5 | 0.040 | 0.000 | 0.000 | 0.001 | -0.005 | 0.952 |
| Jev | shuffle 3 | 5 | 0.041 | 0.001 | 0.000 | 0.001 | -0.000 | 0.950 |
| Kev-9B | one code per call | 1 | 0.002 | -0.000 | -0.003 | 0.000 | -0.000 | 0.997 |
| Kev-9B | neutral labels | 3 | 0.102 | 0.000 | -0.000 | 0.007 | -0.013 | 0.899 |
| Kev-9B | no previous turn | 2 | 0.175 | 0.010 | -0.012 | -0.016 | -0.020 | 0.747 |
| Kev-9B | paraphrase A | 3 | 0.064 | 0.008 | -0.021 | 0.012 | 0.006 | 0.926 |
| Kev-9B | paraphrase B | 3 | 0.076 | 0.004 | -0.020 | 0.009 | 0.012 | 0.920 |
| Kev-9B | options reversed | 4 | 0.306 | 0.119 | -0.099 | 0.046 | 0.044 | 0.813 |
| Kev-9B | shuffle 1 | 4 | 0.192 | 0.066 | -0.042 | -0.006 | -0.026 | 0.842 |
| Kev-9B | shuffle 2 | 4 | 0.204 | 0.072 | -0.053 | 0.007 | -0.034 | 0.843 |
| Kev-9B | shuffle 3 | 3 | 0.129 | 0.035 | -0.023 | -0.014 | -0.002 | 0.864 |
| Luna verbal | neutral labels | 5 | 0.081 | 0.016 | -0.006 | -0.038 | 0.006 | 0.805 |
| Luna verbal | options reversed | 5 | 0.100 | -0.011 | 0.031 | -0.022 | -0.015 | 0.783 |
| Luna verbal | shuffle 1 | 5 | 0.106 | -0.004 | 0.025 | -0.026 | -0.012 | 0.785 |
| OpenAI Decisions | one code per call | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – |
| OpenAI Decisions | neutral labels | 5 | 0.096 | -0.005 | -0.004 | 0.001 | 0.008 | 0.889 |
| OpenAI Decisions | no previous turn | 4 | 0.188 | 0.011 | -0.040 | -0.002 | -0.016 | 0.794 |
| OpenAI Decisions | paraphrase A | 5 | 0.072 | -0.002 | 0.012 | -0.005 | 0.035 | 0.899 |
| OpenAI Decisions | paraphrase B | 5 | 0.076 | 0.001 | 0.017 | -0.024 | 0.026 | 0.895 |
| OpenAI Decisions | options reversed | 5 | 0.103 | -0.005 | 0.021 | -0.014 | 0.022 | 0.873 |
| OpenAI Decisions | shuffle 1 | 5 | 0.089 | -0.005 | 0.012 | -0.001 | 0.018 | 0.875 |
| OpenAI Decisions | shuffle 2 | 5 | 0.083 | -0.004 | 0.021 | -0.018 | 0.015 | 0.861 |
| OpenAI Decisions | shuffle 3 | 5 | 0.088 | 0.000 | 0.017 | -0.019 | 0.015 | 0.870 |
| pplx-decider | one code per call | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – |
| pplx-decider | neutral labels | 5 | 0.071 | -0.004 | -0.002 | 0.009 | 0.012 | 0.890 |
| pplx-decider | no previous turn | 4 | 0.154 | 0.023 | -0.041 | -0.012 | 0.029 | 0.801 |
| pplx-decider | paraphrase A | 5 | 0.038 | -0.001 | 0.008 | -0.007 | -0.002 | 0.932 |
| pplx-decider | paraphrase B | 5 | 0.037 | -0.001 | 0.006 | -0.003 | 0.000 | 0.930 |
| pplx-decider | options reversed | 5 | 0.064 | -0.019 | 0.012 | 0.021 | 0.010 | 0.875 |
| pplx-decider | shuffle 1 | 5 | 0.064 | -0.012 | -0.002 | 0.014 | 0.009 | 0.853 |
| pplx-decider | shuffle 2 | 5 | 0.062 | -0.013 | 0.006 | 0.009 | 0.007 | 0.852 |
| pplx-decider | shuffle 3 | 5 | 0.065 | -0.012 | 0.003 | 0.013 | 0.008 | 0.850 |
| Sonnet verbal | neutral labels | 5 | 0.049 | -0.002 | -0.016 | 0.011 | 0.004 | 0.915 |
| Sonnet verbal | options reversed | 5 | 0.052 | -0.003 | -0.033 | 0.016 | 0.012 | 0.882 |
| Sonnet verbal | shuffle 1 | 5 | 0.059 | 0.000 | -0.020 | 0.006 | -0.002 | 0.887 |

_Flip rate and ΔAURC are the pre-specified primary outcomes; tab06b_rq3_intervals gives per-task 95% intervals for selected arms. 'No previous turn' removes information rather than rewording the request._
