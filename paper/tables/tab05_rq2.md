**Table 5. RQ2: model uncertainty and human disagreement, by coder (test items with ≥ 3 annotators)**

| task | signal | Spearman: uncertainty vs human split | AUROC: flags split items | model κ (mean over references) | annotators below model |
|---|---|---|---|---|---|
| MFRC | Jev | 0.326 | 0.718 | 0.269 | 2 of 6 |
| MFRC | OpenAI Decisions | 0.242 | 0.659 | 0.260 | 2 of 6 |
| MFRC | Luna verbal | 0.339 | 0.721 | 0.250 | 2 of 6 |
| MFRC | Haiku verbal | 0.375 | 0.742 | 0.189 | 1 of 6 |
| MFRC | gpt-oss verbal | 0.307 | 0.673 | 0.310 | 3 of 6 |
| MFRC | Luna logprob | 0.138 | 0.582 | 0.250 | 2 of 6 |
| MFRC | 3-LLM agreement | 0.199 | 0.591 | 0.280 | 2 of 6 |
| MFRC | Sonnet verbal | 0.427 | 0.784 | 0.361 | 5 of 6 |
| MFRC | Clef (27B) | 0.404 | 0.771 | 0.282 | 2 of 6 |
| MFRC | Clef-flash (9B) | 0.395 | 0.764 | 0.306 | 3 of 6 |
| MFRC | pplx-decider | 0.398 | 0.767 | 0.246 | 1 of 6 |
| MFRC | Kev-9B | 0.325 | 0.717 | 0.239 | 2 of 6 |
| MFRC | Kev-27B | 0.394 | 0.764 | 0.249 | 2 of 6 |
| AnnoMI client | Jev | 0.547 | 0.859 | 0.706 | 7 of 10 |
| AnnoMI client | OpenAI Decisions | 0.517 | 0.812 | 0.696 | 7 of 10 |
| AnnoMI client | Luna verbal | 0.514 | 0.803 | 0.701 | 7 of 10 |
| AnnoMI client | Haiku verbal | 0.568 | 0.875 | 0.590 | 3 of 10 |
| AnnoMI client | gpt-oss verbal | 0.522 | 0.817 | 0.655 | 7 of 10 |
| AnnoMI client | Luna logprob | -0.100 | 0.462 | 0.701 | 7 of 10 |
| AnnoMI client | 3-LLM agreement | 0.324 | 0.604 | 0.648 | 7 of 10 |
| AnnoMI client | Sonnet verbal | 0.647 | 0.872 | 0.714 | 9 of 10 |
| AnnoMI client | Clef (27B) | 0.636 | 0.904 | 0.508 | 1 of 10 |
| AnnoMI client | Clef-flash (9B) | 0.600 | 0.912 | 0.483 | 1 of 10 |
| AnnoMI client | pplx-decider | 0.652 | 0.864 | 0.738 | 8 of 10 |
| AnnoMI client | Kev-9B | 0.179 | 0.627 | 0.462 | 1 of 10 |
| AnnoMI client | Kev-27B | 0.572 | 0.872 | 0.672 | 7 of 10 |
| AnnoMI therapist | Jev | 0.574 | 0.773 | 0.891 | 10 of 10 |
| AnnoMI therapist | OpenAI Decisions | 0.438 | 0.729 | 0.914 | 10 of 10 |
| AnnoMI therapist | Luna verbal | 0.560 | 0.797 | 0.899 | 10 of 10 |
| AnnoMI therapist | Haiku verbal | 0.359 | 0.670 | 0.834 | 6 of 10 |
| AnnoMI therapist | gpt-oss verbal | 0.379 | 0.648 | 0.880 | 8 of 10 |
| AnnoMI therapist | Luna logprob | -0.030 | 0.490 | 0.899 | 10 of 10 |
| AnnoMI therapist | 3-LLM agreement | 0.208 | 0.570 | 0.898 | 10 of 10 |
| AnnoMI therapist | Sonnet verbal | 0.511 | 0.738 | 0.883 | 8 of 10 |
| AnnoMI therapist | Clef (27B) | 0.327 | 0.664 | 0.807 | 4 of 10 |
| AnnoMI therapist | Clef-flash (9B) | 0.346 | 0.691 | 0.809 | 4 of 10 |
| AnnoMI therapist | pplx-decider | 0.542 | 0.771 | 0.908 | 10 of 10 |
| AnnoMI therapist | Kev-9B | 0.435 | 0.715 | 0.721 | 1 of 10 |
| AnnoMI therapist | Kev-27B | 0.505 | 0.766 | 0.793 | 3 of 10 |

_Model κ: against the majority of the other annotators, averaged over the leave-one-annotator-out references. The human range for each task is in tab05b_rq2_humans._
