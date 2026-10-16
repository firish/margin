**Table 21. Our corpus LTT procedure: hand-coded units needed before it certifies a coder that is never wrong**

| task | sampling unit | units in corpus | mean unit weight / B | frozen split certifies it | lower bound | our procedure (median) | lower bound  | our procedure (median)  |
|---|---|---|---|---|---|---|---|---|
| MFRC | comment | 2000 | 1.00 | yes | 119 | 123 | 62 | 64 |
| AnnoMI client | transcript | 128 | 0.12 | no | unreachable | not reached | unreachable | not reached |
| AnnoMI client | utterance | 1500 | 0.85 | – | 119 | 146.5 | 62 | 75 |
| AnnoMI therapist | transcript | 132 | 0.13 | no | unreachable | not reached | unreachable | not reached |
| AnnoMI therapist | utterance | 1500 | 0.87 | – | 119 | 145 | 62 | 75 |
| TalkMoves teacher | lesson | 444 | 0.27 | no | 388 | not reached | 114 | not reached |
| TalkMoves teacher | sentence | 1500 | 0.78 | – | 119 | 163 | 62 | 81.5 |
| TalkMoves student | lesson | 436 | 0.08 | no | unreachable | not reached | unreachable | not reached |
| TalkMoves student | sentence | 1500 | 0.90 | – | 119 | 139 | 62 | 70.5 |

_B: the largest unit's weight. Lower bound: a necessary number over every set of hand-coded units and every visiting order (each unit's bet can multiply the evidence by at most 1 + 0.05 × its weight / B, and certification needs a factor of 320 with the grid, 20 with one threshold); fewer units cannot suffice, and reaching the bound does not guarantee certification. Unreachable: not reached even with every unit but one hand-coded. Our procedure: median number of units at which it first certifies, over 20 random hand-coding sequences (a description of typical runs, not a bound). Transcripts and lessons are the units of our splits; utterances and sentences drawn at random are an alternative design._
