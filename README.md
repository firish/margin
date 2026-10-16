# margin

Code, model outputs and paper source for **"Can Model Confidence Triage Qualitative Coding? Decision Models, LLMs,
and What a 5% Error Guarantee Actually Costs"** (Rishi Gulati, 2026). The paper is in [paper/main.pdf](paper/main.pdf).

The study compares seven decision models (Jev, OpenAI Decisions, Clef, Clef-flash, pplx-decider, Kev-9B, Kev-27B) with
twelve LLM confidence signals on five human-coded qualitative-coding tasks (MFRC, AnnoMI client and therapist, TalkMoves
teacher and student) plus a replication of Dunivin (2025). All models were called on 10–12 October 2026 (UTC).

## What is here

| Path | Contents |
|---|---|
| `src/margin/` | The harness: data loaders, request rendering, coders, metrics, analyses and the paper report |
| `codebooks/` | The codebooks, taken from each dataset's own annotation guidelines; sources and terms in `codebooks/README.md` (not covered by the MIT licence) |
| `configs/` | Models, prices and every experiment |
| `docs/ANALYSIS_PLAN.md` | The prospectively specified analysis plan (kept in this repository, not externally registered), its four dated amendments and the deviations log |
| `data/runs/` | Every model output: one row per decision with the probability and the model version returned |
| `data/samples/` | The stratified samples, inclusion weights and calibration/test splits |
| `data/adjudication/` | Adjudication keys and verdicts (ids and labels) |
| `data/ledger.csv.gz` | Text-free spend ledger: request hash, provider, model, tokens, cost and time for all 405,339 requests ($78.56) |
| `results/` | Every analysis output the paper reads |
| `paper/` | LaTeX source, figures and tables |
| `deploy/` | The Modal deployment used to self-host Kev (vendored from [jaredpalmer/kev](https://github.com/jaredpalmer/kev), Apache-2.0) |

Not included: the datasets themselves and the raw response cache, which contains dataset text. Download the datasets from
their sources (below); the model outputs in `data/runs/` and the ledger are enough to recompute every number in the paper,
including costs (`margin a3` reads costs from the ledger when there is no local cache). The guaranteed threshold rule is
corpus LTT (`triage.cluster_ltt` with `n_pool`): a without-replacement betting test whose final capital is averaged over
random visiting orders. `docs/ANALYSIS_PLAN.md` deviations 16-18 explain why it replaced decision-level LTT and why the
visiting order must be random.

## Reproduce the paper's numbers (no API calls)

```bash
uv sync
# put the raw datasets under data/raw/ (see Data below), then:
uv run margin prepare        # raw -> data/processed/<task>/{segments,labels}.parquet
uv run margin analyze        # RQ1-RQ3 (frozen plan) -> results/
uv run margin analyze --adjudication plan_v1  # RQ1 against the adjudicated reference
uv run margin rq5            # exploratory per-code analysis
uv run margin replicate      # Dunivin replication
uv run margin a3             # recalibration, threshold rules, risk control, budget, cascades
uv run margin repeatability  # run-to-run stability
uv run margin prompt-sensitivity  # second-prompt check (amendment v1.4)
uv run margin prevalence     # MFRC: prevalence baseline and recall of applying codes
uv run margin controls       # category-only control, LTT feasibility, simulation, utterance-sampling design
uv run margin report         # figures and tables -> paper/figures, paper/tables
cd paper && tectonic -X compile main.tex
uv run pytest
```

## Re-run the models

```bash
cp .env.example .env              # fill in the keys for the providers you want to run
uv run margin estimate jev_main   # planned calls and cost, counting the local cache
uv run margin run jev_main        # checks budget caps, runs, writes data/runs/<experiment>/
```

Every response is cached in `data/cache.sqlite`, keyed by its exact request, so reruns are free and the `cost_usd`
column is the spend ledger. Each experiment in `configs/experiments/` corresponds to a part of the paper; the analysis
plan says which.

## Data

| Dataset | Source | Licence |
|---|---|---|
| MFRC (Trager et al., 2022) | [Hugging Face](https://huggingface.co/datasets/USC-MOLA-Lab/MFRC) (`train_dedup` parquet) | CC BY 4.0 |
| AnnoMI (Wu et al., 2022) | [GitHub](https://github.com/uccollab/AnnoMI) (`AnnoMI-full.csv`) | none stated; cite the authors |
| TalkMoves (Suresh et al., 2022) | [GitHub](https://github.com/SumnerLab/TalkMoves) (repo extracted to `data/raw/talkmoves/repo`) | CC BY-NC-SA 4.0 |
| Dunivin (2025) | [OSF](https://osf.io/k4fg9) (`gold_standard_coding.csv`, `passages.csv`, `output/`) | none stated; cite the author |

## Licence

Code: MIT (see `LICENSE`). Kev deployment code: Apache-2.0 (`deploy/KEV_LICENSE`). Codebook text quotes published coding
guidelines and keeps their terms (the TalkMoves-derived codebooks are CC BY-NC-SA 4.0; see `codebooks/README.md`). Datasets
keep their own licences.
