# margin: Can Jev triage qualitative coding?

## Context

**Goal:** a paper (plus the harness that produces it) on whether TypeSafe's **Jev** can act as a calibrated second coder for **deductive qualitative coding**. That is the codebook → coding → statistics workflow common in CS-ed and social science.

**Prior work, rechecked today (2026-10-09).** Since our earlier scoping chat, three relevant projects have appeared:

- **Denney & DiGiuseppe (arXiv 2610.06625, Oct 5).** Jev 1.13 vs GPT-6 Luna and Qwen on 7 political-science *classification* replications, measuring accuracy, ECE/AUROC (using logprobs), and cost.
  - Jev wasn't cheaper than *batch-priced* Luna, so we can't lead with a cost argument.
  - They did no human-disagreement modeling, no leave-one-coder-out, and no verbalized-confidence or multi-model baselines.
- **Li et al., CMU (arXiv 2609.26550).** A Jev-as-a-Judge accept/escalate cascade on LLM evaluation benchmarks. It compares only against verbalized confidence and doesn't touch qualitative coding.
- **`gbesse/jev-codebook`.** A CLI that applies a codebook with Jev. It has 0 stars and synthetic fixtures only, no benchmark and no LLM comparison.
- **RAND (arXiv 2512.00009).** Never released its cleaned subsets: it used 60k-word samples, its own chunking, and only the top four codes per dataset. **Its κ table can't be reused as a free baseline.** We run our own baselines and cite RAND for context only.

**The gap (our headline):** *triage and disagreement*. On real qualitative codebooks (multi-label and mutually exclusive, dialogue and Reddit), we ask:

1. Does Jev's probability flag its own mistakes better than the three LLM confidence signals (verbalized, logprob, multi-model agreement), and what share of coding decisions can be auto-accepted at ≤5% error?
2. Does Jev's uncertainty sit where *human coders disagree*?

**Constraints from the user:**

- About $100 total API spend (a soft cap).
- Keys for OpenAI, Anthropic, and Gemini, plus hosted open-weight models (Together/OpenRouter). Nothing runs locally.
- A private coded dataset might come later, so the importer stays generic.

## Research questions (frozen before test-split runs)

**RQ1 Triage (headline).** For each coder × signal, measure on (segment, code) decisions:

- AUROC of confidence vs. correctness.
- The risk-coverage curve and AURC.
- **Coverage@5% error**, with the threshold picked on the *calibration split* and the realized error and coverage reported on the *test split*.
- ECE and Brier.
- Cost per 1k segments.

The signals compared:

| Signal | How it's computed |
|---|---|
| Jev | Noul p → \|2p−1\|, or Choice `confidence` |
| LLM verbalized | 1–10 score per code (RAND-style) |
| LLM logprob | P(yes) read from each answer token's logprob in the JSON output (OpenAI top-5; gpt-oss via Together) |
| Multi-model agreement | Vote share across the 4 cheap LLMs. Reuses the verbalized runs at no extra cost. |
| Self-consistency | 5 samples × 1 cheap model, on a subset |

**RQ2 Disagreement.** Uses datasets that keep each annotator's labels (MFRC; AnnoMI-full if it has overlapping annotators):

- Spearman correlation between each signal's uncertainty and human soft-label entropy.
- AUROC for flagging items that humans split on.
- **Leave-one-coder-out:** κ(annotator *a*, majority of the others) vs κ(Jev, the same majority). This shows whether Jev falls within the spread of human κs.

**RQ3 Robustness: do surface changes hurt accuracy and calibration?**

Each perturbation is measured against the canonical run on the same items. We report:

- Flip rate.
- Change in κ.
- Change in ECE and Brier.
- Change in AUROC.
- Change in coverage@5%, i.e. whether triage still holds up.
- Whether flips concentrate in low-confidence items, meaning the instability shows up as uncertainty.

Because Jev is cheap, every perturbation runs on Jev over the full test split. The LLM comparison uses one cheap LLM (GPT-6 Luna) on a ~500-segment subset, and only for the order and label arms.

| Perturbation | Variants | What it tests |
|---|---|---|
| Choice option order | canonical, reversed, 3 seeded shuffles | jev-1.13's documented first-option bias |
| Noul question order within a call | canonical, reversed, shuffled | Whether "parallel questions are independent" holds |
| All codes in one call vs. one code per call | batched vs. isolated | The "multi-label for free" claim (TypeSafe's cookbook claims no change) |
| Code label names | meaningful label vs. neutral id (`code_A`) with the same definition | Whether Jev keys on the label or the definition |
| Definition paraphrase | 2 paraphrases (LLM-generated once, cached, spot-checked by hand) | Sensitivity to wording |
| Context | segment alone vs. plus previous turn | Context dependence (AnnoMI and TalkMoves only) |

**RQ4 Public vs. private: does it hold on data no model has seen?**

The user will try to obtain a **private coded interview dataset with its codebook**. We run the identical pipeline (all coders and signals, frozen thresholds from the analysis plan) and compare against the public sets:

- κ, coverage@5%, and AUROC on private vs. public.
- **The contamination test:** if the LLMs lose more than Jev when moving from public to private data, that points to memorization of the public sets. TypeSafe says Jev was trained only on synthetic data.
- This is also the only real long-form *interview* data in v1, which makes it the most realistic test of the target workflow.

**Gate before any call:** confirm the data's consent and IRB terms allow sending excerpts to third-party APIs (TypeSafe, OpenAI, Anthropic, Google, Together). If only some providers are allowed, run just those, and keep the data out of the repo (`data/private/` gitignored).

**RQ5 Where it breaks.**

- Per-code gaps by code type: concrete vs. implied meaning, prevalence, and definition length.
- A blinded adjudication sample of about 200 disagreements. This is human work; the harness exports and imports it.

**Supporting result:** κ, α, and F1 vs. human-human agreement, with bootstrap CIs clustered by conversation or transcript.

## Datasets (v1 = 3 public sets, ~2,000 sampled segments each, 30/70 calib/test split, fixed seed)

| Dataset | Source / license | Unit | Question type | Why |
|---|---|---|---|---|
| **MFRC** | HF `USC-MOLA-Lab/MFRC`, CC-BY-4.0, one row per annotator (53,827 rows, 16,123 comments, ≥3 annotators each) | comment | 8 codes, multi-label → **Noul per code** | RQ2 core: per-annotator labels |
| **AnnoMI** | GitHub `uccollab/AnnoMI` (simple + full CSVs, `annotator_id`) | utterance (+ previous turn as context) | client talk: change/neutral/sustain; therapist behaviour (4 options) → **Choice** | Expert-coded counselling dialogue; mutually exclusive codes |
| **TalkMoves** | GitHub `SumnerLab/TalkMoves`, CC-BY-NC-SA, with Coding Manual.pdf | sentence | teacher moves (+none), student moves (+none) → **Choice** | K-12 classroom discourse, close to CS-ed; large and has rare codes |

**Added 2026-10-09: Dunivin (2025), task six.** OSF k4fg9: 120 gold-coded NYT passages about
W.E.B. Du Bois, 9 multi-label codes (Noul per code), with the published GPT-4/3.5 predictions on the
paper's 111-passage test set (our pipeline reproduces its Table 2 exactly). Framed as a
published-baseline replication, not ground truth: the gold standard's construction is undocumented,
and the definitions were tuned for GPT-4 (a portability test for RQ5).

**Private set (RQ4), whenever it arrives.** A generic importer covers it: either CSV (`segment_id, text, [context], coder, codes`) plus a codebook YAML, or REFI-QDA `.qdpx` (most qualitative-analysis tools can export this). The private set is analyzed as a whole, not sampled, unless it's large.

**Phase 2 (after v1 works):**

- **Reddit eating-disorder topic annotations.** UVA Dataverse 10.18130/V3/NENELT; it returned a 500 today. 15 codes with definitions; RAND reported κ≈0.78 here.
- **QDR interview sets.** These come as `.qdpx` and need a free QDR account. Loading them requires a REFI-QDA importer.
- **Moral Foundations Subjective Corpus** (24 annotators per post). No public download found yet, so we'd ask the authors.
- **The user's private dataset**, which also serves as the contamination control.

Codebooks use each dataset's **own** annotation guidelines, copied verbatim where possible, with nothing invented. The same renderer feeds both Jev and the LLMs, so any difference comes from the model, not the prompt.

## Architecture (new Python project in `margin/`, managed with uv)

```
margin/
  pyproject.toml   .env.example   README.md
  docs/RESEARCH_PLAN.md  docs/ANALYSIS_PLAN.md      # frozen before any test-split run
  configs/models.yaml                               # provider, model id, $/M in/out, signals supported
  configs/experiments/{pilot,main,robustness}.yaml
  codebooks/{mfrc,annomi_client,annomi_therapist,talkmoves_teacher,talkmoves_student}.yaml
  src/margin/
    schema.py          # Segment, Code, Codebook (exclusive|multi), HumanLabel, Prediction (pydantic)
    datasets/          # mfrc.py annomi.py talkmoves.py csv_generic.py qdpx.py (generic importers serve RQ4)
                       #  → data/processed/<ds>/{segments,labels}.parquet (labels: segment×annotator×code)
    sampling.py        # stratified sample (keeps rare codes), calib/test split, clustered by conversation
    render.py          # codebook → Jev questions / LLM prompt text (single source of truth)
    perturb.py         # option/question order, batched vs isolated, neutral labels, paraphrase, context
                       #  → each a deterministic transform of (codebook, segment), keyed into the cache
    coders/
      jev.py           # AsyncTypeSafeClient.system_one; all codes in ONE call per segment; pinned jev-1.13.0
      llm_verbalized.py  llm_logprob.py  llm_selfconsistency.py
      providers/{openai,anthropic,gemini,together}.py
    cache.py           # sqlite keyed by sha256(provider, model, payload, params) → reruns cost $0
    budget.py          # pre-flight token estimate × price table; hard abort above per-run and global cap
    run.py             # experiment yaml → data/runs/<exp>/<ds>/<coder>.parquet (+ cost ledger)
    metrics/{agreement,calibration,selective,disagreement,robustness}.py
    adjudicate.py      # blinded disagreement export (CSV) and import
    report.py          # figures (reliability, risk–coverage, LOO forest plot) + tables
    cli.py             # margin prepare | estimate | run | evaluate | report | adjudicate
  tests/
```

**Jev question mapping** (this follows the jev-1.13 known-issues notes on literal reading, small state, and no double negatives):

- **Multi-label codes:** one `Noul(instructions=..., criteria={"true": definition + include examples, "false": exclusions + exclude examples})` per code, all in one request.
- **Exclusive codes:** `Choice(criteria={code: {"what", "not_for", "examples"}, "none": ...})`. Option order is randomized per segment with a seed, and the LLMs get the same order.
- **State:** `{"context": prev_turn, "segment": text}`, kept small.
- **Recorded per call:** `response.model`, the probabilities, confidence, and token usage.

**LLM baselines** (model IDs live in config and get checked against provider docs at build time; I'll load the `claude-api` skill before writing the Anthropic adapter):

- **Cheap tier, used for every signal on the full sample:** GPT-6 Luna, Claude Haiku 4.5, Gemini 3 Flash, gpt-oss-120b (Together).
- **Frontier, on a ~1,500-segment stratified subset only:** Claude Sonnet 5.5.
- **Logprob arm:** Luna and gpt-oss-120b. It's worth checking whether Denney & DiGiuseppe's finding holds here (74% of OpenAI probabilities fell at <0.01 or >0.99).
- **Cost controls:** prompt caching on the codebook prefix, and the batch APIs (OpenAI/Anthropic/Gemini, about 50% off) for the main run if the estimate is over budget.

**Budget design (~$100):**

- Jev costs about $0.042/M input tokens, so under $1 for everything.
- At roughly 6k segments × ~1.2k input tokens, each cheap model sees about 7M input tokens.
- Self-consistency is limited to 1 model × 5 samples × 2k segments. The frontier model sees only the subset.
- `margin estimate` prints the exact dollar cost per arm from `models.yaml` prices before any paid call. `run` refuses to go over `budget.cap_usd` (default 100) and keeps a running ledger.

## Build order

1. **Scaffold.** uv project, config, sqlite cache, budget estimator, CLI skeleton, `.env.example`. Smoke test: one real Jev call to confirm the SDK response shape (`result.nouls[k].noul`, `result.choices[k].probabilities`).
2. **Data.** Write the loaders for MFRC, AnnoMI, and TalkMoves into canonical parquet. Check whether AnnoMI-full has overlapping annotators. Write the codebooks from the original manuals, then add sampling and splits.
3. **Jev pilot.** 200 segments per dataset, calib split only, giving κ, a reliability diagram, and cost. Sanity-check before spending on LLMs.
4. **LLM coders.** Providers, verbalized and logprob arms, then self-consistency. Every paid run goes through `estimate` first.
5. **Metrics and tests.** κ, α, ECE, risk-coverage, coverage@ε, leave-one-coder-out, disagreement correlation.
6. **Perturbations (`perturb.py`).** A Jev pilot on the calib split first, to check that every variant renders correctly.
7. **Freeze `ANALYSIS_PLAN.md`,** then run the main and robustness experiments on the test split.
8. **Report.** Figures, tables, and the blinded adjudication export.
9. **Private set (RQ4) when it arrives.** Run the CSV/qdpx importer and the data-use check, then the same frozen pipeline. Then the phase-2 public datasets.

## Verification

- **Unit tests (`pytest`).** κ and α are checked against `sklearn.metrics.cohen_kappa_score` and the `krippendorff` package. ECE, risk-coverage, and coverage@ε are checked on hand-computed fixtures. Leave-one-coder-out is checked on a toy 3-annotator table.
- **Coders.** Tests run against mocked SDK/HTTP clients and check question rendering, answer parsing, and logprob extraction from saved responses.
- **Live smoke test.** `margin run --experiment pilot --limit 20` makes real Jev and LLM calls. Running it again must hit 100% cache and cost $0 in the ledger.
- **Sanity anchors.**
  - Human-human agreement on MFRC computed by our pipeline should roughly match the agreement reported in the MFRC paper.
  - Jev's run-to-run repeatability is checked on 50 repeated calls (Li et al. saw 0 changes in 96).
- **Budget.** `margin estimate --experiment main` must come in under the cap before `run` is allowed.
- **Perturbations.** Unit tests check that each transform is deterministic, keeps the definitions unchanged (only order or labels change), and that the inverse mapping restores the canonical code ids before scoring.
- **Private importer.** Tested on a synthetic `.qdpx` and CSV fixture, never on the real data.
