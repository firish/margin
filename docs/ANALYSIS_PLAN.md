# Analysis plan

**Status: FROZEN v1.4 (v1.0 approved 2026-10-10; amendment v1.1 approved by the author 2026-10-10, recorded before any Decisions API or Qwen data was collected; amendment v1.2 approved 2026-10-11, recorded before any Clef, Perplexity or Kev data; amendment v1.3 (memorization probe) approved 2026-10-11 before any probe call; amendment v1.4 (LLM prompt sensitivity) approved 2026-10-11 before any call with the new prompt). Changes after this point go in the deviations log.**

## 0. What has and has not been seen

| Data | Seen | Role |
|---|---|---|
| Jev on calib: 200 segments per task (pilot) and question-format pilot | κ, calibration, triage, confusion matrices | development |
| All coders on calib: bug check | κ and error AUROC per coder × task | development |
| Dunivin, all 120 passages (Jev; published GPT runs) | everything | **exploratory / replication only** |
| LLM signals on test | collected, **not scored** | confirmatory |
| Jev on test (5 main tasks) | **not run** | confirmatory |

Label-free diagnostics (parse rates, stop reasons, confidence saturation) have been inspected for all coders.

## 1. Scope

- **Confirmatory:** RQ1–RQ3 on the test split of `mfrc`, `annomi_client`, `annomi_therapist`, `talkmoves_teacher`, `talkmoves_student`.
- **Exploratory:** RQ5, Dunivin, and anything else.
- **RQ4:** run on the private set with this plan, unchanged, when it arrives.

## 2. Data, splits, weights (fixed)

- **Sample:** shared sample `main` (`configs/samples.yaml`, seed 0). Exclusive tasks are stratified with a floor of 60 per code. AnnoMI's 10-annotator subset is always included.
- **Split:** calib/test by hash of transcript (or comment), ~30/70. Calib is used only to fit thresholds and recalibration.
- **Weights:** headline rates use Horvitz-Thompson weights; per-code results are unweighted.

## 3. Reference labels

- **Human reference (primary):** majority of annotators. Ties are dropped: exact 50% for multi-label, a tied plurality for exclusive.
- **Adjudicated reference (secondary, section 9):** blinded verdicts replace the human label on adjudicated items. Unadjudicated disagreements keep the human label, which is conservative against every coder.

## 4. Signals (definitions frozen as implemented in `src/margin`)

**Decisions:**
- **Exclusive:** one per segment. Decision = argmax; confidence = top probability.
- **Multi-label:** one per (segment, code). Decision = p ≥ 0.5; confidence = max(p, 1 − p).
- **Ties** are broken by codebook order.

| Signal | Definition |
|---|---|
| `jev` (**primary Jev arm**) | Choice for exclusive tasks, one Noul per code for multi-label |
| `jev` `noul_format` (secondary, exclusive only) | One Noul per option, combined by odds normalization (`coders.jev.exactly_one`) |
| `{luna,haiku,oss}-verbal`, `sonnet-verbal` | Stated probability. For multi-label: decision = stated yes/no, confidence = distance of the stated probability from 50 (reading-agnostic) |
| `{luna,oss}-logprob` | Answer-token probability from the same call |
| `luna_sc-vote` | Share of 5 samples (temperature 1) |
| `agree3` | Vote share across luna/haiku/oss verbal decisions |

Sonnet and self-consistency are compared only on their nested subsets.

## 5. RQ1, triage (headline)

**Primary metrics:**
- **RQ1a, ranking:** error AUROC, the probability that a correct decision has higher confidence than an incorrect one.
- **RQ1b, payoff:** coverage@5%. Take the lowest per-task threshold whose calib error is ≤ 5%; report the coverage it achieves on test and the realized test error.

**Secondary:** AURC; coverage@10%; coverage@5% with the Clopper-Pearson rule (one-sided, α = 0.05); ECE (10 equal-width bins, weighted); Brier; κ; macro-F1; cost per 1,000 segments; share of confidence ≥ 0.99; number of distinct confidence values.

**Contrasts:**
- **Jev (primary) minus each signal,** paired on identical items. Reported per task and as an equal-weight mean over the five tasks.
- **A superiority claim** ("Jev better/worse than S") requires the pooled 95% interval to exclude 0.
- **No "best LLM" is selected after the fact.**

**Choices fixed for the primary analysis:**
- one threshold per task
- decision threshold 0.5 for multi-label
- raw, uncalibrated confidence

**Secondary analyses:** per-code thresholds fitted on calib; temperature scaling fitted on calib.

## 6. Question format (secondary, pre-registered)

- **Contrast:** Jev `noul_format` vs Choice on the four exclusive tasks, test split.
- **Directional hypothesis (from the calib pilot):** Noul has lower AURC and a lower share of confidence ≥ 0.99.
- **Also reported:** change in κ, ECE, error AUROC, coverage@5%.

## 7. RQ2, human disagreement

Data: MFRC (all items) and the AnnoMI 10-annotator subset (both AnnoMI tasks).

- **Alignment:** Spearman correlation between each signal's normalized entropy and human label entropy; AUROC of signal entropy for "humans not unanimous".
- **Leave-one-coder-out:** for each annotator, κ(annotator, majority of the others) and κ(signal, same majority) on the same items.
  - "Within the human range" means between the lowest and highest annotator.
  - "Above a typical coder" means above the median annotator.
- Every signal gets the same treatment.

## 8. RQ3, robustness

**Jev, full test split:**
- reversed; shuffle1–3
- isolated (multi-label tasks only)
- neutral_labels
- no_context (AnnoMI, TalkMoves)
- paraphrase0 and paraphrase1 (`codebooks/paraphrases/`; meaning-preserving; definitions only; spot-checked)
- noul_format

**Luna, 510 test segments:** reversed, shuffle1, neutral_labels.

**Metrics:** each arm vs canonical on identical items.
- **Primary:** flip rate and change in AURC.
- **Secondary:** change in κ, ECE, error AUROC and coverage@5%; AUROC of canonical confidence for predicting flips.

## 9. Adjudication protocol

- **One adjudicator** (the author), blinded: random item ids, shuffled candidates, no sources or confidences shown.
- **Pool:** disagreements from `jev`, `jev` noul_format, luna/haiku/oss verbal, and sonnet (on its subset). A unit enters if any of these disagrees with the human reference.
- **Selection per task (40 items):**
  - all disagreements where any coder had confidence ≥ 0.9 (randomly subsampled if more)
  - a random sample of the remaining disagreements
  - 10% hidden controls (all coders agree with the human)
- **Intra-rater check:** 10% of items re-presented blind in a second sitting; report the self-agreement rate.
- **Reporting:** upheld rates by source and stratum; control overturn rate; adjudicated-reference metrics (section 3).

## 10. RQ5, where it breaks (exploratory)

- **Per-code results** (κ, error AUROC, ECE).
- **Code types** come from `docs/code_types.csv`: inference level S/M/I, context dependence, residual. They were labelled from definitions before per-code test results were opened; the author checks them.
- **Inference type is confounded with task.** Comparisons are therefore within task, or use task fixed effects. Only Dunivin and the S-vs-M codes in two tasks vary within task, so this analysis is suggestive.
- **Adjudication notes** are coded into RAND-style categories: under-specified code, label error, missing context, model error.

## 11. Dunivin replication section

Reported separately:
- exact reproduction of the paper's Table 2
- Jev and our LLM signals vs the published GPT-4 runs on the paper's 111 test passages
- triage metrics

The section is framed as exploratory: results were seen, n is small, and the definitions were tuned for GPT-4.

## 12. Statistics

- **Bootstrap:** clustered by transcript (AnnoMI, TalkMoves) or by segment (MFRC, Dunivin); 1,000 resamples; seed 0; paired for contrasts; 95% percentile intervals.
- **Reporting:** estimates with intervals and directions; no p-value lists.
- **Robustness of the error-rate claims:** realized test error is shown next to every coverage figure.

## 13. Execution order

1. ~~LLM calib bug check~~ (done; no bugs; logprob/self-consistency AUROC near 0.5 explained by saturation)
2. ~~Freeze this plan.~~ (2026-10-10)
3. Jev runs: `main` (canonical, all 8,120 segments), `noul_format` (exclusive tasks, all), robustness arms incl. paraphrases (test split). Estimated ≈ $3.
4. Test-split evaluation, human reference.
5. Pooled adjudication export; adjudication; adjudicated evaluation.
6. RQ5, Dunivin section, report.

## 14. Known limitations

- Human labels are noisy (MFRC α ≈ 0.17), and only one adjudicator is available.
- All datasets are public and may be in LLM training data (RQ4 addresses this).
- The AnnoMI codebook was reconstructed from MITI/MISC sources.
- Jev's API rounds probabilities to 0.01.
- Token logprobs are near-saturated for both logprob models.
- Code type is confounded with dataset.

## Amendment v1.1 (2026-10-10, before any new data)

**Why:** OpenAI's Decisions API (public beta 2026-10-06) is a decision model in the same category as
Jev. Comparing the two is the most direct test of whether Jev's confidence behaviour is specific to Jev
or general to decision-style APIs.

### A1. New coder: OpenAI Decisions API (`oaid`) — confirmatory

- **Endpoint and model:** `/v1/decisions` with `gpt-6-luna`, the only model it supports. Each response
  records its model id; the run date is recorded too, because the API is in beta.
- **Same content as Jev, mapped to the Decisions types:**
  - Jev Choice → `choice`. Option descriptions are Jev's `code_spec` serialized as JSON text.
  - Jev Noul → `predicate`. The true/false criteria are appended to the instructions as JSON text.
  - Jev's state object → `input` as a JSON string.
  - Option and question order, labels and wording are identical to Jev's for every arm.
- **Arms:** identical to Jev's.
  - primary: native format on the full sample
  - secondary: yes/no-per-option format for exclusive tasks, combined with the same odds rule
  - RQ3 robustness arms on the test split
- **Refusals:** a refused question is recorded and treated as missing for that unit.
- **Contrasts, pooled and per task, with the section 12 bootstrap:**
  - **Confirmatory:** Jev − oaid on the primary metrics (error AUROC, coverage@5%). Two-sided, no
    directional hypothesis.
  - **Secondary:** oaid vs each LLM signal; RQ2 and RQ3 computed for oaid exactly as for Jev.

### A2. New secondary comparator: non-reasoning open-weight LLM with logprobs (`qwen`)

- **Model:** a Qwen instruct model on Together serverless, chosen by availability *without* enabling
  third-party data sharing.
- **Setup:** the same prompt and signals as the other LLMs (verbalized and logprob) on the full sample.
- **Purpose:** tests whether token probabilities are uninformative only for reasoning models.

### A3. Additional analyses (all coders)

- **Recalibration** (temperature/Platt scaling fitted on calib) and **per-code thresholds.** Both were
  v1.0 secondary analyses not yet run.
- **Conformal risk control for 5% error** (finite-sample guarantee, δ = 0.05; threshold fitted on
  calib, applied to test).
- **Cascade simulation:** a decision model first (Jev or oaid), escalating below threshold to an LLM
  and then to a human. Reports accuracy against the human and adjudicated references, cost per 1,000
  segments, and human workload.
- **Labelling:** v1.0 test results for Jev and the LLMs were seen before this amendment, so the A3
  analyses are **exploratory** for those coders. For oaid and qwen they are specified before data.

### A4. Deferred

- Sonnet on the full sample: the author will decide later.

## Amendment v1.2 (2026-10-11, before any data from these coders)

Approved by the author on 2026-10-11. Written before any study item was sent to these coders. Only
connectivity smoke tests on the harness's synthetic smoke segment preceded it.

### A5. Other decision models ("System One"-compatible): confirmatory family

- **Coders.** All are sent Jev's exact requests (state and questions, same order, labels and wording,
  every arm):
  - `clef`: Cloudflare Workers AI `@cf/cloudflare/clef` (27B, built on Qwen3.8-27B).
  - `clef_flash`: `@cf/cloudflare/clef-flash` (9B, built on Qwen3.5-9B).
  - `pplx`: Perplexity `pplx-decider-v1.1-27b`, the current version at run time.
  - `kev9b` and `kev27b`: open-source Kev 1.0 (`jaredpalmer/kev-9b` on Qwen3.5-9B-Base,
    `jaredpalmer/kev-27b` on Qwen3.8-27B). Self-deployed on Modal from the kev repository's
    `kev_serve.py` (kev package commit fc4a17e), bf16, with the checkpoint's fitted temperature.
- **One mapping rule.** If a provider rejects Jev's structured option specs, each spec is sent as
  JSON text, the same rule as oaid (A1). Any such provider is recorded.
- **Arms.**
  - All five: primary (native format, full sample) and the yes/no-per-option format arm.
  - RQ3 robustness arms (test split): `clef_flash`, `pplx`, `kev9b` and `kev27b`. `clef` gets the
    main and format arms only (budget).
- **Confirmatory contrasts.** Jev − each new coder on error AUROC and coverage@5%: two-sided, pooled
  over the five tasks, using the section 12 bootstrap. The 10 tests form one family, Holm-adjusted;
  a claim also needs the pooled 95% CI to exclude 0.
- **Pre-specified secondary: same base model.**
  - clef_flash − qwen-verbal, clef_flash − qwen-logprob, kev9b − qwen-verbal and
    kev9b − qwen-logprob, on error AUROC. Does decision-model training add triage value over the
    same base model's own confidence?
  - clef_flash − kev9b, comparing full calibration training with an adapter plus temperature.
  - Caveat: our Qwen arm is the instruct model Qwen3.5-9B, with thinking off. Kev-9B is built on the
    base checkpoint, and Clef-flash's exact base variant is not documented.
- **Also reported.** RQ2 and RQ3 are computed exactly as for Jev. The A3 analyses stay exploratory.
- **Contamination check for Kev.** Kev's published training-data list is checked against our five
  datasets before results are read. Any overlap is reported, and that coder is flagged on the
  affected task.
- **Cost accounting.** Cloudflare and Perplexity are billed per token through the ledger. Kev's cost
  is the Modal GPU time (H100 at $3.95/h), reported from Modal's usage page, not per token.

## Amendment v1.3 (2026-10-11, before any probe data): memorization probe (substitute for RQ4's contamination test)

Approved by the author on 2026-10-11 and written before any probe call. Exploratory, but its rules are
fixed here.

- **Question.** Did the LLM coders see our public datasets (texts or labels) in training? Decision models
  are excluded, because they cannot generate text.
- **Models:** Luna, Haiku, Sonnet, gpt-oss-120b, Qwen3.5-9B and Qwen3.8-2.4T, at the same settings as
  their coding runs, with plain text output.
- **Completion items.** 100 per dataset, drawn with a fixed seed from the main sample:
  - MFRC: comments of 30 or more words;
  - AnnoMI: client and therapist utterances of 30 or more words;
  - TalkMoves: teacher and student sentences of 20 or more words;
  - Dunivin: passages of 30 or more words.
  - Each prompt holds the first half of the words (for AnnoMI and TalkMoves, the previous turn too).
    The target is the second half.
- **Two conditions per item (Golchin & Surdeanu's guided vs general instruction).** The guided prompt
  names the dataset and asks for the exact original continuation. The general prompt asks only to
  complete the text.
- **Completion scores.** Word-level ROUGE-L F1 against the true second half, and the longest common run
  of identical words. "Near-verbatim recall" is a run of at least 10 words that also covers at least
  half of the true continuation.
- **Label items.** 100 per confirmatory task, from the same sample. The model gets the item and the
  bare code labels, without definitions, and is asked which label(s) the dataset's annotators
  assigned. The guided condition names the dataset; the general condition asks which label fits. The
  score is agreement with the majority reference (exact match for exclusive tasks, Jaccard for MFRC).
- **Decision rule.** Evidence of contamination for a model × dataset if either:
  - guided ROUGE-L exceeds general ROUGE-L (one-sided paired Wilcoxon, Bonferroni over all model ×
    dataset cells, α = 0.05); or
  - two or more items show near-verbatim recall in the guided condition.

  Label leakage is flagged if guided accuracy exceeds general accuracy (one-sided exact McNemar test,
  Bonferroni). A null result means "no evidence of verbatim or label recall", not "never seen".
- **If contamination is found:** RQ1 accuracy is compared on recalled vs not-recalled items for that
  model and dataset.

## Amendment v1.4 (2026-10-11, before any data from the new prompt): LLM prompt sensitivity

Approved by the author on 2026-10-11 and written before any call with the new prompt. Secondary.

- **Question.** Is the LLM baselines' ranking result specific to our confidence-elicitation prompt?
- **Variant `dist_prompt`.** A verbalized distribution (Tian et al., 2023).
  - Exclusive tasks: the model gives an integer probability (0–100) for every option, summing to 100.
    The decision is the argmax; confidence is its normalized probability.
  - Multi-label tasks: the model gives a single probability (0–100) per code that the code applies.
    The decision is p ≥ 50, with no separate yes/no.
  - Everything else matches the canonical prompt: codebook text, option order, rationale field.
- **Coders and items.** Sonnet and Haiku (verbal), on the first 150 test segments per confirmatory
  task in the harness's fixed order (`prompt_robust`). The canonical answers for the same items come
  from the cache.
- **Contrasts.** dist_prompt − canonical on error AUROC (paired cluster bootstrap over test, 1,000
  replicates, two-sided), with κ and ECE reported alongside. Per task and pooled.
- **Interpretation.** A pooled CI excluding 0 means the LLM baseline's ranking is prompt-sensitive.
  The paper then reports both prompts and states which one the main comparisons used. The canonical
  prompt stays the primary analysis.

## Deviations log

1. **2026-10-10: bug fix, no analysis decision changed.** The response cache keyed requests with
   `sort_keys=True`, so a Choice question whose options differed only in order hashed to the same key
   as the canonical request. The RQ3 option-order arms (`reversed`, `shuffle1–3`) on the four exclusive
   tasks were therefore served the canonical Jev response and showed 0 flips. Keys are now
   order-sensitive. All 80,003 existing rows were re-keyed from their stored (order-preserving)
   requests, verified by every completed experiment estimating $0 afterwards. Only those 16 arms were
   re-run (15,702 Jev calls, ≈$0.71). The bug was found on reading the first RQ3 table, before any RQ3
   interpretation. Multi-label (Noul) order arms and all LLM arms were unaffected (their requests
   differed in content, not only in key order).
2. **2026-10-10: adjudication protocol (section 9), author's decision.** A full single-human pass over
   200 items was not feasible. Each blinded item now goes to a two-family LLM panel: Opus 5.5
   (`claude-opus-5-5`, adaptive thinking, effort medium) and OpenAI `gpt-6-sol` (reasoning effort
   medium). Both see exactly the human handout: instructions, the task's codebook, and the item row;
   no sources or confidences. The author adjudicates every item where the panel disagrees, plus a
   blind audit of 20 randomly chosen panel-agreed items, with 10% repeats in a later sitting.
   Final verdict: the author's where given, otherwise the panel's when both models agree, otherwise
   unresolved and excluded. Reported alongside: panel agreement rate, author–panel agreement on the
   audit, author self-agreement. Adjudicated results are labelled "panel + author". Caveat: the panel
   shares a model family with two of the evaluated coders (Opus with Haiku/Sonnet); the cross-family
   agreement rule and the human audit are the mitigations.
3. **2026-10-10: partial author adjudication, plus one exploratory addition.**
   - **Partial completion.** The author judged 48 of the 72 assigned items: the first 50 rows of a
     randomly shuffled workbook, 2 left blank. That covers 36 of 52 panel-disputed items and 12 of 20
     audit items. No second sitting was done, so author self-agreement is not available.
   - **Final verdicts.** 184 of 200 items are resolved (48 author, 136 panel); 16 panel-disputed items
     are unresolved and keep the human label (conservative).
   - **Audit.** Author–panel agreement is 10/12. The hidden controls' human label was upheld on 19/20.
   - **Exploratory addition (not in v1.0).** Head-to-head upheld rates on the sampled high-confidence
     disagreements (who was right when a coder disagreed with the human label), shown separately for
     author-only and panel-only verdicts as a check on panel bias.
   - **Reporting-code fix.** Missing labels from sources that did not code an item were counted as wrong
     in the adjudication summary; they are now excluded.
4. **2026-10-10: RQ5 code types not author-reviewed.** The code-type labels in `docs/code_types.csv`
   were assigned by the assistant from definition text alone. The assistant had already seen Jev's
   per-code Dunivin results and the calib confusion matrices. The author did not review the labels, so
   RQ5 type comparisons carry that caveat in addition to being exploratory. Implied codes only occur
   alongside non-implied codes in Dunivin, so the task-fixed-effects "implied" estimate is identified
   almost entirely from Dunivin's nine codes.

5. **2026-10-10: oaid collection note, no analysis decision changed.** All oaid arms (A1) were
   collected on 2026-10-10 with model `gpt-6-luna` (Decisions API public beta). The run took five passes
   because the account hit its 2M tokens/min limit (concurrency was lowered from 32 to 8) and later ran
   out of credits. Failed calls were never cached and every arm finished with zero failures. Four
   questions were refused, all in the TalkMoves-student `no_context` robustness arm; they are treated as
   missing, as A1 specifies. Observation: on multi-label tasks oaid gives each predicate exactly the same
   probability regardless of question order or batching. We checked this against the raw cached
   responses from separate calls, and it is a property of the endpoint, not of the cache.
6. **2026-10-11: Qwen model choice (A2), one exploratory arm, and a parse fix.**
   - **A2 model.** We probed every priced Qwen chat model on Together on 2026-10-10. Qwen3.6-Plus,
     3.7-Plus, 3.7-Max and 3.8-Flash need third-party data sharing, which stays disabled (author's
     decision). The older Qwens and Qwen3-Next are not serverless. That leaves `Qwen/Qwen3.5-9B` as the
     A2 model, run with thinking disabled (`reasoning.enabled=false`, 0 reasoning tokens) on the full
     sample. This follows A2's rule of choosing by availability.
   - **Exploratory addition (author-approved, not in v1.1).** `Qwen/Qwen3.8-2.4T-A95B` is the largest
     Qwen, has open weights and is hosted by Together without data sharing. It runs with thinking off,
     the same prompt, and both verbal and logprob signals, on the nested Sonnet subset (`llm_frontier`,
     270 segments per task). The purpose is to rule out "the logprob result only holds for a small
     model." The full sample was not affordable ($23 estimated). Results are labelled exploratory.
   - **Parse fix, made before any full-run data.** Together returns a null logprob for top-5
     alternatives that the JSON grammar masks. These are now read as probability 0; previously the
     resulting error discarded the whole response. This was found in a 2-item smoke test and no
     earlier run was affected: no other run's diagnostics contain an error.
7. **2026-10-11: A3 implementation (`src/margin/triage.py`, `margin a3`).** These choices were fixed
   after the oaid and qwen primary results had been seen, but before any A3 output had been computed
   for any coder. A3 therefore counts as pre-specified for oaid and qwen only at the level of the
   v1.1 text; its implementation details are post hoc for every coder.
   - **Recalibration** is fitted on calib with design weights.
     - **Methods:** one temperature per task (softmax over options; sigmoid per code, which leaves
       the 0.5 decisions unchanged), plus Platt and isotonic maps from decision confidence to
       P(correct).
     - **Reported:** test ECE, Brier of confidence, error AUROC, and the realized coverage and
       error of the fixed rules confidence ≥ 0.95 and ≥ 0.90.
   - **Risk control.** Three threshold rules, at eps in {0.02, 0.05, 0.10}, on the frozen split:
     - **Naive empirical** (the primary rule).
     - **Pointwise Clopper-Pearson** (v1.0).
     - **Learn-then-Test.** Fixed-sequence exact binomial tests, alpha = delta = 0.05, from the most
       confident tie block down. Blocks with fewer than 59 decisions are skipped, since no smaller
       set can pass. Errors are scaled to Kish's effective n because of the design weights.
     - **Validity check:** 500 random cluster re-splits per task and signal (30% of clusters to
       calib) estimate each rule's test violation rate. This is a check on the guarantee, which
       assumes exchangeable decisions; that assumption does not hold within clusters.
   - **Calibration budget.** Whole calib clusters are drawn until 50, 100, 200 or 400 segments are
     reached, 250 draws each, and scored on the frozen test set.
   - **Threshold transfer.** A threshold fitted on the other four tasks' calib data (each decision
     weighted equally), compared with the fixed rules 0.95 and 0.99.
   - **Per-code thresholds.** Codes with at least 50 calib decisions get their own threshold; codes
     below that use the task threshold. Exclusive tasks group by predicted code. A per-code Platt
     ranking, cross-fitted over 5 cluster folds of calib, is included.
   - **Cascades.**
     - **Stage thresholds:** fitted per stage on the calib decisions still open at that stage. Both
       the empirical and the LTT rule are reported.
     - **Agreement stage:** accepts where two coders give the same label, using the lower
       confidence.
     - **API cost:** the realized ledger cost per segment, charged once per segment that reaches
       a stage.
     - **Evaluation:** against the human reference and against the adjudicated reference. In the
       adjudicated case the human label is scored like a coder, so human-coded items can be wrong.
     - **Items:** the full-sample cascades use the items every pool coder answered; the subset
       cascades use the llm_frontier items.
     - **Contrasts:** paired cluster bootstrap with 1,000 replicates, thresholds refitted in every
       replicate, pooled as an equal-weight mean over tasks.
8. **2026-10-11: A3 revisions made after seeing the first A3 output.** Because of these, all A3
   results are labelled exploratory for every coder, including oaid and qwen.
   - **LTT start.** The sequence now starts at the first block with ≥ 124 decisions instead of 59.
     124 is the smallest set that a coder with 1% true error passes with 80% power at
     eps = delta = 0.05. The change was prompted by LTT stopping at a 74-decision block with one error
     (Jev, MFRC). The start uses counts only, so the guarantee is unaffected; the simulation test
     still passes.
   - **Oracle rule.** Thresholds fitted on test, as a risk-matched upper bound. This was added after
     naive thresholds turned out to overshoot 5% by different amounts for different coders, so
     their coverages were not comparable.
   - **Per-code head-to-head.** The frontier subset's calib sets (~80 segments) are too small for
     per-code maps on exclusive tasks. Jev's per-code maps now come from its full calib set and are
     compared with the frontier coders' raw confidence on the same test items.
   - **Per-code calibrated ranking.** Now also reported with LTT and oracle thresholds, plus a check
     over 250 cluster re-splits per task and signal.
   - **Cascade contrasts.** Now run under the empirical, LTT and oracle rules. Three agreement-gate
     contrasts were added after the agreement gate's frozen-split results had been seen.
9. **2026-10-11: A4 resolved. Sonnet runs on the full sample (author's decision).** Sonnet 5.5 (same
   prompt and settings as `llm_frontier`) is now run on the full shared sample (`sonnet_main`); the
   nested subset is reused from the cache. The author decided this after seeing the subset results and
   the A3 output, so it is not blind. The design is unchanged and only the sample grows. Reporting:
   - The v1.0 pre-specified subset results stay as they were (snapshot in `results/pre_sonnet_full/`).
   - The full-sample Sonnet results are reported alongside them as an extension.
   - Every analysis that uses `sonnet-verbal` (RQ1 contrasts, RQ2, RQ5, Dunivin replication, A3)
     now uses the full sample.
   - A3 also gains full-sample Sonnet cascades and contrasts, which are exploratory.
   - **Interpretation note.** Per-code recalibration needs each code's map fitted on enough calib
     decisions (≥ 50). The earlier finding that the frontier coders gain nothing from per-code maps
     came from the subset's ~80-segment calib sets, where the maps fall back to one task-level map.
     On the full sample Sonnet gains +0.025 [0.019, 0.033]. The `qwen_big` per-code results stay
     uninterpretable for the same reason.
10. **2026-10-11: pre-write-up additions (exploratory, author-approved).**
    - **Repeatability.** Experiment `repeat1`: 100 test segments per task, identical canonical
      requests re-sent fresh through a cache tag, for Jev, oaid, Luna, Haiku and Sonnet. Run-to-run
      flips and drift are reported.
    - **Sonnet RQ3 arms.** Experiment `sonnet_robust`: reversed, shuffle1 and neutral_labels on the
      same 85 test segments per task as Luna's `llm_robust`.
    - **Holm sensitivity analysis.** Holm adjustment within each pre-registered contrast family,
      using two-sided bootstrap p-values from the existing replicates. The plan's decision rule is
      unchanged; Holm is reported alongside it. One claim changes: Jev vs Luna verbal on error AUROC
      becomes "no claim" (Holm p = 0.20).
    - **Paper figures and tables** come from `margin report`.
11. **2026-10-11: Kev-27B scope (author's decision, before any Kev-27B data).**
    - **Arms.** Kev-27B runs the primary and yes/no-format arms only. Its RQ3 robustness arms are
      dropped: they would add GPU cost and complexity, and the 27B tier is already covered by Clef and
      pplx. It stays in the v1.2 confirmatory family (Jev − Kev-27B on the primary metrics).
    - **Kev deployment as served.** Kev-9B: Qwen3.5-9B-Base plus a rank-16 LoRA, temperature 2.19.
      Kev-27B: full fine-tune of Qwen3.8-27B, temperature 1.32. Both run in bf16 on Modal, as
      reported by `/v1/models` (saved in `data/runs/warm_kev*.json`).
    - **Contamination check (A5) passed.** None of MFRC, AnnoMI, TalkMoves or Dunivin appears in the
      Kev model cards' training-data lists, or anywhere in the Kev repository.
12. **2026-10-11: Kev-9B RQ3 coverage gap (author's decision: accept).** The Modal workspace hosting
    Kev was disabled partway through the Kev-9B robustness arms, when its free credits ran out.
    Autoscaling to two GPUs per app had doubled the burn rate; the app is now capped at one
    container.
    - **Complete for Kev-9B:** the primary and format arms, and all RQ3 arms on MFRC, AnnoMI client
      and AnnoMI therapist.
    - **Missing:** the RQ3 arms on TalkMoves teacher and student, about 13.5k calls. The partial
      files were removed, so RQ3 for Kev-9B covers 3 of 5 tasks. The responses already cached are
      kept, and the arms can be completed later at no cost to what exists.
13. **2026-10-11: exploratory additions after the v1.2 decision models, made before computing them.**
    - **Decision-model committee (`dm_committee`).** The mean probability over all seven decision
      models (Jev, oaid, Clef, Clef-flash, pplx, Kev-9B, Kev-27B), on units every member answered.
      Membership is fixed as "all" before any committee result, so no member is chosen by its
      result.
    - **New cascades:**
      - single stages: Kev-27B → human, pplx → human, Clef → human, committee → human;
      - agreement gates: Jev + Kev-27B, pplx + Kev-27B;
      - Kev-27B → Sonnet → human.
    - **Repeatability** for Clef, Clef-flash and pplx (100 test segments per task, `repeat1`). Kev
      is excluded because its endpoint is offline.
    - All of these are exploratory.
14. **2026-10-11: decoy-name control for the flagged label-probe cells (exploratory, logged before
    running).**
    - **Why.** Three cells met the v1.3 label-leakage rule: Haiku on MFRC, Luna on MFRC, and gpt-oss on
      AnnoMI therapist. But the guided prompt both names the dataset and describes its annotation
      scheme, so a gain can come from task framing rather than memory.
    - **Control.** For the same items, a decoy prompt gives the identical description with a fictitious
      name and citation: "the Hartwell Online Morality Corpus (HOMC; Hartwell et al., 2021)" for MFRC,
      and "the Brennan Counselling Dialogue Set (BCDS; Brennan et al., 2021)" for AnnoMI.
    - **Rule.** Memory is supported only if the real name beats the decoy (one-sided paired test,
      Bonferroni over the three cells, α = 0.05).
15. **2026-10-11: deviation 4 resolved.** The author reviewed the code-type labels in
    `docs/code_types.csv` and approved them unchanged. RQ5 type comparisons therefore no longer carry
    the "not author-reviewed" caveat. The review came after the RQ5 results had been computed, but
    no label was changed.
16. **2026-10-14: changes after an external review of the first draft (author-approved).** All of
    these were made after every result in the draft had been seen, so everything they touch is
    exploratory.
    - **Status of this plan.** It was frozen and amended in this repository before the data each part
      governs, but it was not registered with an external service, so its timing cannot be verified
      independently. The paper calls it a prospectively specified analysis plan, not a pre-registration.
    - **Cluster-level LTT replaces decision-level LTT as the guaranteed rule.** The A3 LTT ran exact
      binomial tests on Kish-effective decision counts, which assumes independent decisions. Decisions
      share comments (MFRC, 8 codes each), transcripts (AnnoMI) and lessons (TalkMoves), and a
      clustered simulation shows the decision-level test certifies unsafe thresholds in up to 34% of
      runs at delta = 0.05. The new rule treats clusters as the sampling unit: for each threshold on
      a fixed grid (16 values, 0.5 to 1.0), the per-cluster weighted loss
      Z = sum w * accepted * (error - eps) is rescaled by B, the largest cluster's total weight in the
      task sample (a design constant), and tested with the Waudby-Smith & Ramdas betting p-value;
      Bonferroni over the grid; the lowest certified threshold is used. It is valid when clusters are
      exchangeable, whatever the dependence within them. The decision-level rule is kept as
      `ltt_dec` for comparison. Multi-stage cascades are certified jointly over the pipeline (8
      candidates per stage, later stages may be switched off; Bonferroni over all combinations).
      Per-code maps with LTT are fitted on half of the calib clusters and tested on the other half.
    - **RQ3 weighting.** Point estimates (flip rate, changes in kappa, ECE, error AUROC, AURC) are now
      design-weighted, as the bootstrap intervals already were.
    - **Added (exploratory):** MFRC prevalence analysis (share of accepted decisions that are
      "absent" calls, present-code recall, a prevalence baseline); a category-only control for
      per-code recalibration; adjudication results split by panel and author verdicts; calibration
      clusters needed to certify even a perfect coder; a text-free spend ledger.
    - **Reporting fixes:** cascade costs are reported per threshold policy; the cascade table's error
      column had been blank; main-text kappa is design-weighted.
17. **2026-10-14: changes after a second external review (author-approved; all exploratory).**
    - **Corpus LTT replaces the deviation-16 cluster rule as the guaranteed rule.** The betting test of
      deviation 16 is valid for independent clusters, not for merely exchangeable ones, and it used the
      largest observed cluster as if it bounded every future cluster. The new rule targets the uncoded
      remainder of the corpus: with clusters hand-coded as a uniformly random subset (our hash split), a
      without-replacement betting test (Waudby-Smith & Ramdas 2024) bounds the remainder's selective risk,
      with the corpus's largest cluster weight B as a known constant and no assumption about dependence. The
      deviation-16 test is kept as "LTT on i.i.d. clusters" for the population target. B is now computed
      from the sample design (all sampled segments) rather than from one coder's scored decisions.
    - **Learned scores are certified honestly.** Per-code maps and the prevalence baseline are fitted on half
      of the calib clusters, frozen, certified on the other half, and applied unchanged to test (the
      per-code route had certified one map and deployed a refit; the baseline had been fitted and certified
      on the same labels).
    - **Inference:** pooled bootstrap intervals now resample each corpus once per replicate, shared by
      related tasks (AnnoMI client and therapist share transcripts; TalkMoves teacher and student share
      lessons). The pre-specified Sonnet subset contrast is recomputed with a p-value and used in the v1.0
      family; the full-sample run is reported as an extension outside the family.
    - **Fixes:** probability floor for log/logit transforms 1e-4 -> 1e-12 (the larger floor erased
      distinctions among near-certain token probabilities); the share of decisions at confidence >= 0.99
      is design-weighted; the ledger route counts each request key once and fails on missing keys; the
      cascade table quotes the right human-only baseline; clusters enter the betting test in a seeded
      hash order of their ids (the order had followed data loading, which moved Kev-27B's certified MFRC
      share between 82.2% and 76.6%). With the smaller floor, temperature scaling no longer calibrates Jev,
      so Platt maps are the recalibration yardstick in EQ1; per-code gains for Jev and OpenAI Decisions,
      which return exact 0/1, depend on the floor (Jev +0.040 at 1e-12, +0.053 at 1e-4).
    - **Wording:** feasibility numbers describe this procedure, not every guarantee; the simulation's
      correlation parameter is a copula loading.
18. **2026-10-15: changes after a third external review (all exploratory).**
    - **Visiting order.** The without-replacement test needs the hand-coded clusters to be visited in a
      uniformly random order, not only to be a random set. Deviation 17 visited them in a fixed hash order
      of their ids, and a corpus whose errors line up with that order was certified wrongly in 200 of 200
      draws (reproduced; now a test). The betting tests (corpus LTT and LTT on i.i.d. clusters, in the
      frozen analyses, re-splits, budget curves and cascades) now average the final capital over 100 random
      visiting orders from a recorded seed (ORDER_SEED = 20261015, set before the rerun and not chosen by
      any outcome). Each order's final capital is an e-value, an average of e-values is an e-value (Vovk &
      Wang 2021), and p = 1 / average is valid by Markov's inequality; the average rises with the null
      mean, so the data-dependent m* argument is unchanged.
    - **Budget curves.** The corpus for a budget draw is now the drawn calib clusters plus the test
      clusters; it had been the full original corpus, whose remainder included unused calib clusters
      that the evaluation never looked at.
    - **Bootstrap streams.** `joint_draws` now gives each corpus its own random stream (it had restarted
      the same seed for every corpus), and the prompt-sensitivity intervals use the shared corpus draws.
      All bootstrap intervals were regenerated.
    - **Feasibility.** An ordering-independent bound (the fewest units that could suffice for any
      hand-coded set and order) replaces the single random path as the basis for impossibility claims; the
      procedure's median over 20 random sequences is reported as a description. A second sampling design
      for the dialogue tasks (single utterances or sentences drawn at random, B = largest segment weight)
      is added, with a re-split diagnostic for the real coders.
    - **Reporting.** Prose intervals are generated from the results files (paper/numbers.tex); table
      cross-references are generated; a whole-corpus reading workload is reported next to the remainder;
      wide appendix tables are split into panels.
    - **After a fourth review (same day; wording, reporting and one diagnostic).** The claims table had
      decided "CI excludes 0" from the printed, rounded interval; it now uses the unrounded endpoints, and
      an endpoint that would print as 0.000 gets one more decimal (no claim changed: the only such endpoint
      is Jev − Luna verbal, lower bound 0.00036, which is no claim after Holm). Policy figures quoted in the
      prose (certified shares, costs, reading workload, single-threshold coverage, utterance-design
      certificate counts) are generated into paper/numbers.tex. `controls.utterance_evidence` records, for
      the first utterance-design re-split, the largest grid threshold whose hand-coded accepted set has
      error <= 5% (results/utterance_evidence.csv), which supports the evidence-size explanation.
    - **After the fifth and sixth reviews (wording and release only; no result changed).** The decoy
      control's item-level scores (300 rows, text-free) were rebuilt from the local response cache without
      any API call and released as results/contamination_decoy_items.csv; `margin decoy` recomputes the
      summary from them, identically. Rates that are nonzero but below display precision print as <0.001.
      The paper's prompt example now uses an invented exchange instead of AnnoMI text, and the release
      statements say that raw response text is withheld because it quotes the datasets.
