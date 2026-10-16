# Paper outline (draft, 2026-10-11)

Working title: **"Can model confidence triage qualitative coding? Decision models, LLMs, and what a
5% error guarantee actually costs"**

Snapshot statement, for the abstract and §3: all models were called 10–12 October 2026, several through
public-beta APIs. Table 18 lists every model's exact version string and call dates. The released
harness and response cache allow re-running against later versions.

## Framing

The question is not "is Jev good" but **what it takes to let a model's confidence decide which
qualitative-coding decisions a human must check**. Seven decision models (Jev, OpenAI Decisions,
Clef, Clef-flash, pplx-decider, Kev-9B, Kev-27B) and LLM confidence signals (verbalized, logprob,
agreement, self-consistency; cheap and frontier) are the cases.

## Question structure

Each research question (RQ) has a confirmatory answer, pre-registered with Holm adjustment within each
family. Each experimental question (EQ) is a targeted experiment that explains or stress-tests an RQ
result; its status is stated next to it. Everything else goes to the exploratory appendix.

### Research questions (main text)

- **RQ1. Triage.** Do decision models' probabilities rank their own coding errors better than LLM
  confidence signals, and what share of decisions can be auto-accepted at ≤ 5% error?
  (Tables 3–4, Figs 1–2, 13; amendments v1.0, v1.1 A1, v1.2 A5.)
- **RQ2. Disagreement.** Does model uncertainty sit where human coders disagree, and do models fall
  within the human inter-annotator range? (Table 5, Fig 8.)
- **RQ3. Robustness.** Are decisions and confidence stable under surface changes to the codebook and
  request (order, labels, paraphrase, context, batching, question format)?
  (Table 6, Fig 9, format contrast.)

### Experimental questions (main text, after the RQs)

- **EQ1. Is calibration error the right yardstick?** Recalibration and the 0.95 rule (Fig 3, Table 7).
  Exploratory (A3).
- **EQ2. Do calibrate-then-threshold rules keep their promise, and what does a guarantee cost?**
  Naive vs Clopper-Pearson vs LTT over cluster re-splits, plus the calibration budget (Figs 5–6,
  Table 7). Exploratory (A3).
- **EQ3. What limits triage: ranking or saturation?** Top-confidence tie blocks and per-code
  recalibration (Figs 4, 10; Tables 10, 13). Exploratory.
- **EQ4. Does decision-model training add value over the same base model?** Qwen3.5-9B vs Clef-flash
  vs Kev-9B (Table 14). Pre-specified secondary (v1.2).
- **EQ5. What is the best automation per dollar?** Single coders, cascades and agreement gates
  (Fig 7, Tables 8–9). Exploratory.
- **EQ6. Are the results artefacts?** Validity checks:
  - repeatability (Fig 11, Table 11);
  - LLM prompt sensitivity (Table 19; v1.4, secondary);
  - memorization probe with decoy control (Tables 15–17, Fig 14; v1.3);
  - adjudicated reference;
  - Dunivin replication.

### Exploratory appendix

- question-format arms per decision model;
- the decision-model committee;
- A2: Qwen non-reasoning logprobs;
- RQ5: per-code agreement by code type (Fig 12);
- who was right in adjudicated disagreements (Table 12);
- Kev GPU cost estimates;
- the full deviations log.

## Contributions

1. **The first multi-vendor evaluation of decision models for deductive qualitative coding.** Seven
   decision models and nine LLM signals on five tasks (multi-label and exclusive; Reddit, counselling
   dialogue, classroom talk), with per-annotator labels, an adjudicated reference, pre-registration
   and a deviations log.
2. **"Decision model" is not one thing.**
   - Jev ranks its errors better than three decision models, ties pplx-decider, and loses to the open
     Kev-27B.
   - Calibration training buys low ECE but not ranking.
   - On a shared base model, decision-model training beats verbalized confidence (+0.05 to +0.065),
     barely beats raw logprobs on ranking, and greatly improves usable coverage.
3. **Calibration error is the wrong yardstick; ranking and top-block resolution decide triage.**
   - One temperature brings every signal's ECE to about 0.04.
   - Top-confidence tie blocks cap triage, and per-code recalibration breaks them.
4. **The calibrate-then-threshold workflow overshoots its target.** Certified (LTT) triage on dialogue
   codebooks needs far more than the ~400 hand-coded segments researchers typically have.
5. **A cost-aware recipe.**
   - Kev-27B, or a pplx + Kev-27B agreement gate, certifies more automation than Sonnet at about 1/6 of
     the cost.
   - pplx-decider alone costs $0.02 per 1,000 segments.

## Claims by question (with status)

**RQ1. Confirmatory (Holm within each family; Table 4)**
- **v1.0:** Jev's confidence ranks errors better than every logprob signal (Luna, gpt-oss: +0.20 AUROC),
  3-LLM agreement (+0.09), self-consistency (+0.15) and gpt-oss verbal (+0.03). It ranks worse than
  Haiku verbal (−0.026) and Sonnet (−0.045; full sample, also coverage@5%). Jev vs Luna verbal: no
  claim (Holm p = 0.20).
- **v1.1:** Jev − OpenAI Decisions on error AUROC: +0.020 [0.004, 0.037] (Holm p = 0.028).
  Coverage@5%: no claim.
- **v1.2:** Jev ranks better than Clef (+0.036), Clef-flash (+0.045) and Kev-9B (+0.062). It ties
  pplx-decider on ranking, with pplx covering more (−0.079). Kev-27B beats Jev on both metrics (−0.032
  AUROC, −0.113 coverage) and is the best decision model (AUROC 0.788 vs Sonnet's 0.801).

**RQ2. Secondary:** uncertainty tracks human splits for the good coders.
- pplx-decider, the committee and Sonnet track best on AnnoMI client; Jev tracks best on AnnoMI
  therapist.
- Models fall inside the human κ range (Table 5, Fig 8).

**RQ3. Secondary.**
- **Option order:** flips 4% of Jev labels, 9% of OpenAI Decisions, about 7% of pplx-decider, about
  10% of Luna and 5–6% of Sonnet, but under 1% of Clef-flash and 12–29% of Kev-9B.
- **OpenAI Decisions** is exactly invariant on multi-label tasks.
- **Format effects differ by model:** yes/no per option helps Clef-flash and pplx-decider, hurts
  Kev-27B, and leaves Jev and OpenAI Decisions unchanged (Table 6, Fig 9).

**EQ1–EQ5.** See the experimental questions above. Headline numbers:
- Naive thresholds break 5% in 26–45% of re-splits.
- LTT certifies about 0% on dialogue for every coder.
- Per-code recalibration: Jev +0.053 [0.041, 0.064]; Jev-per-code vs Sonnet-per-code −0.017
  [−0.029, −0.006], a tie on the four dialogue tasks.
- Kev-27B → human certifies 26.7% vs Sonnet's 22.4%.
- A pplx + Kev-27B agreement gate certifies 27.6% and matches Sonnet's oracle automation (40.5% vs
  40.1%).

**EQ6: validity checks.**
- **Repeatability:** OpenAI Decisions, Clef, Clef-flash and pplx are fully deterministic; Jev changes
  2.1% of labels, Sonnet 3.3%, and Luna 6.6% despite temperature 0.
- **Memorization:** no verbatim recall of any dataset. One label-familiarity cell (Luna on MFRC)
  survives the decoy control, with no effect on coding accuracy.
- **Prompt sensitivity (v1.4):** a verbalized-distribution prompt does not improve the LLM baselines'
  ranking. Sonnet −0.034 [−0.056, −0.011], Haiku −0.010 (n.s.), κ unchanged. The canonical prompt is
  not a weak baseline.

## Section plan

1. Introduction: the triage promise, and why calibration error is the wrong yardstick.
2. Related work:
   - LLM qualitative coding (Dunivin; RAND 2512.00009; 2512.21041; Agreement Is Not Quality
     2607.28890);
   - decision models (Jev; OpenAI Decisions; Clef; pplx-decider; Kev; Denney & DiGiuseppe 2610.06625;
     Li et al. 2609.26550);
   - selective prediction and risk control (LTT, conformal).
3. Data, codebooks and coders (Tables 1–2, 18). Design: frozen plan and amendments, splits,
   clustering, weights, metrics, adjudication.
4. RQ1–RQ3 results.
5. EQ1–EQ6.
6. Discussion: practical recipe; cost; when to prefer a frontier LLM or an open decision model.
7. Limitations and follow-ups:
   - public, short-segment data only, with no long-form interviews (follow-up: QDR projects or private
     data under IRB);
   - beta model versions dated October 2026;
   - partial human adjudication;
   - LTT exchangeability within clusters;
   - the Sonnet extension decided after seeing subset results;
   - the Kev-9B RQ3 gap on TalkMoves;
   - decision models not probed for memorization.
8. Appendix: exploratory analyses and the full deviations log.
