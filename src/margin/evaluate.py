"""Run directory -> per (task, coder, variant, reference) summary of agreement, calibration, triage,
and disagreement. reference = "human" (majority label) or "adjudicated" (blinded verdicts swapped in)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from margin import sampling
from margin.datasets.base import Task
from margin.metrics.agreement import kappa, krippendorff_nominal, macro_f1
from margin.metrics.calibration import brier, ece
from margin.metrics.core import apply_adjudication, decisions, reference
from margin.metrics.disagreement import leave_one_coder_out, uncertainty_vs_humans
from margin.metrics.selective import aurc, coverage_at_risk, error_auroc
from margin.run import RUNS, Experiment
from margin.schema import Codebook

RESULTS = Path("results")


def human_alpha(labels: pd.DataFrame, codebook: Codebook) -> float:
    """Human-human reliability on multiply-coded items (macro over codes for multi)."""
    multi = labels.groupby("segment_id").size()
    lab = labels[labels.segment_id.isin(multi[multi >= 2].index)]
    if lab.empty:
        return np.nan
    if codebook.mode == "exclusive":
        return krippendorff_nominal(lab, codebook)
    return float(np.nanmean([krippendorff_nominal(lab, codebook, c) for c in codebook.code_ids]))


def decision_metrics(dec: pd.DataFrame, codebook: Codebook) -> dict:
    """Metrics that only need the decision frame, so they work for any reference. With both splits
    present (the confirmatory case), calib only fits thresholds and every number is computed on
    test; otherwise everything is in-sample and flagged."""
    calib, test = dec[dec.split == "calib"], dec[dec.split == "test"]
    in_sample = calib.empty or test.empty
    fit, apply = (dec, dec) if in_sample else (calib, test)
    tri = coverage_at_risk(fit, apply)
    dec = apply
    top = dec.conf >= 0.995  # Jev rounds probabilities to 0.01, so this is its single top tie block
    return {
        "decisions": len(dec),
        "kappa": kappa(dec, codebook),
        "kappa_w": kappa(dec, codebook, weighted=True),
        "macro_f1": macro_f1(dec, codebook),
        "accuracy_w": float(np.average(dec.correct, weights=dec.weight)),
        "ece": ece(dec.conf, dec.correct, dec.weight),
        "error_auroc": error_auroc(dec.conf, dec.correct, dec.weight),
        "aurc": aurc(dec.conf, dec.correct, dec.weight),
        "cov@5%": tri.test_coverage,
        "risk@5%": tri.test_risk,
        "cov@5%_ucb": coverage_at_risk(fit, apply, method="ucb").test_coverage,
        "cov@10%": coverage_at_risk(fit, apply, eps=0.10).test_coverage,
        "share_conf_1": float(top.mean()),
        "err_conf_1": float(1 - dec.correct[top].mean()) if top.any() else np.nan,
        "triage_in_sample": in_sample,
    }


def summarize_unit(
    preds: pd.DataFrame, task: Task, codebook: Codebook, sample: pd.DataFrame, verdicts: pd.DataFrame | None = None
) -> list[dict]:
    ids = set(preds.segment_id)
    labels = task.labels[task.labels.segment_id.isin(ids)]
    ref = reference(labels, codebook)
    dec = decisions(preds, ref, codebook, sample).merge(sample[["segment_id", "split"]], on="segment_id")
    # Context metrics follow the same rule as decision_metrics: test only when both splits exist.
    splits = set(dec.split)
    scope = set(sample[sample.split == "test"].segment_id) & ids if splits == {"calib", "test"} else ids
    labels, preds = labels[labels.segment_id.isin(scope)], preds[preds.segment_id.isin(scope)]

    human = {"reference": "human", "segments": len(scope)} | decision_metrics(dec, codebook)
    human |= {"brier": brier(preds, reference(labels, codebook), codebook), "human_alpha": human_alpha(labels, codebook)}
    per_seg = labels.groupby("segment_id").size()
    if (per_seg >= 3).sum() >= 30:
        u = uncertainty_vs_humans(preds, labels, codebook)
        human |= {"rq2_n": u["n"], "rq2_spearman": u["spearman"], "rq2_auroc_split": u["auroc_split"]}
        loo = leave_one_coder_out(preds[preds.segment_id.isin(per_seg[per_seg >= 3].index)], labels, codebook)
        human |= {
            "loo_annotators": len(loo),
            "loo_mean_model_minus_human": float(loo.model_minus_human.mean()),
            "loo_model_beats_human": int((loo.model_minus_human > 0).sum()),
        }
    rows = [human]
    if verdicts is not None:
        v = verdicts[verdicts.task == task.name]
        adj = apply_adjudication(dec, v, codebook)
        rows.append({"reference": "adjudicated", "segments": len(ids), "adjudicated_items": len(v)} | decision_metrics(adj, codebook))
    return rows


def evaluate(experiment: str, adjudication: str | None = None) -> pd.DataFrame:
    from margin.adjudicate import load_verdicts

    exp = Experiment.load(experiment)
    verdicts = load_verdicts(adjudication) if adjudication else None
    rows = []
    for task_name in exp.tasks:
        codebook = Codebook.load(f"codebooks/{task_name}.yaml")
        task = Task.load(task_name)
        sample = sampling.load(task_name, exp.sample)
        for path in sorted((RUNS / exp.name / task_name).glob("*__*.parquet")):
            if path.name.endswith(".diag.parquet"):
                continue
            coder, variant = path.stem.split("__")
            preds = pd.read_parquet(path)
            if preds.empty:
                continue
            for r in summarize_unit(preds, task, codebook, sample, verdicts):
                rows.append({"task": task_name, "coder": coder, "variant": variant} | r)
    table = pd.DataFrame(rows)
    out = RESULTS / exp.name
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / ("summary.csv" if adjudication is None else f"summary_adj_{adjudication}.csv"), index=False)
    return table
