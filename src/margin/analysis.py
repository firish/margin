"""Confirmatory analyses from the frozen plan (docs/ANALYSIS_PLAN.md): RQ1 signals and contrasts,
the question-format contrast, RQ2, and RQ3. Every number is computed on the test split; calib only
fits thresholds. Bootstrap intervals resample whole clusters within calib and within test separately
and refit thresholds in each replicate, so threshold uncertainty is included.
"""

from __future__ import annotations

import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from margin import sampling
from margin.datasets.base import Task
from margin.evaluate import decision_metrics
from margin.metrics.core import apply_adjudication, decisions, reference
from margin.metrics.disagreement import leave_one_coder_out, uncertainty_vs_humans
from margin.metrics.robustness import compare
from margin.run import RUNS
from margin.schema import Codebook

RESULTS = Path("results")
CONFIRMATORY = ["mfrc", "annomi_client", "annomi_therapist", "talkmoves_teacher", "talkmoves_student"]
SIGNALS = {
    "jev": ("jev_main", "jev__canonical"),
    "jev_noul": ("jev_format", "jev__noul_format"),
    "oaid": ("oaid_main", "oaid__canonical"),  # amendment v1.1 (A1)
    "oaid_noul": ("oaid_format", "oaid__noul_format"),
    "luna-verbal": ("llm_main", "luna-verbal__canonical"),
    "haiku-verbal": ("llm_main", "haiku-verbal__canonical"),
    "oss-verbal": ("llm_main", "oss-verbal__canonical"),
    "luna-logprob": ("llm_main", "luna-logprob__canonical"),
    "oss-logprob": ("llm_main", "oss-logprob__canonical"),
    "agree3": ("llm_main", "agree3__canonical"),
    "luna_sc-vote": ("llm_sc", "luna_sc-vote__canonical"),
    "sonnet-verbal": ("sonnet_main", "sonnet-verbal__canonical"),  # A4, full sample since 2026-10-11 (v1.0 subset: results/pre_sonnet_full/)
    "qwen-verbal": ("qwen_main", "qwen-verbal__canonical"),  # amendment v1.1 (A2)
    "qwen-logprob": ("qwen_main", "qwen-logprob__canonical"),
    "qwen_big-verbal": ("qwen_big_frontier", "qwen_big-verbal__canonical"),  # deviation 6 (exploratory)
    "qwen_big-logprob": ("qwen_big_frontier", "qwen_big-logprob__canonical"),
    # Amendment v1.2 (A5): other System One decision models.
    **{dm: ("dm_main", f"{dm}__canonical") for dm in ("clef", "clef_flash", "pplx", "kev9b", "kev27b")},
    **{f"{dm}_noul": ("dm_format", f"{dm}__noul_format") for dm in ("clef", "clef_flash", "pplx", "kev9b", "kev27b")},
}
SIGNALS["dm_committee"] = ("dm_main", "dm_committee__canonical")  # deviation 13 (exploratory)
SIGNALS["sonnet_sub-verbal"] = ("llm_frontier", "sonnet-verbal__canonical")  # v1.0 as specified: Sonnet on the nested subset
NEW_DMS = ["clef", "clef_flash", "pplx", "kev9b", "kev27b"]
SAME_BASE = [("clef_flash", "qwen-verbal"), ("clef_flash", "qwen-logprob"), ("kev9b", "qwen-verbal"), ("kev9b", "qwen-logprob"),
             ("clef_flash", "kev9b")]
"""v1.2 pre-specified secondary: decision-model training vs the same base model (Qwen3.5-9B)."""
COMPARATORS = [s for s in SIGNALS if s != "jev" and not s.endswith("_noul")]
"""Jev minus each of these; includes oaid (the v1.1 confirmatory contrast)."""
OAID_COMPARATORS = [s for s in COMPARATORS if s != "oaid"]
"""oaid minus each LLM signal (v1.1 secondary)."""
REPS, SEED = 1000, 0
CORPUS = {"mfrc": "mfrc", "annomi_client": "annomi", "annomi_therapist": "annomi",
          "talkmoves_teacher": "talkmoves", "talkmoves_student": "talkmoves", "dunivin": "dunivin"}
"""Tasks coded on the same corpus share clusters: AnnoMI's two roles share transcripts, TalkMoves' two share lessons."""


@lru_cache(maxsize=None)
def joint_draws(task: str, reps: int, seed: int = SEED) -> dict[str, list[np.ndarray]]:
    """Bootstrap draws of cluster ids, shared by every task coded on the same corpus, separately for calib
    and test clusters. Resampling the corpus once per replicate keeps the dependence between related tasks
    in pooled intervals (an equal-weight mean of per-task replicate statistics)."""
    corpus = CORPUS.get(task, task)
    members = [t for t, c in CORPUS.items() if c == corpus] or [task]
    groups = pd.concat([sampling.load(t, "main")[["group", "split"]] for t in members]).drop_duplicates("group")
    rng = np.random.default_rng([seed, zlib.crc32(corpus.encode())])  # one stream per corpus, independent across corpora
    out = {}
    for s, label in (("c", "calib"), ("t", "test")):
        ids = np.sort(groups[groups.split == label].group.astype(str).unique())
        out[s] = [ids[rng.integers(0, len(ids), len(ids))] for _ in range(reps)]
    return out


def resample(members: dict[str, np.ndarray], drawn: np.ndarray) -> np.ndarray:
    """Row indices for one bootstrap replicate: each drawn cluster's rows, as often as it was drawn."""
    picked = [members[g] for g in drawn if g in members]
    return np.concatenate(picked) if picked else np.array([], int)


@dataclass
class TaskData:
    name: str
    codebook: Codebook
    task: Task
    sample: pd.DataFrame
    ref: pd.DataFrame
    verdicts: pd.DataFrame | None = None
    """Adjudicated verdicts for this task (secondary analysis); None = human reference."""

    @classmethod
    def load(cls, name: str, adjudication: str | None = None) -> TaskData:
        cb, task = Codebook.load(f"codebooks/{name}.yaml"), Task.load(name)
        sample = sampling.load(name, "main")
        ref = reference(task.labels[task.labels.segment_id.isin(sample.segment_id)], cb)
        verdicts = None
        if adjudication:
            from margin.adjudicate import load_verdicts

            v = load_verdicts(adjudication)
            verdicts = v[(v.task == name)].copy()
            if cb.mode == "exclusive":
                verdicts = verdicts.drop(columns=["code_id"], errors="ignore")
        return cls(name, cb, task, sample, ref, verdicts)

    @property
    def key(self) -> list[str]:
        return ["segment_id"] + (["code_id"] if self.codebook.mode == "multi" else [])

    def preds(self, signal: str) -> pd.DataFrame | None:
        exp, stem = SIGNALS[signal]
        path = RUNS / exp / self.name / f"{stem}.parquet"
        return pd.read_parquet(path) if path.exists() else None

    def decisions(self, preds: pd.DataFrame) -> pd.DataFrame:
        d = decisions(preds, self.ref, self.codebook, self.sample)
        if self.verdicts is not None:
            d = apply_adjudication(d, self.verdicts, self.codebook)
        return d.merge(self.sample[["segment_id", "split"]], on="segment_id")


# --- fast metrics on arrays (same semantics as metrics.selective) ---------------------------------

def _rc(conf: np.ndarray, err: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(thresholds, coverage, risk) at the end of each confidence tie block, most confident first."""
    order = np.argsort(-conf, kind="stable")
    c, e, ww = conf[order], err[order], w[order]
    cw, ce = np.cumsum(ww), np.cumsum(ww * e)
    last = np.r_[c[1:] != c[:-1], True]
    return c[last], cw[last] / ww.sum(), ce[last] / cw[last]


def fast_aurc(conf, correct, w) -> float:
    _, cov, risk = _rc(conf, 1 - correct, w)
    return float((np.diff(np.r_[0.0, cov]) * risk).sum())


def fast_coverage(cc, kc, wc, ct, kt, wt, eps: float) -> tuple[float, float]:
    """Fit the lowest threshold with calib risk <= eps; return (test coverage, test risk)."""
    thr, _, risk = _rc(cc, 1 - kc, wc)
    ok = np.flatnonzero(risk <= eps)
    if ok.size == 0:
        return 0.0, float("nan")
    acc = ct >= thr[ok[-1]]
    if not acc.any():
        return 0.0, float("nan")
    return float(wt[acc].sum() / wt.sum()), float((wt[acc] * (1 - kt[acc])).sum() / wt[acc].sum())


def fast_auroc(conf, correct, w) -> float:
    if correct.min() == correct.max():
        return float("nan")
    return float(roc_auc_score(correct, conf, sample_weight=w))


def triage(cc, kc, wc, ct, kt, wt) -> dict[str, float]:
    cov5, risk5 = fast_coverage(cc, kc, wc, ct, kt, wt, 0.05)
    return {"error_auroc": fast_auroc(ct, kt, wt), "cov@5%": cov5, "risk@5%": risk5,
            "aurc": fast_aurc(ct, kt, wt), "cov@10%": fast_coverage(cc, kc, wc, ct, kt, wt, 0.10)[0]}


METRICS = ["error_auroc", "cov@5%", "aurc", "cov@10%"]


# --- paired contrast with cluster bootstrap --------------------------------------------------------

def _arrays(m: pd.DataFrame, suffix: str, mask: np.ndarray):
    return (m[f"conf{suffix}"].to_numpy(float)[mask], m[f"correct{suffix}"].to_numpy(float)[mask], m["weight"].to_numpy(float)[mask])


def paired(a: pd.DataFrame, b: pd.DataFrame, key: list[str], reps: int = REPS, seed: int = SEED, task: str | None = None) -> dict:
    """Signal a vs b on identical units. -> {"n", metric: (a, b, a-b), metric+"_reps": array of a-b}."""
    m = a[key + ["conf", "correct", "weight", "group", "split"]].merge(b[key + ["conf", "correct"]], on=key, suffixes=("_a", "_b"))
    cal, tst = (m.split == "calib").to_numpy(), (m.split == "test").to_numpy()
    out: dict = {"n_test": int(tst.sum()), "n_calib": int(cal.sum())}

    def stats(ci: np.ndarray, ti: np.ndarray) -> tuple[dict, dict]:
        ra = triage(*_arrays(m, "_a", ci), *_arrays(m, "_a", ti))
        rb = triage(*_arrays(m, "_b", ci), *_arrays(m, "_b", ti))
        return ra, rb

    ra, rb = stats(cal, tst)
    for k in METRICS:
        out[k] = (ra[k], rb[k], ra[k] - rb[k])
    out["risk@5%"] = (ra["risk@5%"], rb["risk@5%"])
    groups = m.group.astype(str).to_numpy()
    by_split = {s: pd.Series(np.flatnonzero(mask)).groupby(groups[mask]).apply(np.array).to_dict() for s, mask in (("c", cal), ("t", tst))}
    shared = joint_draws(task, reps, seed) if task else None
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in METRICS}
    for r in range(reps):
        idx = {}
        for s, members in by_split.items():
            if shared is not None:
                idx[s] = resample(members, shared[s][r])
            else:
                keys = list(members)
                idx[s] = resample(members, np.array(keys)[rng.integers(0, len(keys), len(keys))])
        ci = np.zeros(len(m), bool)
        ti = np.zeros(len(m), bool)
        # Resampled indices can repeat; build per-replicate frames by index rather than masks.
        sub = m.iloc[np.r_[idx["c"], idx["t"]]]
        n_c = len(idx["c"])
        ci = np.r_[np.ones(n_c, bool), np.zeros(len(idx["t"]), bool)]
        ti = ~ci
        ra_ = triage(*_arrays(sub, "_a", ci), *_arrays(sub, "_a", ti))
        rb_ = triage(*_arrays(sub, "_b", ci), *_arrays(sub, "_b", ti))
        for k in METRICS:
            draws[k].append(ra_[k] - rb_[k])
    for k in METRICS:
        out[f"{k}_reps"] = np.array(draws[k], float)
    return out


def _ci(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    return (float("nan"), float("nan")) if x.size == 0 else tuple(np.percentile(x, [2.5, 97.5]).tolist())


def _p(x: np.ndarray) -> float:
    """Two-sided bootstrap p-value for a difference of 0 (percentile method)."""
    x = x[np.isfinite(x)]
    return float("nan") if x.size == 0 else float(min(1.0, 2 * min((x <= 0).mean(), (x >= 0).mean())))


def _contrast_job(args):
    task, a_name, b_name, reps, adjudication = args
    td = TaskData.load(task, adjudication)
    pa, pb = td.preds(a_name), td.preds(b_name)
    if pa is None or pb is None:
        return task, a_name, b_name, None
    return task, a_name, b_name, paired(td.decisions(pa), td.decisions(pb), td.key, reps, task=task)


def contrasts(pairs: list[tuple[str, str]], tasks: list[str], reps: int = REPS, adjudication: str | None = None) -> pd.DataFrame:
    """Per-task and pooled (equal-weight mean over tasks) a-b contrasts with 95% intervals."""
    jobs = [(t, a, b, reps, adjudication) for t in tasks for a, b in pairs]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        done = list(ex.map(_contrast_job, jobs))
    rows, pooled = [], {}
    for task, a, b, r in done:
        if r is None:
            continue
        row = {"task": task, "a": a, "b": b, "n_test": r["n_test"]}
        for k in METRICS:
            row[f"{k}_a"], row[f"{k}_b"], row[f"{k}_diff"] = r[k]
            row[f"{k}_lo"], row[f"{k}_hi"] = _ci(r[f"{k}_reps"])
            row[f"{k}_p"] = _p(r[f"{k}_reps"])
            pooled.setdefault((a, b, k), []).append((r[k][2], r[f"{k}_reps"]))
        row["risk@5%_a"], row["risk@5%_b"] = r["risk@5%"]
        rows.append(row)
    for (a, b) in pairs:
        prow = {"task": "POOLED", "a": a, "b": b}
        for k in METRICS:
            parts = pooled.get((a, b, k), [])
            if not parts:
                continue
            prow[f"{k}_diff"] = float(np.nanmean([p[0] for p in parts]))
            pooled_reps = np.nanmean(np.vstack([p[1] for p in parts]), axis=0)
            prow[f"{k}_lo"], prow[f"{k}_hi"] = _ci(pooled_reps)
            prow[f"{k}_p"] = _p(pooled_reps)
            prow["n_tasks"] = len(parts)
        rows.append(prow)
    return pd.DataFrame(rows)


# --- standalone signal table ------------------------------------------------------------------------

def signal_table(tasks: list[str], adjudication: str | None = None) -> pd.DataFrame:
    rows = []
    for t in tasks:
        td = TaskData.load(t, adjudication)
        for s in SIGNALS:
            p = td.preds(s)
            if p is None:
                continue
            d = td.decisions(p)
            m = decision_metrics(d, td.codebook)
            test = d[d.split == "test"]
            rows.append({"task": t, "signal": s, "segments_test": test.segment_id.nunique()} | m |
                        {"share_conf>=0.99": float(np.average(test.conf >= 0.99, weights=test.weight)),
                         "distinct_conf": int(test.conf.round(4).nunique())})
    return pd.DataFrame(rows)


# --- RQ2 ----------------------------------------------------------------------------------------------

def rq2() -> pd.DataFrame:
    rows = []
    for t in ["mfrc", "annomi_client", "annomi_therapist"]:
        td = TaskData.load(t)
        test = set(td.sample[td.sample.split == "test"].segment_id)
        per = td.task.labels.groupby("segment_id").size()
        ids = test & set(per[per >= 3].index)
        labels = td.task.labels[td.task.labels.segment_id.isin(ids)]
        for s in [x for x in SIGNALS if not x.endswith("_noul")]:
            p = td.preds(s)
            if p is None:
                continue
            p = p[p.segment_id.isin(ids)]
            if p.segment_id.nunique() < 20:
                continue
            u = uncertainty_vs_humans(p, labels, td.codebook)
            loo = leave_one_coder_out(p, labels, td.codebook)
            loo = loo[loo.n >= 20]
            rows.append({"task": t, "signal": s, "segments": p.segment_id.nunique(), "spearman": u["spearman"],
                         "auroc_split": u["auroc_split"], "share_split": u["share_split"], "annotators": len(loo),
                         "human_kappa_min": loo.kappa_human.min(), "human_kappa_median": loo.kappa_human.median(),
                         "human_kappa_max": loo.kappa_human.max(), "model_kappa_mean": loo.kappa_model.mean(),
                         "model_minus_human_mean": loo.model_minus_human.mean(),
                         "beats_n_annotators": int((loo.model_minus_human > 0).sum())})
    return pd.DataFrame(rows)


# --- RQ3 ----------------------------------------------------------------------------------------------

def _boot_pair(base: pd.DataFrame, var: pd.DataFrame, key: list[str], reps: int, seed: int = SEED):
    m = base[key + ["conf", "correct", "weight", "group", "pred"]].merge(var[key + ["conf", "correct", "pred"]], on=key, suffixes=("_b", "_v"))
    flips = (m.pred_b != m.pred_v).to_numpy(float)
    members = pd.Series(np.arange(len(m))).groupby(m.group.to_numpy()).apply(np.array).to_list()
    rng = np.random.default_rng(seed)
    fr, da = [], []
    for _ in range(reps):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        s = m.iloc[idx]
        fr.append(np.average(flips[idx], weights=m.weight.to_numpy(float)[idx]))
        da.append(fast_aurc(s.conf_v.to_numpy(float), s.correct_v.to_numpy(float), s.weight.to_numpy(float))
                  - fast_aurc(s.conf_b.to_numpy(float), s.correct_b.to_numpy(float), s.weight.to_numpy(float)))
    return _ci(np.array(fr)), _ci(np.array(da))


def rq3(reps: int = REPS) -> pd.DataFrame:
    rows = []
    arms = {"jev": ("robustness", "jev", ["reversed", "shuffle1", "shuffle2", "shuffle3", "isolated", "neutral_labels", "no_context", "paraphrase0", "paraphrase1"]),
            "luna-verbal": ("llm_robust", "luna-verbal", ["reversed", "shuffle1", "neutral_labels"]),
            "sonnet-verbal": ("sonnet_robust", "sonnet-verbal", ["reversed", "shuffle1", "neutral_labels"]),
            "oaid": ("oaid_robustness", "oaid", ["reversed", "shuffle1", "shuffle2", "shuffle3", "isolated", "neutral_labels", "no_context", "paraphrase0", "paraphrase1"]),
            **{dm: ("dm_robustness", dm, ["reversed", "shuffle1", "shuffle2", "shuffle3", "isolated", "neutral_labels", "no_context", "paraphrase0", "paraphrase1"])
               for dm in ("clef_flash", "pplx", "kev9b")}}
    for t in CONFIRMATORY:
        td = TaskData.load(t)
        for signal, (exp, stem, variants) in arms.items():
            base_path = RUNS / exp / t / f"{stem}__canonical.parquet"
            if not base_path.exists():
                continue
            base = td.decisions(pd.read_parquet(base_path))
            # Coverage at the canonical arm's calib-fitted 5% threshold (arms run on test only).
            full = td.decisions(td.preds(signal))
            cal = full[full.split == "calib"]
            thr_ok = None
            if not cal.empty:
                thr, _, risk = _rc(cal.conf.to_numpy(float), 1 - cal.correct.to_numpy(float), cal.weight.to_numpy(float))
                ok = np.flatnonzero(risk <= 0.05)
                thr_ok = thr[ok[-1]] if ok.size else None
            for v in variants:
                path = RUNS / exp / t / f"{stem}__{v}.parquet"
                if not path.exists():
                    continue
                var = td.decisions(pd.read_parquet(path))
                c = compare(base, var, td.codebook)
                fr_ci, da_ci = _boot_pair(base, var, td.key, reps)
                cov = {}
                for name, d in (("base", base), ("var", var)):
                    w = d.weight.to_numpy(float)
                    cov[name] = float(w[d.conf.to_numpy() >= thr_ok].sum() / w.sum()) if thr_ok is not None else 0.0
                rows.append({"task": t, "signal": signal, "arm": v, **c, "flip_lo": fr_ci[0], "flip_hi": fr_ci[1],
                             "d_aurc_lo": da_ci[0], "d_aurc_hi": da_ci[1], "d_cov@5%_at_canonical_threshold": cov["var"] - cov["base"]})
    return pd.DataFrame(rows)


def run_all(reps: int = REPS) -> dict[str, pd.DataFrame]:
    RESULTS.mkdir(exist_ok=True)
    out = {
        "rq1_signals": signal_table(CONFIRMATORY),
        "rq1_contrasts": contrasts([("jev", s) for s in COMPARATORS], CONFIRMATORY, reps),
        "format_contrast": contrasts([(f"{dm}_noul", dm) for dm in ["jev", "oaid"] + NEW_DMS], [t for t in CONFIRMATORY if t != "mfrc"], reps),
        "same_base_contrasts": contrasts(SAME_BASE, CONFIRMATORY, reps),
        "oaid_contrasts": contrasts([("oaid", s) for s in OAID_COMPARATORS], CONFIRMATORY, reps),
        "rq2": rq2(),
        "rq3": rq3(reps),
    }
    for name, df in out.items():
        df.to_csv(RESULTS / f"{name}.csv", index=False)
    return out


def run_adjudicated(batch: str, reps: int = REPS) -> dict[str, pd.DataFrame]:
    """Secondary analysis (plan sections 3 and 9): RQ1 against the adjudicated reference."""
    RESULTS.mkdir(exist_ok=True)
    out = {
        f"rq1_signals_adj_{batch}": signal_table(CONFIRMATORY, batch),
        f"rq1_contrasts_adj_{batch}": contrasts([("jev", s) for s in COMPARATORS], CONFIRMATORY, reps, batch),
    }
    for name, df in out.items():
        df.to_csv(RESULTS / f"{name}.csv", index=False)
    return out
