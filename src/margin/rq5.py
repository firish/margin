"""RQ5 (exploratory, plan section 10): where coding breaks, per code and by code type.

Per code, every coder is scored one-vs-rest on the test split (unweighted): binary kappa, F1, detection
AUROC (does p for this code separate items that have it from items that do not), and probability
calibration (ECE of p against the 0/1 reference). Code types come from docs/code_types.csv; because
type is confounded with task, comparisons are made within task.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

from margin.analysis import RESULTS, TaskData

TASKS = ["mfrc", "annomi_client", "annomi_therapist", "talkmoves_teacher", "talkmoves_student", "dunivin"]
SIGNALS = ["jev", "luna-verbal", "haiku-verbal", "oss-verbal", "sonnet-verbal"]
LLM_CHEAP = ["luna-verbal", "haiku-verbal", "oss-verbal"]


def prob_ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1], right=True), 0, bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum() for b in range(bins) if (idx == b).any()) / len(p))


def code_rows(td: TaskData, preds: pd.DataFrame) -> list[dict]:
    test = set(td.sample[td.sample.split == "test"].segment_id)
    if td.codebook.mode == "exclusive":
        ref = td.ref[td.ref.segment_id.isin(test)].set_index("segment_id").ref_code
        wide = preds.pivot_table(index="segment_id", columns="code_id", values="p").reindex(index=ref.index, columns=td.codebook.code_ids)
        keep = wide.notna().all(axis=1)
        wide, ref = wide[keep], ref[keep]
        pred = wide.idxmax(axis=1)
        items = [(c, (ref == c).to_numpy(int), (pred == c).to_numpy(int), wide[c].to_numpy(float)) for c in td.codebook.code_ids]
    else:
        ref = td.ref[td.ref.segment_id.isin(test)][["segment_id", "code_id", "y"]]
        d = ref.merge(preds[["segment_id", "code_id", "p"]], on=["segment_id", "code_id"])
        items = [(c, g.y.to_numpy(int), (g.p >= 0.5).to_numpy(int), g.p.to_numpy(float)) for c, g in d.groupby("code_id")]
    rows = []
    for code, y, pred, p in items:
        if len(y) == 0:
            continue
        both = y.min() != y.max()
        rows.append({
            "code_id": code, "n": len(y), "n_pos": int(y.sum()), "prevalence": float(y.mean()),
            "kappa": float(cohen_kappa_score(y, pred, labels=[0, 1])) if both or pred.min() != pred.max() else np.nan,
            "f1": float(f1_score(y, pred, zero_division=0)),
            "detect_auroc": float(roc_auc_score(y, p)) if both else np.nan,
            "ece": prob_ece(p, y),
        })
    return rows


def per_code() -> pd.DataFrame:
    rows = []
    for t in TASKS:
        td = TaskData.load(t)
        for s in SIGNALS:
            p = td.preds(s)
            if p is None:
                continue
            rows += [{"task": t, "signal": s} | r for r in code_rows(td, p)]
    out = pd.DataFrame(rows)
    types = pd.read_csv("docs/code_types.csv")[["task", "code_id", "label", "inference", "implied", "context_dependent", "residual", "def_words", "n_include", "n_examples"]]
    return out.merge(types, on=["task", "code_id"], how="left")


def type_summary(pc: pd.DataFrame) -> pd.DataFrame:
    """Within-task means by inference level for Jev, and Jev's kappa gap to the cheap LLMs."""
    wide = pc.pivot_table(index=["task", "code_id"], columns="signal", values="kappa").reset_index()
    wide["llm_mean"] = wide[LLM_CHEAP].mean(axis=1)
    wide["jev_minus_llm"] = wide["jev"] - wide["llm_mean"]
    meta = pc.drop_duplicates(["task", "code_id"])[["task", "code_id", "label", "inference", "implied", "context_dependent", "residual", "prevalence", "def_words"]]
    wide = wide.merge(meta, on=["task", "code_id"])
    return wide


def fixed_effects(wide: pd.DataFrame, outcome: str) -> dict:
    """OLS of a per-code outcome on code features with task fixed effects (descriptive; ~36 codes)."""
    d = wide.dropna(subset=[outcome]).copy()
    X = pd.get_dummies(d.task, prefix="task", drop_first=False).astype(float)
    X["implied"] = d.implied.astype(float)
    X["context_dependent"] = d.context_dependent.astype(float)
    X["residual"] = d.residual.astype(float)
    X["log_prevalence"] = np.log(d.prevalence.clip(lower=1e-3))
    beta, *_ = np.linalg.lstsq(X.to_numpy(), d[outcome].to_numpy(float), rcond=None)
    resid = d[outcome].to_numpy(float) - X.to_numpy() @ beta
    dof = max(1, len(d) - np.linalg.matrix_rank(X.to_numpy()))
    cov = np.linalg.pinv(X.to_numpy().T @ X.to_numpy()) * (resid @ resid / dof)
    se = np.sqrt(np.diag(cov))
    names = list(X.columns)
    return {k: (float(beta[i]), float(se[i])) for i, k in enumerate(names) if not k.startswith("task_")} | {"n_codes": len(d), "dof": dof}


def adjudication_categories(batch: str = "plan_v1") -> pd.DataFrame:
    """For adjudicated items where a coder disagreed with the human label: who was wrong."""
    from margin.adjudicate import load_verdicts

    v = load_verdicts(batch)
    v = v[v.stratum != "control"]
    rows = []
    for src in ["jev", "luna", "haiku", "oss"]:
        d = v[v[f"{src}_label"].notna() & (v[f"{src}_label"] != v.human)]
        # Verdicts by the LLM panel and by the author are also reported separately (the review asked for it).
        for t, g in list(d.groupby("task")) + [("ALL", d)] + [(f"ALL ({who} verdicts)", d[d.source == who]) for who in ("panel", "author")]:
            ok_c = np.array([(not amb) and (lab in acc) for lab, acc, amb in zip(g[f"{src}_label"], g.acceptable, g.ambiguous)])
            ok_h = np.array([(not amb) and (lab in acc) for lab, acc, amb in zip(g.human, g.acceptable, g.ambiguous)])
            amb = g.ambiguous.to_numpy(bool)
            n = len(g)
            rows.append({"coder": src, "task": t, "n": n,
                         "label_error (coder right, human wrong)": float((ok_c & ~ok_h).sum() / n) if n else np.nan,
                         "model_error (human right, coder wrong)": float((~ok_c & ok_h & ~amb).sum() / n) if n else np.nan,
                         "both_defensible (under-specified)": float(((ok_c & ok_h) | amb).sum() / n) if n else np.nan,
                         "neither_right": float((~ok_c & ~ok_h & ~amb).sum() / n) if n else np.nan})
    return pd.DataFrame(rows)


def run() -> dict[str, pd.DataFrame]:
    pc = per_code()
    wide = type_summary(pc)
    fe_rows = []
    for outcome in ["jev", "jev_minus_llm"]:
        res = fixed_effects(wide, outcome)
        for term, val in res.items():
            if isinstance(val, tuple):
                fe_rows.append({"outcome": outcome, "term": term, "beta": val[0], "se": val[1], "n_codes": res["n_codes"], "dof": res["dof"]})
    fe = pd.DataFrame(fe_rows)
    cats = adjudication_categories()
    out = {"rq5_per_code": pc, "rq5_codes_wide": wide, "rq5_fixed_effects": fe, "rq5_adjudication_categories": cats}
    for name, df in out.items():
        df.to_csv(RESULTS / f"{name}.csv", index=False)
    return out
