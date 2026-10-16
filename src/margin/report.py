"""Paper figures and tables, regenerated from results/*.csv and the run files.

`margin report` -> paper/figures/*.png|pdf and paper/tables/*.md|csv. Every number comes from an
analysis output; nothing here fits or tunes anything except the plotted recalibration maps, which
reuse margin.triage exactly.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from margin import sampling, triage  # noqa: E402
from margin.analysis import CONFIRMATORY, RESULTS  # noqa: E402
from margin.evaluate import human_alpha  # noqa: E402

FIG, TAB = Path("paper/figures"), Path("paper/tables")
TASK = {"mfrc": "MFRC", "annomi_client": "AnnoMI client", "annomi_therapist": "AnnoMI therapist",
        "talkmoves_teacher": "TalkMoves teacher", "talkmoves_student": "TalkMoves student", "dunivin": "Dunivin", "POOLED": "Pooled"}
SIG = {"jev": "Jev", "jev_noul": "Jev (yes/no format)", "oaid": "OpenAI Decisions", "oaid_noul": "OpenAI Decisions (yes/no)",
       "luna-verbal": "Luna verbal", "haiku-verbal": "Haiku verbal", "oss-verbal": "gpt-oss verbal", "qwen-verbal": "Qwen3.5-9B verbal",
       "luna-logprob": "Luna logprob", "oss-logprob": "gpt-oss logprob", "qwen-logprob": "Qwen3.5-9B logprob",
       "agree3": "3-LLM agreement", "luna_sc-vote": "Luna self-consistency", "sonnet-verbal": "Sonnet verbal",
       "qwen_big-verbal": "Qwen3.8-2.4T verbal", "qwen_big-logprob": "Qwen3.8-2.4T logprob", "jev per-code": "Jev, per-code recalibrated",
       "clef": "Clef (27B)", "clef_flash": "Clef-flash (9B)", "pplx": "pplx-decider", "kev9b": "Kev-9B", "kev27b": "Kev-27B",
       "clef_noul": "Clef (yes/no)", "clef_flash_noul": "Clef-flash (yes/no)", "pplx_noul": "pplx-decider (yes/no)",
       "kev9b_noul": "Kev-9B (yes/no)", "kev27b_noul": "Kev-27B (yes/no)", "dm_committee": "Decision-model committee (7)",
       "sonnet_sub-verbal": "Sonnet verbal (v1.0 subset)"}
COLOR = {"jev": "#1b6ca8", "jev per-code": "#5fa8e0", "oaid": "#e07b39", "luna-verbal": "#6a994e", "luna-logprob": "#a7c957",
         "haiku-verbal": "#8e5ea2", "oss-verbal": "#7f7f7f", "oss-logprob": "#bdbdbd", "qwen-verbal": "#c9a227", "qwen-logprob": "#e9d985",
         "sonnet-verbal": "#b5179e", "qwen_big-verbal": "#d4a373", "agree3": "#2a9d8f", "luna_sc-vote": "#264653",
         "clef": "#f4a261", "clef_flash": "#e9c46a", "pplx": "#20b2aa", "kev9b": "#9d4edd", "kev27b": "#5a189a", "dm_committee": "#003049",
         "sonnet_sub-verbal": "#b5179e"}
RULE = {"empirical": "naive (calib error ≤ 5%)", "cp": "pointwise Clopper-Pearson", "ltt_dec": "LTT on decisions (assumes independence)",
        "ltt_iid": "LTT on clusters (assumes i.i.d. clusters)", "ltt": "LTT on the corpus (random choice of clusters)", "oracle": "oracle (fit on test)"}
RULE_COLOR = {"empirical": "#d62828", "cp": "#f77f00", "ltt_dec": "#8ecae6", "ltt_iid": "#457b9d", "ltt": "#003049"}
DIALOGUE = [t for t in CONFIRMATORY if t != "mfrc"]
TASK2 = {k: v.replace(" ", "\n", 1) if k != "POOLED" else v for k, v in TASK.items()} | {"annomi": "AnnoMI", "talkmoves": "TalkMoves"}  # two-line tick labels
ARM = {"reversed": "options\nreversed", "shuffle1": "shuffle 1", "shuffle2": "shuffle 2", "shuffle3": "shuffle 3",
       "neutral_labels": "neutral\nlabels", "paraphrase0": "paraphrase\nA", "paraphrase1": "paraphrase\nB",
       "isolated": "one code\nper call", "no_context": "no previous\nturn"}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.dpi": 100})


# --- output helpers -----------------------------------------------------------------------------------

def fmt(x, digits: int = 3) -> str:
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return "–"
        return f"{x:.{digits}f}"
    return str(x)


def num(x: float, digits: int = 3, plus: bool = False) -> str:
    """A number for an interval: one more decimal when a nonzero value would otherwise print as 0, so an endpoint
    that excludes zero never looks like zero."""
    d = digits + 1 if round(float(x), digits) == 0 and float(x) != 0 else digits
    return f"{float(x):+.{d}f}" if plus else f"{float(x):.{d}f}"


def ci(lo: float, hi: float, digits: int = 3, plus: bool = False) -> str:
    return f"[{num(lo, digits, plus)}, {num(hi, digits, plus)}]"


def rate(v: float, digits: int = 3) -> str:
    """A rate for display: 0 only when nothing changed, '<0.001' when nonzero but below the displayed precision."""
    if pd.isna(v):
        return "–"
    if v == 0:
        return "0"
    return f"<{10 ** -digits:.{digits}f}" if round(float(v), digits) == 0 else f"{float(v):.{digits}f}"


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c], digits) for c in cols) + " |")
    return "\n".join(lines)


def tex_escape(x) -> str:
    s = fmt(x, LATEX_DIGITS) if not isinstance(x, str) else x
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
                 ("≤", r"$\leq$"), ("≥", r"$\geq$"), ("−", "--"), ("→", r"$\rightarrow$"), ("κ", r"$\kappa$"), ("α", r"$\alpha$"),
                 ("Δ", r"$\Delta$"), ("δ", r"$\delta$"), ("–", "--"), ("×", r"$\times$")):
        s = s.replace(a, b)
    return s


LATEX_DIGITS = 3


def tex_refs(s: str) -> str:
    """Cross-references in escaped notes: [[tab:name]] and [[sec:name]] become LaTeX references, so notes never
    quote a hard-coded table number."""
    kinds = {"tab": "Table", "sec": "Section", "fig": "Figure"}
    return re.sub(r"\[\[(tab|sec|fig):([A-Za-z0-9\\_]+)\]\]",
                  lambda m: f"{kinds[m.group(1)]}~\\ref{{{m.group(1)}:{m.group(2).replace(chr(92) + '_', '_')}}}", s)


def md_refs(s: str) -> str:
    return re.sub(r"\[\[(tab|sec|fig):([A-Za-z0-9_]+)\]\]", lambda m: m.group(2), s)


def tex_header(c, align: str, width: int = 13) -> str:
    """A column header, wrapped onto several lines (makecell) so wide tables need less shrinking."""
    lines, cur = [], ""
    for w in str(c).split(" "):
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    if len(lines) == 1:
        return tex_escape(lines[0])
    return rf"\makecell[b{align}]{{" + r" \\{} ".join(tex_escape(x) for x in lines) + "}"  # {} stops "\\[" parsing as a skip


def _tabular(df: pd.DataFrame, cols: list[str], breaks: tuple = (), width: int = 13) -> list[str]:
    align = "".join("r" if pd.api.types.is_numeric_dtype(df[c]) else "l" for c in cols)
    lines = [r"\begin{adjustbox}{max width=\linewidth}", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(tex_header(c, a, width) for c, a in zip(cols, align)) + r" \\", r"\midrule"]
    for i, (_, r) in enumerate(df.iterrows()):
        if i in breaks:
            lines.append(r"\midrule")
        lines.append(" & ".join(tex_escape(r[c]) for c in cols) + r" \\")
    return lines + [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]


def tex_table(df: pd.DataFrame, caption: str, label: str, note: str = "", digits: int = 3, breaks: tuple = (),
              panels: list[tuple[str, list[str]]] | None = None, width: int = 13) -> str:
    """A booktabs table environment (no jinja2 needed); numbers formatted like the markdown tables. With panels,
    each (title, columns) group is set as its own tabular, stacked, so a wide table is split rather than shrunk."""
    global LATEX_DIGITS
    LATEX_DIGITS = digits
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{tex_escape(caption)}}}", rf"\label{{{label}}}"]
    for k, (title, cols) in enumerate(panels or [("", list(df.columns))]):
        if title:
            lines += ([r"\medskip"] if k else []) + [rf"\par\textit{{{tex_escape(title)}}}\par\smallskip"]
        lines += _tabular(df, cols, breaks, width)
    if note:
        lines.append(rf"\par\smallskip\footnotesize\raggedright {tex_refs(tex_escape(note))}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, name: str, caption: str, note: str = "", digits: int = 3,
               panels: list[tuple[str, list[str]]] | None = None, width: int = 13) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.tex").write_text(tex_table(df, caption.split(". ", 1)[-1].rstrip("*"), f"tab:{name}", note, digits,
                                               panels=panels, width=width))
    text = f"**{caption}**\n\n{md_table(df, digits)}\n"
    if note:
        text += f"\n_{md_refs(note)}_\n"
    (TAB / f"{name}.md").write_text(text)


def save_fig(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"{name}.csv")


def pooled_risk(per_task: pd.DataFrame, cov: str, risk: str) -> float:
    """Error among all accepted decisions with tasks weighted equally (per-task risks weighted by coverage).

    A plain mean of per-task risks lets a task where the rule accepts a handful of decisions dominate."""
    w = per_task[cov].fillna(0).to_numpy(float)
    return float((w * per_task[risk].fillna(0).to_numpy(float)).sum() / w.sum()) if w.sum() > 0 else np.nan


def holm(p: pd.Series) -> pd.Series:
    order = np.argsort(p.to_numpy())
    m, adj, running = len(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p.iloc[i]))
        adj[i] = running
    return pd.Series(adj, index=p.index)


# --- figures --------------------------------------------------------------------------------------------

def fig_risk_coverage() -> None:
    sigs = ["jev", "jev per-code", "kev27b", "pplx", "oaid", "haiku-verbal", "sonnet-verbal", "luna-logprob"]
    fig, grid = plt.subplots(2, 3, figsize=(11, 6.2), sharey=True)
    axes = grid.ravel()
    for ax, task in zip(axes, CONFIRMATORY):
        key = "code_id" if triage.task_data(task).codebook.mode == "multi" else "pred"
        for s in sigs:
            d = triage.load(task, "jev" if s == "jev per-code" else s)
            if d is None:
                continue
            cal, tst = d[d.split == "calib"], d[d.split == "test"]
            conf = triage.per_code_platt(cal, tst, key, 50) if s == "jev per-code" else tst.conf.to_numpy(float)
            cv = triage.curve(conf, 1 - tst.correct.to_numpy(float), tst.weight.to_numpy(float))
            ax.step(cv.cov, cv.risk, where="post", color=COLOR[s], lw=1.6 if s.startswith("jev") else 1.1,
                    ls="--" if s == "jev per-code" else "-", label=SIG[s])
        ax.axhspan(0, 0.05, color="#2ca02c", alpha=0.08, lw=0, zorder=0)  # the region a 5% target accepts
        ax.axhline(0.05, color="k", lw=0.8, ls=":")
        ax.set_title(TASK[task]); ax.set_xlabel("coverage (share auto-accepted)"); ax.set_xlim(0, 1); ax.set_ylim(0, 0.45)
    for row in grid:
        row[0].set_ylabel("error among accepted (test)")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].axis("off")
    axes[-1].legend(handles, labels, loc="center", fontsize=9)
    fig.subplots_adjust(hspace=0.45)
    save_fig(fig, "fig01_risk_coverage")


def fig_forest() -> None:
    c = read("rq1_contrasts")
    c = c[(c.a == "jev") & (c.b != "dm_committee")]  # the committee is exploratory (deviation 13)
    groups = [["oaid", "clef", "clef_flash", "pplx", "kev9b", "kev27b"],
              ["luna-verbal", "haiku-verbal", "oss-verbal", "qwen-verbal", "sonnet_sub-verbal", "sonnet-verbal", "qwen_big-verbal"],
              ["luna-logprob", "oss-logprob", "qwen-logprob", "qwen_big-logprob"], ["agree3", "luna_sc-vote"]]
    names = {"decision models": 0, "verbalized": 1, "token probability": 2, "agreement": 3}
    pooled = c[c.task == "POOLED"].set_index("b")
    order = [b for g in groups for b in sorted([x for x in g if x in pooled.index], key=lambda x: pooled.loc[x, "error_auroc_diff"])]
    label = {b: SIG.get(b, b) + (" *" if b in ("qwen-verbal", "qwen-logprob", "qwen_big-verbal", "qwen_big-logprob", "sonnet-verbal") else "") for b in order}
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.36 * len(order) + 1.4), sharey=True)
    y = np.arange(len(order))[::-1]
    bounds = np.cumsum([len([x for x in g if x in pooled.index]) for g in groups])[:-1]
    for ax, k, title in [(axes[0], "error_auroc", "Error AUROC: Jev − comparator"), (axes[1], "cov@5%", "Accepted under naive rule: Jev − comparator")]:
        for yi, b in zip(y, order):
            r = pooled.loc[b]
            per = c[(c.task != "POOLED") & (c.b == b)]
            ax.scatter(per[f"{k}_diff"], np.full(len(per), yi), s=9, color="#bbbbbb", zorder=1)
            sig = not (r[f"{k}_lo"] <= 0 <= r[f"{k}_hi"])
            ax.errorbar(r[f"{k}_diff"], yi, xerr=[[r[f"{k}_diff"] - r[f"{k}_lo"]], [r[f"{k}_hi"] - r[f"{k}_diff"]]],
                        fmt="o", color="#1b6ca8" if sig else "#888888", ms=5, capsize=2, zorder=2)
        for bnd in bounds:
            ax.axhline(len(order) - bnd - 0.5, color="#dddddd", lw=0.8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y, [label[b] for b in order]); ax.set_title(title)
        ax.set_xlabel("difference (pooled; grey dots = tasks)")
    save_fig(fig, "fig02_forest_jev_contrasts")


def fig_reliability() -> None:
    sigs = ["jev", "oaid", "pplx", "kev27b", "sonnet-verbal", "luna-logprob"]
    fig, grid = plt.subplots(2, 3, figsize=(11, 7), sharey=True)
    axes = grid.ravel()
    for ax, s in zip(axes, sigs):
        raw, scaled = [], []
        for task in CONFIRMATORY:
            td = triage.task_data(task)
            preds = td.preds(s)
            if preds is None:
                continue
            d = td.decisions(preds)
            c_ = (d.split == "calib").to_numpy()
            t = (d.split == "test").to_numpy()
            f = triage.platt(d.conf.to_numpy(float)[c_], d.correct.to_numpy(float)[c_], d.weight.to_numpy(float)[c_])
            w = d.weight.to_numpy(float)[t]
            w = w / w.sum()  # each task counts equally (MFRC has ~10x more decisions)
            raw.append(pd.DataFrame({"conf": d.conf.to_numpy(float)[t], "correct": d.correct.to_numpy(float)[t], "w": w}))
            scaled.append(pd.DataFrame({"conf": f(d.conf.to_numpy(float)[t]), "correct": d.correct.to_numpy(float)[t], "w": w}))
        for frame, label, style in [(pd.concat(raw), "raw", "-o"), (pd.concat(scaled), "Platt-recalibrated", "--s")]:
            bins = np.clip(np.digitize(frame.conf, np.linspace(0, 1, 11)[1:-1], right=True), 0, 9)
            g = frame.assign(b=bins).groupby("b").apply(lambda x: pd.Series({"conf": np.average(x.conf, weights=x.w), "acc": np.average(x.correct, weights=x.w), "n": len(x)}))
            g = g[g.n >= 20]
            ax.plot(g.conf, g.acc, style, ms=4, lw=1.2, color=COLOR[s] if label == "raw" else "#444444", label=label)
        ax.plot([0, 1], [0, 1], color="k", lw=0.6, ls=":")
        ax.set_title(SIG[s]); ax.set_xlim(0.3, 1.01); ax.set_ylim(0, 1.01)
    for row in grid:
        row[0].set_ylabel("accuracy (test; tasks\nweighted equally)")
    for ax in grid[-1]:
        ax.set_xlabel("confidence")
    fig.subplots_adjust(hspace=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend([handles[0], handles[1]], ["raw (colour: coder)", "Platt map fitted on calib"], ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    save_fig(fig, "fig03_reliability")


def top_block_table() -> pd.DataFrame:
    rows = []
    for task in DIALOGUE:
        for s in ["jev", "oaid", "clef", "pplx", "kev27b", "luna-verbal", "haiku-verbal", "sonnet-verbal"]:
            d = triage.load(task, s)
            if d is None:
                continue
            t = d[d.split == "test"]
            top = (t.conf >= t.conf.max() - 1e-9).to_numpy()
            w = t.weight.to_numpy(float)
            rows.append({"task": task, "signal": s, "top_confidence": float(t.conf.max()), "n_at_top": int(top.sum()),
                         "share_at_top": float(w[top].sum() / w.sum()),
                         "error_at_top": float((w[top] * (1 - t.correct.to_numpy(float)[top])).sum() / w[top].sum())})
    return pd.DataFrame(rows)


def fig_top_block(tb: pd.DataFrame) -> None:
    sigs = list(dict.fromkeys(tb.signal))
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    x = np.arange(len(DIALOGUE))
    wdt = 0.8 / len(sigs)
    for i, s in enumerate(sigs):
        sub = tb[tb.signal == s].set_index("task").reindex(DIALOGUE)
        axes[0].bar(x + i * wdt, sub.share_at_top, wdt, color=COLOR[s], label=SIG[s])
        err = sub.error_at_top.where(sub.n_at_top >= 10)  # blocks of < 10 decisions are not interpretable
        axes[1].bar(x + i * wdt, err, wdt, color=COLOR[s])
        for j, small in enumerate((sub.n_at_top < 10).to_numpy()):
            if small:
                axes[1].text(x[j] + i * wdt, 0.002, "<10", ha="center", va="bottom", fontsize=5, rotation=90, color="#666666")
    axes[1].axhline(0.05, color="k", lw=0.8, ls=":")
    axes[0].set_ylabel("share of decisions at\nthe coder's top confidence"); axes[1].set_ylabel("error among them")
    axes[1].set_xticks(x + 0.4 - wdt / 2, [TASK2[t] for t in DIALOGUE])
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.3), fontsize=8)
    save_fig(fig, "fig04_top_confidence_blocks")


def fig_threshold_rules() -> None:
    """MFRC only: the one task where any guaranteed rule accepts anything under cluster sampling (per-task rates: tab07b, tab07d)."""
    r = read("a3_resplits")
    r = r[r.task == "mfrc"]
    sigs = [s for s in ["jev", "oaid", "clef", "clef_flash", "pplx", "kev9b", "kev27b", "luna-verbal", "haiku-verbal", "sonnet-verbal"] if s in set(r.signal)]
    oracle = read("a3_risk_control")
    oracle = oracle[(oracle.task == "mfrc") & (oracle.rule == "oracle") & (oracle.eps == 0.05)].set_index("signal").test_coverage
    rules = ["empirical", "ltt_dec", "ltt_iid", "ltt"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    x = np.arange(len(sigs))
    wdt = 0.8 / len(rules)
    for i, rule in enumerate(rules):
        sub_ = r[r.rule == rule].set_index("signal").reindex(sigs)
        axes[0].bar(x + i * wdt, 100 * sub_.violation_rate, wdt, color=RULE_COLOR[rule], label=RULE[rule])
        axes[1].bar(x + i * wdt, 100 * sub_.mean_coverage, wdt, color=RULE_COLOR[rule])
    axes[1].scatter(x + 0.4 - wdt / 2, 100 * oracle.reindex(sigs), marker="_", s=300, color="k", label="oracle on the frozen split", zorder=3)
    axes[0].axhline(5, color="k", lw=0.8, ls=":")
    axes[0].set_ylabel("re-splits with held-out error > 5% (%)"); axes[1].set_ylabel("mean share of decisions accepted (%)")
    for ax in axes:
        ax.set_xticks(x + 0.4 - wdt / 2, [SIG[s] for s in sigs], rotation=30, ha="right")
    h0, l0 = axes[0].get_legend_handles_labels(); h1, l1 = axes[1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08), fontsize=8)
    save_fig(fig, "fig05_threshold_rules")


def fig_budget() -> None:
    """MFRC: coverage and held-out exceedance when only n hand-coded comments are available."""
    b = read("a3_budget")
    b = b[b.task == "mfrc"]
    sigs = ["jev", "pplx", "kev27b", "haiku-verbal", "luna-verbal"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for s in sigs:
        for rule, ls in [("empirical", "-"), ("ltt", "--")]:
            sub = b[(b.signal == s) & (b.rule == rule)].sort_values("calib_segments")
            lab = f"{SIG[s]}, {'naive' if rule == 'empirical' else 'corpus LTT'}"
            axes[0].plot(sub.realized_segments, 100 * sub.mean_coverage, ls, marker="o", ms=3, color=COLOR[s], label=lab)
            axes[1].plot(sub.realized_segments, 100 * sub.violation_rate, ls, marker="o", ms=3, color=COLOR[s])
    axes[1].axhline(5, color="k", lw=0.8, ls=":")
    axes[0].set_ylabel("mean share of test decisions accepted (%)"); axes[1].set_ylabel("draws with held-out error > 5% (%)")
    for ax in axes:
        ticks = [50, 100, 200, 400, 608]
        ax.set_xscale("log"); ax.set_xticks(ticks, [str(n) for n in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xlabel("hand-coded calibration comments (MFRC)")
    axes[1].legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7)
    save_fig(fig, "fig06_calibration_budget")


LABEL7 = {  # MFRC, cluster-level LTT: cascade -> (label, text offset in points)
    "pplx -> human": ("pplx-decider", (6, -3)), "jev -> human": ("Jev", (6, -3)), "kev9b -> human": ("Kev-9B", (6, -3)),
    "clef_flash -> human": ("Clef-flash", (6, -3)), "clef -> human": ("Clef", (-22, 6)), "kev27b -> human": ("Kev-27B", (-10, 8)),
    "oss-verbal -> human": ("gpt-oss", (6, -3)), "luna-verbal -> human": ("Luna", (6, -3)), "haiku-verbal -> human": ("Haiku", (6, 3)),
    "sonnet-verbal -> human": ("Sonnet", (6, 2)), "agree:pplx+kev27b -> human": ("pplx + Kev-27B agree", (8, -3)),
    "agree:jev+kev27b -> human": ("Jev + Kev-27B agree", (8, -4)), "dm_committee -> human": ("decision-model committee (7)", (-30, 9)),
    "kev27b -> sonnet-verbal -> human": ("Kev-27B → Sonnet", (-90, -3)), "jev -> sonnet-verbal -> human": ("Jev → Sonnet", (8, -3)),
    "agree:jev+haiku-verbal -> human": ("Jev + Haiku agree", (8, -9)),
}


def fig_cost_automation() -> None:
    """EQ5, deployable and guaranteed: MFRC (the only task where anything is certified under cluster sampling), corpus LTT, each
    configuration at the cost of that policy. Right: how many of the codes that apply the certified decisions find."""
    c = read("a3_cascades")
    c = c[(c.task == "mfrc") & (c.pool == "full") & (c.reference_for_eval == "human") & (c.rule == "ltt") & c.cascade.isin(LABEL7)]
    prev = pd.read_csv(RESULTS / "mfrc_prevalence.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    for _, r in c.iterrows():
        if r.automation <= 0:
            continue
        name = r.cascade
        single = name.count("->") == 1 and not name.startswith(("agree:", "dm_committee"))
        col = COLOR.get(name.split(" -> ")[0], "#555555") if single else "#444444"
        x, y = max(r.api_usd_per_1k_segments, 0.02), 100 * r.automation
        ax.scatter(x, y, color=col, s=44, zorder=3, edgecolor="k", lw=0.4, marker="o" if single else "D")
        text, offset = LABEL7[name]
        ax.annotate(text, (x, y), fontsize=7, xytext=offset, textcoords="offset points",
                    arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "#777777"} if abs(offset[0]) > 30 or abs(offset[1]) > 12 else None)
    ax.scatter([], [], color="#777777", edgecolor="k", lw=0.4, s=44, label="single coder")
    ax.scatter([], [], color="#444444", marker="D", s=40, label="two-coder cascade or agreement gate (certified jointly)")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xscale("log"); ax.set_xlim(0.017, 8); ax.set_ylim(45, 92)
    ax.set_xlabel("API cost per 1,000 MFRC comments under that policy (USD; log scale)")
    ax.set_ylabel("MFRC decisions certified at ≤5% error (%)")
    ax2 = axes[1]
    single = c[(c.cascade.str.count("->") == 1) & ~c.cascade.str.startswith(("agree:", "dm_committee"))]
    for _, r in single.iterrows():
        sig = r.cascade.split(" -> ")[0]
        row = prev[(prev.signal == sig) & (prev.rule == "ltt")]
        if row.empty or r.automation <= 0:
            continue
        x, y = max(r.api_usd_per_1k_segments, 0.02), 100 * row.present_recall_accepted.iloc[0]
        ax2.scatter(x, y, color=COLOR.get(sig, "#555"), s=44, edgecolor="k", lw=0.4, zorder=3)
        ax2.annotate(SIG.get(sig, sig).replace(" verbal", "").split(" (")[0], (x, y), fontsize=7,
                     xytext={"clef": (6, -10)}.get(sig, (5, 3)), textcoords="offset points")
    ax2.set_xscale("log"); ax2.set_xlim(0.017, 8)
    ax2.set_xlabel("API cost per 1,000 MFRC comments (USD; log scale)")
    ax2.set_ylabel("recall of applying codes (%)")
    save_fig(fig, "fig07_cost_vs_automation")


def fig_rq2() -> None:
    r = read("rq2")
    sigs = [s for s in ["jev", "oaid", "pplx", "kev27b", "luna-verbal", "haiku-verbal", "sonnet-verbal"] if s in set(r.signal)]
    tasks = list(dict.fromkeys(r.task))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
    x = np.arange(len(tasks))
    wdt = 0.8 / len(sigs)
    for i, s in enumerate(sigs):
        sub = r[r.signal == s].set_index("task").reindex(tasks)
        axes[0].bar(x + i * wdt, sub.spearman, wdt, color=COLOR[s], label=SIG[s])
        axes[1].scatter(x + i * wdt, sub.model_kappa_mean, color=COLOR[s], s=30, zorder=3)
    hum = r.drop_duplicates("task").set_index("task").reindex(tasks)
    for j, t in enumerate(tasks):
        axes[1].plot([j + 0.4 - wdt / 2] * 2, [hum.loc[t, "human_kappa_min"], hum.loc[t, "human_kappa_max"]], color="#cccccc", lw=10, zorder=1, solid_capstyle="butt")
        axes[1].plot(j + 0.4 - wdt / 2, hum.loc[t, "human_kappa_median"], "_", color="k", ms=20, zorder=2)
    axes[0].set_ylabel("Spearman: uncertainty vs human split"); axes[0].legend(fontsize=7)
    axes[1].set_ylabel("κ with the majority of the other annotators")
    for ax in axes:
        ax.set_xticks(x + 0.4 - wdt / 2, [TASK2[t] for t in tasks])
    save_fig(fig, "fig08_rq2_disagreement")


def fig_rq3() -> None:
    """Coder x perturbation matrix: mean flip rate, with the number of tasks behind each cell; arms a coder
    did not run are marked, not drawn as zero. Context removal is separated: it removes information."""
    r = read("rq3")
    g = r.groupby(["signal", "arm"]).agg(flip=("flip_rate", "mean"), tasks=("task", "nunique")).reset_index()
    arms = ["reversed", "shuffle1", "shuffle2", "shuffle3", "neutral_labels", "paraphrase0", "paraphrase1", "isolated", "no_context"]
    sigs = [s for s in ["jev", "oaid", "clef_flash", "pplx", "kev9b", "luna-verbal", "sonnet-verbal"] if s in set(g.signal)]
    flip = g.pivot(index="signal", columns="arm", values="flip").reindex(index=sigs, columns=arms)
    tasks = g.pivot(index="signal", columns="arm", values="tasks").reindex(index=sigs, columns=arms)
    fig, ax = plt.subplots(figsize=(12, 4.4))
    im = ax.imshow(100 * flip.to_numpy(float), cmap="Oranges", vmin=0, vmax=30, aspect="auto")
    for i in range(len(sigs)):
        for j in range(len(arms)):
            v, k = flip.iat[i, j], tasks.iat[i, j]
            if pd.isna(v):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f2f2f2", edgecolor="white", hatch="///", lw=0))
                ax.text(j, i, "not run", ha="center", va="center", fontsize=8, color="#888888")
            else:
                ax.text(j, i, f"{100 * v:.1f}%\n({int(k)} task{'s' if k != 1 else ''})", ha="center", va="center", fontsize=8,
                        color="white" if v > 0.2 else "black")
    ax.axvline(len(arms) - 1.5, color="k", lw=1.2)
    ax.set_xticks(range(len(arms)), [ARM[a] for a in arms], fontsize=8)
    ax.set_yticks(range(len(sigs)), [SIG[s] for s in sigs])
    ax.text(len(arms) - 1, -0.75, "information\nremoved", ha="center", va="bottom", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="decisions that change (%)")
    save_fig(fig, "fig09_rq3_flips")


def fig_per_code_gain() -> None:
    g = read("a3_per_code_gain")
    g = g[(g.task == "POOLED") & g.comparison.str.endswith("per-code calibrated - raw")].copy()
    g["signal"] = g.comparison.str.split(":").str[0]
    g = g.sort_values("auroc_a")
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(g) + 1))
    for i, (_, r) in enumerate(g.iterrows()):
        ax.plot([r.auroc_b, r.auroc_a], [i, i], color="#999999", lw=1.5)
        ax.scatter(r.auroc_b, i, color="white", edgecolor=COLOR.get(r.signal, "k"), s=40, zorder=3)
        ax.scatter(r.auroc_a, i, color=COLOR.get(r.signal, "k"), s=40, zorder=3)
        ax.annotate(f"{r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]", (max(r.auroc_a, r.auroc_b), i), xytext=(6, -3), textcoords="offset points", fontsize=7)
    ax.set_yticks(range(len(g)), [SIG.get(s, s) for s in g.signal])
    ax.set_xlabel("error AUROC on test (hollow: raw; filled: per-code Platt maps fitted on calib)")
    save_fig(fig, "fig10_per_code_recalibration")


def fig_repeatability() -> None:
    path = RESULTS / "repeatability.csv"
    if not path.exists():
        return
    r = pd.read_csv(path)
    p = r[r.task == "POOLED"].set_index("signal")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, k, lab in [(axes[0], "flip_rate", "share of decisions whose label changes"), (axes[1], "mean_abs_dconf", "mean |Δ confidence|")]:
        bars = ax.bar(range(len(p)), p[k], color=[COLOR.get(s, "#555") for s in p.index])
        ax.bar_label(bars, labels=[rate(v) for v in p[k]], fontsize=7)
        ax.set_xticks(range(len(p)), [SIG.get(s, s) for s in p.index], rotation=30, ha="right"); ax.set_ylabel(lab)
    save_fig(fig, "fig11_repeatability")


def fig_rq5() -> None:
    w = read("rq5_codes_wide")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    tasks = [t for t in CONFIRMATORY + ["dunivin"] if t in set(w.task)]
    for j, t in enumerate(tasks):
        sub = w[w.task == t]
        jitter = np.linspace(-0.25, 0.25, len(sub)) if len(sub) > 1 else np.zeros(1)
        for (_, r), dx in zip(sub.iterrows(), jitter):
            ax.scatter(j + dx, r.jev_minus_llm, marker="^" if r.implied else "o", color="#d62828" if r.implied else "#1b6ca8", s=22)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(tasks)), [TASK2[t] for t in tasks]); ax.set_ylabel("κ(Jev) − mean κ(cheap LLMs), per code")
    ax.scatter([], [], marker="o", color="#1b6ca8", label="explicit code"); ax.scatter([], [], marker="^", color="#d62828", label="implied code")
    ax.legend()
    save_fig(fig, "fig12_rq5_code_types")


# --- tables ---------------------------------------------------------------------------------------------

SOURCES = {"mfrc": ("MFRC (Trager et al. 2022)", "Reddit comment", "CC-BY-4.0"),
           "annomi_client": ("AnnoMI (Wu et al. 2022)", "client utterance (+ previous turn)", "public GitHub"),
           "annomi_therapist": ("AnnoMI (Wu et al. 2022)", "therapist utterance (+ previous turn)", "public GitHub"),
           "talkmoves_teacher": ("TalkMoves (Suresh et al. 2022)", "teacher sentence (+ previous turn)", "CC-BY-NC-SA"),
           "talkmoves_student": ("TalkMoves (Suresh et al. 2022)", "student sentence (+ previous turn)", "CC-BY-NC-SA"),
           "dunivin": ("Dunivin (2025) replication", "historical passage", "OSF")}


def table_datasets() -> None:
    rows = []
    for t in CONFIRMATORY + ["dunivin"]:
        td = triage.task_data(t)
        s = sampling.load(t, "main")
        lab = td.task.labels[td.task.labels.segment_id.isin(s.segment_id)]
        per = lab.groupby("segment_id").size()
        cb = td.codebook
        rows.append({"task": TASK[t], "source": SOURCES[t][0], "unit": SOURCES[t][1], "license": SOURCES[t][2],
                     "codebook": "multi-label (yes/no per code)" if cb.mode == "multi" else "mutually exclusive",
                     "codes": len(cb.code_ids), "calib segments": int((s.split == "calib").sum()), "test segments": int((s.split == "test").sum()),
                     "clusters": s.group.nunique(), "annotators/item (median)": float(per.median()),
                     "human α (multiply-coded items)": human_alpha(lab, cb)})
    save_table(pd.DataFrame(rows), "tab01_datasets", "Table 1. Datasets, codebooks and samples",
               "Exclusive codebooks include their 'none' option among the codes. α: Krippendorff's alpha on items coded by ≥ 2 annotators (macro over codes for multi-label).", 2)


CODERS = [  # signal, model, kind, settings, sample
    ("jev", "TypeSafe jev-1.13.0", "decision model", "Choice (exclusive) / Noul per code (multi)", "full"),
    ("oaid", "OpenAI Decisions API, gpt-6-luna", "decision model", "Jev's requests mapped to choice / predicate", "full"),
    ("luna-verbal", "OpenAI gpt-6-luna", "LLM", "reasoning effort none; verbal 0–100", "full"),
    ("luna-logprob", "OpenAI gpt-6-luna", "LLM", "answer-token logprobs, same call", "full"),
    ("luna_sc-vote", "OpenAI gpt-6-luna", "LLM", "5 samples at T = 1, vote share", "350/task"),
    ("haiku-verbal", "Anthropic claude-haiku-4-5", "LLM", "T = 0; verbal", "full"),
    ("oss-verbal", "openai/gpt-oss-120b (Together)", "LLM", "reasoning low; verbal", "full"),
    ("oss-logprob", "openai/gpt-oss-120b (Together)", "LLM", "answer-token logprobs", "full"),
    ("qwen-verbal", "Qwen/Qwen3.5-9B (Together)", "LLM", "thinking off; verbal", "full"),
    ("qwen-logprob", "Qwen/Qwen3.5-9B (Together)", "LLM", "thinking off; logprobs", "full"),
    ("agree3", "Luna + Haiku + gpt-oss", "ensemble", "vote share of three verbal runs", "full"),
    ("sonnet-verbal", "Anthropic claude-sonnet-5-5", "LLM (frontier)", "thinking off, effort low; verbal", "full"),
    ("qwen_big-verbal", "Qwen/Qwen3.8-2.4T-A95B (Together)", "LLM (frontier)", "thinking off; verbal", "270/task"),
    ("qwen_big-logprob", "Qwen/Qwen3.8-2.4T-A95B (Together)", "LLM (frontier)", "thinking off; logprobs", "270/task"),
    ("clef", "Cloudflare @cf/cloudflare/clef (27B)", "decision model", "Jev's exact requests", "full"),
    ("clef_flash", "Cloudflare @cf/cloudflare/clef-flash (9B)", "decision model", "Jev's exact requests", "full"),
    ("pplx", "Perplexity pplx-decider-v1.1-27b", "decision model", "Jev's exact requests", "full"),
    ("kev9b", "jaredpalmer/kev-9b (Modal, H100, bf16)", "decision model (open)", "Jev's exact requests", "full"),
    ("kev27b", "jaredpalmer/kev-27b (Modal, B200, bf16)", "decision model (open)", "Jev's exact requests", "full"),
]
BATCH = {"luna-verbal", "luna-logprob", "luna_sc-vote", "haiku-verbal", "sonnet-verbal"}  # OpenAI Batch / Anthropic Message Batches: 50% off


def table_coders() -> None:
    cost = read("a3_costs")
    cost = cost[cost.task.isin(CONFIRMATORY)].groupby("signal").usd_per_segment.mean() * 1000
    rows = [{"signal": SIG[s], "model": m, "kind": k, "settings": st, "sample": sm, "USD / 1k segments": cost.get(s, np.nan),
             "with batch API": cost.get(s, np.nan) / 2 if s in BATCH else np.nan} for s, m, k, st, sm in CODERS]
    save_table(pd.DataFrame(rows), "tab02_coders", "Table 2. Coders, confidence signals and realized API cost",
               "Cost: realized ledger cost of the canonical calls (list prices incl. prompt-cache discounts), mean over the five confirmatory tasks. "
               "Batch column assumes the 50% batch discount (OpenAI Batch, Anthropic Message Batches); '–' = no batch tier used or available.", 3)


def risk_by_rule(rule: str) -> pd.DataFrame:
    """Per task and signal: frozen-split coverage and realized error of a threshold rule at 5%."""
    rc = read("a3_risk_control")
    return rc[rc.task.isin(CONFIRMATORY) & (rc.eps == 0.05) & (rc.rule == rule)]


def table_rq1() -> None:
    s = read("rq1_signals")
    s = s[s.task.isin(CONFIRMATORY)]
    m = s.groupby("signal").mean(numeric_only=True)
    rc = read("a3_risk_control")
    rc = rc[(rc.task == "POOLED") & (rc.eps == 0.05)].pivot_table(index="signal", columns="rule", values="test_coverage")
    ltt = risk_by_rule("ltt")
    order = [c[0] for c in CODERS[:1]] + ["oaid", "jev_noul", "oaid_noul"] + [c[0] for c in CODERS[2:]] + \
        [f"{d}_noul" for d in ["clef", "clef_flash", "pplx", "kev9b", "kev27b"]]
    rows = []
    for sig in order:
        if sig not in m.index:
            continue
        r = m.loc[sig]
        sample = {c[0]: c[4] for c in CODERS}.get(sig, "full")
        rows.append({"signal": SIG[sig], "tasks": int(s[s.signal == sig].task.nunique()), "sample": sample, "κ (weighted)": r.kappa_w,
                     "accuracy": r.accuracy_w, "ECE": r.ece, "error AUROC": r.error_auroc, "AURC": r.aurc,
                     "cov@5% (naive)": r["cov@5%"], "realized error": pooled_risk(s[s.signal == sig], "cov@5%", "risk@5%"),
                     "cov@5% (oracle)": rc.loc[sig, "oracle"] if sig in rc.index else np.nan,
                     "cov@5% (LTT, decision-level)": rc.loc[sig, "ltt_dec"] if sig in rc.index else np.nan,
                     "cov@5% (LTT, cluster-level)": rc.loc[sig, "ltt"] if sig in rc.index else np.nan,
                     "its error": pooled_risk(ltt[ltt.signal == sig], "test_coverage", "test_risk"),
                     "share conf ≥ 0.99": r["share_conf>=0.99"]})
    save_table(pd.DataFrame(rows), "tab03_rq1_signals", "Table 3. RQ1: agreement, calibration and triage on the test split (mean over tasks)",
               "Means are over the tasks listed (yes/no-format arms exist only for the four exclusive tasks), so compare rows with the same task set. "
               "κ uses design weights. cov@5% (naive): threshold with calib error ≤ 5%, applied to test (primary analysis); realized error = error among all accepted "
               "test decisions (tasks weighted equally). Oracle: threshold fitted on test (risk-matched upper bound). LTT, decision-level: binomial tests that "
               "treat decisions as independent; LTT, cluster-level: valid for exchangeable clusters (comments, transcripts, lessons), δ = 0.05.")


def table_claims() -> None:
    c = read("rq1_contrasts")
    c = c[(c.task == "POOLED") & (c.a == "jev")]
    v10 = ["luna-verbal", "haiku-verbal", "oss-verbal", "luna-logprob", "oss-logprob", "agree3", "luna_sc-vote", "sonnet_sub-verbal"]
    rows = []
    for fam, comps in [("v1.0 confirmatory: Jev − comparator", v10), ("v1.1 A1 confirmatory: Jev − OpenAI Decisions", ["oaid"]),
                       ("v1.2 A5 confirmatory: Jev − other decision models", ["clef", "clef_flash", "pplx", "kev9b", "kev27b"]),
                       ("v1.1 A2 secondary: Jev − Qwen", ["qwen-verbal", "qwen-logprob"])]:
        part = []
        for b in comps:
            r = c[c.b == b]
            if r.empty:
                continue
            r = r.iloc[0]
            for k, lab in [("error_auroc", "error AUROC"), ("cov@5%", "coverage@5%")]:
                part.append({"family": fam, "comparison": f"Jev − {SIG[b]}", "metric": lab, "difference": r[f"{k}_diff"],
                             "95% CI": ci(r[f'{k}_lo'], r[f'{k}_hi']), "p (bootstrap)": r[f"{k}_p"], "_lo": r[f"{k}_lo"], "_hi": r[f"{k}_hi"]})
        part = pd.DataFrame(part)
        if part.empty:
            continue
        has_p = part["p (bootstrap)"].notna()
        part["p (Holm, within family)"] = np.nan
        if "confirmatory" in fam:
            part.loc[has_p, "p (Holm, within family)"] = holm(part.loc[has_p, "p (bootstrap)"].clip(lower=0.001)).to_numpy()
        excl = (part._lo > 0) | (part._hi < 0)  # unrounded endpoints, not the printed ones
        holm_ok = (part["p (Holm, within family)"] < 0.05) | ~has_p | ("confirmatory" not in fam)
        part["claim"] = np.where(excl & holm_ok, np.where(part.difference > 0, "Jev better", "Jev worse"), "no claim")
        rows.append(part)
    ext = c[c.b == "sonnet-verbal"]  # Sonnet on the full sample: an exploratory extension (deviation 9), outside the Holm family
    if not ext.empty:
        r = ext.iloc[0]
        rows.append(pd.DataFrame([{"family": "v1.0 extension (exploratory)", "comparison": "Jev − Sonnet verbal (full sample)", "metric": lab,
                                   "difference": r[f"{k}_diff"], "95% CI": ci(r[f'{k}_lo'], r[f'{k}_hi']), "p (bootstrap)": r[f"{k}_p"],
                                   "p (Holm, within family)": np.nan,
                                   "claim": ("Jev better" if r[f"{k}_lo"] > 0 else "Jev worse" if r[f"{k}_hi"] < 0 else "no claim")}
                                  for k, lab in [("error_auroc", "error AUROC"), ("cov@5%", "coverage@5%")]]))
    out = pd.concat(rows, ignore_index=True).drop(columns=["_lo", "_hi"], errors="ignore")
    for col in ["p (bootstrap)", "p (Holm, within family)"]:
        out[col] = [("<0.001" if v < 0.001 else f"{v:.3f}") if pd.notna(v) else "–" for v in out[col]]
    save_table(out, "tab04_claims", "Table 4. Confirmatory contrasts with Holm adjustment (pooled over the five tasks)",
               "Bootstrap p: two-sided, 1,000 clustered replicates with thresholds refitted (p < 0.001 shown as 0.001 before adjustment). "
               "Intervals resample each corpus's clusters jointly for related tasks. A claim needs the pooled 95% CI to exclude 0 and Holm p < 0.05 "
               "within the family. Holm is part of the plan's rule for v1.2 and a sensitivity analysis added for v1.0 and v1.1 (deviation 10). Sonnet "
               "enters the v1.0 family as specified, on its nested subset; its full-sample extension (deviation 9) is reported separately, outside the family.")


def table_rq2() -> None:
    r = read("rq2")
    r = r[r.signal.isin(["jev", "oaid", "clef", "clef_flash", "pplx", "kev9b", "kev27b", "luna-verbal", "haiku-verbal", "oss-verbal",
                         "sonnet-verbal", "luna-logprob", "agree3"])]
    out = r.assign(task=r.task.map(TASK), signal=r.signal.map(SIG))
    out["annotators below model"] = [f"{int(b)} of {int(n)}" for b, n in zip(out.beats_n_annotators, out.annotators)]
    out = out[["task", "signal", "spearman", "auroc_split", "model_kappa_mean", "annotators below model"]]
    out.columns = ["task", "signal", "Spearman: uncertainty vs human split", "AUROC: flags split items", "model κ (mean over references)",
                   "annotators below model"]
    save_table(out, "tab05_rq2", "Table 5. RQ2: model uncertainty and human disagreement, by coder (test items with ≥ 3 annotators)",
               "Model κ: against the majority of the other annotators, averaged over the leave-one-annotator-out references. The human range for each "
               "task is in [[tab:tab05b_rq2_humans]].", width=12)
    h = r.drop_duplicates("task")[["task", "segments", "annotators", "human_kappa_min", "human_kappa_median", "human_kappa_max"]]
    h = h.assign(task=h.task.map(TASK))
    h.columns = ["task", "items", "annotators", "human κ min", "human κ median", "human κ max"]
    save_table(h, "tab05b_rq2_humans", "Table 5b. RQ2: agreement of each annotator with the majority of the other annotators (the human range)",
               "Leave-one-annotator-out κ on the same items as [[tab:tab05_rq2]]. AnnoMI's multi-annotated items come from four transcripts.")


def table_rq3() -> None:
    r = read("rq3")
    g = r.groupby(["signal", "arm"]).agg(tasks=("task", "nunique"), flip=("flip_rate", "mean"), d_aurc=("d_aurc", "mean"),
                                         d_kappa=("d_kappa", "mean"), d_auroc=("d_error_auroc", "mean"), d_ece=("d_ece", "mean"),
                                         flips_at_low_conf=("flips_track_uncertainty_auroc", "mean")).reset_index()
    g["signal"] = g.signal.map(SIG)
    g["arm"] = g.arm.map(lambda a: ARM.get(a, a).replace("\n", " "))
    g.columns = ["signal", "arm", "tasks", "flip rate", "ΔAURC", "Δκ", "Δ error AUROC", "ΔECE", "flips at low confidence (AUROC)"]
    save_table(g, "tab06_rq3", "Table 6. RQ3: robustness to perturbations (design-weighted within tasks; equal-weight mean over the tasks each arm covers; "
               "vs the canonical arm on identical items)",
               "Flip rate and ΔAURC are the pre-specified primary outcomes; [[tab:tab06b_rq3_intervals]] gives per-task 95% intervals for selected arms. "
               "'No previous turn' removes information rather than rewording the request.")
    sel = r[r.arm.isin(["reversed", "no_context"]) & r.signal.isin(["jev", "oaid", "clef_flash", "pplx", "kev9b", "sonnet-verbal"])].copy()
    sel["signal"], sel["task"] = sel.signal.map(SIG), sel.task.map(TASK)
    sel["arm"] = sel.arm.map(lambda a: ARM.get(a, a).replace("\n", " "))
    sel["flip rate [95% CI]"] = [f"{a:.3f} [{lo:.3f}, {hi:.3f}]" for a, lo, hi in zip(sel.flip_rate, sel.flip_lo, sel.flip_hi)]
    sel["ΔAURC [95% CI]"] = [f"{a:+.3f} [{lo:+.3f}, {hi:+.3f}]" for a, lo, hi in zip(sel.d_aurc, sel.d_aurc_lo, sel.d_aurc_hi)]
    save_table(sel[["arm", "signal", "task", "flip rate [95% CI]", "ΔAURC [95% CI]"]].sort_values(["arm", "signal", "task"]),
               "tab06b_rq3_intervals", "Table 6b. RQ3 primary outcomes with per-task 95% intervals (cluster bootstrap) for two arms",
               "Positive ΔAURC means a larger area under the risk–coverage curve for the perturbed run (worse selective risk); AURC "
               "reflects the error rate as well as the ranking of errors.")


def table_triage_rules() -> None:
    rec_tasks = read("a3_recalibration")
    rec_tasks = rec_tasks[rec_tasks.task.isin(CONFIRMATORY) & (rec_tasks.method == "raw")]
    rec = read("a3_recalibration")
    rec = rec[rec.task == "POOLED"].pivot_table(index="signal", columns="method", values=["ece", "error_auroc"])
    sigs = ["jev", "oaid", "clef", "clef_flash", "pplx", "kev9b", "kev27b", "luna-verbal", "haiku-verbal", "oss-verbal", "qwen-verbal",
            "luna-logprob", "oss-logprob", "qwen-logprob", "sonnet-verbal", "qwen_big-verbal"]
    rows = []
    for s in sigs:
        if s not in rec.index:
            continue
        rows.append({"signal": SIG[s], "ECE raw": rec.loc[s, ("ece", "raw")], "ECE Platt": rec.loc[s, ("ece", "platt")],
                     "ECE isotonic": rec.loc[s, ("ece", "isotonic")], "ECE temperature": rec.loc[s, ("ece", "temperature")],
                     "error AUROC raw": rec.loc[s, ("error_auroc", "raw")], "error AUROC Platt": rec.loc[s, ("error_auroc", "platt")],
                     "share accepted by '≥ 0.95'": rec_tasks[rec_tasks.signal == s]["rule0.95_cov"].mean(),
                     "its error": pooled_risk(rec_tasks[rec_tasks.signal == s], "rule0.95_cov", "rule0.95_risk")})
    rc7 = pd.DataFrame(rows)
    c7 = list(rc7.columns)
    save_table(rc7, "tab07_triage_rules", "Table 7. EQ1 (exploratory): recalibration and the fixed rule 'accept if confidence ≥ 0.95' (pooled)",
               "Maps fitted per task on calib. Platt and isotonic maps act on the decision's confidence; temperature scaling acts on the full distribution and "
               "is sensitive to how exact 0s and 1s are floored ([[tab:tab24_temperature_floor]]; here 1e-12). Errors are among accepted test decisions.",
               panels=[("A. Expected calibration error, raw and after recalibration", c7[:5]),
                       ("B. Error ranking, and the fixed rule 'accept if confidence ≥ 0.95' applied to raw confidence", [c7[0]] + c7[5:])], width=14)
    rc = read("a3_risk_control")
    rc = rc[(rc.task == "POOLED") & (rc.eps == 0.05)].pivot_table(index="signal", columns="rule", values="test_coverage")
    rs = read("a3_resplits")
    rs = rs[rs.task == "mfrc"].pivot_table(index="signal", columns="rule", values="violation_rate")
    pc = read("a3_per_code")
    pc = pc[(pc.task == "mfrc") & (pc.method == "per-code calibrated ranking")].set_index("signal")
    emp, ltt = risk_by_rule("empirical"), risk_by_rule("ltt")
    rows = []
    for s in sigs:
        if s not in rc.index:
            continue
        v = lambda rule: rs.loc[s, rule] if s in rs.index else np.nan  # noqa: E731
        rows.append({"signal": SIG[s], "naive: accepted": rc.loc[s, "empirical"], "naive: error": pooled_risk(emp[emp.signal == s], "test_coverage", "test_risk"),
                     "corpus LTT: accepted": rc.loc[s, "ltt"], "corpus LTT: error": pooled_risk(ltt[ltt.signal == s], "test_coverage", "test_risk"),
                     "oracle: accepted": rc.loc[s, "oracle"],
                     "MFRC re-splits > 5%: naive": v("empirical"), "decisions LTT": v("ltt_dec"), "i.i.d.-cluster LTT": v("ltt_iid"), "corpus LTT": v("ltt"),
                     "MFRC per-code maps: corpus LTT accepted": pc.loc[s, "ltt_coverage"] if s in pc.index else np.nan})
    tr = pd.DataFrame(rows)
    tr.iloc[:, 1:] = tr.iloc[:, 1:] * 100
    tr.columns = ["signal", "naive: accepted", "naive: error", "corpus LTT: accepted", "corpus LTT: error", "oracle: accepted",
                  "naive", "LTT on decisions", "LTT on i.i.d. clusters", "corpus LTT", "per-code maps, corpus LTT: accepted"]
    save_table(tr, "tab07c_threshold_rules", "Table 8. EQ2 (exploratory): threshold rules at a 5% target (percent)",
               "Panel A: frozen split, equal-weight means over the five tasks; oracle: fitted on test. Panel B: 500 random cluster re-splits of MFRC; "
               "for corpus LTT each test split is the uncoded remainder of its re-split, so exceedance there is a direct check of its guarantee. "
               "Per-code maps are fitted on half of the calib clusters, frozen, certified on the other half and applied unchanged to test.", 1,
               panels=[("A. Frozen split, pooled over tasks", ["signal", "naive: accepted", "naive: error", "corpus LTT: accepted",
                                                               "corpus LTT: error", "oracle: accepted"]),
                       ("B. MFRC: share of re-splits with held-out error above 5%, and corpus LTT with per-code maps",
                        ["signal", "naive", "LTT on decisions", "LTT on i.i.d. clusters", "corpus LTT", "per-code maps, corpus LTT: accepted"])],
               width=14)
    r = read("a3_resplits")
    r = r[r.task.isin(CONFIRMATORY) & r.signal.isin(["jev", "oaid", "clef", "clef_flash", "pplx", "kev9b", "kev27b", "sonnet-verbal"])]
    t = r.pivot_table(index=["signal", "task"], columns="rule", values=["mean_coverage", "share_zero_coverage", "violation_rate"])
    for name, rules, title in [("tab07b_resplits_by_task", [("empirical", "naive"), ("ltt_dec", "LTT on decisions")],
                                 "Table 7b. Naive rule and LTT on decisions by task over 500 cluster re-splits (percent)"),
                                ("tab07d_resplits_guaranteed", [("ltt_iid", "i.i.d.-cluster LTT"), ("ltt", "corpus LTT")],
                                 "Table 7d. LTT on i.i.d. clusters and corpus LTT by task over 500 cluster re-splits (percent)")]:
        out = pd.DataFrame({"signal": [SIG[i[0]] for i in t.index], "task": [TASK[i[1]] for i in t.index]})
        for rule, lab in rules:
            out[f"{lab}: accepted"] = 100 * t[("mean_coverage", rule)].to_numpy()
            out[f"{lab}: none accepted"] = 100 * t[("share_zero_coverage", rule)].to_numpy()
            out[f"{lab}: error > 5%"] = 100 * t[("violation_rate", rule)].to_numpy()
        save_table(out, name, title, "Accepted: mean share of test decisions accepted; none accepted: share of re-splits accepting nothing; error > 5%: share "
                   "of re-splits whose held-out error exceeds 5%. A re-split that accepts nothing cannot exceed 5%; read exceedance with acceptance.", 1,
                   width=11)


def pretty_cascade(name: str) -> str:
    """'agree:jev+kev27b -> human' -> 'Jev + Kev-27B agree → human'."""
    def one(part: str) -> str:
        if part.startswith("agree:"):
            return " + ".join(SIG.get(x, x) for x in part[6:].split("+")) + " agree"
        return SIG.get(part, part)
    return " → ".join(one(x) for x in name.split(" -> "))


def table_cascades() -> None:
    c = read("a3_cascades")
    keep = ["jev -> human", "oaid -> human", "clef -> human", "clef_flash -> human", "pplx -> human", "kev9b -> human", "kev27b -> human",
            "dm_committee -> human", "agree:pplx+kev27b -> human", "agree:jev+kev27b -> human", "kev27b -> sonnet-verbal -> human",
            "luna-verbal -> human", "haiku-verbal -> human", "sonnet-verbal -> human",
            "jev -> haiku-verbal -> human", "haiku-verbal -> jev -> human", "agree:jev+haiku-verbal -> human", "agree:jev+luna-verbal -> human",
            "jev -> sonnet-verbal -> human", "agree:jev+sonnet-verbal -> human", "agree:haiku-verbal+sonnet-verbal -> human"]
    full = c[(c.pool == "full") & (c.reference_for_eval == "human")]
    pooled_rows = full[full.task == "POOLED"]
    tasks = full[full.task.isin(CONFIRMATORY)]
    mfrc = full[full.task == "mfrc"]
    adj = c[(c.task == "mfrc") & (c.pool == "full") & (c.reference_for_eval == "adjudicated") & (c.rule == "ltt")].set_index("cascade")
    val = lambda frame, k, rule: frame[frame.rule == rule].set_index("cascade")[k].reindex(keep).to_numpy()  # noqa: E731
    err = lambda rule: [pooled_risk(tasks[(tasks.cascade == k) & (tasks.rule == rule)], "automation", "auto_risk") for k in keep]  # noqa: E731
    names = [pretty_cascade(k) for k in keep]
    usd = lambda v: [f"{x:.3f}" for x in v]  # noqa: E731
    a = pd.DataFrame({"configuration": names,
                      "naive: accepted": 100 * val(pooled_rows, "automation", "empirical"), "naive: error": 100 * np.array(err("empirical")),
                      "naive: USD/1k": usd(val(pooled_rows, "api_usd_per_1k_segments", "empirical")),
                      "oracle: accepted": 100 * val(pooled_rows, "automation", "oracle"),
                      "oracle: USD/1k": usd(val(pooled_rows, "api_usd_per_1k_segments", "oracle"))})
    save_table(a, "tab08_cascades", "Table 9. A3 (exploratory): configurations under the naive and oracle rules, pooled over the five tasks",
               "Accepted: percent of decisions auto-accepted; error: percent wrong among accepted decisions (tasks weighted equally); USD/1k: API cost "
               "per 1,000 segments under that policy. 'A + B agree' accepts where both give the same label (confidence = the lower).", 1)
    m_err = [mfrc[(mfrc.cascade == k) & (mfrc.rule == "ltt")].auto_risk.mean() for k in keep]
    human_only = c[(c.task == "mfrc") & (c.pool == "full") & (c.reference_for_eval == "adjudicated")].human_only_error_vs_adjudicated.iloc[0]
    n_cal = int(triage.load("mfrc", "jev").query("split == 'calib'").group.nunique())
    n_test = mfrc[mfrc.rule == "ltt"].set_index("cascade").n_test_decisions.reindex(keep).to_numpy(float) / 8
    still = val(mfrc, "human_segments", "ltt")
    b = pd.DataFrame({"configuration": names,
                      "certified: accepted": 100 * val(mfrc, "automation", "ltt"), "error": 100 * np.array(m_err),
                      "USD per 1k comments": usd(val(mfrc, "api_usd_per_1k_segments", "ltt")),
                      "remainder comments still read": 100 * still,
                      "all comments still read": 100 * (n_cal + still * n_test) / (n_cal + n_test),
                      "overall error vs adjudicated": 100 * adj.final_error_vs_adjudicated.reindex(keep).to_numpy()})
    save_table(b, "tab08b_cascades_certified", "Table 10. A3 (exploratory): configurations certified by corpus LTT on MFRC (percent; the only task where anything "
               "is certified when whole clusters are hand-coded)",
               "Multi-stage configurations are certified jointly over the pipeline. Accepted, error and remainder comments still read refer to the uncoded "
               "remainder (the test split); a comment is still read if at least one of its decisions is left for a person. All comments still read adds "
               f"the {n_cal} hand-coded calibration comments to the numerator and denominator. "
               f"Overall error vs adjudicated: certified decisions scored against the adjudicated reference, the rest taken as human-coded; human-only coding "
               f"of MFRC has {100 * human_only:.1f}% error against that reference.", 1)


def table_cascade_contrasts() -> None:
    """One row per contrast: naive and oracle pooled over the five tasks, corpus LTT on MFRC (where it certifies)."""
    c = read("a3_cascade_contrasts")
    c = c[((c.task == "POOLED") & c.rule.isin(["empirical", "oracle"])) | ((c.task == "mfrc") & (c.rule == "ltt"))]
    c = c[c.pool == "full"]
    cell = lambda r: f"{100 * r.automation_diff:+.1f} [{100 * r.automation_lo:+.1f}, {100 * r.automation_hi:+.1f}]"  # noqa: E731
    rows = []
    for (a, b), g in c.groupby(["a", "b"], sort=False):
        by = g.set_index("rule")
        row = {"a": pretty_cascade(a).replace(" → human", ""), "b": pretty_cascade(b).replace(" → human", "")}
        for rule, lab in [("empirical", "naive"), ("oracle", "oracle"), ("ltt", "corpus LTT")]:
            row[lab] = cell(by.loc[rule]) if rule in by.index else "–"
        row["USD per 1k, a − b"] = by.loc["ltt", "api_usd_per_1k_segments_diff"] if "ltt" in by.index else np.nan
        rows.append(row)
    save_table(pd.DataFrame(rows), "tab09_cascade_contrasts",
               "Table 9. A3 (exploratory): paired configuration contrasts, accepted share a − b in percentage points [95% CI]",
               "1,000 clustered bootstrap replicates with thresholds refitted; related tasks share cluster draws. Naive and oracle: equal-weight means over the "
               "five tasks. Corpus LTT: MFRC only, the one task where it certifies anything when whole clusters are hand-coded. Every configuration ends with a person coding what it does not "
               "accept; 'A + B agree' accepts where both give the same label.", 2, width=16,
               panels=[("A. Naive rule and oracle, pooled over the five tasks", ["a", "b", "naive", "oracle"]),
                       ("B. Corpus LTT on MFRC", ["a", "b", "corpus LTT", "USD per 1k, a − b"])])


def table_per_code() -> None:
    g = read("a3_per_code_gain")
    g = g[g.task == "POOLED"][["comparison", "auroc_a", "auroc_b", "diff", "lo", "hi"]]
    g["comparison"] = g.comparison.map(lambda c: " ".join(SIG.get(t.rstrip(":"), t.rstrip(":")) + (":" if t.endswith(":") else "")
                                                          if t.rstrip(":") in SIG and t != "per-code" else t for t in c.split(" ")))
    g.columns = ["comparison", "AUROC a", "AUROC b", "a − b", "95% CI low", "95% CI high"]
    save_table(g, "tab10_per_code_recalibration", "Table 10. A3 (exploratory): per-code recalibration of confidence (error AUROC on test, pooled)",
               "Per-code Platt maps fitted on calib (codes with ≥ 50 calib decisions; others use the task map). CIs resample test clusters with the "
               "fitted maps held fixed, so they are conditional on those maps.")


def table_repeatability() -> None:
    path = RESULTS / "repeatability.csv"
    if not path.exists():
        return
    r = pd.read_csv(path)
    r = r[r.task == "POOLED"].drop(columns=["task"])
    r["signal"] = r.signal.map(SIG)
    r["flip_rate"] = r.flip_rate.map(rate)
    r = r.rename(columns={"flip_rate": "label flip rate", "identical_p": "share identical", "mean_abs_dp": "mean |Δp|",
                          "mean_abs_dconf": "mean |Δ confidence|", "accuracy_1": "accuracy, run 1", "accuracy_2": "accuracy, run 2",
                          "flips_at_low_conf_auroc": "flips at low confidence (AUROC)"})
    for c in ["segments", "decisions"]:
        if c in r:
            r[c] = r[c].astype(int)
    cols = list(r.columns)
    stab = [c for c in cols if c in ("signal", "segments", "decisions", "label flip rate", "share identical", "mean |Δp|", "mean |Δ confidence|")]
    rest = ["signal"] + [c for c in cols if c not in stab]
    save_table(r, "tab11_repeatability", "Table 11. Run-to-run repeatability (identical requests sent twice; 100 test segments per task; mean over tasks)",
               "Flips at low confidence: AUROC of low confidence for the decisions whose label changed (– : no changes).",
               panels=[("A. How much the outputs change", stab), ("B. Accuracy in each run, and where the changes fall", rest)], width=12)


def table_adjudication() -> None:
    a = read("rq5_adjudication_categories")
    a = a[a.task.str.startswith("ALL")].copy()
    a["verdicts"] = a.task.map({"ALL": "all", "ALL (panel verdicts)": "LLM panel", "ALL (author verdicts)": "author"})
    a = a.drop(columns=["task"])[["coder", "verdicts"] + [c for c in a.columns if c not in ("coder", "verdicts", "task")]]
    a["coder"] = a.coder.map(lambda s: SIG.get(s, SIG.get(f"{s}-verbal", s)))
    a = a.rename(columns={"label_error (coder right, human wrong)": "coder right, human wrong", "model_error (human right, coder wrong)": "human right, coder wrong",
                          "both_defensible (under-specified)": "both defensible (codebook under-specified)", "neither_right": "neither right"})
    save_table(a, "tab12_adjudication", "Table 12. Who was right when a coder disagreed with the human label (adjudicated sample)",
               "Shares of adjudicated disagreements on a high-confidence-enriched sample drawn from six label sources (Jev in both formats, Luna, Haiku, "
               "gpt-oss and Sonnet's subset run); the four coders shown have enough disagreements to break down. The panel resolved the items on which its "
               "two members agreed; the author resolved 36 items the panel disputed plus a 12-item audit of panel-agreed items (deviations 2–3).")


def table_same_base() -> None:
    path = RESULTS / "same_base_contrasts.csv"
    if not path.exists():
        return
    c = pd.read_csv(path)
    c = c[c.task == "POOLED"]
    if c.empty:
        return
    rows = []
    for _, r in c.iterrows():
        for k, lab in [("error_auroc", "error AUROC"), ("cov@5%", "coverage@5%")]:
            rows.append({"comparison": f"{SIG.get(r.a, r.a)} − {SIG.get(r.b, r.b)}", "metric": lab, "difference": r[f"{k}_diff"],
                         "95% CI": ci(r[f'{k}_lo'], r[f'{k}_hi']), "p (bootstrap)": r.get(f"{k}_p", np.nan)})
    save_table(pd.DataFrame(rows), "tab14_same_base", "Table 14. Same base model (Qwen3.5-9B): decision-model training vs the base LLM's own confidence (pooled)",
               "Pre-specified secondary (amendment v1.2). Our Qwen arm is the instruct model with thinking off; Kev-9B uses the base checkpoint; Clef-flash's exact base variant is undocumented.")


def fig_decision_models() -> None:
    s = read("rq1_signals")
    s = s[s.task.isin(CONFIRMATORY)]
    sigs = [x for x in ["jev", "oaid", "clef", "clef_flash", "pplx", "kev27b", "kev9b", "qwen-verbal", "qwen-logprob", "sonnet-verbal"] if x in set(s.signal)]
    if len(sigs) < 4:
        return
    m = s.groupby("signal")[["error_auroc", "kappa", "ece"]].mean().reindex(sigs)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, k, lab in [(axes[0], "error_auroc", "error AUROC (ranking)"), (axes[1], "kappa", "κ (agreement)"), (axes[2], "ece", "ECE (lower is better)")]:
        ax.barh(range(len(sigs)), m[k], color=[COLOR.get(x, "#777") for x in sigs])
        for i, v in enumerate(m[k]):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=7)
        ax.set_title(lab); ax.set_xlim(0, max(m[k].max() * 1.15, 0.1))
    axes[0].set_yticks(range(len(sigs)), [SIG.get(x, x) for x in sigs]); axes[0].invert_yaxis()
    save_fig(fig, "fig13_decision_models")


MODEL_NAME = {"luna": "Luna", "haiku": "Haiku", "sonnet": "Sonnet", "oss": "gpt-oss-120b", "qwen": "Qwen3.5-9B", "qwen_big": "Qwen3.8-2.4T"}


def table_contamination() -> None:
    path = RESULTS / "contamination_summary.csv"
    if not path.exists():
        return
    c = pd.read_csv(path)
    comp = c[c.probe == "completion"].assign(model=lambda d: d.model.map(MODEL_NAME))
    comp = comp[["model", "set", "items", "guided", "general", "difference", "p_bonferroni", "verbatim_guided", "verbatim_general", "max_run_guided", "flag"]]
    for col in ["items", "verbatim_guided", "verbatim_general", "max_run_guided"]:
        comp[col] = comp[col].astype(int)  # counts, not rates
    comp.columns = ["model", "dataset", "items", "ROUGE-L, named", "ROUGE-L, unnamed", "difference", "p (Bonferroni)",
                    "near-verbatim items, named", "near-verbatim items, unnamed", "longest run, named (words)", "flagged"]
    cols = list(comp.columns)
    save_table(comp[cols[:7]], "tab15_memorization_completion",
               "Table 15. Memorization probe: completing the second half of dataset items, overlap with the true continuation (amendment v1.3)",
               "Named: the dataset is named and the exact original text requested. Unnamed: plain continuation. Difference: named − unnamed mean ROUGE-L; "
               "p: one-sided Wilcoxon signed-rank test for named > unnamed, Bonferroni-adjusted. Near-verbatim matches are in [[tab:tab15b_memorization_verbatim]].", width=12)
    save_table(comp[cols[:3] + cols[7:]], "tab15b_memorization_verbatim",
               "Table 15b. Memorization probe: near-verbatim reproduction (counts of items, amendment v1.3)",
               "Near-verbatim: a run of ≥ 10 identical words covering ≥ 50% of the true continuation; counts of the 100 items. Longest run: the longest "
               "identical run under the named prompt only. Flagged: the pre-specified rule (Bonferroni p < 0.05 for named > unnamed in "
               "[[tab:tab15_memorization_completion]], or ≥ 2 near-verbatim items under the named prompt). Matches include a predictable counting "
               "sequence (TalkMoves) and Du Bois quotations (Dunivin).", width=12)
    lab = c[c.probe == "label"].assign(model=lambda d: d.model.map(MODEL_NAME), set=lambda d: d.set.map(TASK))
    lab = lab[["model", "set", "items", "guided", "general", "difference", "p_bonferroni", "flag"]]
    lab.columns = ["model", "task", "items", "accuracy guided", "accuracy general", "difference", "p (Bonferroni)", "evidence of label leakage"]
    save_table(lab, "tab16_memorization_labels", "Table 16. Memorization probe: recalling annotated labels without the codebook (amendment v1.3)",
               "Accuracy against the majority label (Jaccard for MFRC). Guided names the dataset and asks what its annotators assigned.")


def table_contamination_followups() -> None:
    d, f = RESULTS / "contamination_decoy.csv", RESULTS / "contamination_followup.csv"
    if not (d.exists() and f.exists()):
        return
    dec = pd.read_csv(d).merge(pd.read_csv(f), on=["model", "task"], how="left")
    dec["model"], dec["task"] = dec.model.map(MODEL_NAME), dec.task.map(TASK)
    cols = list(dec.columns)
    save_table(dec, "tab17_memorization_followups", "Table 17. Flagged label-probe cells: decoy-name control and coding accuracy on 'recalled' items",
               "Decoy: the same dataset description with a fictitious name (deviation 14). Memory is supported only if the real name beats the decoy. "
               "Recalled items: guided better than no name. Coding accuracy is the model's RQ1 accuracy on those items.",
               panels=[("A. Label recall with the real name, a decoy name and no name", cols[:9]),
                       ("B. Coding accuracy on 'recalled' and other items", cols[:2] + cols[9:])], width=12)


def fig_contamination() -> None:
    path = RESULTS / "contamination_summary.csv"
    if not path.exists():
        return
    c = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={"wspace": 0.25})
    for ax, probe, col, title in [(axes[0], "completion", "set", "completion: ROUGE-L guided − general"),
                                  (axes[1], "label", "set", "labels: accuracy guided − general")]:
        piv = c[c.probe == probe].pivot_table(index="model", columns=col, values="difference").reindex(list(MODEL_NAME))
        flags = c[c.probe == probe].pivot_table(index="model", columns=col, values="flag", aggfunc="max").reindex(list(MODEL_NAME))
        im = ax.imshow(piv.to_numpy(float), cmap="RdBu_r", vmin=-0.15, vmax=0.15, aspect="auto")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.iat[i, j]
                ax.text(j, i, ("" if pd.isna(v) else f"{v:+.3f}") + (" *" if flags.iat[i, j] else ""), ha="center", va="center", fontsize=7)
        ax.set_yticks(range(len(piv)), [MODEL_NAME[m] for m in piv.index])
        ax.set_xticks(range(piv.shape[1]), [TASK2.get(x, x) for x in piv.columns], fontsize=7)
        ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.02)
    save_fig(fig, "fig14_memorization")


SNAPSHOT = [  # coder, provider, requested model, run experiment/stem used to read the returned version string
    ("Jev", "typesafe", "jev-1.13.0", "jev_main", "jev__canonical"),
    ("OpenAI Decisions", "openai_decisions", "gpt-6-luna", "oaid_main", "oaid__canonical"),
    ("Clef", "cloudflare", "@cf/cloudflare/clef", "dm_main", "clef__canonical"),
    ("Clef-flash", "cloudflare", "@cf/cloudflare/clef-flash", "dm_main", "clef_flash__canonical"),
    ("pplx-decider", "perplexity", "pplx-decider-v1.1-27b", "dm_main", "pplx__canonical"),
    ("Kev-9B", "kev", "jaredpalmer/kev-9b", "dm_main", "kev9b__canonical"),
    ("Kev-27B", "kev", "jaredpalmer/kev-27b", "dm_main", "kev27b__canonical"),
    ("Luna", "openai", "gpt-6-luna", "llm_main", "luna-verbal__canonical"),
    ("Haiku", "anthropic", "claude-haiku-4-5", "llm_main", "haiku-verbal__canonical"),
    ("Sonnet", "anthropic", "claude-sonnet-5-5", "sonnet_main", "sonnet-verbal__canonical"),
    ("gpt-oss-120b", "together", "openai/gpt-oss-120b", "llm_main", "oss-verbal__canonical"),
    ("Qwen3.5-9B", "together", "Qwen/Qwen3.5-9B", "qwen_main", "qwen-verbal__canonical"),
    ("Qwen3.8-2.4T", "together", "Qwen/Qwen3.8-2.4T-A95B", "qwen_big_frontier", "qwen_big-verbal__canonical"),
]


def table_snapshot() -> None:
    """Every model as called: requested id, version string returned, and the UTC dates of its calls."""
    import sqlite3

    from margin.cache import DEFAULT_PATH
    from margin.run import RUNS

    if not Path(DEFAULT_PATH).exists():  # the response cache is not distributed; keep the committed table
        return
    db = sqlite3.connect(DEFAULT_PATH)
    rows = []
    for name, provider, model, exp, stem in SNAPSHOT:
        n, t0, t1 = db.execute("SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM responses WHERE model = ? AND (provider = ? OR provider LIKE ?)",
                               (model, provider, provider + "#%")).fetchone()
        versions = set()
        for path in (RUNS / exp).glob(f"*/{stem}.parquet"):
            versions |= set(pd.read_parquet(path, columns=["model_version"]).model_version.dropna().astype(str))
        day = lambda t: pd.Timestamp(t, unit="s", tz="UTC").strftime("%Y-%m-%d") if t else "–"  # noqa: E731
        rows.append({"coder": name, "provider": provider.replace("_", " "), "model requested": model,
                     "version returned": ", ".join(sorted(versions)) or "–", "calls (all arms)": int(n), "first call (UTC)": day(t0), "last call (UTC)": day(t1)})
    db.close()
    save_table(pd.DataFrame(rows), "tab18_model_snapshot", "Table 18. Model snapshot: what was called, which version answered, and when",
               "Several APIs were in public beta (OpenAI Decisions from 2026-10-06; decision models launched September–October 2026). "
               "Kev weights: Hugging Face revisions db029f0 (kev-9b) and af0e6d5 (kev-27b), the latest on the call dates; kev package commit fc4a17e. "
               "Results describe these versions on these dates; the released harness allows re-running against later versions.")


def table_prompt_sensitivity() -> None:
    path = RESULTS / "prompt_sensitivity.csv"
    if not path.exists():
        return
    p = pd.read_csv(path)
    p["signal"] = p.signal.map(SIG)
    p["task"] = p.task.map(TASK)
    keep = ["task", "signal", "decisions", "error_auroc_canonical", "error_auroc_dist", "error_auroc_diff", "lo", "hi",
            "kappa_canonical", "kappa_dist", "ece_canonical", "ece_dist"]
    p = p[keep].rename(columns={"error_auroc_canonical": "err. AUROC canonical", "error_auroc_dist": "err. AUROC distribution",
                               "error_auroc_diff": "difference", "kappa_canonical": "κ canonical", "kappa_dist": "κ distribution",
                               "ece_canonical": "ECE canonical", "ece_dist": "ECE distribution"})
    save_table(p, "tab19_prompt_sensitivity", "Table 19. LLM prompt sensitivity: verbalized distribution vs canonical prompt (amendment v1.4)",
               "Same test items (first 150 per task). Difference = distribution − canonical error AUROC, paired cluster bootstrap 95% CI; related tasks "
               "share cluster draws.",
               panels=[("A. Error ranking", list(p.columns[:8])),
                       ("B. Agreement and calibration", list(p.columns[:3]) + list(p.columns[8:]))], width=12)


# --- condensed main-text tables (paper/tables/main_*.tex) ---------------------------------------------

def main_tables() -> None:
    def tex(df, name, caption, note="", digits=3):
        (TAB / f"{name}.tex").write_text(tex_table(df, caption, f"tab:{name}", note, digits))
        df.to_csv(TAB / f"{name}.csv", index=False)

    d = pd.read_csv(TAB / "tab01_datasets.csv")
    d = d[["task", "unit", "codebook", "codes", "calib segments", "test segments", "clusters", "annotators/item (median)", "human α (multiply-coded items)"]]
    d.columns = ["task", "unit", "codebook", "codes", "calib", "test", "clusters", "annot./item", "human α"]
    d["unit"] = d.unit.str.replace(" (+ previous turn)", "", regex=False).str.replace("Reddit comment", "Reddit comment", regex=False)
    d["codebook"] = d.codebook.map({"multi-label (yes/no per code)": "multi-label", "mutually exclusive": "exclusive"})
    d["annot./item"] = d["annot./item"].astype(int).astype(str)
    d.loc[d.task.str.startswith("AnnoMI"), "annot./item"] = "1 (10 on a subset)"
    tex(d, "main_datasets", "Tasks, codebooks and samples. Calib and test are segments; clusters are conversations or transcripts (comments for MFRC). Dunivin is the replication task.",
        "Multi-label: one yes/no decision per code; exclusive: one choice among the codes (counts include any 'none' or 'other' option). Dialogue units include the previous turn as context. "
        "α: Krippendorff's alpha on multiply-coded items (macro over codes for multi-label); – where items are single-coded. RQ2 uses MFRC and AnnoMI's ten-annotator subset.", 2)

    cost = read("a3_costs")
    cost = cost[cost.task.isin(CONFIRMATORY)].groupby("signal").usd_per_segment.mean() * 1000
    models = [  # one row per underlying model: name, as called, kind, confidence signals, sample, cost key
        ("Jev", "TypeSafe jev-1.13.0", "decision model", "probability", "full", "jev"),
        ("OpenAI Decisions", "OpenAI Decisions API (gpt-6-luna)", "decision model", "probability", "full", "oaid"),
        ("Clef", "Cloudflare @cf/cloudflare/clef (27B)", "decision model", "probability", "full", "clef"),
        ("Clef-flash", "Cloudflare @cf/cloudflare/clef-flash (9B)", "decision model", "probability", "full", "clef_flash"),
        ("pplx-decider", "Perplexity pplx-decider-v1.1-27b", "decision model", "probability", "full", "pplx"),
        ("Kev-9B", "jaredpalmer/kev-9b, self-hosted (2 H100)", "open decision model", "probability", "full", "kev9b"),
        ("Kev-27B", "jaredpalmer/kev-27b, self-hosted (2 B200)", "open decision model", "probability", "full", "kev27b"),
        ("Luna", "OpenAI gpt-6-luna, reasoning none, T = 0", "LLM", "verbalized; token prob.; self-consistency (350/task, $0.28)", "full", "luna-verbal"),
        ("Haiku", "Anthropic claude-haiku-4-5, T = 0", "LLM", "verbalized", "full", "haiku-verbal"),
        ("gpt-oss", "openai/gpt-oss-120b (Together), reasoning low", "LLM", "verbalized; token prob.", "full", "oss-verbal"),
        ("Qwen3.5-9B", "Qwen/Qwen3.5-9B instruct (Together), thinking off", "LLM", "verbalized; token prob.", "full", "qwen-verbal"),
        ("Sonnet", "Anthropic claude-sonnet-5-5, thinking off", "LLM (frontier)", "verbalized", "full", "sonnet-verbal"),
        ("Qwen3.8-2.4T", "Qwen/Qwen3.8-2.4T-A95B (Together), thinking off", "LLM (frontier)", "verbalized; token prob.", "270/task", "qwen_big-verbal"),
        ("3-LLM agreement", "Luna + Haiku + gpt-oss verbal runs", "ensemble", "vote share", "full", "agree3"),
    ]
    c = pd.DataFrame([{"model": n, "kind": k_, "confidence signals": sg, "sample": sm, "USD / 1k segments": cost.get(key, np.nan)}
                      for n, a_, k_, sg, sm, key in models])
    tex(c, "main_coders", "Models and confidence signals, with realized API cost per 1,000 segments (October 2026 prices).",
        "Kev: estimated serving cost from observed throughput on Modal (Kev-9B 30 calls/s on 2 H100s at $3.95/h; Kev-27B 11 calls/s on 2 B200s "
        "at $6.25/h), excluding cold starts and idle time. Model identifiers, settings, versions returned and call dates are in Table TABSNAPSHOT (appendix).")
    t = TAB / "main_coders.tex"
    t.write_text(t.read_text().replace("Table TABSNAPSHOT", r"Table~\ref{tab:tab18_model_snapshot}"))

    s = pd.read_csv(TAB / "tab03_rq1_signals.csv")
    s = s[~s.signal.str.contains("yes/no")]
    groups = [["Jev", "OpenAI Decisions", "Clef (27B)", "Clef-flash (9B)", "pplx-decider", "Kev-9B", "Kev-27B"],
              ["Luna verbal", "Haiku verbal", "gpt-oss verbal", "Qwen3.5-9B verbal", "Sonnet verbal", "Qwen3.8-2.4T verbal"],
              ["Luna logprob", "gpt-oss logprob", "Qwen3.5-9B logprob", "Qwen3.8-2.4T logprob"], ["3-LLM agreement", "Luna self-consistency"]]
    order = [g for grp in groups for g in grp if g in set(s.signal)]
    breaks = tuple(np.cumsum([len([g for g in grp if g in set(s.signal)]) for grp in groups])[:-1])
    s = s.set_index("signal").reindex(order).reset_index()
    a1 = s[["signal", "sample", "κ (weighted)", "ECE", "error AUROC", "AURC", "share conf ≥ 0.99"]]
    a1.columns = ["signal", "sample", "κ (weighted)", "ECE", "error AUROC", "AURC", "conf ≥ 0.99"]
    (TAB / "main_rq1.tex").write_text(tex_table(a1, "RQ1: agreement, calibration and error ranking on the test split (mean over the five tasks, design-weighted). "
        "Groups: decision models, verbalized LLM confidence, token probabilities, agreement signals.", "tab:main_rq1",
        "Error AUROC: probability that a correct decision is more confident than an incorrect one (ties count half). AURC: area under the "
        "risk-coverage curve, lower is better. conf ≥ 0.99: share of decisions with confidence ≥ 0.99.", 3, breaks))
    a1.to_csv(TAB / "main_rq1.csv", index=False)

    rc = read("a3_risk_control")
    rc = rc[rc.task.isin(CONFIRMATORY) & (rc.eps == 0.05)]
    dia = [t_ for t_ in CONFIRMATORY if t_ != "mfrc"]
    rows = []
    for sig in ["jev", "oaid", "clef", "clef_flash", "pplx", "kev9b", "kev27b", "luna-verbal", "haiku-verbal", "oss-verbal", "sonnet-verbal", "qwen_big-verbal"]:
        g = rc[rc.signal == sig]
        if g.empty:
            continue
        get = lambda rule, tasks, k="test_coverage": g[(g.rule == rule) & g.task.isin(tasks)]  # noqa: E731
        rows.append({"signal": SIG[sig],
                     "naive: accepted": get("empirical", CONFIRMATORY).test_coverage.mean(),
                     "naive: error": pooled_risk(get("empirical", CONFIRMATORY), "test_coverage", "test_risk"),
                     "oracle: accepted": get("oracle", CONFIRMATORY).test_coverage.mean(),
                     "MFRC: decision-level LTT": get("ltt_dec", ["mfrc"]).test_coverage.mean(),
                     "MFRC: cluster-level LTT": get("ltt", ["mfrc"]).test_coverage.mean(),
                     "its error": pooled_risk(get("ltt", ["mfrc"]), "test_coverage", "test_risk"),
                     "dialogue: naive": get("empirical", dia).test_coverage.mean(),
                     "dialogue: decision-level LTT": get("ltt_dec", dia).test_coverage.mean(),
                     "dialogue: cluster-level LTT": get("ltt", dia).test_coverage.mean()})
    pol = pd.DataFrame(rows)
    pol.columns = [c.replace("cluster-level LTT", "corpus LTT").replace("decision-level LTT", "LTT on decisions") for c in pol.columns]
    pct_cols = pol.columns[1:]
    pol[pct_cols] = pol[pct_cols] * 100
    (TAB / "main_policies.tex").write_text(tex_table(pol, "Threshold policies at a 5% error target on the frozen split (percent): share of test decisions "
        "accepted, and error among them. Pooled columns average the five tasks; dialogue columns average the four dialogue tasks.", "tab:main_policies",
        "Naive: threshold with calibration error ≤ 5% (the pre-specified rule). Oracle: fitted on test, an unattainable upper bound. LTT on decisions treats "
        "decisions as independent and is shown only for comparison; corpus LTT certifies the error of the uncoded remainder ([[sec:rules]]).",
        1, (7,)))
    pol.to_csv(TAB / "main_policies.csv", index=False)

    mp = pd.read_csv(RESULTS / "mfrc_prevalence.csv")
    piv = mp.pivot_table(index="signal", columns="rule", values=["accepted", "error_among_accepted", "accepted_absent_share",
                                                                  "present_recall_accepted", "present_sent_to_human"], sort=False)
    order = ["jev", "oaid", "clef", "clef_flash", "pplx", "kev9b", "kev27b", "luna-verbal", "haiku-verbal", "oss-verbal", "sonnet-verbal", "prevalence baseline"]
    piv = piv.reindex([o for o in order if o in piv.index])
    mf = pd.DataFrame({"coder": [SIG.get(i, i) for i in piv.index],
                       "naive: accepted": piv[("accepted", "empirical")].to_numpy(), "naive: error": piv[("error_among_accepted", "empirical")].to_numpy(),
                       "naive: present codes found": piv[("present_recall_accepted", "empirical")].to_numpy(),
                       "LTT: accepted": piv[("accepted", "ltt")].to_numpy(), "LTT: error": piv[("error_among_accepted", "ltt")].to_numpy(),
                       "LTT: share 'absent'": piv[("accepted_absent_share", "ltt")].to_numpy(),
                       "LTT: present codes found": piv[("present_recall_accepted", "ltt")].to_numpy(),
                       "LTT: present codes to human": piv[("present_sent_to_human", "ltt")].to_numpy()})
    mf.columns = [c.replace("present codes found", "applying codes found").replace("present codes to human", "applying codes deferred")
                  .replace("LTT:", "corpus LTT:") for c in mf.columns]
    mf[mf.columns[1:]] = mf[mf.columns[1:]] * 100
    (TAB / "main_mfrc.tex").write_text(tex_table(mf, "MFRC: what the accepted decisions are (percent). 11% of test decisions are codes that apply. The "
        "prevalence baseline reads no text: it predicts 'absent' for every code, with confidence equal to that code's share of 'absent' labels.",
        "tab:main_mfrc",
        "Accepted: share of test decisions; error: among them; share 'absent': accepted decisions that say a code does not apply; applying codes found: "
        "applying codes accepted with the right label, over all applying codes; deferred: applying codes left for a person. The baseline's naive scores are "
        "fitted on all calibration comments; for corpus LTT they are fitted on half and certified on the other half. Design-weighted.",
        1, (len(mf) - 1,)))
    mf.to_csv(TAB / "main_mfrc.csv", index=False)

    k = pd.read_csv(TAB / "tab04_claims.csv")
    fam = {"v1.0 confirmatory: Jev − comparator": "v1.0", "v1.1 A1 confirmatory: Jev − OpenAI Decisions": "v1.1",
           "v1.2 A5 confirmatory: Jev − other decision models": "v1.2", "v1.1 A2 secondary: Jev − Qwen": "v1.1 (sec.)",
           "v1.0 extension (exploratory)": "ext."}
    k["family"] = k.family.map(fam)
    verdict = {"Jev better": "Jev", "Jev worse": "X", "no claim": "–"}
    rows = []
    for (f, comp), g in k.groupby(["family", "comparison"], sort=False):
        a_, c_ = g[g.metric == "error AUROC"].iloc[0], g[g.metric == "coverage@5%"].iloc[0]
        rows.append({"family": f, "X": comp.replace("Jev − ", ""),
                     "Δ err. AUROC [95% CI]": f"{a_.difference:+.3f} {a_['95% CI']}", "Holm p": a_["p (Holm, within family)"], "better": verdict[a_.claim],
                     "Δ cov@5% [95% CI]": f"{c_.difference:+.3f} {c_['95% CI']}", "Holm p ": c_["p (Holm, within family)"], "better ": verdict[c_.claim]})
    k = pd.DataFrame(rows)
    fam_breaks = tuple(i for i in range(1, len(k)) if k.family.iloc[i] != k.family.iloc[i - 1])
    (TAB / "main_claims.tex").write_text(tex_table(k, "Confirmatory contrasts, Jev minus each comparator X, pooled over the five tasks (paired cluster bootstrap, "
        "thresholds refitted in every replicate). Positive values favour Jev. Coverage is the naive rule's share accepted.", "tab:main_claims",
        "'better' names the winner (Jev, or X for the comparator) when the pooled 95% CI excludes 0 and Holm p < 0.05 within the family; – : no claim. "
        "Holm is part of the plan's rule for v1.2 and a sensitivity analysis for v1.0 and v1.1. Sonnet enters v1.0 as specified, on its nested subset; "
        "the full-sample extension (ext.) is exploratory and outside the family. v1.1 secondary rows are unadjusted. Intervals resample each corpus's "
        "clusters jointly for related tasks; coverage intervals are asymmetric because refitted thresholds jump between tie blocks.", 3, fam_breaks))
    k.to_csv(TAB / "main_claims.csv", index=False)

    b = pd.read_csv(TAB / "tab14_same_base.csv")
    b["p (bootstrap)"] = ["<0.001" if v < 0.001 else f"{v:.3f}" for v in b["p (bootstrap)"]]
    tex(b, "main_same_base", "EQ4: two 9B decision models built on Qwen3.5-9B vs the Qwen3.5-9B instruct model's own confidence (pooled over tasks).",
        "Coverage is the naive rule's share accepted. Kev-9B starts from the base checkpoint and Clef-flash's exact variant is unspecified, so these "
        "contrasts compare related systems with different training and read-out, not a controlled training ablation.")

    rp = pd.read_csv(RESULTS / "repeatability.csv")
    rp = rp[rp.task == "POOLED"].set_index("signal").flip_rate
    r3 = read("rq3")
    order = r3[r3.arm.isin(["reversed", "shuffle1"])].groupby("signal").flip_rate.mean()
    mem = pd.read_csv(RESULTS / "contamination_summary.csv")
    dec = pd.read_csv(RESULTS / "contamination_decoy.csv")
    model_sig = {"luna": "luna-verbal", "haiku": "haiku-verbal", "sonnet": "sonnet-verbal", "oss": "oss-verbal", "qwen": "qwen-verbal", "qwen_big": "qwen_big-verbal"}
    rows = []
    for sig in ["jev", "oaid", "clef", "clef_flash", "pplx", "kev9b", "kev27b", "luna-verbal", "haiku-verbal", "sonnet-verbal", "oss-verbal", "qwen-verbal", "qwen_big-verbal"]:
        m = next((k for k, v in model_sig.items() if v == sig), None)
        comp = mem[(mem.probe == "completion") & (mem.model == m)] if m else None
        lab = mem[(mem.probe == "label") & (mem.model == m)] if m else None
        held = dec[(dec.model == m) & (dec["p (Bonferroni, 3 cells)"] < 0.05)] if m else None
        rows.append({"coder": SIG[sig], "run-to-run label flips": rate(rp.get(sig, np.nan)), "flips under option order": rate(order.get(sig, np.nan)),
                     "completion-probe flag": ("not flagged" if not comp["flag"].any() else "flagged") if m else "not assessed",
                     "label recall beyond decoy": ((", ".join(TASK[t] for t in held.task) or "no evidence") if lab is not None and lab["flag"].any() else "no evidence") if m else "not assessed"})
    ps = pd.read_csv(RESULTS / "prompt_sensitivity.csv").query("task == 'POOLED'").set_index("signal")
    psd = lambda s_: f"{num(ps.loc[s_, 'error_auroc_diff'], 3, True)} {ci(ps.loc[s_, 'lo'], ps.loc[s_, 'hi'], 3, True)}"  # noqa: E731
    tex(pd.DataFrame(rows), "main_validity", "EQ6: validity checks. Run-to-run flips: identical requests sent twice (100 test segments per task; 0 means no "
        "observed label changes in these repeats, <0.001 a nonzero rate below display precision); option-order flips: reversed and shuffled options (RQ3 "
        "arms); memorization probe with decoy-name control (amendment v1.3, deviation 14). Decision models return no text, so they could not be probed.",
        "Completion-probe flag: the pre-specified rule ([[tab:tab15b_memorization_verbatim]]); not flagged does not mean no literal matches, which "
        "occurred rarely under both prompts. Prompt sensitivity (amendment v1.4): a verbalized-distribution prompt changes error AUROC by "
        f"{psd('sonnet-verbal')} for Sonnet and {psd('haiku-verbal')} for Haiku.")


def table_ltt_checks() -> None:
    sim = read("ltt_simulation")
    best = sim[sim.rule == "best safe threshold"].mean_coverage.iloc[0]
    sim = sim[sim.rule != "best safe threshold"].copy()
    sim["rule"] = sim.rule.map({"ltt_dec": "decisions", "ltt_iid": "i.i.d. clusters", "ltt": "corpus"})
    sim = sim.astype({"clusters": int, "cluster_size": int})[["clusters", "cluster_size", "copula_loading", "error_correlation", "rule",
                                                                "p_remainder_risk_above_eps", "p_population_risk_above_eps", "mean_coverage"]]
    sim.columns = ["clusters in corpus", "decisions per cluster", "copula loading", "within-cluster error correlation", "LTT on",
                   "P(remainder risk > 5%)", "P(population risk > 5%)", "mean share accepted"]
    save_table(sim, "tab20_ltt_simulation", "Table 20. LTT on synthetic clustered corpora (300 simulations per row, 30% of clusters hand-coded, δ = 0.05)",
               f"Calibrated confidence conf = 1 − 0.5·Beta(1, 6); errors correlated within a cluster through a Gaussian copula (the loading is not the error "
               f"correlation, which is shown). Remainder: the uncoded clusters of the corpus; population: closed form. The best threshold with population "
               f"risk ≤ 5% accepts {best:.3f} of decisions.")
    f = read("ltt_feasibility")
    unit = {("mfrc", "comment"): "comment", ("annomi_client", "transcript or lesson"): "transcript", ("annomi_therapist", "transcript or lesson"): "transcript",
            ("talkmoves_teacher", "transcript or lesson"): "lesson", ("talkmoves_student", "transcript or lesson"): "lesson",
            ("annomi_client", "utterance or sentence"): "utterance", ("annomi_therapist", "utterance or sentence"): "utterance",
            ("talkmoves_teacher", "utterance or sentence"): "sentence", ("talkmoves_student", "utterance or sentence"): "sentence"}
    yn = lambda v: "–" if pd.isna(v) else ("yes" if str(v) == "True" else "no")  # noqa: E731
    count = lambda v: "unreachable" if pd.isna(v) else f"{int(v)}"  # noqa: E731
    med = lambda v: "not reached" if pd.isna(v) else f"{v:g}"  # noqa: E731
    out = pd.DataFrame({"task": f.task.map(TASK), "sampling unit": [unit[(t, u)] for t, u in zip(f.task, f.unit)],
                        "units in corpus": f.units_in_corpus.astype(int), "mean unit weight / B": f.mean_unit_share_of_bound,
                        "frozen split certifies it": f["frozen split certifies a perfect coder (grid of 16)"].map(yn),
                        "lower bound": f["fewest possible (grid of 16)"].map(count),
                        "our procedure (median)": f["procedure, median of 20 sequences (grid of 16)"].map(med),
                        "lower bound ": f["fewest possible (one threshold)"].map(count),
                        "our procedure (median) ": f["procedure, median of 20 sequences (one threshold)"].map(med)})
    cols = list(out.columns)
    save_table(out, "tab21_ltt_feasibility", "Table 21. Our corpus LTT procedure: hand-coded units needed before it certifies a coder that is never wrong",
               "B: the largest unit's weight. Lower bound: a necessary number over every set of hand-coded units and every visiting order (each unit's bet "
               "can multiply the evidence by at most 1 + 0.05 × its weight / B, and certification needs a factor of 320 with the grid, 20 with one "
               "threshold); fewer units cannot suffice, and reaching the bound does not guarantee certification. Unreachable: not reached even with every "
               "unit but one hand-coded. Our procedure: median number of units at which it first certifies, over 20 random hand-coding sequences (a "
               "description of typical runs, not a bound). Transcripts and lessons are the units of our splits; utterances and sentences drawn at random "
               "are an alternative design.", 2,
               panels=[("A. Grid of 16 thresholds (as used throughout)", cols[:7]),
                       ("B. One threshold fixed in advance (0.95)", cols[:3] + cols[7:])], width=12)
    if (RESULTS / "utterance_sampling.csv").exists():
        u = read("utterance_sampling")
        tested = ["jev", "oaid", "clef", "clef_flash", "pplx", "kev9b", "kev27b", "luna-verbal", "haiku-verbal", "oss-verbal", "sonnet-verbal"]
        u = u[u.signal.isin(tested)]
        tasks = [t for t in CONFIRMATORY if t != "mfrc"]
        piv = lambda rule, v: u[u.rule == rule].pivot_table(index="signal", columns="task", values=v, sort=False)  # noqa: E731
        acc, zero, naive = piv("ltt", "mean_coverage") * 100, piv("ltt", "share_zero_coverage"), piv("empirical", "violation_rate") * 100
        reps = int(u.reps.iloc[0])
        out = pd.DataFrame({"signal": [SIG[x] for x in acc.index]})
        for t in tasks:
            out[f"{TASK[t]}"] = acc[t].to_numpy()
        for t in tasks:
            out[f"{TASK[t]} "] = [f"{int(round(reps * (1 - z)))} of {reps}" for z in zero[t]]
        for t in tasks:
            out[f"{TASK[t]}  "] = naive[t].to_numpy()
        cols = list(out.columns)
        viol = u[u.rule == "ltt"].violation_rate.max()
        save_table(out, "tab21b_utterance_sampling", "Table 21b. Dialogue tasks under a different hand-coding design: corpus LTT with single utterances or sentences "
                   f"drawn at random (exploratory; {reps} re-splits, 30% hand-coded)",
                   "Each re-split hand-codes a uniformly random 30% of the task's segments, with their context as the coder saw it; the uncoded segments are "
                   "the remainder, and the sampling unit is the segment, so B is the largest segment weight. Dependence among segments of one transcript is "
                   f"allowed. Coders tested: the seven decision models and Luna's, Haiku's, gpt-oss's and Sonnet's verbalized confidence (44 coder–task "
                   f"pairs). Corpus LTT's accepted remainder "
                   f"decisions exceeded 5% error in {100 * viol:.0f}% of re-splits in every cell. Panel C: the naive rule on the same re-splits, for comparison.", 1,
                   panels=[("A. Corpus LTT: mean share of the remainder's decisions accepted (percent; re-splits with no certificate count as 0)", cols[:5]),
                           ("B. Corpus LTT: re-splits with any certified threshold", [cols[0]] + cols[5:9]),
                           ("C. Naive rule: share of re-splits whose accepted decisions exceed 5% error (percent)", [cols[0]] + cols[9:])], width=12)
    st = read("ltt_single_threshold")
    st = st[st.task == "mfrc"][["signal", "grid_coverage", "grid_risk", "single_coverage", "single_risk"]]
    st["signal"] = st.signal.map(SIG)
    st.columns = ["signal", "grid of 16: accepted", "error", "one threshold (0.95): accepted", "error "]
    save_table(st, "tab23_ltt_single_threshold", "Table 23. MFRC: corpus LTT over the 16-threshold grid vs one threshold fixed in advance (0.95)",
               "On the four dialogue tasks, with the original whole-cluster split, neither version certifies anything for any coder.")
    tf = read("temperature_floor")
    tf = tf.pivot_table(index="signal", columns=["method", "floor"], values=["ece", "error_auroc"])
    out = pd.DataFrame({"signal": [SIG.get(x, x) for x in tf.index],
                        "ECE raw": tf[("ece", "raw", 1e-12)].to_numpy(),
                        "ECE temperature, floor 1e-4": tf[("ece", "temperature", 1e-4)].to_numpy(),
                        "ECE temperature, floor 1e-12": tf[("ece", "temperature", 1e-12)].to_numpy(),
                        "error AUROC raw": tf[("error_auroc", "raw", 1e-12)].to_numpy(),
                        "error AUROC temperature, floor 1e-4": tf[("error_auroc", "temperature", 1e-4)].to_numpy(),
                        "error AUROC temperature, floor 1e-12": tf[("error_auroc", "temperature", 1e-12)].to_numpy(),
                        "error AUROC per-code Platt, floor 1e-4": tf[("error_auroc", "per-code Platt", 1e-4)].to_numpy(),
                        "error AUROC per-code Platt, floor 1e-12": tf[("error_auroc", "per-code Platt", 1e-12)].to_numpy()})
    out.columns = ["signal", "raw", "temperature, floor 1e-4", "temperature, floor 1e-12", "raw ", "temperature, floor 1e-4 ",
                   "temperature, floor 1e-12 ", "per-code Platt, floor 1e-4", "per-code Platt, floor 1e-12"]
    save_table(out, "tab24_temperature_floor", "Table 24. Temperature scaling and per-code maps depend on how exact 0s and 1s are floored before taking logarithms (pooled)",
               "One temperature per task, fitted by weighted log-loss on calib. A task-level Platt or isotonic map is monotone and leaves error ranking unchanged at "
               "any floor; per-code Platt maps (EQ3) are not monotone across codes, so the floor changes their ranking for coders that return exact 0s and 1s.",
               panels=[("A. ECE", list(out.columns[:4])), ("B. Error AUROC", [out.columns[0]] + list(out.columns[4:]))], width=14)
    c = read("category_only")
    c = c[c.task == "POOLED"][["signal", "auroc_raw", "auroc_category_only"]]
    g = read("a3_per_code_gain")
    g = g[(g.task == "POOLED") & g.comparison.str.endswith("per-code calibrated - raw")].assign(signal=lambda d: d.comparison.str.split(":").str[0])
    c = c.merge(g[["signal", "auroc_a"]].rename(columns={"auroc_a": "auroc_per_code"}), on="signal", how="left")
    c["signal"] = c.signal.map(SIG)
    c.columns = ["signal", "raw confidence", "predicted code only", "per-code maps"]
    save_table(c, "tab22_category_only", "Table 22. Error AUROC from raw confidence, from the predicted code alone, and from per-code maps (pooled)",
               "'Predicted code only' scores each decision by the calibration accuracy of its predicted code (multi-label: of its code), with no confidence.")


def _signed(x: float, digits: int, plus: bool = True) -> str:
    if round(float(x), digits) == 0 and float(x) != 0:
        digits += 1  # a nonzero endpoint never prints as zero
    v = round(float(x), digits)
    if v == 0:
        v = 0.0  # no "-0.000"
    body = f"{abs(v):.{digits}f}"
    return f"$-${body}" if v < 0 else (f"$+${body}" if plus else body)


def write_numbers() -> Path:
    """paper/numbers.tex: every bootstrap estimate the prose quotes, keyed, so text and tables come from the same
    results files. In the paper, \\res{key} prints one; an unknown key stops the build."""
    out: dict[str, str] = {}
    pval = lambda v: "$<$0.001" if v < 0.001 else f"{v:.3f}"  # noqa: E731

    def interval(prefix: str, d: float, lo: float, hi: float, digits: int, scale: float = 1.0, plus_ci: bool = False) -> None:
        out[f"{prefix}/d"] = _signed(scale * d, digits)
        out[f"{prefix}/ci"] = f"[{_signed(scale * lo, digits, plus_ci)}, {_signed(scale * hi, digits, plus_ci)}]"
        out[f"{prefix}/dci"] = f"{out[f'{prefix}/d']} {out[f'{prefix}/ci']}"

    for name in ["rq1_contrasts", "oaid_contrasts", "same_base_contrasts", "format_contrast", "rq1_contrasts_adj_plan_v1"]:
        if not (RESULTS / f"{name}.csv").exists():
            continue
        for _, r in read(name).query("task == 'POOLED'").iterrows():
            for k, lab in (("error_auroc", "auroc"), ("cov@5%", "cov")):
                key = f"{name.replace('_contrasts', '').replace('_contrast', '')}/{r.a}/{r.b}/{lab}"
                interval(key, r[f"{k}_diff"], r[f"{k}_lo"], r[f"{k}_hi"], 3)
                out[f"{key}/p"] = pval(r[f"{k}_p"])
    clean = lambda x: re.sub(r"[ ()]+", "-", str(x)).strip("-")  # noqa: E731
    for _, r in read("a3_per_code_gain").query("task == 'POOLED'").iterrows():
        interval(f"pc/{clean(r.a)}/{clean(r.b)}", r["diff"], r.lo, r.hi, 3)
    for _, r in read("a3_cascade_contrasts").query("task == 'mfrc' and rule == 'ltt' and pool == 'full'").iterrows():
        interval(f"cas/{r.a.replace(' -> ', '>')}/{r.b.replace(' -> ', '>')}", r.automation_diff, r.automation_lo, r.automation_hi, 1, 100, True)
    if (RESULTS / "prompt_sensitivity.csv").exists():
        for _, r in read("prompt_sensitivity").query("task == 'POOLED'").iterrows():
            interval(f"ps/{r.signal}", r.error_auroc_diff, r.lo, r.hi, 3)
    # Policy results quoted in the prose (percent unless stated).
    pct = lambda v, d=0: f"{100 * v:.{d}f}"  # noqa: E731
    c = read("a3_cascades").query("task == 'mfrc' and pool == 'full' and reference_for_eval == 'human' and rule == 'ltt'")
    n_cal = int(triage.load("mfrc", "jev").query("split == 'calib'").group.nunique())
    for _, r in c.iterrows():
        k = f"pol/{r.cascade.replace(' -> ', '>')}"
        n_test = r.n_test_decisions / 8
        out[f"{k}/acc"], out[f"{k}/err"] = pct(r.automation), pct(r.auto_risk, 1)
        out[f"{k}/usd"], out[f"{k}/read"] = f"{r.api_usd_per_1k_segments:.2f}", pct(r.human_segments)
        out[f"{k}/readall"] = pct((n_cal + r.human_segments * n_test) / (n_cal + n_test))
    out["pol/ncal"] = str(n_cal)
    for _, r in read("ltt_single_threshold").query("task == 'mfrc'").iterrows():
        out[f"st/{r.signal}/grid"], out[f"st/{r.signal}/single"] = pct(r.grid_coverage), pct(r.single_coverage)
    for _, r in read("mfrc_prevalence").query("rule == 'ltt'").iterrows():
        out[f"rec/{r.signal}"] = pct(r.present_recall_accepted)
    if (RESULTS / "utterance_sampling.csv").exists():
        u = read("utterance_sampling").query("rule == 'ltt'")
        for _, r in u.iterrows():
            certs = int(round(r.reps * (1 - r.share_zero_coverage)))
            out[f"ut/{r.task}/{r.signal}/certs"] = str(certs)
            out[f"ut/{r.task}/{r.signal}/mean"] = pct(r.mean_coverage, 1)
            out[f"ut/{r.task}/{r.signal}/cond"] = pct(r.mean_coverage / (1 - r.share_zero_coverage)) if certs else "–"
        out["ut/reps"] = str(int(u.reps.iloc[0]))
        out["ut/maxmean"] = pct(u.mean_coverage.max(), 1)
        out["ut/nonzero"] = str(int((u.share_zero_coverage < 1).sum()))
        out["ut/cells"] = str(len(u))
    lines = ["% Generated by `margin report` (report.write_numbers); do not edit.",
             r"\newcommand{\res}[1]{\ifcsname res@#1\endcsname\csname res@#1\endcsname\else"
             r"\PackageError{margin}{unknown result key #1}{Regenerate paper/numbers.tex with margin report.}\fi}"]
    lines += [rf"\expandafter\def\csname res@{k}\endcsname{{{v}}}" for k, v in sorted(out.items())]
    path = Path("paper/numbers.tex")
    path.write_text("\n".join(lines) + "\n")
    return path


def run() -> list[Path]:
    tb = top_block_table()
    save_table(tb.assign(task=tb.task.map(TASK), signal=tb.signal.map(SIG)).rename(columns={"top_confidence": "top confidence", "n_at_top": "decisions at top",
               "share_at_top": "share at top", "error_at_top": "error at top"}), "tab13_top_blocks",
               "Table 13. Top-confidence tie blocks on the dialogue tasks (test split)")
    for f in [fig_risk_coverage, fig_forest, fig_reliability, lambda: fig_top_block(tb), fig_threshold_rules, fig_budget,
              fig_cost_automation, fig_rq2, fig_rq3, fig_per_code_gain, fig_repeatability, fig_rq5,
              table_datasets, table_coders, table_rq1, table_claims, table_rq2, table_rq3, table_triage_rules,
              table_cascades, table_cascade_contrasts, table_per_code, table_repeatability, table_adjudication,
              table_same_base, fig_decision_models, table_contamination, fig_contamination,
              table_contamination_followups, table_snapshot, table_prompt_sensitivity, table_ltt_checks,
              main_tables, write_numbers]:
        f()
    return sorted(FIG.glob("*.png")) + sorted(TAB.glob("*.md"))
