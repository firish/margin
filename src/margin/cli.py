"""`margin` command line."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from dotenv import load_dotenv

from margin.budget import Registry, check_cap
from margin.cache import Cache
from margin.coders import jev
from margin.perturb import CANONICAL
from margin.schema import Code, Codebook, Segment

load_dotenv(override=True)  # keys in .env win over stale shell variables
app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def smoke() -> None:
    """One real Jev call per question type, to confirm the key, response shape, and ledger."""
    registry, cache = Registry.load(), Cache()
    spec = registry["jev"]
    talk = Codebook(
        name="smoke-exclusive",
        dataset="smoke",
        unit="client utterance",
        mode="exclusive",
        instructions="Which kind of talk about changing their behaviour is the client using?",
        codes=[
            Code(id="change", label="Change talk", definition="The client argues for, or commits to, changing."),
            Code(id="sustain", label="Sustain talk", definition="The client argues for keeping things as they are."),
        ],
        none_option=Code(id="none", label="Neutral", definition="Neither for nor against change."),
    )
    topics = Codebook(
        name="smoke-multi",
        dataset="smoke",
        unit="client utterance",
        mode="multi",
        instructions="Which topics does the client mention?",
        codes=[
            Code(id="health", label="Health", definition="Mentions physical or mental health."),
            Code(id="money", label="Money", definition="Mentions cost, prices, or finances."),
        ],
    )
    seg = Segment(
        segment_id="smoke-1",
        text="Honestly I want to quit, the cough is getting worse and cigarettes cost a fortune.",
        context="How are you feeling about smoking these days?",
        group="smoke",
    )
    calls = jev.plan_calls([seg], talk, CANONICAL) + jev.plan_calls([seg], topics, CANONICAL)
    n_todo, usd = jev.estimate_usd(calls, spec, cache)
    check_cap(registry, cache, usd)
    typer.echo(f"{n_todo} uncached call(s), estimated ${usd:.6f}")
    preds, failures = asyncio.run(jev.run(calls, spec, cache))
    typer.echo(preds.to_string(index=False))
    if failures:
        typer.echo(json.dumps(failures, indent=2))
        raise typer.Exit(1)
    typer.echo(f"ledger total: ${cache.spent():.6f}")


@app.command()
def prepare(dataset: list[str] = typer.Argument(None, help="mfrc, annomi, talkmoves (default: all)")) -> None:
    """Raw downloads -> data/processed/<task>/{segments,labels}.parquet, validated against codebooks."""
    from margin.datasets import annomi, dunivin, mfrc, talkmoves

    loaders = {"mfrc": mfrc.prepare, "annomi": annomi.prepare, "talkmoves": talkmoves.prepare, "dunivin": dunivin.prepare}
    for name in dataset or list(loaders):
        for task in loaders[name]():
            task.validate(Codebook.load(f"codebooks/{task.name}.yaml"))
            out = task.save()
            typer.echo(f"{task.name:18s} {len(task.segments):7d} segments  {len(task.labels):7d} label rows -> {out}")
            typer.echo("  " + json.dumps(task.report, default=str)[:400])


@app.command()
def sample(
    config: str = "configs/samples.yaml",
    task: list[str] = typer.Option(None, "--task", help="only these tasks (default: all in the config)"),
) -> None:
    """Draw the shared stratified sample for every task (data/samples/<task>-<tag>.parquet)."""
    import yaml

    from margin import sampling
    from margin.datasets.base import Task

    cfg = yaml.safe_load(open(config))
    for task_name, n in cfg["n"].items():
        if task and task_name not in task:
            continue
        loaded = Task.load(task_name)
        s = sampling.draw(loaded, Codebook.load(f"codebooks/{task_name}.yaml"), n, floor=cfg["floor"], seed=cfg["seed"])
        path = sampling.save(s, task_name, cfg["tag"])
        by_split = s.split.value_counts().to_dict()
        typer.echo(f"{task_name:18s} {len(s):5d} -> {path}  split={by_split}")
        typer.echo("   strata " + json.dumps(s.stratum.value_counts().to_dict()))


@app.command()
def estimate(
    experiments: list[str] = typer.Argument(..., help="one or more experiment names"),
    limit: int = typer.Option(None, help="override segments per task"),
) -> None:
    """Planned calls and estimated cost per (task, coder, variant), counting the cache."""
    import pandas as pd

    from margin import run as runner

    registry, cache = Registry.load(), Cache()
    units = [u for e in experiments for u in runner.plan(runner.Experiment.load(e), registry, limit)]
    table = runner.estimate(units, cache)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        typer.echo(table.to_string(index=False))
        typer.echo(table.groupby(["experiment", "coder"])[["calls", "uncached", "est_usd"]].sum().round(4).to_string())
    typer.echo(f"total estimate ${table.est_usd.sum():.4f}; spent so far ${cache.spent():.4f} of ${registry.cap_usd:.2f}")


@app.command(name="run")
def run_cmd(
    experiments: list[str] = typer.Argument(..., help="one or more experiments, run concurrently"),
    limit: int = typer.Option(None, help="override segments per task"),
    yes: bool = typer.Option(False, "--yes", "-y", help="skip the confirmation prompt"),
) -> None:
    """Estimate, check the budget caps, then run and write predictions to data/runs/."""
    from margin import run as runner

    registry, cache = Registry.load(), Cache()
    units = [u for e in experiments for u in runner.plan(runner.Experiment.load(e), registry, limit)]
    table = runner.estimate(units, cache)
    usd = float(table.est_usd.sum())
    check_cap(registry, cache, usd)
    typer.echo(f"{int(table.uncached.sum())} uncached calls, estimated ${usd:.4f}")
    if not yes and usd > 0:
        typer.confirm("Proceed?", abort=True)
    for row in runner.execute(units, cache):
        typer.echo(json.dumps(row))
    typer.echo(f"ledger total ${cache.spent():.4f}")


@app.command(name="evaluate")
def evaluate_cmd(
    experiment: str,
    adjudication: str = typer.Option(None, help="batch name: also score against adjudicated labels"),
) -> None:
    """Agreement, calibration, triage, and disagreement metrics -> results/<experiment>/summary*.csv."""
    import pandas as pd

    from margin.evaluate import evaluate

    table = evaluate(experiment, adjudication)
    with pd.option_context("display.width", 200, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
        typer.echo(table.to_string(index=False))


@app.command()
def ensemble(
    experiment: str,
    coders: list[str] = typer.Option(..., "--coder", help="coder files to pool, e.g. luna-verbal"),
    name: str = typer.Option("agree", help="name of the new coder file"),
) -> None:
    """Multi-model agreement signal: vote share across coders, written as another coder file."""
    from margin import ensemble as ens
    from margin.run import Experiment

    exp = Experiment.load(experiment)
    for task in exp.tasks:
        out = ens.write(exp.name, task, coders, name)
        typer.echo(f"{task:18s} {out.segment_id.nunique():5d} segments -> {name}__canonical.parquet")


@app.command()
def paraphrase(task: list[str] = typer.Option(None, "--task", help="default: all six tasks")) -> None:
    """Write two meaning-preserving paraphrases per definition (Sonnet) + docs/paraphrase_review.md."""
    from margin.paraphrase import generate

    tasks = task or ["mfrc", "annomi_client", "annomi_therapist", "talkmoves_teacher", "talkmoves_student", "dunivin"]
    issues = generate(tasks)
    typer.echo(f"wrote codebooks/paraphrases/*.yaml and docs/paraphrase_review.md for {len(tasks)} codebooks")
    typer.echo("automatic flags:\n  " + ("\n  ".join(issues) if issues else "none"))


@app.command()
def analyze(
    reps: int = typer.Option(1000, help="bootstrap replicates"),
    adjudication: str = typer.Option(None, help="batch: run the adjudicated-reference RQ1 analysis instead"),
) -> None:
    """Frozen-plan analyses (test split): RQ1 signals + contrasts, format contrast, RQ2, RQ3 -> results/*.csv."""
    from margin.analysis import run_adjudicated, run_all

    results = run_adjudicated(adjudication, reps) if adjudication else run_all(reps)
    for name, df in results.items():
        typer.echo(f"results/{name}.csv  ({len(df)} rows)")


@app.command()
def rq5() -> None:
    """Exploratory per-code analysis and code-type comparisons -> results/rq5_*.csv."""
    from margin.rq5 import run

    for name, df in run().items():
        typer.echo(f"results/{name}.csv  ({len(df)} rows)")


@app.command()
def replicate(reps: int = typer.Option(1000, help="bootstrap replicates")) -> None:
    """Dunivin replication section -> results/dunivin_*.csv."""
    from margin.replication import run

    for name, df in run(reps).items():
        typer.echo(f"results/{name}.csv  ({len(df)} rows)")


@app.command()
def a3(
    reps: int = typer.Option(500, help="cluster re-splits per task and signal"),
    boot: int = typer.Option(1000, help="bootstrap replicates for cascade contrasts"),
) -> None:
    """Amendment v1.1 A3: recalibration, threshold rules, risk control, calib budget, cascades -> results/a3_*.csv."""
    from margin.triage import run

    for name, df in run(reps, boot).items():
        typer.echo(f"results/{name}.csv  ({len(df)} rows)")


@app.command()
def controls() -> None:
    """Review checks: category-only control, LTT feasibility, clustered LTT simulation -> results/."""
    from margin.controls import run

    for name, df in run().items():
        typer.echo(f"results/{name}.csv  ({len(df)} rows)")


@app.command()
def ledger() -> None:
    """Export the text-free spend ledger from the response cache -> data/ledger.csv.gz."""
    from margin.triage import LEDGER, export_ledger

    typer.echo(f"{LEDGER}  ({export_ledger()} requests)")


@app.command()
def prevalence() -> None:
    """MFRC: share of accepted decisions that are "absent" calls, present-code recall, prevalence baseline."""
    from margin.prevalence import run

    df = run()
    typer.echo(f"results/mfrc_prevalence.csv  ({len(df)} rows)")


@app.command()
def repeatability() -> None:
    """Run-to-run stability from the repeat1 experiment -> results/repeatability.csv."""
    from margin.repeatability import run

    df = run()
    typer.echo(f"results/repeatability.csv  ({len(df)} rows)")


@app.command()
def report() -> None:
    """Paper figures and tables from results/ -> paper/figures, paper/tables."""
    from margin.report import run

    for path in run():
        typer.echo(str(path))


@app.command()
def contamination(estimate_only: bool = typer.Option(False, "--estimate", help="print the cost estimate and stop")) -> None:
    """Memorization probe (amendment v1.3) -> results/contamination_*.csv."""
    from margin import contamination as probe

    if estimate_only:
        typer.echo(probe.estimate().to_string(index=False))
        return
    for name, df in probe.run().items():
        typer.echo(f"results/{name}.csv  ({len(df)} rows)")


@app.command()
def decoy(from_cache: bool = typer.Option(False, "--from-cache", help="first rebuild the item scores from the local response cache")) -> None:
    """Decoy-name control (deviation 14) -> results/contamination_decoy.csv, from results/contamination_decoy_items.csv."""
    from margin import contamination as probe
    from margin.analysis import RESULTS

    if from_cache:
        items = probe.decoy_items_from_cache()
        items.to_csv(RESULTS / "contamination_decoy_items.csv", index=False)
        typer.echo(f"results/contamination_decoy_items.csv  ({len(items)} rows)")
    typer.echo(probe.decoy_from_items().to_string(index=False))


@app.command("prompt-sensitivity")
def prompt_sensitivity(reps: int = typer.Option(1000, help="bootstrap replicates")) -> None:
    """Amendment v1.4: distribution prompt vs canonical prompt for Sonnet and Haiku -> results/prompt_sensitivity.csv."""
    from margin.prompt_sensitivity import run

    typer.echo(f"results/prompt_sensitivity.csv  ({len(run(reps))} rows)")


adjudicate_app = typer.Typer(no_args_is_help=True, help="Blinded adjudication of coder/human disagreements.")
app.add_typer(adjudicate_app, name="adjudicate")


@adjudicate_app.command("export")
def adjudicate_export(
    batch: str = typer.Option(..., help="name for this batch (data/adjudication/<batch>/)"),
    plan: bool = typer.Option(True, help="pool the sources and tasks fixed in docs/ANALYSIS_PLAN.md section 9"),
    experiment: str = typer.Option(None, help="instead of --plan: one experiment ..."),
    coders: list[str] = typer.Option(None, "--coder", help="... and its coder file stems (variant canonical)"),
    split: str = typer.Option("test", help="calib, test, or all"),
    n: int = typer.Option(40, help="items per task"),
    hi_conf: float = typer.Option(0.9, help="disagreements at or above this confidence are taken first"),
    control_share: float = typer.Option(0.1, help="share of hidden agreed-upon controls"),
    repeat_share: float = typer.Option(0.1, help="share re-presented blind in a second sitting"),
    seed: int = 0,
) -> None:
    """Write the blinded workbook(s) + codebooks + instructions; keep the answer key separately."""
    from margin.adjudicate import PLAN_SOURCES, PLAN_TASKS, ExportConfig, export
    from margin.run import Experiment

    if experiment:
        exp = Experiment.load(experiment)
        sources, tasks = tuple((c, exp.name, f"{c}__canonical") for c in coders), tuple(exp.tasks)
    else:
        sources, tasks = PLAN_SOURCES, PLAN_TASKS
    cfg = ExportConfig(sources, tasks, None if split == "all" else split, n, hi_conf, control_share, repeat_share, seed)
    out = export(cfg, batch)
    typer.echo(f"handout: {out / 'to_adjudicator'}  (give the adjudicator only this folder)")
    typer.echo((out / "manifest.json").read_text())


@adjudicate_app.command("panel")
def adjudicate_panel(batch: str, estimate_only: bool = typer.Option(False, "--estimate")) -> None:
    """Two-family LLM panel (opus + sol) judges every item blind; builds the author's handout."""
    from margin import panel
    from margin.cache import Cache

    typer.echo(f"panel estimate: ${panel.estimate_usd(batch, Cache()):.2f}")
    if estimate_only:
        return
    answers = panel.run_panel(batch)
    for name, df in answers.items():
        typer.echo(f"{name}: {int((df.answer != '').sum())}/{len(df)} answered")
    agree = panel.panel_agreement(batch)
    typer.echo(f"panel agreement: {int(agree.agree.sum())}/{len(agree)}")
    typer.echo(agree.groupby("task").agree.agg(["sum", "size"]).to_string())
    out = panel.build_author_handout(batch)
    typer.echo(f"author handout: {out}")


@adjudicate_app.command("resolve")
def adjudicate_resolve(batch: str) -> None:
    """Combine author + panel into verdicts_final.parquet (used by evaluate --adjudication)."""
    from margin import panel

    typer.echo(json.dumps(panel.resolve(batch), indent=2))


@adjudicate_app.command("import")
def adjudicate_import(
    batch: str,
    answers: list[str] = typer.Argument(..., help="filled workbook(s): adjudication.xlsx [adjudication_part2_later.xlsx]"),
    adjudicator: str = typer.Option(..., help="short name; one verdicts file per adjudicator"),
) -> None:
    """Read filled workbooks, unblind against the key, and summarize who was upheld."""
    import pandas as pd

    from margin.adjudicate import import_answers, intra_rater, summary

    v = import_answers(batch, [Path(a) for a in answers], adjudicator)
    coders = [c[: -len("_label")] for c in v.columns if c.endswith("_label")]
    typer.echo(f"{len(v)} verdicts saved; intra-rater: {intra_rater(v)}")
    with pd.option_context("display.width", 200):
        typer.echo(summary(v, coders).to_string(index=False))


@app.command()
def spend() -> None:
    """Spend so far, per provider and model, from the cache ledger."""
    registry, cache = Registry.load(), Cache()
    for provider, model, n, usd in cache.spend_by_model():
        typer.echo(f"{provider:10s} {model:28s} {n:7d} calls  ${usd:9.4f}")
    typer.echo(f"total ${cache.spent():.4f} of ${registry.cap_usd:.2f} cap")


if __name__ == "__main__":
    app()
