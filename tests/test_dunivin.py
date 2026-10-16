"""Regression: our loader + kappa reproduce Dunivin (2025) Table 2 exactly. Needs the OSF raw files."""

from pathlib import Path

import pytest

RAW = Path("data/raw/dunivin")
pytestmark = pytest.mark.skipif(not (RAW / "output").exists(), reason="Dunivin raw data not downloaded")

PAPER_MEANS = {
    "per-code-with-justification_t=0_top_p=1_model=gpt-4": 0.68,
    "per-code-without-justification_t=0_top_p=1_model=gpt-4": 0.59,
    "full-codebook_with_justification_t=0_top_p=1_model=gpt-4": 0.60,
    "full-codebook_without_justification_t=0_top_p=1_model=gpt-4": 0.46,
    "per-code-with-justification_t=0_top_p=1_model=gpt-3.5": 0.34,
    "per-code-without-justification_t=0_top_p=1_model=gpt-3.5": 0.29,
}


@pytest.mark.parametrize("config,expected", PAPER_MEANS.items())
def test_reproduces_table_2(config, expected):
    from margin.datasets.dunivin import TEST_IDS, prepare, published_predictions
    from margin.metrics.agreement import per_code_kappa
    from margin.metrics.core import decisions, reference
    from margin.schema import Codebook

    (task,) = prepare()
    cb = Codebook.load("codebooks/dunivin.yaml")
    task.validate(cb)
    test = {f"dunivin-{i}" for i in TEST_IDS}
    ref = reference(task.labels[task.labels.segment_id.isin(test)], cb)
    kappas = per_code_kappa(decisions(published_predictions(config), ref, cb))
    assert len(kappas) == 9
    assert sum(kappas.values()) / 9 == pytest.approx(expected, abs=0.005)
