import pandas as pd

from margin.datasets.base import Task
from margin.sampling import allocate, draw, reference_labels, split_of
from margin.schema import Code, Codebook

CB = Codebook(
    name="x",
    dataset="x",
    unit="u",
    mode="exclusive",
    instructions="?",
    codes=[Code(id="common", label="C", definition="c"), Code(id="rare", label="R", definition="r")],
)


def make_task(n_common=950, n_rare=50):
    ids = [f"s{i}" for i in range(n_common + n_rare)]
    segs = pd.DataFrame(
        {"segment_id": ids, "text": "t", "context": None, "setting": None, "group": [f"g{i // 10}" for i in range(len(ids))]}
    )
    segs["multi_annotated"] = [i < 5 for i in range(len(ids))]
    labels = pd.DataFrame(
        {"segment_id": ids, "annotator": "gold", "codes": [["common"]] * n_common + [["rare"]] * n_rare}
    )
    return Task("x", segs, labels)


def test_allocate_respects_floor_and_total():
    take = allocate(pd.Series({"a": 900, "b": 40, "c": 5}), 200, floor=30)
    assert take["b"] >= 30 and take["c"] == 5 and take.sum() == 200


def test_reference_plurality_breaks_ties_by_codebook_order():
    labels = pd.DataFrame(
        {"segment_id": ["a", "a", "b", "b", "b"], "annotator": [1, 2, 1, 2, 3], "codes": [["rare"], ["common"], ["rare"], ["rare"], ["common"]]}
    )
    ref = reference_labels(labels, CB)
    assert ref["a"] == "common" and ref["b"] == "rare"


def test_draw_weights_forced_and_split():
    task = make_task()
    s = draw(task, CB, n=200, floor=40, seed=1)
    assert len(s) == 200 and s.segment_id.is_unique
    assert set(s.segment_id) >= {f"s{i}" for i in range(5)}  # forced subset always in
    assert (s[s.stratum == "multi_annotated"].weight == 1).all()
    assert (s.stratum == "rare").sum() >= 40
    # Horvitz-Thompson weights recover the non-forced population size.
    assert abs(s[s.stratum != "multi_annotated"].weight.sum() - 995) < 1e-6
    # Same group, same split; deterministic across draws.
    assert s.groupby("group").split.nunique().max() == 1
    assert s.equals(draw(task, CB, n=200, floor=40, seed=1))
    assert split_of("g1") in {"calib", "test"}
