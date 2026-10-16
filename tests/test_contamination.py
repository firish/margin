import pytest

from margin import contamination as c


def call(kind, item):
    return c.Call(kind, "guided", item, "", 50)


def test_lcs_and_longest_run():
    a, b = "the cat sat on the mat".split(), "a cat sat on a mat".split()
    assert c.lcs(a, b) == 4 and c.longest_run(a, b) == 3


def test_completion_scoring_flags_near_verbatim_recall():
    target = "one two three four five six seven eight nine ten eleven twelve"
    item = {"dataset": "mfrc", "task": "mfrc", "segment_id": "s", "target": target}
    exact = c.score(call("completion", item), "One two three four five six seven eight nine ten eleven twelve.")
    assert exact["rouge_l"] == pytest.approx(1.0) and exact["verbatim"]
    para = c.score(call("completion", item), "one two three something else entirely different here now")
    assert not para["verbatim"] and para["longest_run"] == 3


def test_label_scoring():
    excl = {"dataset": "annomi", "task": "annomi_client", "segment_id": "s", "mode": "exclusive",
            "labels": {"change": "Change talk", "sustain": "Sustain talk", "neutral": "Neutral talk"}, "reference": ["sustain"]}
    assert c.score(call("label", excl), "Sustain talk")["label_accuracy"] == 1.0
    assert c.score(call("label", excl), "Change talk (not sustain talk)")["label_accuracy"] == 0.0  # first label mentioned counts
    multi = {"dataset": "mfrc", "task": "mfrc", "segment_id": "s", "mode": "multi",
             "labels": {"care": "Care", "loyalty": "Loyalty"}, "reference": ["care", "loyalty"]}
    assert c.score(call("label", multi), "Care")["label_accuracy"] == pytest.approx(0.5)
    assert c.score(call("label", {**multi, "reference": []}), "none")["label_accuracy"] == 1.0
