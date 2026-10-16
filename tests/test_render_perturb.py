import pytest

from margin.perturb import STANDARD_ARMS, Variant, apply_labels_and_wording
from margin.render import jev_calls
from margin.schema import Code, Codebook, Segment


@pytest.fixture
def exclusive() -> Codebook:
    return Codebook(
        name="talk",
        dataset="t",
        unit="client utterance",
        mode="exclusive",
        instructions="Which talk type?",
        codes=[
            Code(id="change", label="Change talk", definition="Favours change.", examples_pos=["I want to quit."]),
            Code(id="sustain", label="Sustain talk", definition="Favours status quo."),
        ],
        none_option=Code(id="none", label="Neutral", definition="Neither."),
    )


@pytest.fixture
def multi() -> Codebook:
    return Codebook(
        name="topics",
        dataset="t",
        unit="comment",
        mode="multi",
        instructions="Which topics?",
        codes=[
            Code(id=c, label=c.title(), definition=f"About {c}.", exclude=[f"Not {c}."])
            for c in ["care", "fairness", "loyalty", "purity"]
        ],
    )


@pytest.fixture
def seg() -> Segment:
    return Segment(segment_id="s1", text="text", context="prev", group="g")


def definitions(call, codebook):
    """code id -> definition as seen by the model, via the keymap (inverse mapping)."""
    q = call.questions
    if codebook.mode == "exclusive":
        crit = q["code"]["criteria"]
        return {call.keymap[label]: crit[label]["definition"] for label in crit}
    return {call.keymap[k]: v["criteria"]["true"]["definition"] for k, v in q.items()}


@pytest.mark.parametrize("arm", ["canonical", "reversed", "shuffle1", "shuffle2", "neutral_labels", "no_context"])
def test_variants_keep_definitions_and_invert(exclusive, multi, seg, arm):
    v = STANDARD_ARMS[arm]
    for cb in (exclusive, multi):
        (call,) = jev_calls(apply_labels_and_wording(cb, v), seg, v)
        assert definitions(call, cb) == {c.id: c.definition for c in cb.all_options}


def test_order_arms(multi, seg):
    canon = [c for c in jev_calls(multi, seg, STANDARD_ARMS["canonical"])[0].keymap.values()]
    rev = [c for c in jev_calls(multi, seg, STANDARD_ARMS["reversed"])[0].keymap.values()]
    assert canon == ["care", "fairness", "loyalty", "purity"]
    assert rev == canon[::-1]


def test_shuffle_is_deterministic_and_segment_specific(multi, seg):
    v = STANDARD_ARMS["shuffle1"]
    a = list(jev_calls(multi, seg, v)[0].keymap.values())
    b = list(jev_calls(multi, seg, v)[0].keymap.values())
    assert a == b
    orders = {
        tuple(jev_calls(multi, Segment(segment_id=f"s{i}", text="t", group="g"), v)[0].keymap.values())
        for i in range(30)
    }
    assert len(orders) > 1


def test_isolated_one_call_per_code(multi, seg):
    calls = jev_calls(multi, seg, STANDARD_ARMS["isolated"])
    assert [list(c.keymap.values()) for c in calls] == [["care"], ["fairness"], ["loyalty"], ["purity"]]


def test_neutral_labels_hide_names(exclusive, multi, seg):
    v = STANDARD_ARMS["neutral_labels"]
    (call,) = jev_calls(apply_labels_and_wording(exclusive, v), seg, v)
    assert set(call.questions["code"]["criteria"]) == {"option_A", "option_B", "option_C"}
    (call,) = jev_calls(apply_labels_and_wording(multi, v), seg, v)
    text = str(call.questions)
    assert "Care" not in text and "code_A" in text


def test_context_arm(exclusive, seg):
    with_ctx = jev_calls(exclusive, seg, STANDARD_ARMS["canonical"])[0]
    without = jev_calls(exclusive, seg, STANDARD_ARMS["no_context"])[0]
    assert with_ctx.state == {"previous_turn": "prev", "segment": "text"}
    assert without.state == {"segment": "text"}
    framed = Segment(segment_id="s2", text="t", context="c", setting="Target: quit smoking", group="g")
    assert list(jev_calls(exclusive, framed, STANDARD_ARMS["no_context"])[0].state) == ["setting", "segment"]


def test_paraphrase(multi, seg):
    paras = {c.id: [f"P0 {c.id}", f"P1 {c.id}"] for c in multi.codes}
    v = Variant(name="p1", paraphrase=1)
    (call,) = jev_calls(apply_labels_and_wording(multi, v, paras), seg, v)
    assert definitions(call, multi) == {c.id: f"P1 {c.id}" for c in multi.codes}
    with pytest.raises(ValueError):
        apply_labels_and_wording(multi, v, {})


def test_noul_format_for_exclusive_keeps_option_descriptions(exclusive, seg):
    choice = jev_calls(exclusive, seg, STANDARD_ARMS["canonical"])[0]
    (nouls,) = jev_calls(exclusive, seg, STANDARD_ARMS["noul_format"])
    assert list(nouls.keymap.values()) == ["change", "sustain", "none"]  # one Noul per option, incl. none
    assert all(q["type"] == "noul" for q in nouls.questions.values())
    by_id = {nouls.keymap[k]: q for k, q in nouls.questions.items()}
    for label, spec in choice.questions["code"]["criteria"].items():
        assert by_id[choice.keymap[label]]["criteria"]["true"]["definition"] == spec["definition"]
    assert '"Change talk", "Sustain talk", "Neutral"' in by_id["sustain"]["instructions"]
    reversed_nouls = jev_calls(exclusive, seg, Variant(name="r", order="reversed", question_format="noul"))[0]
    assert list(reversed_nouls.keymap.values()) == ["none", "sustain", "change"]


def test_noul_format_leaves_multi_unchanged(multi, seg):
    assert jev_calls(multi, seg, STANDARD_ARMS["noul_format"])[0].request == jev_calls(multi, seg, STANDARD_ARMS["canonical"])[0].request
