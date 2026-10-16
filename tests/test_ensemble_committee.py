import pandas as pd
import pytest

from margin.ensemble import mean_probability


def test_mean_probability_keeps_complete_units_and_averages():
    a = pd.DataFrame({"segment_id": ["s1", "s1", "s2", "s2"], "code_id": ["x", "y", "x", "y"], "p": [0.8, 0.2, 0.6, 0.4]})
    b = pd.DataFrame({"segment_id": ["s1", "s1", "s2"], "code_id": ["x", "y", "x"], "p": [0.4, 0.6, 0.5]})  # s2 incomplete
    out = mean_probability({"a": a, "b": b}).set_index(["segment_id", "code_id"]).p
    assert set(out.index.get_level_values(0)) == {"s1"}
    assert out[("s1", "x")] == pytest.approx(0.6) and out[("s1", "y")] == pytest.approx(0.4)
