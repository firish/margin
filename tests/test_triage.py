from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from margin import triage
from margin.metrics.selective import coverage_at_risk


def test_empirical_rule_matches_primary_analysis():
    rng = np.random.default_rng(1)
    conf = np.round(rng.uniform(0.3, 1, 800), 2)
    correct = (rng.random(800) < conf).astype(float)
    w = rng.uniform(1, 4, 800)
    cal = pd.DataFrame({"conf": conf[:300], "correct": correct[:300], "weight": w[:300]})
    tst = pd.DataFrame({"conf": conf[300:], "correct": correct[300:], "weight": w[300:]})
    want = coverage_at_risk(cal, tst)
    thr = triage.fit_threshold(conf[:300], 1 - correct[:300], w[:300])
    assert thr == pytest.approx(want.threshold)
    cov, risk = triage.apply_threshold(conf[300:], 1 - correct[300:], w[300:], thr)
    assert cov == pytest.approx(want.test_coverage) and risk == pytest.approx(want.test_risk)


def test_binom_cdf_matches_scipy_on_integers():
    for n, k in [(59, 0), (100, 3), (400, 12)]:
        assert triage.binom_cdf(np.array([k]), np.array([n]), 0.05)[0] == pytest.approx(binom.cdf(k, n, 0.05))


def test_decision_level_ltt_needs_enough_decisions():
    ones = lambda n: (np.linspace(1, 0.6, n), np.zeros(n), np.ones(n))  # noqa: E731
    assert triage.N_MIN == 59 and triage.ltt_start() == 124
    assert triage.fit_threshold(*ones(123), rule="ltt_dec") == np.inf  # sequence starts at 124 decisions
    assert triage.fit_threshold(*ones(124), rule="ltt_dec") == pytest.approx(0.6)


def test_decision_level_ltt_controls_risk_for_independent_decisions():
    rng = np.random.default_rng(0)
    viol = {"empirical": 0, "ltt_dec": 0}
    sims = 300
    for _ in range(sims):
        conf = rng.uniform(0.5, 1, 6000)
        err = (rng.random(6000) > conf).astype(float)  # calibrated: P(correct) = conf
        w = np.ones(6000)
        for rule in viol:
            thr = triage.fit_threshold(conf[:600], err[:600], w[:600], rule)
            cov, risk = triage.apply_threshold(conf[600:], err[600:], w[600:], thr)
            viol[rule] += cov > 0 and risk > 0.05
    assert viol["ltt_dec"] / sims <= 0.06
    assert viol["empirical"] / sims >= 0.25


def test_wsr_pvalue_is_valid_at_the_null():
    rng = np.random.default_rng(3)
    y = (rng.random((2000, 200)) < 0.05).astype(float)  # mean exactly alpha
    p = triage.wsr_pvalues(y, 0.05, 0.05)
    assert (p <= 0.05).mean() <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 2000)  # Ville: <= delta, up to Monte Carlo error
    assert triage.wsr_pvalues(np.full((1, 500), 0.0), 0.05, 0.05)[0] < 1e-6  # clearly below alpha
    assert triage.wsr_pvalues(np.full((1, 500), 0.05), 0.05, 0.05)[0] == 1.0  # no evidence


def test_iid_cluster_ltt_refuses_what_correlated_clusters_cannot_support():
    # The first review's counterexample: 16 clusters of 8 perfectly correlated decisions, none wrong in this
    # draw (probability 0.94^16 = 0.37 when each cluster is wrong with probability 0.06).
    conf, err, w = np.full(128, 0.9), np.zeros(128), np.ones(128)
    groups = np.repeat(np.arange(16), 8)
    assert triage.fit_threshold(conf, err, w, "ltt_dec") == pytest.approx(0.9)  # 0.95^128 = 0.0014: certified
    assert triage.fit_threshold(conf, err, w, "ltt_iid", groups=groups, bound=8.0) == np.inf


def test_wor_pvalue_is_valid_at_a_finite_population_null():
    rng = np.random.default_rng(7)
    pop = np.r_[np.ones(10), np.zeros(190)]  # mean exactly 0.05
    rejections = 0
    sims = 3000
    for _ in range(sims):
        draw = rng.permutation(pop)[:120]
        rejections += triage.wor_pvalues(draw[None, :], np.array([0.05]), 200, 0.05)[0] <= 0.05
    assert rejections / sims <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / sims)
    # Monotone in the null mean, which the corpus certificate relies on.
    y = rng.random((1, 80)) * 0.1
    p = [triage.wor_pvalues(y, np.array([m]), 200, 0.05)[0] for m in (0.02, 0.05, 0.08, 0.2)]
    assert all(a >= b for a, b in zip(p, p[1:]))


def test_corpus_ltt_controls_the_remainder_under_within_cluster_correlation():
    from scipy.stats import norm
    rng = np.random.default_rng(4)
    n_clusters, size, rho = 400, 8, 0.9
    g = np.repeat(np.arange(n_clusters), size)
    conf = 1 - 0.5 * rng.beta(1, 6, n_clusters * size)
    z = rho * np.repeat(rng.standard_normal(n_clusters), size) + np.sqrt(1 - rho**2) * rng.standard_normal(n_clusters * size)
    err = (norm.cdf(z) > conf).astype(float)  # one fixed corpus with strongly correlated errors
    w = np.ones_like(conf)
    viol, sims = 0, 200
    for _ in range(sims):  # the randomness is which clusters get hand-coded
        cal = (rng.random(n_clusters) < 0.3)[g]
        thr = triage.fit_threshold(conf[cal], err[cal], w[cal], "ltt", groups=g[cal], bound=float(size), n_pool=n_clusters)
        cov, risk = triage.apply_threshold(conf[~cal], err[~cal], w[~cal], thr)
        viol += cov > 0 and risk > 0.05
    assert viol / sims <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / sims)


def test_corpus_ltt_does_not_depend_on_a_fixed_visiting_order():
    # The third review's counterexample: errors lined up with a fixed order of cluster ids. A random set of
    # hand-coded clusters visited in that fixed order certified the unsafe remainder every time; visiting
    # them in random orders must not.
    n_pool, n = 2000, 600
    ids = np.array([f"cluster-{i:04d}" for i in range(n_pool)])
    err = (np.arange(n_pool) >= 1600).astype(float)  # the sorted-id order puts every wrong cluster last
    conf, w = np.full(n_pool, 0.9), np.ones(n_pool)
    rng = np.random.default_rng(402)
    unsafe, sims = 0, 100
    for _ in range(sims):
        cal = np.zeros(n_pool, bool)
        cal[rng.choice(n_pool, n, replace=False)] = True
        thr = triage.cluster_ltt(conf[cal], err[cal], w[cal], ids[cal], bound=1.0, n_pool=n_pool)
        cov, risk = triage.apply_threshold(conf[~cal], err[~cal], w[~cal], thr)
        unsafe += cov > 0 and risk > 0.05
    assert unsafe / sims <= 0.05


def test_order_averaged_pvalue_is_reproducible_and_monotone():
    rng = np.random.default_rng(8)
    y = rng.random((3, 150)) * 0.08
    a = triage.wor_pvalues(y, np.array([0.05, 0.05, 0.05]), 400, 0.05)
    assert np.array_equal(a, triage.wor_pvalues(y, np.array([0.05, 0.05, 0.05]), 400, 0.05))
    p = [triage.wor_pvalues(y[:1], np.array([m]), 400, 0.05)[0] for m in (0.03, 0.05, 0.07, 0.1)]
    assert all(x >= z for x, z in zip(p, p[1:]))


def test_corpus_ltt_certifies_easy_data_and_needs_a_remainder():
    rng = np.random.default_rng(5)
    conf = rng.uniform(0.9, 1, 4000)
    err = (rng.random(4000) < 0.005).astype(float)
    g = np.arange(4000) // 4
    thr = triage.fit_threshold(conf, err, np.ones(4000), "ltt", groups=g, bound=4.0, n_pool=3000)
    assert thr == pytest.approx(0.5)  # every grid threshold is certified, so the lowest is used
    with pytest.raises(ValueError):
        triage.fit_threshold(conf, err, np.ones(4000), "ltt", groups=g, bound=4.0, n_pool=1000)


def test_segment_cost_keys_are_counted_once():
    keys = ["a", "b", "a", "c", "b"]
    assert list(dict.fromkeys(keys)) == ["a", "b", "c"]


def fake_td(mode: str, ids: list[str], ref: pd.DataFrame, codes: list[str]):
    sample = pd.DataFrame({"segment_id": ids, "split": "calib", "weight": 1.0})
    return SimpleNamespace(sample=sample, ref=ref, codebook=SimpleNamespace(mode=mode, code_ids=codes))


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(2)
    n, codes = 4000, ["a", "b", "c"]
    z = rng.normal(0, 1.5, (n, 3))
    q = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    y = np.array([rng.choice(3, p=row) for row in q])
    shown = np.exp(2 * z) / np.exp(2 * z).sum(1, keepdims=True)  # reported probabilities are too sharp (T = 2)
    ids = [f"s{i}" for i in range(n)]
    preds = pd.DataFrame({"segment_id": np.repeat(ids, 3), "code_id": codes * n, "p": shown.ravel()})
    ref = pd.DataFrame({"segment_id": ids, "ref_code": [codes[k] for k in y]})
    t, scaled = triage.temperature(fake_td("exclusive", ids, ref, codes), preds)
    assert t == pytest.approx(2.0, rel=0.1)
    assert scaled.groupby("segment_id").p.sum().to_numpy() == pytest.approx(np.ones(n))

    zm = rng.normal(0, 1.5, n)
    ym = (rng.random(n) < 1 / (1 + np.exp(-zm))).astype(int)
    preds_m = pd.DataFrame({"segment_id": ids, "code_id": "x", "p": 1 / (1 + np.exp(-3 * zm))})
    ref_m = pd.DataFrame({"segment_id": ids, "code_id": "x", "y": ym})
    t_m, scaled_m = triage.temperature(fake_td("multi", ids, ref_m, ["x"]), preds_m)
    assert t_m == pytest.approx(3.0, rel=0.1)
    assert ((scaled_m.p >= 0.5) == (preds_m.p >= 0.5)).all()  # decisions unchanged


def test_cascade_accounting():
    m = pd.DataFrame({
        "segment_id": [f"c{i}" for i in range(4)] + [f"t{i}" for i in range(4)],
        "weight": 1.0,
        "a|conf": [0.95, 0.9, 0.6, 0.55, 0.99, 0.7, 0.7, 0.2],
        "a|correct": [1, 1, 0, 1, 1, 0, 0, 0],
        "b|conf": [0.1, 0.1, 0.8, 0.3, 0.1, 0.85, 0.5, 0.9],
        "b|correct": [0, 0, 1, 0, 0, 1, 1, 0],
    })
    m["a|correct_adj"], m["b|correct_adj"], m["human|correct_adj"] = m["a|correct"], m["b|correct"], 1.0
    cal = np.r_[np.ones(4, bool), np.zeros(4, bool)]
    out = triage.simulate(m, ("a", "b"), {"a": 1.0, "b": 10.0}, cal, ~cal)
    # Thresholds: a at 0.9 (calib risk 0 down to 0.9); b at 0.8 on the calib items a left open.
    assert out["stage1_share"] == 0.25 and out["stage2_share"] == 0.5
    assert out["automation"] == 0.75 and out["auto_risk"] == pytest.approx(1 / 3)
    assert out["final_error_vs_human"] == 0.25 and out["human_decisions"] == 0.25
    assert out["api_usd_per_1k_segments"] == pytest.approx(1000 * (1.0 + 0.75 * 10.0))


def test_cascade_ltt_is_joint_and_turns_off_unhelpful_stages():
    rng = np.random.default_rng(6)
    n = 3000
    m = pd.DataFrame({"segment_id": [f"s{i}" for i in range(n)], "weight": 1.0, "group": np.arange(n) // 3,
                      "a|conf": rng.uniform(0.95, 1, n), "a|correct": (rng.random(n) > 0.002).astype(float),
                      "b|conf": rng.uniform(0.95, 1, n), "b|correct": (rng.random(n) > 0.5).astype(float)})
    cal = (m.group < 700).to_numpy()  # 700 calib clusters out of a corpus of 1,000
    t = triage.cascade_ltt(m, ("a", "b"), cal, bound=3.0)
    assert t[0] == pytest.approx(0.6) and t[1] == np.inf  # stage a takes everything; b (50% error) is switched off
    assert triage.cascade_ltt(m, ("b",), cal, bound=3.0) == (np.inf,)


def test_agreement_stage_uses_lower_confidence_and_drops_disagreements(monkeypatch):
    frames = {
        "a": pd.DataFrame({"segment_id": ["s1", "s2"], "pred": ["x", "x"], "conf": [0.9, 0.8], "correct": [1, 1], "correct_adj": [1, 1]}),
        "b": pd.DataFrame({"segment_id": ["s1", "s2"], "pred": ["x", "y"], "conf": [0.7, 0.99], "correct": [1, 0], "correct_adj": [1, 0]}),
    }
    real = triage.stage_columns
    monkeypatch.setattr(triage, "stage_columns", lambda t, s, k, a: frames[s] if s in frames else real(t, s, k, a))
    out = real("t", "agree:a+b", ["segment_id"], None)
    assert out.conf.tolist() == [0.7, -1.0] and out.pred.tolist() == ["x", "x"]


def test_human_preds_score_the_reference_itself():
    ref = pd.DataFrame({"segment_id": ["s1", "s2"], "ref_code": ["b", "a"]})
    td = SimpleNamespace(ref=ref, codebook=SimpleNamespace(mode="exclusive", code_ids=["a", "b"]))
    p = triage.human_preds(td).pivot(index="segment_id", columns="code_id", values="p")
    assert p.idxmax(axis=1).to_dict() == {"s1": "b", "s2": "a"}


def test_iid_cluster_ltt_controls_population_risk_where_decision_level_ltt_does_not():
    from scipy.stats import norm
    rng = np.random.default_rng(4)
    viol = {"ltt_dec": 0, "ltt_iid": 0}
    sims, n_clusters, size, rho = 150, 60, 40, 0.9
    g = np.repeat(np.arange(n_clusters), size)
    for _ in range(sims):  # independent clusters; true selective risk at t is (1 - t) / 2
        conf = rng.uniform(0.5, 1, n_clusters * size)
        z = rho * np.repeat(rng.standard_normal(n_clusters), size) + np.sqrt(1 - rho**2) * rng.standard_normal(n_clusters * size)
        err = (norm.cdf(z) > conf).astype(float)
        for rule in viol:
            viol[rule] += triage.fit_threshold(conf, err, np.ones_like(conf), rule, groups=g, bound=float(size)) < 0.9 - 1e-9
    assert viol["ltt_iid"] / sims <= 0.05
    assert viol["ltt_dec"] / sims >= 0.08


def test_feasibility_bound_is_a_lower_bound_on_the_procedure():
    from margin import controls
    delta = triage.DELTA / len(triage.GRID)
    r = np.ones(400)  # equal-weight units, as MFRC comments
    bound = controls._fewest_possible(r, delta)
    assert bound == int(np.ceil(np.log(1 / delta) / np.log1p(triage.EPS)))
    first = next(n for n in range(5, 400) if controls._perfect_coder_certified(r, n, 400, delta))
    assert first >= bound
    # Units that are small relative to the largest one cannot reach the evidence threshold at all.
    assert np.isnan(controls._fewest_possible(np.r_[1.0, np.full(150, 0.05)], delta))
