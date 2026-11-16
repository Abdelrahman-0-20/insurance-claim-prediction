"""Metric correctness on synthetic data (perfect / random / degenerate cases)."""
import numpy as np
import pytest

from src.evaluation.metrics import (
    calibration_bins,
    cumulative_lift,
    frequency_report,
    normalized_gini,
    pr_auc,
    pure_premium_report,
    roc_auc,
    tweedie_deviance,
)


def test_perfect_ranking_metrics():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert normalized_gini(y, s) == pytest.approx(1.0)
    assert pr_auc(y, s) == pytest.approx(1.0)
    assert roc_auc(y, s) == pytest.approx(1.0)
    # perfectly inverted ranking
    assert normalized_gini(y, 1.0 - s) == pytest.approx(-1.0)


def test_gini_works_on_cost_scaled_targets():
    """Pure-premium targets are cost-scaled (claim x severity), not binary."""
    rng = np.random.RandomState(5)
    claim = (rng.rand(1000) < 0.1).astype(float)
    cost = claim * 3000.0                      # {0, 3000} cost target
    score = claim * 0.5 + rng.rand(1000) * 0.5
    g = normalized_gini(cost, score)
    assert np.isfinite(g), "cost-scaled targets must not trip the degenerate guard"
    assert g > 0.3
    # all-equal target stays undefined (no ranking signal to normalize against)
    assert np.isnan(normalized_gini(np.full(50, 7.0), rng.rand(50)))


def test_random_predictor_near_baseline():
    rng = np.random.RandomState(0)
    y = (rng.rand(5000) < 0.1).astype(int)
    s = rng.rand(5000)
    assert abs(pr_auc(y, s) - 0.1) < 0.05
    assert abs(roc_auc(y, s) - 0.5) < 0.05
    assert abs(normalized_gini(y, s)) < 0.1


def test_tweedie_deviance_properties():
    # perfect predictions (strictly positive) -> zero deviance
    y_pos = np.array([1.0, 10.0, 2.0, 5.0])
    assert tweedie_deviance(y_pos, y_pos, power=1.5) == pytest.approx(0.0, abs=1e-8)
    # zero-claim rows with near-zero mu stay finite and ~0
    y = np.array([0.0, 10.0, 0.0, 5.0])
    mu_near = np.array([1e-12, 10.0, 1e-12, 5.0])
    assert tweedie_deviance(y, mu_near, power=1.5) == pytest.approx(0.0, abs=1e-4)
    # wrong predictions -> positive, finite
    d = tweedie_deviance(y, np.full(4, 2.0), power=1.5)
    assert d > 0 and np.isfinite(d)
    # non-negative on random data
    rng = np.random.RandomState(1)
    yr = rng.rand(500)
    mr = rng.rand(500) + 0.1
    assert tweedie_deviance(yr, mr, power=1.5) > 0
    # exposure weights accepted
    w = rng.uniform(0.1, 10.0, 500)
    assert np.isfinite(tweedie_deviance(yr, mr, power=1.5, sample_weight=w))


def test_calibration_bins_consistent():
    rng = np.random.RandomState(2)
    n = 1000
    y = (rng.rand(n) < 0.2).astype(float)
    p = np.clip(rng.rand(n), 0.001, 0.999)
    w = rng.uniform(0.5, 5.0, n)
    c = calibration_bins(y, p, w, n_bins=10)
    assert sum(c["count"]) == n
    assert len(c["mean_predicted"]) == len(c["observed_rate"]) == len(c["count"])
    assert all(0.0 <= m <= 1.0 for m in c["mean_predicted"])
    assert all(0.0 <= o <= 1.0 for o in c["observed_rate"])


def test_cumulative_lift_signal_and_constant():
    rng = np.random.RandomState(3)
    n = 5000
    y = (rng.rand(n) < 0.1).astype(float)
    # score correlated with outcome -> strong top-decile lift, ends at 1.0
    score = y * 0.6 + rng.rand(n) * 0.4
    rows = cumulative_lift(y, score, None, n_deciles=10)
    assert rows[0]["lift"] > 1.5
    assert rows[-1]["lift"] == pytest.approx(1.0, abs=1e-9)
    # constant score -> lift near baseline everywhere
    rows_c = cumulative_lift(y, np.full(n, 0.5), None, n_deciles=10)
    assert all(abs(r["lift"] - 1.0) < 0.15 for r in rows_c)


def test_report_builders_have_required_keys():
    rng = np.random.RandomState(4)
    n = 800
    y = (rng.rand(n) < 0.15).astype(int)
    p = rng.rand(n)
    w = rng.uniform(0.5, 3.0, n)
    freq = frequency_report(y, p, w, calibration_bins_n=5, lift_deciles=5)
    for key in ("pr_auc", "roc_auc", "brier", "normalized_gini", "calibration", "cumulative_lift"):
        assert key in freq
    pp = pure_premium_report(y.astype(float), p * 3000.0, w, power=1.5)
    assert "tweedie_deviance" in pp and np.isfinite(pp["tweedie_deviance"])
