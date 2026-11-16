"""Approved adjustment 3: risk-tier thresholds come from out-of-fold scores."""
import numpy as np
import pandas as pd
import pytest

from src.models.predict import compute_tier_thresholds


def test_thresholds_are_p25_p75_of_given_scores():
    scores = np.arange(1, 101, dtype=float) / 100.0   # 0.01 .. 1.00
    low, high = compute_tier_thresholds(scores, (0.25, 0.75))
    assert np.isclose(low, np.quantile(scores, 0.25))
    assert np.isclose(high, np.quantile(scores, 0.75))
    assert low < high


def test_thresholds_empty_array_raises():
    with pytest.raises(ValueError, match="empty"):
        compute_tier_thresholds([], (0.25, 0.75))


def test_bundle_uses_stored_thresholds_not_recomputed(small_bundle, small_bundle_raw):
    """Tier assignment follows the STORED OOF cutoffs, never thresholds recomputed
    from the input batch's own score distribution."""
    from src.models.predict import ClaimsModelBundle

    # contrived case: stored thresholds that no input-quantile rule would produce
    meta = dict(small_bundle.meta)
    meta["tier_low_max"] = 0.0      # nothing scores <= 0  -> no Low rows
    meta["tier_high_min"] = 1e9     # nothing scores >= 1e9 -> no High rows
    bundle = ClaimsModelBundle(
        small_bundle.preprocessor, small_bundle.estimator, meta, None
    )
    out = bundle.predict(small_bundle_raw)
    assert (out["risk_tier"] == "Medium").all()

    # while the input's own P25/P75 would have produced Low and High rows
    p = out["claim_probability"].to_numpy()
    input_q = np.quantile(p, [0.25, 0.75])
    assert input_q[0] > 0.0 and input_q[1] < 1e9

    # and with the real stored thresholds, assignment is exactly the stored cut
    out0 = small_bundle.predict(small_bundle_raw)
    low = small_bundle.meta["tier_low_max"]
    high = small_bundle.meta["tier_high_min"]
    expected = np.where(p <= low, "Low", np.where(p >= high, "High", "Medium"))
    np.testing.assert_array_equal(out0["risk_tier"].to_numpy(), expected)
