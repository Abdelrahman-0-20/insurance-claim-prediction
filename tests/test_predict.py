"""Inference wrapper contract: outputs, ordering, validation, explanations, I/O."""
import numpy as np
import pandas as pd
import pytest

OUT_COLS = [
    "claim_probability",
    "expected_pure_premium",
    "risk_tier",
    "claim_frequency_per_exposure",
]


def test_predict_output_contract(small_bundle, small_bundle_raw):
    out = small_bundle.predict(small_bundle_raw)
    assert list(out.columns) == OUT_COLS
    prob = out["claim_probability"].to_numpy()
    assert np.all((prob >= 0.0) & (prob <= 1.0))
    # approved adjustment 1: pure premium = probability x severity (exact)
    sev = small_bundle.meta["severity"]
    np.testing.assert_allclose(
        out["expected_pure_premium"].to_numpy(), prob * sev, rtol=1e-12
    )
    # frequency per exposure is probability / subscription_length
    exp = small_bundle_raw["subscription_length"].to_numpy(dtype=float)
    np.testing.assert_allclose(
        out["claim_frequency_per_exposure"].to_numpy(), prob / exp, rtol=1e-10
    )


def test_risk_tiers_partition_all_rows(small_bundle, small_bundle_raw):
    out = small_bundle.predict(small_bundle_raw)
    tiers = out["risk_tier"]
    assert set(tiers.unique()) <= {"Low", "Medium", "High"}
    assert len(tiers) == len(small_bundle_raw)
    # tiers must be contiguous cuts around the stored thresholds
    p = out["claim_probability"]
    assert (p[tiers == "Low"] <= small_bundle.meta["tier_low_max"] + 1e-12).all()
    assert (p[tiers == "High"] >= small_bundle.meta["tier_high_min"] - 1e-12).all()
    mid = tiers == "Medium"
    assert (
        (p[mid] > small_bundle.meta["tier_low_max"])
        & (p[mid] < small_bundle.meta["tier_high_min"])
    ).all()


def test_row_order_and_count_preserved(small_bundle, small_bundle_raw):
    shuffled = small_bundle_raw.sample(frac=1.0, random_state=7).reset_index(drop=True)
    out = small_bundle.predict(shuffled)
    assert len(out) == len(shuffled)
    # per-row predictions must follow the input rows, not any canonical order
    out_orig = small_bundle.predict(small_bundle_raw)
    idx = shuffled.index[:5]
    # recompute expected for the first shuffled row directly
    single = small_bundle.predict(shuffled.iloc[:1])
    row0 = shuffled.iloc[:1]
    prob_from_full = out.loc[row0.index[0], "claim_probability"]
    assert np.isclose(prob_from_full, single["claim_probability"].iloc[0])


def test_validate_missing_column_is_actionable(small_bundle, small_bundle_raw):
    bad = small_bundle_raw.drop(columns=["customer_age"])
    rep = small_bundle.validate_input(bad)
    assert not rep.ok
    assert any("customer_age" in e for e in rep.errors)
    assert any("schema" in e or "provide" in e for e in rep.errors)
    with pytest.raises(ValueError, match="customer_age"):
        small_bundle.predict(bad)


def test_validate_extra_column_warns_not_fails(small_bundle, small_bundle_raw):
    extra = small_bundle_raw.copy()
    extra["some_future_column"] = 1
    rep = small_bundle.validate_input(extra)
    assert rep.ok
    assert any("some_future_column" in w for w in rep.warnings)
    out = small_bundle.predict(extra)
    assert len(out) == len(extra)


def test_validate_unknown_category_warns(small_bundle, small_bundle_raw):
    df = small_bundle_raw.copy()
    df.loc[df.index[0], "region_code"] = "ZZ9"
    rep = small_bundle.validate_input(df)
    assert rep.ok                      # scored as unknown, not rejected
    assert any("region_code" in w for w in rep.warnings)


def test_explain_local_shape_and_ordering(small_bundle, small_bundle_raw):
    contrib = small_bundle.explain_local(small_bundle_raw, row=0)
    assert list(contrib.columns) == ["feature", "contribution"]
    assert len(contrib) > 0
    assert contrib["feature"].is_unique
    # sorted by absolute contribution, descending
    abs_c = contrib["contribution"].abs().to_numpy()
    assert np.all(np.diff(abs_c) <= 1e-12)
    # contributions should aggregate to something on the model's scale
    assert np.isfinite(contrib["contribution"]).all()


def test_save_load_roundtrip(small_bundle, small_bundle_raw, tmp_path):
    path = tmp_path / "bundle.joblib"
    small_bundle.save(path)
    from src.models.predict import ClaimsModelBundle
    loaded = ClaimsModelBundle.load(path)
    a = small_bundle.predict(small_bundle_raw)
    b = loaded.predict(small_bundle_raw)
    pd.testing.assert_frame_equal(a, b)
    assert loaded.meta["model_name"] == small_bundle.meta["model_name"]


def test_load_missing_bundle_names_fix(tmp_path):
    from src.models.predict import ClaimsModelBundle
    with pytest.raises(FileNotFoundError, match="make train"):
        ClaimsModelBundle.load(tmp_path / "nope.joblib")


def test_exposure_feature_used_for_frequency(small_bundle, small_bundle_raw):
    """Approved adjustment 2: subscription_length is the exposure input."""
    df = small_bundle_raw.copy()
    df["subscription_length"] = df["subscription_length"] * 2.0
    out = small_bundle.predict(df)
    base = small_bundle.predict(small_bundle_raw)
    # probability itself moves with duration (feature), frequency per exposure falls
    prob_changed = not np.allclose(
        out["claim_probability"].to_numpy(), base["claim_probability"].to_numpy()
    )
    assert prob_changed
    np.testing.assert_allclose(
        out["claim_frequency_per_exposure"].to_numpy(),
        out["claim_probability"].to_numpy() / df["subscription_length"].to_numpy(),
        rtol=1e-10,
    )
