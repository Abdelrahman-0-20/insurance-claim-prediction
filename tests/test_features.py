"""Domain feature engineering determinism, ratio guards, preprocessor determinism,
and unknown-category safety across encoding modes."""
import numpy as np
import pandas as pd

from src.features.build_features import build_domain_features, build_preprocessor


def test_domain_features_deterministic(synthetic_claims):
    out1 = build_domain_features(synthetic_claims)
    out2 = build_domain_features(synthetic_claims)
    pd.testing.assert_frame_equal(out1, out2)


def test_domain_features_adds_expected_columns(synthetic_claims):
    out = build_domain_features(synthetic_claims)
    for col in (
        "torque_nm", "power_bhp", "driver_vehicle_age_ratio",
        "vehicle_customer_age_ratio", "specific_power", "weight_power_ratio",
    ):
        assert col in out.columns
    # parsed values are correct for the fixture strings
    assert np.isclose(out["torque_nm"].iloc[0] % 1, 0.0)
    assert out["torque_nm"].dropna().isin([100.0, 250.0]).all()
    assert out["power_bhp"].dropna().isin([80.0, 113.45]).all()


def test_ratio_guards_zero_denominator():
    df = pd.DataFrame({
        "customer_age": [40, 50],
        "vehicle_age": [0.0, 3.0],
        "max_power": ["80bhp@4000rpm", "0bhp@0rpm"],
        "max_torque": ["100Nm@3000rpm", "250Nm@2750rpm"],
        "displacement": [0, 1500],
        "gross_weight": [1200, 1500],
    })
    out = build_domain_features(df)
    for col in ("driver_vehicle_age_ratio", "vehicle_customer_age_ratio",
                "specific_power", "weight_power_ratio"):
        assert not np.isinf(out[col].to_numpy(dtype=float)).any(), col
    # zero displacement -> NaN (not inf); zero vehicle age -> finite ratio
    assert np.isnan(out["specific_power"].iloc[0])
    assert np.isfinite(out["driver_vehicle_age_ratio"].iloc[0])


def _toy_frame():
    rng = np.random.RandomState(1)
    n = 80
    return pd.DataFrame({
        "num1": rng.rand(n),
        "num2": rng.randint(0, 100, n).astype(float),
        "cat1": rng.choice(["a", "b", "c"], n),
        "cat2": rng.choice(["u", "v"], n),
    })


def test_preprocessor_fit_transform_deterministic():
    X = _toy_frame()
    y = (X["num1"] > 0.5).astype(int).to_numpy()
    outs = []
    for _ in range(2):
        pre = build_preprocessor("target", ["num1", "num2"], ["cat1", "cat2"], seed=42)
        outs.append(pre.fit_transform(X, y))
    np.testing.assert_allclose(outs[0], outs[1])


def test_unknown_categories_do_not_crash():
    X = _toy_frame()
    y = (X["num1"] > 0.5).astype(int).to_numpy()
    X_new = X.copy()
    X_new.loc[0, "cat1"] = "UNSEEN_CATEGORY"
    X_new.loc[1, "cat2"] = "ALSO_UNSEEN"
    for encoding in ("target", "native", "onehot"):
        pre = build_preprocessor(encoding, ["num1", "num2"], ["cat1", "cat2"], seed=42)
        pre.fit(X, y)
        transformed = pre.transform(X_new)
        assert np.isfinite(np.asarray(transformed, dtype=float)).all(), encoding
