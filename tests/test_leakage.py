"""Zero data leakage: train-only fitted statistics, no target in features,
and a split module that fits nothing."""
import numpy as np
import pandas as pd

from src.features.build_features import (
    build_domain_features,
    build_preprocessor,
    resolve_columns,
    transform_feature_names,
)


def test_preprocessor_statistics_are_train_only():
    """Statistics must equal TRAIN medians, not full-data medians."""
    rng = np.random.RandomState(0)
    a_train = rng.rand(50)                 # ~[0, 1]
    a_holdout = 100.0 + rng.rand(50)       # ~[100, 101]
    X = pd.DataFrame({
        "a": np.concatenate([a_train, a_holdout]),
        "c": ["x", "y"] * 50,
    })
    pre = build_preprocessor("onehot", ["a"], ["c"], seed=42)
    pre.fit(X.iloc[:50])                   # fit on train rows ONLY
    stats = pre.named_transformers_["num"].named_steps["imputer"].statistics_
    train_median = float(np.median(a_train))
    full_median = float(np.median(X["a"]))
    assert np.isclose(stats[0], train_median)
    assert not np.isclose(stats[0], full_median)  # fixture guarantees the two differ


def test_target_and_weight_columns_never_in_features(synthetic_claims, cfg):
    df = synthetic_claims.copy()
    df["exposure"] = df["subscription_length"]
    domain = build_domain_features(df.drop(columns=["claim_status"]))
    numeric, categorical = resolve_columns(
        domain, set(cfg.features["exclude_columns"])
    )
    pre = build_preprocessor("onehot", numeric, categorical, seed=42)
    pre.fit(domain[numeric + categorical])
    names, _ = transform_feature_names(pre, numeric + categorical)
    for forbidden in ("claim_status", "exposure", "policy_id", "claim_amount"):
        assert not any(forbidden == n or n.endswith("__" + forbidden) for n in names)


def test_split_module_fits_nothing():
    """src/data/split.py must not contain any estimator .fit call."""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "src" / "data" / "split.py").read_text()
    assert ".fit(" not in src
    assert "StandardScaler" not in src
    assert "TargetEncoder" not in src


def test_domain_features_are_stateless(synthetic_claims):
    """Pure function: same input -> identical output, no stored state."""
    a = build_domain_features(synthetic_claims)
    b = build_domain_features(synthetic_claims)
    pd.testing.assert_frame_equal(a, b)
