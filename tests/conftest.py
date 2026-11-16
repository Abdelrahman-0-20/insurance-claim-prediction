"""Shared fixtures: synthetic claims data and a small trained bundle."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config


@pytest.fixture(scope="session")
def cfg():
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    return load_config(path)


@pytest.fixture(scope="session")
def synthetic_claims() -> pd.DataFrame:
    """Small claims frame mirroring the raw schema, with a learnable signal."""
    rng = np.random.RandomState(42)
    n = 600
    airbags = rng.choice(["Yes", "No"], n)
    df = pd.DataFrame({
        "policy_id": [f"P{i:04d}" for i in range(n)],
        "subscription_length": rng.uniform(0.5, 12.0, n).round(1),
        "vehicle_age": rng.uniform(0.0, 15.0, n).round(1),
        "customer_age": rng.randint(30, 75, n),
        "region_code": rng.choice(["C1", "C2", "C3", "C4"], n),
        "region_density": rng.randint(1000, 30000, n),
        "segment": rng.choice(["A1", "B2"], n),
        "max_torque": rng.choice(["100Nm@3000rpm", "250Nm@2750rpm"], n),
        "max_power": rng.choice(["80bhp@4000rpm", "113.45bhp@4000rpm"], n),
        "displacement": rng.randint(800, 2500, n),
        "gross_weight": rng.randint(900, 2500, n),
        "airbags": airbags,
        "ncap_rating": rng.randint(0, 5, n),
    })
    logit = (
        0.04 * (df["customer_age"].to_numpy() - 45)
        + 0.06 * df["vehicle_age"].to_numpy()
        - 0.45 * (airbags == "Yes")
        + rng.normal(0.0, 1.0, n)
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    df["claim_status"] = (rng.rand(n) < p).astype(int)
    return df


@pytest.fixture(scope="session")
def dataset(synthetic_claims):
    from src.data.ingest import ClaimsDataset
    df = synthetic_claims.copy()
    df["exposure"] = df["subscription_length"]
    features = df.drop(columns=["claim_status"])
    return ClaimsDataset(
        features=features,
        exposure=df["exposure"].astype(float),
        target=df["claim_status"].astype(int),
        claim_amount=None,
        source="synthetic",
    )


EXCLUDE = {"policy_id", "claim_status", "claim_amount", "exposure"}


def make_bundle(severity: float = 3000.0):
    """Fit a small logistic bundle on the synthetic data (fast, deterministic)."""
    from src.features.build_features import (
        build_domain_features,
        build_preprocessor,
        resolve_columns,
        transform_feature_names,
    )
    from src.models.predict import ClaimsModelBundle, compute_tier_thresholds
    from src.models.tune import build_estimator, categorical_indices, fit_estimator, predict_score

    rng = np.random.RandomState(42)
    n = 600
    airbags = rng.choice(["Yes", "No"], n)
    df = pd.DataFrame({
        "policy_id": [f"P{i:04d}" for i in range(n)],
        "subscription_length": rng.uniform(0.5, 12.0, n).round(1),
        "vehicle_age": rng.uniform(0.0, 15.0, n).round(1),
        "customer_age": rng.randint(30, 75, n),
        "region_code": rng.choice(["C1", "C2", "C3", "C4"], n),
        "region_density": rng.randint(1000, 30000, n),
        "segment": rng.choice(["A1", "B2"], n),
        "max_torque": rng.choice(["100Nm@3000rpm", "250Nm@2750rpm"], n),
        "max_power": rng.choice(["80bhp@4000rpm", "113.45bhp@4000rpm"], n),
        "displacement": rng.randint(800, 2500, n),
        "gross_weight": rng.randint(900, 2500, n),
        "airbags": airbags,
        "ncap_rating": rng.randint(0, 5, n),
    })
    logit = (
        0.04 * (df["customer_age"].to_numpy() - 45)
        + 0.06 * df["vehicle_age"].to_numpy()
        - 0.45 * (airbags == "Yes")
        + rng.normal(0.0, 1.0, n)
    )
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.rand(n) < p).astype(int)
    w = df["subscription_length"].to_numpy(dtype=float)

    raw = df.copy()
    domain = build_domain_features(raw)
    numeric, categorical = resolve_columns(domain, EXCLUDE)
    feature_columns = [c for c in domain.columns if c not in EXCLUDE]
    required_columns = [c for c in raw.columns if c not in EXCLUDE]

    pre = build_preprocessor(
        "onehot", numeric, categorical, seed=42, scale_numeric=True
    )
    X = pre.fit_transform(domain, y)
    names, groups = transform_feature_names(pre, numeric + categorical)
    cat_idx = categorical_indices(names)
    est = build_estimator("glm", "frequency", {"C": 0.5, "max_iter": 5000}, 42)
    fit_estimator(est, X, y, w, cat_idx, "glm")
    scores = predict_score(est, X, cat_idx, "glm", "frequency")
    low_max, high_min = compute_tier_thresholds(scores, (0.25, 0.75))

    allowed = {
        c: sorted(domain[c].astype(str).unique().tolist()) for c in categorical
    }
    meta = {
        "model_name": "small_glm",
        "task": "frequency",
        "family": "glm",
        "encoding": "onehot",
        "feature_columns": feature_columns,
        "required_columns": required_columns,
        "numeric_cols": numeric,
        "categorical_cols": categorical,
        "feature_names": names,
        "feature_groups": groups,
        "severity": severity,
        "tier_low_max": low_max,
        "tier_high_min": high_min,
        "tier_quantiles": [0.25, 0.75],
        "oof_summary": {"n": n, "mean": float(scores.mean()),
                        "min": float(scores.min()), "max": float(scores.max())},
        "seed": 42,
        "allowed_categories": allowed,
        "exposure_feature": "subscription_length",
        "trained_at": "test",
    }
    return ClaimsModelBundle(pre, est, meta, None)


@pytest.fixture(scope="session")
def small_bundle():
    return make_bundle()


@pytest.fixture(scope="session")
def small_bundle_raw(synthetic_claims) -> pd.DataFrame:
    df = synthetic_claims.copy()
    return df.drop(columns=["claim_status"])
