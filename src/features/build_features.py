"""Domain feature engineering and the sklearn ColumnTransformer factory.

Design (docs/ARCHITECTURE.md sections 4-5):
- build_domain_features: pure pandas, deterministic, no fitting (safe before splitting).
- build_preprocessor: ColumnTransformer with Pipeline branches, fitted ONLY on training
  data by callers (leakage prevention).
- build_feature_groups: maps transformed feature names back to raw source columns for SHAP.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)

EPS = 1e-9


def _parse_leading_number(series: pd.Series, pattern: str) -> pd.Series:
    """Extract a leading numeric value from strings like '250Nm@2750rpm'."""
    extracted = series.astype(str).str.extract(pattern, expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Guarded division: non-positive/NaN denominators -> NaN (imputer handles them)."""
    num = pd.to_numeric(num, errors="coerce").astype(float)
    den = pd.to_numeric(den, errors="coerce").astype(float)
    den = den.where(den.abs() > EPS, other=np.nan)
    ratio = num / den
    return ratio.replace([np.inf, -np.inf], np.nan)


def build_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add parsed mechanics and domain ratios. Pure, deterministic, no fitting."""
    out = df.copy()

    if "max_torque" in out.columns:
        out["torque_nm"] = _parse_leading_number(out["max_torque"], r"([0-9.]+)\s*Nm")
    if "max_power" in out.columns:
        out["power_bhp"] = _parse_leading_number(out["max_power"], r"([0-9.]+)\s*bhp")

    if "customer_age" in out.columns and "vehicle_age" in out.columns:
        out["driver_vehicle_age_ratio"] = _safe_ratio(
            out["customer_age"], out["vehicle_age"] + 1.0
        )
        out["vehicle_customer_age_ratio"] = _safe_ratio(
            out["vehicle_age"], out["customer_age"] + 1.0
        )
    if "power_bhp" in out.columns and "displacement" in out.columns:
        out["specific_power"] = _safe_ratio(out["power_bhp"], out["displacement"])
    if "power_bhp" in out.columns and "gross_weight" in out.columns:
        out["weight_power_ratio"] = _safe_ratio(out["gross_weight"], out["power_bhp"])

    # normalize dtypes: bool -> str, object NaN -> explicit missing category
    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].astype(str)
        elif out[col].dtype == object:
            out[col] = out[col].fillna("__missing__").astype(str)
    return out


def resolve_columns(
    df: pd.DataFrame, exclude: Iterable[str] = ()
) -> tuple[list[str], list[str]]:
    """Deterministically resolve (numeric_cols, categorical_cols) from the frame."""
    exclude = set(exclude)
    numeric, categorical = [], []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def build_preprocessor(
    encoding: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    seed: int = 42,
    scale_numeric: bool | None = None,
) -> ColumnTransformer:
    """Build the per-model ColumnTransformer (branches are Pipelines).

    encoding: 'target' (GLMs) | 'native' (LightGBM/CatBoost) | 'onehot' (XGBoost)
    scale_numeric: defaults to True for GLM encodings, False for tree encodings.
    """
    if scale_numeric is None:
        scale_numeric = encoding == "target"

    num_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    if encoding == "target":
        cat_encoder: Any = TargetEncoder(
            cv=KFold(n_splits=5, shuffle=True, random_state=seed)
        )
    elif encoding == "native":
        # integer codes; unseen categories -> -1 (estimators treat as unknown/missing)
        cat_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
    elif encoding == "onehot":
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        raise ValueError(
            f"unknown encoding '{encoding}' -- expected target|native|onehot in config"
        )

    cat_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("encoder", cat_encoder),
    ]

    transformers: list[tuple[str, Any, list[str]]] = [
        ("num", Pipeline(num_steps), numeric_cols),
    ]
    if categorical_cols:
        transformers.append(("cat", Pipeline(cat_steps), categorical_cols))
    return ColumnTransformer(
        transformers, remainder="drop", sparse_threshold=0.0
    )


def build_feature_groups(
    raw_columns: list[str], feature_names_out: list[str]
) -> dict[str, str]:
    """Map each transformed feature name to its raw source column (SHAP aggregation).

    Handles transformer prefixes (num__/cat__), indicator suffixes, and one-hot
    expansions via longest-prefix matching against the raw column names.
    """
    raw_sorted = sorted(raw_columns, key=len, reverse=True)
    groups: dict[str, str] = {}
    for name in feature_names_out:
        local = name.split("__", 1)[1] if "__" in name else name
        match = next(
            (c for c in raw_sorted if local == c or local.startswith(c + "_")),
            None,
        )
        groups[name] = match or local
    return groups


def transform_feature_names(
    preprocessor: ColumnTransformer, raw_columns: list[str]
) -> tuple[list[str], dict[str, str]]:
    """Output feature names and the output->raw column group map."""
    names = list(preprocessor.get_feature_names_out())
    return names, build_feature_groups(raw_columns, names)
