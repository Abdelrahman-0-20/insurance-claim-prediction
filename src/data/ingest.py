"""Data ingestion: local claims.csv or OpenML freMTPL2 fallback -> canonical schema.

Canonical schema (see docs/ARCHITECTURE.md section 3):
  policy_id (str), subscription_length/exposure (float), claim_status (int {0,1}),
  optional claim_amount (float), plus all raw feature columns.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ClaimsDataset:
    """Canonical in-memory dataset."""
    features: pd.DataFrame          # raw feature frame (includes subscription_length)
    exposure: pd.Series             # sample_weight column (canonical)
    target: pd.Series               # claim_status {0,1}
    claim_amount: pd.Series | None  # severity target when available (OpenML path)
    source: str = ""
    dropped_zero_exposure: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.features)


def _drop_zero_exposure(df: pd.DataFrame, exposure_col: str) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df[df[exposure_col] > 0]
    return df.copy(), before - len(df)


def _coerce_types(df: pd.DataFrame, target_col: str, exposure_col: str) -> pd.DataFrame:
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    df[exposure_col] = pd.to_numeric(df[exposure_col], errors="coerce").fillna(0.0).astype(float)
    bad = sorted(set(df[target_col].unique()) - {0, 1})
    if bad:
        raise ValueError(
            f"target '{target_col}' must be binary {{0,1}}, found {bad} -- check the raw data"
        )
    return df


def _assert_positive_exposure(df: pd.DataFrame, exposure_col: str) -> None:
    if (df[exposure_col] <= 0).any():
        n = int((df[exposure_col] <= 0).sum())
        raise ValueError(
            f"column '{exposure_col}' still contains {n} non-positive row(s) after cleaning "
            f"-- set data.drop_zero_exposure: true or fix the raw data"
        )


def load_local_claims(path: str | Path, cfg_data: dict[str, Any]) -> ClaimsDataset:
    """Load data/raw/claims.csv (local motor-portfolio source)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"claims file not found at {p} -- run 'make setup' to copy "
            "'Insurance claims data.csv' into data/raw/claims.csv"
        )
    df = pd.read_csv(p)
    target_col = cfg_data.get("target_column", "claim_status")
    id_col = cfg_data.get("id_column", "policy_id")
    aliases: dict[str, str] = cfg_data.get("aliases", {})
    for src, dst in aliases.items():
        if src in df.columns and dst not in df.columns:
            # canonical exposure column is a copy; source column stays as a feature
            df[dst] = pd.to_numeric(df[src], errors="coerce")
    if "exposure" not in df.columns:
        raise ValueError(
            f"no exposure column found in {p}; expected 'exposure' or an alias "
            f"(config data.aliases) -- check configs/config.yaml"
        )
    if id_col not in df.columns:
        df[id_col] = [f"row_{i}" for i in range(len(df))]
    if target_col not in df.columns:
        raise ValueError(
            f"target column '{target_col}' not found in {p} -- check config data.target_column"
        )
    df = _coerce_types(df, target_col, "exposure")
    dropped = 0
    if cfg_data.get("drop_zero_exposure", True):
        df, dropped = _drop_zero_exposure(df, "exposure")
        if dropped:
            logger.info("dropped %d rows with zero exposure", dropped)
    _assert_positive_exposure(df, "exposure")
    features = df.drop(columns=[c for c in (target_col, "claim_amount") if c in df.columns])
    return ClaimsDataset(
        features=features,
        exposure=df["exposure"].astype(float),
        target=df[target_col].astype(int),
        claim_amount=None,
        source=str(p),
        dropped_zero_exposure=dropped,
        meta={"n_rows": len(df)},
    )


def _fetch_openml(name: str) -> pd.DataFrame:
    from sklearn.datasets import fetch_openml
    return fetch_openml(name, as_frame=True, parser="auto").frame


def load_openml_claims(cfg_data: dict[str, Any]) -> ClaimsDataset:
    """Auto-fetch OpenML freMTPL2freq + freMTPL2sev and join into the canonical schema."""
    freq = _fetch_openml(cfg_data.get("openml_freq", "freMTPL2freq"))
    sev = _fetch_openml(cfg_data.get("openml_sev", "freMTPL2sev"))

    freq = freq.rename(columns={"IDpol": "policy_id"})
    sev = sev.rename(columns={"IDpol": "policy_id"})
    freq["policy_id"] = freq["policy_id"].astype(str)
    sev["policy_id"] = sev["policy_id"].astype(str)

    sev_agg = (
        sev.groupby("policy_id", as_index=False)
        .agg(claim_amount=("ClaimAmount", "sum"), n_claims_sev=("ClaimAmount", "size"))
    )
    df = freq.merge(sev_agg, on="policy_id", how="left")
    df["claim_amount"] = df["claim_amount"].fillna(0.0)

    # canonical columns
    df["exposure"] = pd.to_numeric(df["Exposure"], errors="coerce").astype(float)
    df["claim_status"] = (
        pd.to_numeric(df.get("ClaimNb", 0), errors="coerce").fillna(0).gt(0).astype(int)
    )
    # Keep "Exposure" as an input FEATURE (approved adjustment 2); canonical "exposure"
    # is the sample_weight copy excluded from features by config.
    drop_cols = [c for c in ("ClaimNb", "ClaimAmount", "n_claims_sev") if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    df = _coerce_types(df, "claim_status", "exposure")
    dropped = 0
    if cfg_data.get("drop_zero_exposure", True):
        df, dropped = _drop_zero_exposure(df, "exposure")
        if dropped:
            logger.info("dropped %d rows with zero exposure (OpenML)", dropped)
    _assert_positive_exposure(df, "exposure")

    claim_amount = df.pop("claim_amount") if "claim_amount" in df.columns else None
    id_col = cfg_data.get("id_column", "policy_id")
    features = df.drop(columns=["claim_status"])
    if id_col not in features.columns:
        features[id_col] = [f"row_{i}" for i in range(len(features))]
    return ClaimsDataset(
        features=features,
        exposure=df["exposure"].astype(float),
        target=df["claim_status"].astype(int),
        claim_amount=claim_amount.astype(float) if claim_amount is not None else None,
        source="openml:freMTPL2freq+freMTPL2sev",
        dropped_zero_exposure=dropped,
        meta={"n_rows": len(df)},
    )


def load_dataset(cfg: Any) -> ClaimsDataset:
    """Load local claims.csv when present, otherwise auto-fetch OpenML fallback."""
    raw_path = Path(cfg.data.get("raw_path", "data/raw/claims.csv"))
    if raw_path.exists():
        return load_local_claims(raw_path, cfg.data)
    logger.info("%s absent -> OpenML freMTPL2 fallback", raw_path)
    return load_openml_claims(cfg.data)
