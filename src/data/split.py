"""Stratified train/val/test splitting. This module fits NOTHING (leakage prevention)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.ingest import ClaimsDataset


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "train": self.train.tolist(),
            "val": self.val.tolist(),
            "test": self.test.tolist(),
        }


def stratified_split(
    dataset: ClaimsDataset,
    ratios: dict[str, Any],
    seed: int,
) -> SplitIndices:
    """Two-stage stratified split on claim_status. Deterministic for a given seed."""
    n = len(dataset)
    if n < 10:
        raise ValueError(
            f"dataset too small to split ({n} rows) -- check the raw data source"
        )
    target = dataset.target.to_numpy()
    idx = np.arange(n)
    test_size = float(ratios["test"])
    val_ratio_of_rest = float(ratios["val"]) / (
        float(ratios["train"]) + float(ratios["val"])
    )
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=target
    )
    y_trainval = target[idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_ratio_of_rest, random_state=seed, stratify=y_trainval
    )
    split = SplitIndices(
        train=np.sort(idx_train), val=np.sort(idx_val), test=np.sort(idx_test)
    )
    _assert_disjoint(split, n)
    return split


def _assert_disjoint(split: SplitIndices, n: int) -> None:
    all_idx = np.concatenate([split.train, split.val, split.test])
    if len(all_idx) != n or len(np.unique(all_idx)) != n:
        raise RuntimeError(
            "train/val/test splits must be disjoint and exhaustive -- "
            "this indicates a bug in src/data/split.py"
        )


def split_hash(split: SplitIndices) -> str:
    payload = json.dumps(split.as_dict(), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def split_report(split: SplitIndices, dataset: ClaimsDataset, seed: int) -> dict[str, Any]:
    """Audit-friendly summary of the split (class rates + hash)."""
    y = dataset.target.to_numpy()
    def rate(a: np.ndarray) -> float:
        return float(y[a].mean()) if len(a) else float("nan")
    return {
        "seed": seed,
        "hash": split_hash(split),
        "n_train": int(len(split.train)),
        "n_val": int(len(split.val)),
        "n_test": int(len(split.test)),
        "claim_rate_train": rate(split.train),
        "claim_rate_val": rate(split.val),
        "claim_rate_test": rate(split.test),
        "n_rows_total": int(len(y)),
    }


def save_split_report(split: SplitIndices, dataset: ClaimsDataset, seed: int, path: str | Path) -> dict[str, Any]:
    report = split_report(split, dataset, seed)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
