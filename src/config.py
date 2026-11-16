"""Typed configuration loading and deterministic seeding."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class AppConfig:
    """Immutable view over configs/config.yaml."""
    seed: int
    data: dict[str, Any]
    split: dict[str, Any]
    features: dict[str, Any]
    tuning: dict[str, Any]
    metrics: dict[str, Any]
    risk_tiers: dict[str, Any]
    models: dict[str, Any]
    app: dict[str, Any]
    raw: dict[str, Any] = field(repr=False)
    path: str = ""

    @property
    def enabled_models(self) -> dict[str, Any]:
        return {k: v for k, v in self.models.items() if v.get("enabled", True)}

    @property
    def expected_severity(self) -> float:
        return float(self.data.get("expected_severity", 3000.0))


def load_config(path: str | Path = "configs/config.yaml") -> AppConfig:
    """Load and validate the YAML configuration."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Config file not found at {p} -- run 'make setup' or pass --config <path>"
        )
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    for section in ("seed", "data", "split", "features", "metrics", "risk_tiers", "models"):
        if section not in raw:
            raise ValueError(
                f"Config section '{section}' missing in {p} -- see configs/config.yaml template"
            )
    tr, va, te = (float(raw["split"][k]) for k in ("train", "val", "test"))
    if abs(tr + va + te - 1.0) > 1e-6:
        raise ValueError(
            f"split ratios must sum to 1.0, got train={tr} val={va} test={te} -- fix configs/config.yaml"
        )
    return AppConfig(
        seed=int(raw["seed"]),
        data=raw["data"],
        split=raw["split"],
        features=raw["features"],
        tuning=raw.get("tuning", {}),
        metrics=raw["metrics"],
        risk_tiers=raw["risk_tiers"],
        models=raw["models"],
        app=raw.get("app", {}),
        raw=raw,
        path=str(p),
    )


def seed_everything(seed: int) -> None:
    """Seed python and numpy RNGs from a single seed."""
    random.seed(seed)
    np.random.seed(seed)
