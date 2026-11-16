"""Inference wrapper: ClaimsModelBundle (predict / validate_input / explain_local).

Contract (docs/ARCHITECTURE.md section 9):
- predict(df) preserves row order and count and returns claim_probability,
  expected_pure_premium (= probability x severity for frequency models),
  risk_tier from OOF-derived thresholds, claim_frequency_per_exposure.
- validate_input returns an actionable ValidationReport (errors name column + fix).
- explain_local returns raw-column-aggregated SHAP contributions for one policy.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import (
    build_domain_features,
    transform_feature_names,
)

logger = logging.getLogger(__name__)

EXPOSURE_FEATURE_CANDIDATES = ("subscription_length", "Exposure", "exposure")


def compute_tier_thresholds(
    scores, quantiles: tuple[float, float] = (0.25, 0.75)
) -> tuple[float, float]:
    """Risk-tier cutoffs from OUT-OF-FOLD validation scores (approved adjustment 3).

    scores must be out-of-fold predictions on train+val, never in-sample scores
    after the final refit.
    """
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        raise ValueError("cannot compute risk-tier thresholds from an empty score array")
    low_max = float(np.quantile(arr, quantiles[0]))
    high_min = float(np.quantile(arr, quantiles[1]))
    return low_max, high_min


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_rows: int = 0

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValueError("Input validation failed:\n" + "\n".join(self.errors))


class ClaimsModelBundle:
    """Serializable bundle: preprocessor + estimator + audit metadata."""

    def __init__(
        self,
        preprocessor: Any,
        estimator: Any,
        meta: dict[str, Any],
        shap_background: np.ndarray | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.meta = dict(meta)
        self.shap_background = shap_background
        self._explainer: Any = None  # lazy, excluded from pickling

    # ------------------------------------------------------------------ #
    # Pickling: drop the lazy explainer
    # ------------------------------------------------------------------ #
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_explainer"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate_input(self, df: pd.DataFrame) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        if not isinstance(df, pd.DataFrame):
            return ValidationReport(False, ["input must be a pandas DataFrame"], [], 0)

        required: list[str] = self.meta["required_columns"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            errors.append(
                "missing required column(s): "
                + ", ".join(missing)
                + " -- provide them in the CSV or form (see artifacts/reports/schema.json)"
            )

        allowed: dict[str, list[str]] = self.meta.get("allowed_categories", {})
        for col in required:
            if col not in df.columns:
                continue
            if col in allowed:
                vals = set(df[col].dropna().astype(str).unique())
                unknown = sorted(vals - set(allowed[col]))
                if unknown:
                    warnings.append(
                        f"column '{col}': {len(unknown)} value(s) not seen in training "
                        f"(e.g. {unknown[:3]}) -- scored as unknown"
                    )
            elif pd.api.types.is_numeric_dtype(self.meta.get("_train_dtypes", {}).get(col, "numeric")):
                pass
        extras = [c for c in df.columns if c not in required and c not in ("policy_id", "claim_status", "claim_amount", "exposure")]
        if extras:
            warnings.append(
                "extra column(s) ignored: " + ", ".join(sorted(extras)[:8])
            )
        return ValidationReport(not errors, errors, warnings, len(df))

    def _coerce(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.meta["numeric_cols"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        for col in self.meta["categorical_cols"]:
            if col in out.columns:
                out[col] = out[col].astype(str)
        return out

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def _transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        raw = build_domain_features(self._coerce(df))
        X = self.preprocessor.transform(raw)
        return raw, np.asarray(X)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        report = self.validate_input(df)
        report.raise_if_failed()
        raw, Xt = self._transform(df)
        task = self.meta["task"]
        severity = float(self.meta["severity"])

        if task == "frequency":
            prob = self._score(Xt)
            expected_pure_premium = prob * severity
        else:
            expected_pure_premium = self._score(Xt)
            prob = np.full(len(expected_pure_premium), np.nan)

        low_max = float(self.meta["tier_low_max"])
        high_min = float(self.meta["tier_high_min"])
        score_for_tier = prob if task == "frequency" else expected_pure_premium
        tiers = np.where(
            score_for_tier <= low_max, "Low",
            np.where(score_for_tier >= high_min, "High", "Medium"),
        )

        exposure = self._exposure(raw, len(df))
        with np.errstate(divide="ignore", invalid="ignore"):
            freq_per_exposure = np.where(
                exposure > 0, prob / exposure, np.nan
            ) if task == "frequency" else np.full(len(df), np.nan)

        return pd.DataFrame({
            "claim_probability": prob,
            "expected_pure_premium": expected_pure_premium,
            "risk_tier": tiers,
            "claim_frequency_per_exposure": freq_per_exposure,
        }, index=df.index)

    def _score(self, Xt: np.ndarray) -> np.ndarray:
        from src.models.tune import categorical_indices, predict_score
        names, _ = transform_feature_names(
            self.preprocessor, self.meta["numeric_cols"] + self.meta["categorical_cols"]
        )
        cat_idx = categorical_indices(names)
        return predict_score(
            self.estimator, Xt, cat_idx, self.meta["family"], self.meta["task"]
        )

    def _exposure(self, raw: pd.DataFrame, n: int) -> np.ndarray:
        feat = self.meta.get("exposure_feature")
        if feat and feat in raw.columns:
            return pd.to_numeric(raw[feat], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return np.full(n, np.nan)

    # ------------------------------------------------------------------ #
    # Local explanation
    # ------------------------------------------------------------------ #
    def _get_explainer(self):
        if self._explainer is None:
            from src.evaluation.explain import make_explainer
            from src.models.tune import categorical_indices
            cat_idx = categorical_indices(self.meta["feature_names"])
            self._explainer = make_explainer(
                self.estimator,
                self.meta["family"],
                self.shap_background,
                cat_idx=cat_idx,
            )
        return self._explainer

    def explain_local(self, df: pd.DataFrame, row: int = 0) -> pd.DataFrame:
        """SHAP contributions for one policy, aggregated to raw source columns."""
        report = self.validate_input(df)
        report.raise_if_failed()
        raw, Xt = self._transform(df)
        explainer = self._get_explainer()
        values = explainer(Xt[row:row + 1])[0]
        names, groups = transform_feature_names(
            self.preprocessor, self.meta["numeric_cols"] + self.meta["categorical_cols"]
        )
        from src.evaluation.explain import aggregate_contributions
        cols, agg = aggregate_contributions(values[None, :], names, groups)
        contrib = pd.DataFrame({
            "feature": cols,
            "contribution": agg[0],
        })
        contrib = contrib.reindex(
            contrib["contribution"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)
        return contrib

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "preprocessor": self.preprocessor,
                "estimator": self.estimator,
                "meta": self.meta,
                "shap_background": self.shap_background,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ClaimsModelBundle":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"model bundle not found at {p} -- run 'make train' first"
            )
        obj = joblib.load(p)
        return cls(
            obj["preprocessor"], obj["estimator"], obj["meta"],
            obj.get("shap_background"),
        )

    def save_meta_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        slim = {k: v for k, v in self.meta.items() if k != "_train_dtypes"}
        path.write_text(json.dumps(slim, indent=2, default=str), encoding="utf-8")
        return path
