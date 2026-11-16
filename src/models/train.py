"""End-to-end training CLI: ingest -> split -> tune -> OOF -> refit -> artifacts.

Usage:
    python -m src.models.train --config configs/config.yaml
    python -m src.models.train --config configs/config.yaml --models logistic_regression,lightgbm_freq

Weighting rules (approved adjustments, docs/PLAN.md section 4):
1. Frequency classifiers: exposure sample_weight only, natural class priors
   (no scale_pos_weight, no class_weight, no SMOTE).
2. subscription_length (or Exposure) is both an input feature and the exposure weight.
3. Risk-tier P25/P75 thresholds come from out-of-fold predictions on train+val.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, seed_everything
from src.data.ingest import ClaimsDataset, load_dataset
from src.data.split import save_split_report, stratified_split
from src.evaluation.metrics import brier, normalized_gini, pr_auc, tweedie_deviance, save_json
from src.features.build_features import (
    build_domain_features,
    build_preprocessor,
    resolve_columns,
    transform_feature_names,
)
from src.models.predict import ClaimsModelBundle, compute_tier_thresholds
from src.models.tune import (
    categorical_indices,
    fit_estimator,
    save_tuning_report,
    tune_model,
    weighted_cv,
)

logger = logging.getLogger("src.models.train")
EXPOSURE_FEATURE_CANDIDATES = ("subscription_length", "Exposure", "exposure")


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "sklearn", "lightgbm", "xgboost", "catboost", "optuna", "shap"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = "not installed"
    return out


def run_training(
    config_path: str = "configs/config.yaml",
    model_names: list[str] | None = None,
    artifacts_dir: str = "artifacts",
    resume: bool = False,
) -> dict[str, Any]:
    """Train all enabled models and persist bundles + reports. Returns train summary.

    With resume=True, models whose bundle and tuning report already exist on disk
    are loaded instead of retrained (interrupted runs continue where they stopped).
    """
    cfg = load_config(config_path)
    seed_everything(cfg.seed)
    art = Path(artifacts_dir)
    (art / "models").mkdir(parents=True, exist_ok=True)
    (art / "reports").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    dataset: ClaimsDataset = load_dataset(cfg)
    raw_features = dataset.features.reset_index(drop=True)
    features = build_domain_features(raw_features)
    logger.info(
        "loaded %d rows from %s (dropped %d zero-exposure)",
        len(dataset), dataset.source, dataset.dropped_zero_exposure,
    )

    split = stratified_split(dataset, cfg.split, cfg.seed)
    split_rep = save_split_report(split, dataset, cfg.seed, art / "reports" / "data_split.json")

    exclude = set(cfg.features.get("exclude_columns", []))
    numeric_cols, categorical_cols = resolve_columns(features, exclude)
    feature_columns = [c for c in features.columns if c not in exclude]
    raw_numeric, raw_categorical = resolve_columns(raw_features, exclude)
    required_columns = [c for c in raw_features.columns if c not in exclude]
    exposure_feature = next(
        (c for c in EXPOSURE_FEATURE_CANDIDATES if c in required_columns), None
    )

    y_freq = dataset.target.to_numpy(dtype=int)
    severity = cfg.expected_severity
    if dataset.claim_amount is not None:
        y_pp = dataset.claim_amount.to_numpy(dtype=float)
    else:
        y_pp = y_freq.astype(float) * severity
    w = dataset.exposure.to_numpy(dtype=float)

    tr_idx, va_idx, te_idx = split.train, split.val, split.test
    tv_idx = np.concatenate([tr_idx, va_idx])
    X_tv = features.iloc[tv_idx].reset_index(drop=True)
    X_tv_raw = raw_features.iloc[tv_idx].reset_index(drop=True)
    y_tv_freq = y_freq[tv_idx]
    y_tv_pp = y_pp[tv_idx]
    w_tv = w[tv_idx]

    # seeded tuning subsample drawn from TRAIN rows only (never val/test)
    rng_sub = np.random.RandomState(cfg.seed)
    n_sub = int(cfg.tuning.get("subsample_rows", 6000))
    sub_rows = np.sort(
        rng_sub.choice(tr_idx, size=min(n_sub, len(tr_idx)), replace=False)
    )
    X_sub = features.iloc[sub_rows].reset_index(drop=True)

    # ------------------------------------------------------- defaults/schema
    allowed_categories: dict[str, list[str]] = {}
    input_defaults: dict[str, Any] = {}
    for col in required_columns:
        if col in raw_categorical:
            vals = sorted(X_tv_raw[col].astype(str).unique().tolist())
            allowed_categories[col] = vals
            input_defaults[col] = {
                "kind": "categorical",
                "default": X_tv_raw[col].astype(str).mode().iloc[0],
                "categories": vals,
            }
        else:
            s = pd.to_numeric(X_tv_raw[col], errors="coerce")
            input_defaults[col] = {
                "kind": "numeric",
                "default": float(s.median()) if s.notna().any() else 0.0,
                "min": float(s.min()) if s.notna().any() else 0.0,
                "max": float(s.max()) if s.notna().any() else 0.0,
            }
    save_json(input_defaults, art / "reports" / "input_defaults.json")
    save_json(
        {
            "required_columns": required_columns,
            "numeric_columns": raw_numeric,
            "categorical_columns": raw_categorical,
            "allowed_categories": allowed_categories,
            "aliases": cfg.data.get("aliases", {}),
            "target_column": cfg.data.get("target_column", "claim_status"),
            "id_column": cfg.data.get("id_column", "policy_id"),
            "exposure_feature": exposure_feature,
        },
        art / "reports" / "schema.json",
    )

    # ----------------------------------------------------------------- models
    folds = int(cfg.tuning.get("cv_folds", 5))
    enabled = cfg.enabled_models
    if model_names:
        unknown = [m for m in model_names if m not in enabled]
        if unknown:
            raise ValueError(
                f"unknown model(s) {unknown} -- available: {sorted(enabled)}"
            )
        enabled = {k: v for k, v in enabled.items() if k in model_names}

    summary: dict[str, Any] = {}
    bundles: dict[str, ClaimsModelBundle] = {}
    rng_bg = np.random.RandomState(cfg.seed + 1)

    for name, spec in enabled.items():
        task = spec["task"]
        family = spec["family"]
        encoding = spec["encoding"]
        y_full = y_freq if task == "frequency" else y_pp

        bundle_path = art / "models" / f"bundle_{name}.joblib"
        report_path = art / "reports" / f"optuna_{name}.json"
        if resume and bundle_path.exists() and report_path.exists():
            summary[name] = json.loads(report_path.read_text())
            bundles[name] = ClaimsModelBundle.load(bundle_path)
            logger.info(
                "resume: %s already trained (PR-AUC OOF %s) -- skipping",
                name,
                summary[name].get("oof", {}).get("pr_auc"),
            )
            continue

        logger.info("=== tuning %s (%s / %s) ===", name, family, task)

        tuning = tune_model(
            name, spec, X_sub, y_full[sub_rows], w[sub_rows],
            numeric_cols=numeric_cols, categorical_cols=categorical_cols,
            seed=cfg.seed, folds=folds,
        )
        best_params = tuning["best_params"]

        # out-of-fold predictions on train+val (each row scored out-of-fold)
        oof, fold_metrics, oof_mean = weighted_cv(
            X_tv, y_full[tv_idx], w_tv,
            encoding=encoding, numeric_cols=numeric_cols, categorical_cols=categorical_cols,
            family=family, task=task, params=best_params, seed=cfg.seed, folds=folds,
        )
        if not np.all(np.isfinite(oof)):
            raise RuntimeError(
                f"OOF predictions for {name} contain non-finite values -- check tuning"
            )

        # approved adjustment 3: tier thresholds from OOF scores
        qs = list(cfg.risk_tiers.get("quantiles", [0.25, 0.75]))
        low_max, high_min = compute_tier_thresholds(oof, (qs[0], qs[1]))

        # final refit on train+val
        pre = build_preprocessor(encoding, numeric_cols, categorical_cols, cfg.seed)
        X_fit = pre.fit_transform(X_tv, y_tv_freq if task == "frequency" else y_tv_pp)
        names_out, groups = transform_feature_names(pre, numeric_cols + categorical_cols)
        cat_idx = categorical_indices(names_out)
        est = build_estimator_safe(family, task, best_params, cfg.seed)
        fit_estimator(est, X_fit, y_full[tv_idx], w_tv, cat_idx, family)

        # small stored background for GLM local explanations
        shap_background = None
        if family == "glm" and len(X_fit):
            take = min(200, len(X_fit))
            shap_background = np.asarray(
                X_fit[rng_bg.choice(len(X_fit), size=take, replace=False)]
            )

        oof_report: dict[str, float]
        if task == "frequency":
            oof_report = {
                "pr_auc": pr_auc(y_full[tv_idx], oof, w_tv),
                "brier": brier(y_full[tv_idx], oof, w_tv),
                "normalized_gini": normalized_gini(y_full[tv_idx], oof, w_tv),
            }
        else:
            oof_report = {
                "tweedie_deviance": tweedie_deviance(
                    y_full[tv_idx],
                    oof,
                    power=float(cfg.metrics["tweedie_power"]),
                    sample_weight=w_tv,
                ),
                "normalized_gini": normalized_gini(y_full[tv_idx], oof, w_tv),
            }

        meta = {
            "model_name": name,
            "task": task,
            "family": family,
            "encoding": encoding,
            "feature_columns": feature_columns,
            "required_columns": required_columns,
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "feature_names": names_out,
            "feature_groups": groups,
            "severity": severity,
            "tier_low_max": low_max,
            "tier_high_min": high_min,
            "tier_quantiles": qs,
            "oof_summary": {
                "n": int(len(oof)),
                "mean": float(np.mean(oof)),
                "min": float(np.min(oof)),
                "max": float(np.max(oof)),
            },
            "seed": cfg.seed,
            "allowed_categories": allowed_categories,
            "exposure_feature": exposure_feature,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        bundle = ClaimsModelBundle(pre, est, meta, shap_background)
        bundle.save(bundle_path)
        bundles[name] = bundle

        tuning["oof"] = oof_report
        tuning["tier_thresholds"] = {"low_max": low_max, "high_min": high_min}
        tuning["fold_metrics"] = [float(m) for m in fold_metrics]
        tuning["oof_mean_cv_metric"] = float(oof_mean)
        save_tuning_report(tuning, art / "reports" / f"optuna_{name}.json")
        summary[name] = tuning
        logger.info(
            "%s done: %s | OOF %s | tiers [%.5f, %.5f]",
            name, {k: round(v, 4) for k, v in oof_report.items()},
            round(float(np.quantile(oof, 0.5)), 5), low_max, high_min,
        )

    # ------------------------------------------------------------- champion
    freq_names = [n for n, s in enabled.items() if s["task"] == "frequency"]
    if not freq_names:
        raise ValueError(
            "no frequency models enabled -- at least one model with task 'frequency' "
            "is required to select a champion (config models.*.task)"
        )
    champion = max(
        freq_names,
        key=lambda n: summary[n]["oof"].get("pr_auc", float("-inf")),
    )
    (art / "models" / "champion_name.txt").write_text(champion + "\n", encoding="utf-8")
    bundles[champion].save(art / "models" / "champion.joblib")

    # ------------------------------------------------------------- metadata
    save_json(summary, art / "reports" / "train_summary.json")
    save_json(
        {
            "seed": cfg.seed,
            "config_path": str(config_path),
            "dataset_source": dataset.source,
            "n_rows": int(len(dataset)),
            "n_features_model": len(feature_columns),
            "dropped_zero_exposure": int(dataset.dropped_zero_exposure),
            "expected_severity": severity,
            "split": split_rep,
            "champion": champion,
            "versions": _versions(),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "exposure_weighting": "sample_weight=exposure; natural class priors for frequency",
            "risk_tier_source": "out-of-fold predictions on train+val (P25/P75)",
        },
        art / "reports" / "run_metadata.json",
    )
    logger.info("champion: %s", champion)
    return {"champion": champion, "summary": summary}


def build_estimator_safe(family: str, task: str, params: dict[str, Any], seed: int):
    from src.models.tune import build_estimator
    return build_estimator(family, task, params, seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and benchmark insurance claims models."
    )
    parser.add_argument("--config", default="configs/config.yaml", help="path to config YAML")
    parser.add_argument(
        "--models", default=None,
        help="comma-separated subset of models to train (default: all enabled)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="skip models whose bundle and tuning report already exist on disk",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    names = [m.strip() for m in args.models.split(",")] if args.models else None
    try:
        result = run_training(args.config, names, resume=args.resume)
    except Exception as exc:
        logger.error("training failed: %s", exc)
        return 1
    print(json.dumps({"champion": result["champion"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
