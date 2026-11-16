"""Estimator builders, exposure-weighted cross-validation, and Optuna tuning.

Weighting rules (docs/PLAN.md section 4, approved adjustments):
- Every estimator is fitted with sample_weight=exposure.
- Frequency classifiers use NATURAL class priors: no scale_pos_weight, no SMOTE.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, TweedieRegressor
from sklearn.model_selection import StratifiedKFold

from src.evaluation.metrics import pr_auc, tweedie_deviance
from src.features.build_features import build_preprocessor, transform_feature_names

logger = logging.getLogger(__name__)

TREE_FAMILIES = ("lightgbm", "xgboost", "catboost")


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def build_estimator(family: str, task: str, params: dict[str, Any], seed: int):
    """Construct an unfitted estimator for (family, task) with merged params."""
    params = dict(params)
    if family == "glm":
        if task == "frequency":
            # natural class priors: no class_weight (approved adjustment 1)
            return LogisticRegression(random_state=seed, **params)
        return TweedieRegressor(**params)
    if family == "lightgbm":
        import lightgbm as lgb
        cls = lgb.LGBMClassifier if task == "frequency" else lgb.LGBMRegressor
        return cls(random_state=seed, **params)
    if family == "xgboost":
        import xgboost as xgb
        cls = xgb.XGBClassifier if task == "frequency" else xgb.XGBRegressor
        params.setdefault("tree_method", "hist")
        if task == "pure_premium" and "tweedie_power" in params:
            # XGBoost's native param is tweedie_variance_power; the sklearn
            # wrapper does NOT alias tweedie_power (it is silently ignored).
            params = dict(params)
            params["tweedie_variance_power"] = params.pop("tweedie_power")
        return cls(random_state=seed, **params)
    if family == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        cls = CatBoostClassifier if task == "frequency" else CatBoostRegressor
        params.setdefault("verbose", False)
        params.setdefault("random_seed", seed)
        return cls(**params)
    raise ValueError(f"unknown family '{family}' -- expected one of glm|lightgbm|xgboost|catboost")


def categorical_indices(feature_names: list[str]) -> list[int]:
    """Positions of transformed features coming from the categorical branch."""
    return [i for i, n in enumerate(feature_names) if n.startswith("cat__")]


def _prep_X(X, family: str, cat_idx: list[int]):
    """Tree learners with categorical indices need integer codes, not floats.

    - catboost: requires an integer-typed categorical column, so hand it a
      DataFrame with those columns cast to int64 (a float ndarray raises).
    - lightgbm: accepts a float ndarray with categorical_feature indices.
    """
    X = np.asarray(X)
    if family == "catboost" and len(cat_idx):
        df = pd.DataFrame(X)
        for j in cat_idx:
            df[j] = np.rint(df[j].to_numpy(dtype=float)).astype(np.int64)
        return df
    if family == "lightgbm" and len(cat_idx):
        out = np.array(X, copy=True)
        out[:, cat_idx] = np.rint(out[:, cat_idx]).astype(np.int64)
        return out
    return X


def fit_estimator(est, X, y, w, cat_idx: list[int], family: str) -> None:
    """Fit with exposure sample weights. Never class weights (natural priors)."""
    Xp = _prep_X(X, family, cat_idx)
    if family == "lightgbm":
        kw: dict[str, Any] = {"sample_weight": w}
        if cat_idx:
            kw["categorical_feature"] = list(cat_idx)
        est.fit(Xp, y, **kw)
    elif family == "catboost":
        est.fit(Xp, y, sample_weight=w, cat_features=list(cat_idx))
    else:
        est.fit(Xp, y, sample_weight=w)


def predict_score(est, X, cat_idx: list[int], family: str, task: str) -> np.ndarray:
    """Score on the natural scale: probability (frequency) or expected cost (premium)."""
    Xp = _prep_X(X, family, cat_idx)
    if task == "frequency":
        if hasattr(est, "predict_proba"):
            return np.asarray(est.predict_proba(Xp))[:, 1]
        if hasattr(est, "decision_function"):
            z = np.asarray(est.decision_function(Xp), dtype=float)
            return 1.0 / (1.0 + np.exp(-z))
        return np.asarray(est.predict(Xp), dtype=float)
    return np.asarray(est.predict(Xp), dtype=float)


# --------------------------------------------------------------------------- #
# Weighted cross-validation (preprocessor refit per fold -> no leakage)
# --------------------------------------------------------------------------- #
def weighted_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    *,
    encoding: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    family: str,
    task: str,
    params: dict[str, Any],
    seed: int,
    folds: int = 5,
) -> tuple[np.ndarray, list[float], float]:
    """Manual 5-fold stratified CV. Returns (oof_scores, fold_metrics, mean_metric)."""
    strata = (np.asarray(y) > 0).astype(int)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.full(len(X), np.nan, dtype=float)
    fold_metrics: list[float] = []
    for tr, va in skf.split(np.zeros(len(X)), strata):
        pre = build_preprocessor(encoding, numeric_cols, categorical_cols, seed)
        X_tr = pre.fit_transform(X.iloc[tr], y[tr])
        X_va = pre.transform(X.iloc[va])
        names, _ = transform_feature_names(pre, numeric_cols + categorical_cols)
        cat_idx = categorical_indices(names)
        est = build_estimator(family, task, params, seed)
        fit_estimator(est, X_tr, y[tr], w[tr], cat_idx, family)
        score = predict_score(est, X_va, cat_idx, family, task)
        oof[va] = score
        if task == "frequency":
            fold_metrics.append(pr_auc(y[va], score, w[va]))
        else:
            fold_metrics.append(
                tweedie_deviance(
                    y[va],
                    score,
                    power=float(params.get("power", 1.5)),
                    sample_weight=w[va],
                )
            )
    finite = [m for m in fold_metrics if np.isfinite(m)]
    mean_metric = float(np.mean(finite)) if finite else float("nan")
    return oof, fold_metrics, mean_metric


# --------------------------------------------------------------------------- #
# Optuna tuning
# --------------------------------------------------------------------------- #
def suggest_params(trial, space: dict[str, Any] | None) -> dict[str, Any]:
    """Map the config search space onto Optuna suggestions."""
    out: dict[str, Any] = {}
    for name, cfg in (space or {}).items():
        t = cfg["type"]
        if t == "int":
            out[name] = trial.suggest_int(
                name, int(cfg["low"]), int(cfg["high"]), log=bool(cfg.get("log", False))
            )
        elif t == "float":
            out[name] = trial.suggest_float(
                name, float(cfg["low"]), float(cfg["high"]), log=bool(cfg.get("log", False))
            )
        elif t == "categorical":
            out[name] = trial.suggest_categorical(name, cfg["choices"])
        else:
            raise ValueError(f"unknown search space type '{t}' for param '{name}' in config")
    return out


def tune_model(
    name: str,
    spec: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    w: np.ndarray,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    seed: int,
    folds: int = 5,
) -> dict[str, Any]:
    """Run a seeded Optuna study for one model. Returns best params + trial stats."""
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    family, task = spec["family"], spec["task"]
    encoding = spec["encoding"]
    fixed = dict(spec.get("params", {}))
    space = spec.get("search_space", {})
    n_trials = int(spec.get("n_trials", 10))
    direction = "maximize" if task == "frequency" else "minimize"

    def objective(trial) -> float:
        suggested = suggest_params(trial, space)
        params = {**fixed, **suggested}
        _, _, mean_metric = weighted_cv(
            X, y, w,
            encoding=encoding,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            family=family,
            task=task,
            params=params,
            seed=seed,
            folds=folds,
        )
        if not np.isfinite(mean_metric):
            raise optuna.TrialPruned()
        return mean_metric

    study = optuna.create_study(
        direction=direction, sampler=TPESampler(seed=seed, multivariate=True)
    )

    import time as _time

    def _callback(study_, trial) -> None:
        logger.info(
            "%s trial %d/%d: %s=%.5f (%.1fs)",
            name, trial.number + 1, n_trials,
            study_.direction.name.lower(), trial.value,
            _time.monotonic() - _trial_start[0],
        )

    _trial_start = [_time.monotonic()]
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        catch=(Exception,),
        callbacks=[_callback],
    )
    best = study.best_trial
    best_params = dict(fixed)
    for pname in space:
        best_params[pname] = best.params[pname]
    logger.info(
        "tuned %s: best %s=%.4f params=%s",
        name, study.direction.name, best.value, best_params,
    )
    return {
        "model": name,
        "task": task,
        "family": family,
        "encoding": encoding,
        "best_params": best_params,
        "best_value": float(best.value),
        "direction": study.direction.name,
        "n_trials": len(study.trials),
        "search_space": space,
    }


def save_tuning_report(report: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return p
