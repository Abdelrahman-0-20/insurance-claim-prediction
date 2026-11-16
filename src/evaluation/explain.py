"""SHAP helpers: per-family explainers, normalization, aggregation, plots.

- TreeExplainer for LightGBM/XGBoost/CatBoost
- LinearExplainer for GLMs with a stored background sample, coefficient fallback
- Contributions aggregate back to raw source columns via the bundle's feature groups
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def shap_to_matrix(vals: Any) -> np.ndarray:
    """Normalize any shap output (list / Explanation / ndarray) to (n_features, k)."""
    if isinstance(vals, list):
        vals = vals[-1]
    v = getattr(vals, "values", vals)
    v = np.asarray(v, dtype=float)
    if v.ndim == 3:
        if v.shape[-1] in (1, 2):
            v = v[..., -1]        # (n, k, classes) -> positive class
        elif v.shape[1] == 2:
            v = v[:, 1, :]        # (n, classes, k) -> positive class
        else:
            raise ValueError(f"unexpected SHAP values shape {v.shape}")
    if v.ndim == 1:
        v = v.reshape(1, -1)
    if v.ndim != 2:
        raise ValueError(f"unexpected SHAP values shape {v.shape}")
    return v


class _CallableExplainer:
    """Wrap shap explainers or the coefficient fallback into a plain callable."""

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, X) -> np.ndarray:
        return shap_to_matrix(self._fn(X))


class CoefficientExplainer:
    """Coefficient-based fallback for GLMs: contribution = coef * (x - mean)."""

    def __init__(self, estimator, background: np.ndarray | None = None):
        coef = np.asarray(getattr(estimator, "coef_"), dtype=float).reshape(-1)
        self.coef_ = coef
        self.intercept_ = float(np.asarray(getattr(estimator, "intercept_", 0.0)).reshape(-1)[0])
        self.mean_ = (
            np.asarray(background, dtype=float).mean(axis=0)
            if background is not None and len(background)
            else np.zeros_like(coef)
        )

    def __call__(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        base = self.intercept_ + float(np.dot(self.coef_, self.mean_))
        return (X - self.mean_) * self.coef_[None, :] + 0.0 * base


def make_explainer(
    estimator: Any,
    family: str,
    background: np.ndarray | None = None,
    cat_idx: list[int] | None = None,
) -> _CallableExplainer:
    """Build a callable explainer returning (n, k) contribution matrices."""
    if family == "glm":
        try:
            import shap
            expl = shap.LinearExplainer(estimator, background)
            return _CallableExplainer(expl)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("LinearExplainer failed (%s); using coefficient fallback", exc)
            return _CallableExplainer(CoefficientExplainer(estimator, background))
    import shap
    try:
        expl = shap.TreeExplainer(estimator)
    except Exception as exc:  # pragma: no cover
        logger.warning("TreeExplainer failed (%s); using coefficient fallback", exc)
        return _CallableExplainer(CoefficientExplainer(estimator, background))

    if family == "catboost" and cat_idx:
        # CatBoost rejects float arrays when categorical features are declared
        from src.models.tune import _prep_X
        idx = list(cat_idx)
        return _CallableExplainer(
            lambda X: expl.shap_values(_prep_X(X, "catboost", idx))
        )
    return _CallableExplainer(expl)


def aggregate_contributions(
    values: np.ndarray, feature_names: list[str], feature_groups: dict[str, str]
) -> tuple[list[str], np.ndarray]:
    """Sum transformed-feature contributions back to raw source columns."""
    groups: dict[str, np.ndarray] = {}
    for j, name in enumerate(feature_names):
        col = feature_groups.get(name, name)
        groups[col] = values[:, j] + groups[col] if col in groups else values[:, j].copy()
    cols = list(groups.keys())
    return cols, np.column_stack([groups[c] for c in cols])


# --------------------------------------------------------------------------- #
# Plots (clean minimal style, no emojis)
# --------------------------------------------------------------------------- #
def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    })


def plot_shap_importance(importance: dict[str, float], path: str | Path, top_n: int = 20) -> Path:
    _style()
    items = sorted(importance.items(), key=lambda kv: kv[1])[-top_n:]
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#3b6fb6")
    ax.set_xlabel("Mean absolute SHAP contribution")
    ax.set_title("Global feature importance (SHAP)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_shap_beeswarm(
    values: np.ndarray, display: pd.DataFrame, path: str | Path, top_n: int = 20
) -> Path:
    _style()
    import shap as sh
    import warnings
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    try:
        with warnings.catch_warnings():
            # shap.summary_plot seeds the global NumPy RNG (library-internal)
            warnings.simplefilter("ignore", FutureWarning)
            sh.summary_plot(values, display, max_display=top_n, show=False, plot_type="dot")
        plt.gcf().axes[-1].set_title("SHAP feature impact on the champion model")
    except Exception as exc:  # pragma: no cover - fallback to bar chart
        logger.warning("beeswarm failed (%s); falling back to bar chart", exc)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7.0, 5.2))
        order = np.argsort(np.abs(values).mean(axis=0))
        ax.barh(
            np.array(display.columns)[order][-top_n:],
            np.abs(values).mean(axis=0)[order][-top_n:],
            color="#3b6fb6",
        )
        ax.set_xlabel("Mean absolute SHAP contribution")
        ax.set_title("Global feature importance (SHAP)")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_shap_local(contrib: pd.DataFrame, path: str | Path, top_n: int = 15) -> Path:
    """Horizontal bar of one policy's top SHAP drivers (positive/negative)."""
    _style()
    c = contrib.reindex(contrib["contribution"].abs().sort_values(ascending=True).index)
    c = c.tail(top_n)
    colors = ["#c0392b" if v < 0 else "#1e8449" for v in c["contribution"]]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * max(6, len(c)) + 1.4))
    ax.barh(c["feature"], c["contribution"], color=colors)
    ax.axvline(0.0, color="#444444", linewidth=1)
    ax.set_xlabel("SHAP contribution to the prediction")
    ax.set_title("Local explanation for one policy")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    return path

# --------------------------------------------------------------------------- #
# CLI: python -m src.evaluation.explain --config configs/config.yaml
# --------------------------------------------------------------------------- #
def bundle_meta_names(bundle) -> tuple[list[str], dict[str, str]]:
    return bundle.meta["feature_names"], bundle.meta["feature_groups"]


def explain_artifacts(
    config_path: str = "configs/config.yaml",
    artifacts_dir: str = "artifacts",
    sample_rows: int = 2000,
) -> dict[str, Any]:
    """Global + local SHAP reports for the champion bundle (test split sample)."""
    import sys as _sys

    from src.config import load_config, seed_everything
    from src.data.ingest import load_dataset
    from src.data.split import stratified_split
    from src.features.build_features import build_domain_features
    from src.models.predict import ClaimsModelBundle

    cfg = load_config(config_path)
    seed_everything(cfg.seed)
    art = Path(artifacts_dir)
    model_dir = art / "models"
    if not (model_dir / "champion.joblib").exists():
        raise FileNotFoundError(
            f"champion bundle missing at {model_dir / 'champion.joblib'} -- run 'make train' first"
        )

    bundle = ClaimsModelBundle.load(model_dir / "champion.joblib")
    dataset = load_dataset(cfg)
    split = stratified_split(dataset, cfg.split, cfg.seed)

    # deterministic sample of the untouched test split
    test_idx = np.sort(split.test)[:sample_rows]
    X_raw = dataset.features.iloc[test_idx].reset_index(drop=True)
    _raw, Xt = bundle._transform(X_raw)

    explainer = bundle._get_explainer()
    values = explainer(Xt)
    names, groups = bundle_meta_names(bundle)
    agg_cols, agg_vals = aggregate_contributions(values, names, groups)

    importance = {
        c: float(np.mean(np.abs(agg_vals[:, j]))) for j, c in enumerate(agg_cols)
    }
    save_json(
        {
            "champion": bundle.meta["model_name"],
            "n_rows_sampled": int(len(test_idx)),
            "mean_abs_shap": dict(
                sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
            ),
        },
        art / "reports" / "shap_global.json",
    )

    plot_shap_importance(importance, art / "plots" / "shap_importance.png")

    display = build_domain_features(X_raw)[agg_cols].copy()
    for c in display.columns:
        if not pd.api.types.is_numeric_dtype(display[c]):
            display[c] = display[c].astype("category").cat.codes
    plot_shap_beeswarm(agg_vals, display, art / "plots" / "shap_summary.png")

    local = bundle.explain_local(X_raw, row=0)
    plot_shap_local(local, art / "explanations" / "shap_local_reference.png")
    save_json(
        {
            "champion": bundle.meta["model_name"],
            "policy_index": int(test_idx[0]),
            "contributions": local.to_dict(orient="records"),
        },
        art / "explanations" / "shap_local_reference.json",
    )
    return {
        "champion": bundle.meta["model_name"],
        "top_features": list(importance)[:5],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import logging as _logging

    parser = argparse.ArgumentParser(description="SHAP reports for the champion model.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--sample-rows", type=int, default=2000)
    args = parser.parse_args(argv)
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        result = explain_artifacts(args.config, args.artifacts, args.sample_rows)
    except Exception as exc:
        logger.error("SHAP reporting failed: %s", exc)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
