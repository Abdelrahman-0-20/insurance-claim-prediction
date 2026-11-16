"""Actuarial + classification metrics, report builders, and plot helpers.

All metrics accept optional exposure weights (sample_weight) and are exposure-weighted
where meaningful (see docs/ARCHITECTURE.md section 8).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #
def pr_auc(y_true: np.ndarray, score: np.ndarray, sample_weight=None) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score, sample_weight=sample_weight))


def roc_auc(y_true: np.ndarray, score: np.ndarray, sample_weight=None) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score, sample_weight=sample_weight))


def brier(y_true: np.ndarray, prob: np.ndarray, sample_weight=None) -> float:
    return float(brier_score_loss(y_true, prob, sample_weight=sample_weight))


def calibration_bins(
    y_true: np.ndarray,
    prob: np.ndarray,
    sample_weight=None,
    n_bins: int = 10,
) -> dict[str, list]:
    """Equal-frequency calibration bins with exposure-weighted observed rates."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if len(y) == 0:
        return {"mean_predicted": [], "observed_rate": [], "count": []}
    n_bins = max(1, min(n_bins, len(y)))
    # equal-frequency edges on the predicted score
    edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) <= 1:
        edges = np.array([p.min() - 1e-9, p.max() + 1e-9])
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    mean_pred, obs_rate, counts = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        wb = w[m]
        sw = wb.sum()
        if sw <= 0:
            continue
        mean_pred.append(float(np.average(p[m], weights=wb)))
        obs_rate.append(float(np.average(y[m], weights=wb)))
        counts.append(int(m.sum()))
    return {"mean_predicted": mean_pred, "observed_rate": obs_rate, "count": counts}


def normalized_gini(y_true: np.ndarray, score: np.ndarray, sample_weight=None) -> float:
    """Ranking Gini normalized by the Gini of the perfect ranking (1.0 = perfect).

    Orders ascending by score (lowest predicted risk first), integrates the Lorenz
    curve from the origin, and divides by the perfect-ranking Gini so that a
    perfectly ranked portfolio scores 1.0 and a perfectly inverted one -1.0.
    """
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(s).all():
        return float("nan")

    def _gini(order: np.ndarray) -> float:
        yy, ww = y[order], w[order]
        total = float((yy * ww).sum())
        if total <= 0:
            return float("nan")
        cum_w = np.concatenate([[0.0], np.cumsum(ww)])
        cum_w = cum_w / cum_w[-1]
        cum_y = np.concatenate([[0.0], np.cumsum(yy * ww)]) / total
        area = float(np.trapezoid(cum_y, cum_w))
        return 1.0 - 2.0 * area

    order_obs = np.argsort(s, kind="stable")     # ascending: lowest score first
    order_perf = np.argsort(y, kind="stable")    # perfect: actual values first
    g_obs, g_perf = _gini(order_obs), _gini(order_perf)
    if not np.isfinite(g_obs) or not np.isfinite(g_perf) or g_perf <= 0:
        # g_perf <= 0 covers degenerate cases: all-zero target, or all-equal target
        return float("nan")
    return float(g_obs / g_perf)


def cumulative_lift(
    y_true: np.ndarray,
    score: np.ndarray,
    sample_weight=None,
    n_deciles: int = 10,
) -> list[dict[str, float]]:
    """Cumulative lift: observed rate in the top-k deciles vs the overall rate."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    overall = float(np.average(y, weights=w)) if w.sum() > 0 else float("nan")
    if overall <= 0:
        return []
    order = np.argsort(-s, kind="stable")
    y, w = y[order], w[order]
    cw = np.cumsum(w)
    cy = np.cumsum(y * w)
    rows = []
    for k in range(1, n_deciles + 1):
        cut = int(round(len(y) * k / n_deciles))
        cut = max(1, min(cut, len(y)))
        if cw[cut - 1] <= 0:
            continue
        rate = float(cy[cut - 1] / cw[cut - 1])
        rows.append(
            {"top_fraction": k / n_deciles, "observed_rate": rate, "lift": rate / overall}
        )
    return rows


def tweedie_deviance(
    y_true: np.ndarray, y_pred: np.ndarray, power: float = 1.5, sample_weight=None
) -> float:
    """Exposure-weighted mean Tweedie deviance (power in (1,2) for pure premium)."""
    y = np.asarray(y_true, dtype=float)
    mu = np.clip(np.asarray(y_pred, dtype=float), 1e-12, None)
    w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    p = float(power)
    if abs(p - 1.0) < 1e-8:
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(y > 0, y * np.log(np.where(y > 0, y / mu, 1.0)), 0.0) - (y - mu)
        d = 2.0 * term
    elif abs(p - 2.0) < 1e-8:
        d = 2.0 * (-np.log(np.clip(y, 1e-12, None) / mu) + (y - mu) / mu)
    else:
        d = 2.0 * (
            (np.power(np.clip(y, 0.0, None), 2.0 - p) - np.power(mu, 2.0 - p))
            / ((2.0 - p) * (1.0 - p))
            - (y - mu) * np.power(mu, 1.0 - p) / (1.0 - p)
        )
    d = np.where(np.isfinite(d), d, np.nan)
    sw = w.sum()
    if sw <= 0:
        return float("nan")
    return float(np.nansum(d * w) / sw)


# --------------------------------------------------------------------------- #
# Report builders
# --------------------------------------------------------------------------- #
def frequency_report(
    y_true, prob, exposure=None, calibration_bins_n: int = 10, lift_deciles: int = 10
) -> dict[str, Any]:
    """Full metric report for a frequency (claim probability) model."""
    y = np.asarray(y_true)
    p = np.asarray(prob, dtype=float)
    w = None if exposure is None else np.asarray(exposure, dtype=float)
    return {
        "pr_auc": pr_auc(y, p, w),
        "roc_auc": roc_auc(y, p, w),
        "brier": brier(y, p, w),
        "normalized_gini": normalized_gini(y, p, w),
        "calibration": calibration_bins(y, p, w, n_bins=calibration_bins_n),
        "cumulative_lift": cumulative_lift(y, p, w, n_deciles=lift_deciles),
    }


def pure_premium_report(y_true, expected_cost, exposure=None, power: float = 1.5) -> dict[str, Any]:
    """Metric report for a pure-premium (expected claim cost) model."""
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(expected_cost, dtype=float)
    w = None if exposure is None else np.asarray(exposure, dtype=float)
    return {
        "tweedie_deviance": tweedie_deviance(y, pred, power=power, sample_weight=w),
        "normalized_gini": normalized_gini(y, pred, w),
    }


# --------------------------------------------------------------------------- #
# Plotting (clean minimal style, no emojis anywhere)
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


def plot_calibration(curves: dict[str, dict[str, list]], path: str | Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1, label="Perfect")
    for name, c in curves.items():
        if c["mean_predicted"]:
            ax.plot(c["mean_predicted"], c["observed_rate"], marker="o", label=name)
    ax.set_xlabel("Mean predicted claim probability")
    ax.set_ylabel("Observed claim rate")
    ax.set_title("Calibration (equal-frequency bins, exposure-weighted)")
    ax.legend(frameon=False, fontsize=8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_lift(lift_rows: dict[str, list[dict[str, float]]], path: str | Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for name, rows in lift_rows.items():
        if rows:
            x = [r["top_fraction"] for r in rows]
            y = [r["lift"] for r in rows]
            ax.plot(x, y, marker="o", label=name)
    ax.axhline(1.0, linestyle="--", color="#888888", linewidth=1, label="Baseline")
    ax.set_xlabel("Population fraction (highest risk first)")
    ax.set_ylabel("Lift (observed rate / overall rate)")
    ax.set_title("Cumulative lift by risk decile")
    ax.legend(frameon=False, fontsize=8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_roc_pr(
    curves: dict[str, tuple[np.ndarray, np.ndarray]], path: str | Path
) -> Path:
    """curves: name -> (y_true, score)."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for name, (y, s) in curves.items():
        y = np.asarray(y)
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        ax1.plot(fpr, tpr, label=f"{name} (AUC={roc_auc(y, s):.3f})")
        ax2.plot(rec, prec, label=f"{name} (AP={pr_auc(y, s):.3f})")
    ax1.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=1)
    ax1.set_xlabel("False positive rate")
    ax1.set_ylabel("True positive rate")
    ax1.set_title("ROC curve")
    ax1.legend(frameon=False, fontsize=7)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-recall curve")
    ax2.legend(frameon=False, fontsize=7)
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
# CLI: python -m src.evaluation.metrics --config configs/config.yaml
# --------------------------------------------------------------------------- #
def evaluate_artifacts(
    config_path: str = "configs/config.yaml", artifacts_dir: str = "artifacts"
) -> dict[str, Any]:
    """Score every bundle on the untouched test split and write reports + plots."""
    from src.config import load_config, seed_everything
    from src.data.ingest import load_dataset
    from src.data.split import stratified_split
    from src.models.predict import ClaimsModelBundle

    cfg = load_config(config_path)
    seed_everything(cfg.seed)
    art = Path(artifacts_dir)
    model_dir = art / "models"
    if not model_dir.exists() or not list(model_dir.glob("bundle_*.joblib")):
        raise FileNotFoundError(
            f"no model bundles in {model_dir} -- run 'make train' first"
        )

    dataset = load_dataset(cfg)
    split = stratified_split(dataset, cfg.split, cfg.seed)
    test_idx = split.test
    X_test = dataset.features.iloc[test_idx].reset_index(drop=True)
    y_test = dataset.target.to_numpy(dtype=int)[test_idx]
    exposure = dataset.exposure.to_numpy(dtype=float)[test_idx]

    cal_curves: dict[str, dict[str, list]] = {}
    lift_rows: dict[str, list[dict[str, float]]] = {}
    roc_pr_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    comparison: dict[str, Any] = {}
    detailed: dict[str, Any] = {}
    power = float(cfg.metrics["tweedie_power"])

    for path in sorted(model_dir.glob("bundle_*.joblib")):
        bundle = ClaimsModelBundle.load(path)
        name = bundle.meta["model_name"]
        task = bundle.meta["task"]
        preds = bundle.predict(X_test)
        severity = float(bundle.meta["severity"])
        y_cost = (
            dataset.claim_amount.to_numpy(dtype=float)[test_idx]
            if dataset.claim_amount is not None
            else y_test.astype(float) * severity
        )

        if task == "frequency":
            prob = preds["claim_probability"].to_numpy(dtype=float)
            rep = frequency_report(
                y_test, prob, exposure,
                calibration_bins_n=int(cfg.metrics["calibration_bins"]),
                lift_deciles=int(cfg.metrics["lift_deciles"]),
            )
            rep["tweedie_deviance"] = tweedie_deviance(
                y_cost, prob * severity, power=power, sample_weight=exposure
            )
            cal_curves[name] = rep["calibration"]
            lift_rows[name] = rep["cumulative_lift"]
            roc_pr_curves[name] = (y_test, prob)
        else:
            cost = preds["expected_pure_premium"].to_numpy(dtype=float)
            rep = pure_premium_report(y_test, cost, exposure, power=power)
            rep["pr_auc"] = pr_auc(y_test, cost, exposure)
            rep["roc_auc"] = roc_auc(y_test, cost, exposure)

        detailed[name] = {
            "task": task,
            "family": bundle.meta["family"],
            "encoding": bundle.meta["encoding"],
            "tier_thresholds": {
                "low_max": bundle.meta["tier_low_max"],
                "high_min": bundle.meta["tier_high_min"],
            },
            "metrics": rep,
        }
        comparison[name] = {
            "task": task,
            "family": bundle.meta["family"],
            "encoding": bundle.meta["encoding"],
            "pr_auc": rep.get("pr_auc"),
            "roc_auc": rep.get("roc_auc"),
            "brier": rep.get("brier"),
            "normalized_gini": rep.get("normalized_gini"),
            "tweedie_deviance": rep.get("tweedie_deviance"),
        }
        logger.info(
            "evaluated %s: %s", name,
            {k: round(v, 5) for k, v in comparison[name].items() if isinstance(v, float)},
        )

    champion = (model_dir / "champion_name.txt").read_text(encoding="utf-8").strip()
    save_json(detailed, art / "reports" / "metrics.json")
    save_json(
        {"champion": champion, "test_rows": int(len(test_idx)), "models": comparison},
        art / "reports" / "model_comparison.json",
    )

    if cal_curves:
        plot_calibration(cal_curves, art / "plots" / "calibration.png")
    if roc_pr_curves:
        plot_roc_pr(roc_pr_curves, art / "plots" / "roc_pr.png")
    if champion in lift_rows:
        plot_lift({champion: lift_rows[champion]}, art / "plots" / "lift.png")
    return {"champion": champion, "models": comparison}


def main(argv: list[str] | None = None) -> int:
    import logging as _logging
    parser = argparse.ArgumentParser(description="Evaluate trained bundles on the test split.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args(argv)
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    try:
        result = evaluate_artifacts(args.config, args.artifacts)
    except Exception as exc:
        logger.error("evaluation failed: %s", exc)
        return 1
    print(json.dumps({"champion": result["champion"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
