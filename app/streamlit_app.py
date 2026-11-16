"""Insurance Claims Prediction - Streamlit application.

Three tabs: Single Policy Simulator, Batch Scoring, Model Diagnostics.
Design rules: clean, minimalist, high contrast, strictly zero emojis.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ART = ROOT / "artifacts"

st.set_page_config(
    page_title="Insurance Claims Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_bundle():
    from src.models.predict import ClaimsModelBundle
    path = ART / "models" / "champion.joblib"
    if not path.exists():
        return None
    return ClaimsModelBundle.load(path)


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_reports() -> dict:
    return {
        "comparison": read_json(ART / "reports" / "model_comparison.json"),
        "metrics": read_json(ART / "reports" / "metrics.json"),
        "metadata": read_json(ART / "reports" / "run_metadata.json"),
        "defaults": read_json(ART / "reports" / "input_defaults.json"),
        "schema": read_json(ART / "reports" / "schema.json"),
        "shap_global": read_json(ART / "reports" / "shap_global.json"),
    }


def score_frame(bundle, df: pd.DataFrame, severity: float) -> pd.DataFrame:
    """Score a frame and apply the sidebar severity assumption to the premium."""
    out = bundle.predict(df).copy()
    if bundle.meta["task"] == "frequency":
        out["expected_pure_premium"] = out["claim_probability"] * severity
    return out


def shap_bar_figure(contrib: pd.DataFrame, top_n: int = 12):
    import plotly.graph_objects as go
    c = contrib.head(top_n).iloc[::-1]
    colors = ["#c0392b" if v < 0 else "#1e8449" for v in c["contribution"]]
    fig = go.Figure(
        go.Bar(
            x=c["contribution"],
            y=c["feature"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Top prediction drivers (SHAP)",
        xaxis_title="SHAP contribution (model link scale)",
        yaxis_title="",
        height=max(360, 34 * len(c) + 120),
        margin=dict(l=10, r=10, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    return fig


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
bundle = load_bundle()
reports = load_reports()

st.title("Insurance Claims Prediction")

if bundle is None:
    st.error(
        "Model artifacts not found in artifacts/models. "
        "Run 'make train' (then 'make evaluate') to train the models, then reload this page."
    )
    st.stop()

meta = bundle.meta
with st.sidebar:
    st.header("Model")
    st.write(f"Champion: {meta['model_name']}")
    st.write(f"Task: {meta['task'].replace('_', ' ')}")
    st.write(f"Encoding: {meta['encoding']}")
    trained = str(meta.get("trained_at", "unknown"))[:19]
    st.caption(f"Trained: {trained}")
    st.divider()
    st.header("Assumptions")
    default_severity = float(meta.get("severity", 3000.0))
    severity = st.number_input(
        "Expected severity (claim cost)",
        min_value=1.0,
        max_value=1_000_000.0,
        value=default_severity,
        step=100.0,
        help="Pure premium = claim probability x expected severity.",
    )
    low_max = float(meta["tier_low_max"])
    high_min = float(meta["tier_high_min"])
    st.caption(
        f"Risk tiers from out-of-fold scores: "
        f"Low <= {low_max:.4f}, High >= {high_min:.4f}"
    )
    st.divider()
    st.caption(
        "Probability calibration: natural class priors with exposure weights "
        "(no class reweighting)."
    )

tab_sim, tab_batch, tab_diag = st.tabs(
    ["Single Policy Simulator", "Batch Scoring", "Model Diagnostics"]
)


# --------------------------------------------------------------------------- #
# Tab 1: Single Policy Simulator
# --------------------------------------------------------------------------- #
with tab_sim:
    st.caption(
        "Adjust policyholder and vehicle attributes, then score the policy."
    )
    defaults = reports["defaults"] or {}
    schema = reports["schema"] or {}
    required = schema.get("required_columns") or meta["required_columns"]

    groups: dict[str, list[str]] = {
        "Policy duration": [],
        "Policyholder and region": [],
        "Vehicle": [],
        "Safety equipment": [],
    }
    for col in required:
        if col in ("subscription_length", "Exposure"):
            groups["Policy duration"].append(col)
        elif col.startswith("is_") or col in ("airbags", "ncap_rating"):
            groups["Safety equipment"].append(col)
        elif col in ("customer_age", "region_code", "region_density"):
            groups["Policyholder and region"].append(col)
        else:
            groups["Vehicle"].append(col)
    # split oversized groups so no column is excessively tall
    parts: list[tuple[str, list[str]]] = []
    for name, cols in groups.items():
        if not cols:
            continue
        if len(cols) > 14:
            mid = (len(cols) + 1) // 2
            parts.append((f"{name} (1)", cols[:mid]))
            parts.append((f"{name} (2)", cols[mid:]))
        else:
            parts.append((name, cols))

    with st.form("policy_form"):
        cols_layout = st.columns(len(parts))
        values: dict[str, object] = {}
        for layout_col, (gname, gcols) in zip(cols_layout, parts):
            with layout_col:
                st.markdown(f"**{gname}**")
                for col in gcols:
                    spec = defaults.get(col, {})
                    kind = spec.get("kind", "numeric")
                    if kind == "numeric":
                        lo = float(spec.get("min", 0.0))
                        hi = float(spec.get("max", 100.0))
                        val = float(spec.get("default", lo))
                        lo, hi, val = min(lo, hi), max(lo, hi), min(max(val, lo), hi)
                        if float(val).is_integer() and float(lo).is_integer() and float(hi).is_integer():
                            values[col] = st.number_input(
                                col, min_value=int(lo), max_value=int(hi),
                                value=int(val), key=f"f_{col}",
                            )
                        else:
                            step = max((hi - lo) / 200.0, 0.01)
                            values[col] = st.number_input(
                                col, min_value=lo, max_value=hi,
                                value=val, step=step, key=f"f_{col}",
                            )
                    else:
                        options = spec.get("categories") or ["(unknown)"]
                        cur = str(spec.get("default", options[0]))
                        idx = options.index(cur) if cur in options else 0
                        values[col] = st.selectbox(
                            col, options, index=idx, key=f"f_{col}"
                        )
        submitted = st.form_submit_button("Score policy", type="primary")

    if submitted:
        row = pd.DataFrame([values])
        try:
            result = score_frame(bundle, row, severity)
        except ValueError as exc:
            st.error(str(exc))
        else:
            prob = float(result["claim_probability"].iloc[0])
            tier = str(result["risk_tier"].iloc[0])
            premium = float(result["expected_pure_premium"].iloc[0])
            k1, k2, k3 = st.columns(3)
            k1.metric("Claim Probability", f"{prob:.2%}")
            k2.metric("Risk Tier", tier)
            k3.metric("Expected Pure Premium", f"{premium:,.0f}")
            st.caption(
                f"Expected pure premium = {prob:.4f} x severity {severity:,.0f}. "
                f"Tier cutoffs: Low <= {low_max:.4f}, High >= {high_min:.4f} "
                "(out-of-fold validation quantiles)."
            )
            contrib = bundle.explain_local(row, row=0)
            st.plotly_chart(
                shap_bar_figure(contrib), width="stretch", key="local_shap"
            )
            with st.expander("Feature contributions (all)"):
                st.dataframe(
                    contrib.assign(contribution=lambda d: d["contribution"].round(5)),
                    width="stretch",
                    hide_index=True,
                )
    else:
        st.info("Set the policy attributes above and select 'Score policy'.")


# --------------------------------------------------------------------------- #
# Tab 2: Batch Scoring
# --------------------------------------------------------------------------- #
with tab_batch:
    st.caption(
        "Upload a CSV with one row per policy. Schema is validated before scoring."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"], key="batch_upload")

    schema = reports["schema"] or {}
    required = schema.get("required_columns") or meta["required_columns"]
    if st.button("Download input template"):
        defaults = reports["defaults"] or {}
        template_row = {
            c: (defaults.get(c, {}).get("default", "")) for c in required
        }
        tmpl = pd.DataFrame([template_row])
        st.download_button(
            "Template CSV",
            data=tmpl.to_csv(index=False).encode("utf-8"),
            file_name="scoring_template.csv",
            mime="text/csv",
            key="template_dl",
        )

    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not parse the CSV file: {exc}. Export a plain UTF-8 CSV and retry.")
            raw = None
        if raw is not None:
            st.write(f"Loaded {len(raw):,} row(s), {raw.shape[1]} column(s).")
            report = bundle.validate_input(raw)
            if report.errors:
                st.error("Schema validation failed:")
                for err in report.errors:
                    st.write(f"- {err}")
            else:
                for warn in report.warnings:
                    st.warning(warn)
                scored = pd.concat(
                    [raw.reset_index(drop=True),
                     score_frame(bundle, raw, severity).reset_index(drop=True)],
                    axis=1,
                )
                counts = scored["risk_tier"].value_counts()
                m1, m2, m3 = st.columns(3)
                m1.metric("Low risk rows", int(counts.get("Low", 0)))
                m2.metric("Medium risk rows", int(counts.get("Medium", 0)))
                m3.metric("High risk rows", int(counts.get("High", 0)))
                st.dataframe(scored.head(100), width="stretch", hide_index=True)
                st.download_button(
                    "Download scored CSV",
                    data=scored.to_csv(index=False).encode("utf-8"),
                    file_name="scored_policies.csv",
                    mime="text/csv",
                    type="primary",
                    key="scored_dl",
                )
    else:
        st.info("Upload a CSV file to score a portfolio.")


# --------------------------------------------------------------------------- #
# Tab 3: Model Diagnostics
# --------------------------------------------------------------------------- #
with tab_diag:
    comparison = reports["comparison"]
    if comparison is None:
        st.warning(
            "model_comparison.json not found. Run 'make evaluate' to generate the reports."
        )
    else:
        st.subheader("Model comparison (untouched test split)")
        rows = []
        for name, m in comparison.get("models", {}).items():
            rows.append({
                "model": name,
                "task": m.get("task"),
                "family": m.get("family"),
                "PR-AUC": m.get("pr_auc"),
                "ROC-AUC": m.get("roc_auc"),
                "Brier": m.get("brier"),
                "Norm. Gini": m.get("normalized_gini"),
                "Tweedie deviance": m.get("tweedie_deviance"),
            })
        df_cmp = pd.DataFrame(rows).set_index("model")
        st.dataframe(df_cmp.round(4), width="stretch")
        st.caption(
            f"Champion: {comparison.get('champion')} "
            f"(selected by out-of-fold PR-AUC on train+val). "
            f"Test rows: {comparison.get('test_rows')}."
        )

    st.subheader("Calibration and ranking")
    c1, c2 = st.columns(2)
    cal = ART / "plots" / "calibration.png"
    lift = ART / "plots" / "lift.png"
    roc_pr = ART / "plots" / "roc_pr.png"
    with c1:
        if cal.exists():
            st.image(str(cal), caption="Calibration (equal-frequency bins)")
        else:
            st.caption("calibration.png missing - run 'make evaluate'.")
    with c2:
        if lift.exists():
            st.image(str(lift), caption="Cumulative lift by risk decile")
        else:
            st.caption("lift.png missing - run 'make evaluate'.")
    if roc_pr.exists():
        st.image(str(roc_pr), caption="ROC and precision-recall curves")
    else:
        st.caption("roc_pr.png missing - run 'make evaluate'.")

    st.subheader("Global feature importance (SHAP)")
    shap_global = reports["shap_global"]
    shap_png = ART / "plots" / "shap_importance.png"
    shap_sum = ART / "plots" / "shap_summary.png"
    sc1, sc2 = st.columns(2)
    with sc1:
        if shap_png.exists():
            st.image(str(shap_png), caption="Mean absolute SHAP importance")
        else:
            st.caption("shap_importance.png missing - run 'make evaluate'.")
    with sc2:
        if shap_sum.exists():
            st.image(str(shap_sum), caption="SHAP summary (beeswarm)")
        else:
            st.caption("shap_summary.png missing - run 'make evaluate'.")
    if shap_global:
        top = list(shap_global.get("mean_abs_shap", {}).items())[:12]
        st.dataframe(
            pd.DataFrame(top, columns=["feature", "mean |SHAP|"]).round(5),
            hide_index=True,
            width="stretch",
        )

    st.subheader("Run metadata")
    if reports["metadata"]:
        st.json(reports["metadata"])
    else:
        st.caption("run_metadata.json missing - run 'make train'.")
