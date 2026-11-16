# Implementation Plan — Insurance Claims Prediction

Status: approved for execution. This document is the implementation plan for the full repository
(see `docs/ARCHITECTURE.md` for the system architecture). It is the source of truth for scope,
ordering, and acceptance criteria.

## 1. Goal and success criteria

**Goal.** A config-driven, modular, production-grade Python repository (seed 42, Python 3.11+) for
Insurance Claims Prediction: a leakage-safe data and feature pipeline, an actuarial modeling
benchmark (interpretable GLMs vs gradient-boosted trees) with exposure weighting and pure-premium
modeling, Optuna tuning, actuarial evaluation and SHAP explainability into `artifacts/`, and a fast,
minimal, strictly zero-emoji Streamlit application with 3 focused tabs. No monolithic scripts.

**Success criteria**

1. `make setup && make test && make train && make evaluate` runs end-to-end and populates
   `artifacts/` (model bundles, JSON reports, PNG plots, SHAP outputs).
2. `streamlit run app/streamlit_app.py` (`make app`) boots cleanly with 3 working tabs and
   strictly zero emojis anywhere (titles, tabs, labels, status messages, charts).
3. `pytest` passes, including explicit tests for preprocessing determinism, zero data leakage,
   and inference-wrapper outputs.
4. `README.md` documents setup, methodology, assumptions, and a benchmark-results table
   generated from the actual pipeline run.

## 2. Verified data and environment facts

**Data** — `Insurance claims data.csv` (58,592 rows x 41 columns):

- Target: `claim_status` (binary), claim rate 6.4% (imbalanced).
- Exposure proxy: `subscription_length` (0-14, mean 6.1); 276 rows with zero exposure.
- Policyholder features: `customer_age` (35-75), `region_code` (22 levels), `region_density`.
- Vehicle features: `vehicle_age` (0-20), `segment` (6), `model` (11), `fuel_type` (3),
  `max_torque`, `max_power`, `engine_type` (11), `displacement`, `cylinder`,
  `transmission_type` (2), `steering_type`, dimensions, `gross_weight`, `ncap_rating`,
  approx. 20 binary safety-feature flags.
- No missing values; low-cardinality categoricals; **no claim-amount column**.

**Environment** — Python 3.12.3 (satisfies 3.11+); pip/PyPI/OpenML reachable. Pre-installed:
scikit-learn 1.9.1 (TargetEncoder available), xgboost 3.4.1, numpy, scipy, pytest, pyyaml, joblib.
Installed by `make setup` into `.venv`: pandas, lightgbm, catboost, optuna, shap, streamlit,
matplotlib.

## 3. Explicit assumptions

1. **Primary data.** The provided CSV is the intended `data/raw/claims.csv`; `make setup` copies
   it there. The OpenML `freMTPL2freq` + `freMTPL2sev` auto-fetch is implemented as the fallback
   when `data/raw/claims.csv` is absent (canonical schema normalization covers both sources).
2. **Exposure.** `subscription_length` maps to the canonical `exposure` weight column and is used
   as `sample_weight` in every fit. The 276 zero-exposure rows are dropped at ingest (zero risk
   time); the count is logged and asserted by tests.
3. **Severity / pure premium.** The local CSV has no claim amounts, so
   `expected_pure_premium = P(claim) x expected_severity`, where `expected_severity` is a config
   value (default 3,000 currency units) editable in the app and documented as an assumption. On the
   OpenML fallback path, severity is estimated from actual `ClaimAmount`s aggregated from
   `freMTPL2sev`.
4. **Runtime.** System Python 3.12 satisfies "Python 3.11+". A project-local `.venv` is created by
   `make setup` and used by every Makefile target.

## 4. Approved modeling adjustments (normative)

The following three adjustments to the modeling design are approved and override any conflicting
earlier wording:

1. **Probability calibration.** Frequency classifiers are trained with **natural class priors** and
   `exposure` sample weights only. `scale_pos_weight` (and any class-prior reweighting) is NOT
   applied to frequency classifiers, because it distorts predicted probabilities and inflates
   `expected_pure_premium = claim_probability x severity`. Cost-sensitive misclassification
   weighting is out of scope for the frequency task. Tests assert that effective training weights for
   frequency models equal `exposure` exactly.
2. **Exposure feature.** `subscription_length` remains in the model's **input features** (in
   addition to `exposure` acting as `sample_weight`) so that changing policy duration in the
   Streamlit simulator updates the predicted claim probability. Batch scoring and the app form
   therefore require `subscription_length` as an input column. (On the OpenML path the analogous
   `Exposure` column plays both roles.)
3. **Risk-tier thresholds.** The Low/Medium/High cutoffs (P25/P75) are computed from **out-of-fold
   validation predictions** (seeded 5-fold stratified OOF over train+val with the best
   hyperparameters), never from in-sample predictions after the final `train + val` refit. The
   OOF scores and derived thresholds are persisted in the bundle metadata for auditability.

## 5. Work breakdown (ordered)

1. **Scaffold.** `configs/config.yaml`, `pyproject.toml`, `Makefile`
   (`setup`, `train`, `evaluate`, `test`, `app`, plus `clean`), `.gitignore`, package
   skeleton under `src/`, `data/raw/` + `artifacts/` directories; copy dataset to
   `data/raw/claims.csv`; create `.venv` and install dependencies.
2. **Config + data layer.** `src/config.py` (typed config loading, seed helpers),
   `src/data/ingest.py` (local loader + OpenML fallback -> canonical schema, zero-exposure drop),
   `src/data/split.py` (stratified train/val/test indices, fits nothing, persisted split hash)
   + tests.
3. **Features.** `src/features/build_features.py` (domain ratios, torque/power parsing, guarded
   divisions; ColumnTransformer factory with target/native/onehot encoding modes) + tests.
4. **Models.** `src/models/tune.py` (Optuna TPE seeded, manual 5-fold weighted stratified CV,
   bounded trials), `src/models/train.py` (end-to-end CLI: tune -> OOF -> tier thresholds ->
   refit on train+val -> artifact persistence), `src/models/predict.py`
   (`ClaimsModelBundle`: predict / validate_input / explain_local / save / load) + tests,
   including tests for the three approved adjustments.
5. **Evaluation.** `src/evaluation/metrics.py` (PR-AUC, ROC-AUC, Brier + calibration, Normalized
   Gini, cumulative lift, Tweedie deviance; JSON reports + plots), `src/evaluation/explain.py`
   (global + local SHAP with per-family explainers and GLM fallback) + tests.
6. **Pipeline run.** `make train` then `make evaluate` end-to-end; verify `artifacts/`
   inventory (bundles, JSON reports, PNGs, SHAP outputs).
7. **App.** `app/streamlit_app.py` (3 tabs: Single Policy Simulator, Batch Scoring, Model
   Diagnostics) + zero-emoji enforcement test + headless boot smoke test.
8. **Docs.** `docs/PLAN.md` (this file), `docs/ARCHITECTURE.md`, `README.md` with quickstart,
   methodology, assumptions, and the actual benchmark-results table from
   `artifacts/reports/model_comparison.json`.

## 6. Test plan (pytest)

| Suite | Proves |
|---|---|
| `test_split.py` | Stratification preserved (class rates within tolerance); same seed -> same indices; train/val/test disjoint and exhaustive |
| `test_leakage.py` | Preprocessor fitted on train has statistics equal to **train-only** medians/modes on a fixture where train and full-data statistics differ; target column never appears in transformed feature names; `split.py` fits nothing |
| `test_features.py` | Domain feature engineering deterministic across runs; ratio guards (zero denominators produce no inf); preprocessor fit_transform deterministic with fixed seed; unknown categories at transform time do not crash (target/native/onehot modes) |
| `test_weights_calibration.py` | Frequency models train with weights exactly equal to `exposure` (no class-prior rescaling, no `scale_pos_weight` anywhere in frequency specs) |
| `test_predict.py` | Inference wrapper contract: expected columns, probabilities in [0,1], `expected_pure_premium == probability x severity` within tolerance, risk tiers partition all rows, batch scoring preserves row order and count, validation errors are actionable |
| `test_risk_tiers.py` | Tier thresholds are quantiles (P25/P75) of out-of-fold scores; thresholds differ from in-sample quantiles on a contrived case; tiers use the stored OOF-derived cutoffs |
| `test_metrics.py` | Synthetic correctness (perfect predictor -> Normalized Gini 1 and PR-AUC 1; random predictor near 0); Tweedie deviance non-negative and finite; calibration bins consistent |
| `test_ingest.py` | Local normalization (exposure mapping, zero-exposure drop, dtype coercion); OpenML fallback with monkeypatched fetch (join + severity aggregation verified) |
| `test_app_strings.py` | No emoji code points in `app/` sources (design rule enforcement) |
| `test_app_smoke.py` | Streamlit AppTest executes the app headlessly: 3 tabs render and form submission yields the KPI cards |

## 7. Edge cases and failure modes

| Case | Handling |
|---|---|
| No `data/raw/claims.csv` | Auto-fetch OpenML freMTPL2freq + freMTPL2sev (documented network requirement) |
| Zero/negative exposure rows (276 present) | Dropped at ingest with logged count; tests assert post-ingest `exposure > 0` |
| Division by zero in domain ratios | Denominators floored at epsilon; `inf -> NaN`; imputer + missingness indicators |
| Unseen categories in batch scoring | OneHot `handle_unknown="ignore"` / TargetEncoder smoothing / native unknown code; validation warns per column |
| Missing/extra columns in uploaded CSV | `validate_input` returns an actionable report naming the columns and expected values |
| No claim amounts in local data | Pure premium uses config `expected_severity` (documented, app-editable) |
| Missing artifacts when the app starts | Actionable message ("Run `make train` first"), no stack trace |
| Nondeterminism | All random states derive from seed 42 (splits, Optuna sampler, estimators, OOF folds); LightGBM deterministic flags; reproducibility recorded in `run_metadata.json` |
| GLM SHAP edge cases | Coefficient-based importance fallback if `LinearExplainer` fails |
| Tuning wall-clock | Bounded Optuna trials per model (config), capped iterations, seeded tuning subsample knob |

## 8. Execution and verification sequence

1. Scaffold repo; `make setup` (venv + deps + data copy).
2. Implement config + data + features layers and their tests; `make test` green.
3. Implement models + evaluation layers and their tests (including adjustment tests);
   `make test` green.
4. `make train`; `make evaluate`; verify complete `artifacts/` inventory.
5. Implement Streamlit app; headless boot smoke test; zero-emoji test green.
6. Write `README.md` with the actual benchmark table; final full re-run of `make test`.

## 9. Acceptance criteria

- `make setup`, `make test`, `make train`, `make evaluate`, `make app` all succeed from a
  clean checkout.
- `pytest` passes with named tests proving preprocessing determinism, zero data leakage
  (train-only fitted statistics), and inference wrapper outputs; plus the three approved
  adjustments.
- `artifacts/` contains model bundles, `metrics.json`, `model_comparison.json`,
  calibration/lift/ROC-PR plots, and global + local SHAP outputs.
- The Streamlit app runs via `streamlit run app/streamlit_app.py` with 3 functional tabs, KPI
  cards (Claim Probability, Risk Tier Low/Medium/High, Expected Pure Premium), local SHAP driver
  chart, batch scoring with schema validation and CSV download, and a diagnostics view; strictly
  zero emojis anywhere.
- `README.md` documents setup, methodology, assumptions, and benchmark results matching the
  artifacts.
