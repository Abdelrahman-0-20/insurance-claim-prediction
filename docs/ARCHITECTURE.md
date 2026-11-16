# Architecture — Insurance Claims Prediction

This document specifies the system architecture: repository layout, data contracts, feature and
preprocessing design, modeling, artifacts, evaluation, the inference API, the Streamlit UI, and the
configuration schema. Design constraints from the approved plan (`docs/PLAN.md`) are normative,
including the three approved modeling adjustments:

1. Frequency classifiers use natural class priors + `exposure` weights only (no
   `scale_pos_weight`) so probabilities remain calibrated.
2. `subscription_length` is both an input feature and (via the canonical `exposure` column) the
   `sample_weight`.
3. Risk-tier thresholds (P25/P75) come from out-of-fold validation predictions, not in-sample
   predictions.

## 1. Repository layout

```
Insurance Claims Prediction/
├── configs/
│   └── config.yaml                # single source of truth: seed=42, data, split, features,
│                                  #   models + search spaces, tuning, metrics, app
├── pyproject.toml                 # setuptools package (src.*), runtime + [dev] deps, pytest config
├── Makefile                       # setup | train | evaluate | test | app | clean
├── README.md                      # quickstart, methodology, assumptions, benchmark results
├── .gitignore
├── docs/
│   ├── PLAN.md                    # implementation plan
│   └── ARCHITECTURE.md            # this document
├── data/
│   └── raw/
│       └── claims.csv             # copy of "Insurance claims data.csv" (made by make setup)
├── src/
│   ├── __init__.py
│   ├── config.py                  # load_config() -> typed AppConfig; seed_everything()
│   ├── data/
│   │   ├── ingest.py              # local CSV + OpenML fallback -> canonical ClaimsDataset
│   │   └── split.py               # stratified train/val/test indices (fits nothing)
│   ├── features/
│   │   └── build_features.py      # domain features + ColumnTransformer factory
│   ├── models/
│   │   ├── tune.py                # Optuna (seeded TPE) + manual weighted 5-fold CV; builders
│   │   ├── train.py               # end-to-end training CLI; artifact persistence
│   │   └── predict.py             # ClaimsModelBundle inference wrapper
│   └── evaluation/
│       ├── metrics.py             # actuarial + classification metrics; reports + plots
│       └── explain.py             # SHAP helpers (per-family explainers), global + local plots
├── app/
│   └── streamlit_app.py           # 3-tab UI (strictly zero emojis)
├── tests/                         # pytest suites (see PLAN.md section 6)
└── artifacts/
    ├── models/                    # bundle_<model>.joblib, champion.joblib, champion_name.txt
    ├── reports/                   # JSON reports (see section 7)
    ├── plots/                     # calibration/lift/roc_pr/shap PNGs
    └── explanations/              # local SHAP reference plot
```

## 2. End-to-end data flow

```
"Insurance claims data.csv" ──make setup──▶ data/raw/claims.csv
        │  (fallback when absent: OpenML freMTPL2freq + freMTPL2sev, joined and aggregated)
        ▼
src/data/ingest.py
        │  canonical ClaimsDataset: raw features + exposure + claim_status [+ claim_amount]
        │  (zero-exposure rows dropped; dtypes coerced; target validated as {0,1})
        ▼
src/data/split.py  ──▶ SplitIndices(train, val, test)   [stratified on claim_status, seed 42,
        │                                                persisted to artifacts/reports/data_split.json]
        ▼
src/features/build_features.py::build_domain_features    [pure pandas, deterministic, no fitting]
        ▼
ColumnTransformer per model   [FIT ON TRAIN ONLY during tuning; on train+val for the final fit;
        │                      test data never seen by any fitted object]
        ▼
Estimator (GLM / LightGBM / XGBoost / CatBoost; frequency or Tweedie pure-premium)
        │   fit(sample_weight = exposure)   — natural class priors; no SMOTE;
        │   no scale_pos_weight on frequency classifiers
        ▼
Optuna tune.py (seeded TPE, manual 5-fold weighted stratified CV on a seeded tuning subsample)
        │   best params → 5-fold OOF predictions on train+val → risk-tier P25/P75 thresholds
        ▼
ClaimsModelBundle (preprocessor + estimator + metadata: feature schema, encoding mode,
        │           severity assumption, OOF tier thresholds, model kind)
        ▼
src/evaluation/metrics.py + explain.py  ──▶ artifacts/reports/*.json, artifacts/plots/*.png,
        │                                   artifacts/explanations/*.png
        ▼
app/streamlit_app.py  (loads champion bundle + artifacts; 3 tabs)
```

## 3. Canonical data schema

`src/data/ingest.py` normalizes every source into one canonical frame:

| Column | Type | Role | Notes |
|---|---|---|---|
| `policy_id` | str | identifier | retained for traceability; excluded from features |
| `subscription_length` | float | **feature** (approved adjustment 2) + source of `exposure` | local source; `Exposure` on the OpenML path |
| `exposure` | float | `sample_weight` | canonical copy of the duration/exposure column; excluded from features to avoid duplication |
| `claim_status` | int {0,1} | target | binarized `ClaimNb > 0` on the OpenML path |
| `claim_amount` | float | optional severity target | present only on the OpenML path (from `freMTPL2sev`) |
| all other raw columns | mixed | features | policyholder, vehicle, safety flags |

Source adapters:

- **Local** (`data/raw/claims.csv`): `subscription_length -> exposure` (the raw column stays in
  the feature set); zero-exposure rows dropped (logged); no severity.
- **OpenML** (`freMTPL2freq` + `freMTPL2sev`): rows joined on policy id; severity aggregated per
  policy (claim count, total claim amount); `Exposure -> exposure` (stays in features as well);
  `ClaimNb -> claim_status` (binarized); `ClaimAmount` retained for pure-premium targets and
  empirical severity.

## 4. Feature engineering (src/features/build_features.py)

**Domain features** (`build_domain_features`; pure pandas, deterministic, no fitting):

| Feature | Definition | Rationale |
|---|---|---|
| `power_bhp` | parsed from `max_power` ("113.45bhp@4000rpm" -> 113.45) | numeric power |
| `torque_nm` | parsed from `max_torque` ("250Nm@2750rpm" -> 250) | numeric torque |
| `driver_vehicle_age_ratio` | `customer_age / (vehicle_age + 1)` | requested driver-to-vehicle age ratio |
| `vehicle_customer_age_ratio` | `vehicle_age / (customer_age + 1)` | inverse view of age mix |
| `specific_power` | `power_bhp / displacement` | engine downsizing indicator |
| `weight_power_ratio` | `gross_weight / power_bhp` | mass-to-power risk proxy |

Guards: denominators floored at epsilon; `inf -> NaN`; NaNs handled by imputers (section 5).
All raw columns are retained.

**Column typing**: numeric = float/int columns (incl. engineered); categorical = object/bool/category
columns. Column lists are resolved deterministically from the training frame and stored in bundle
metadata.

## 5. Preprocessing (ColumnTransformer factory)

Numeric branch: `SimpleImputer(strategy=median, add_indicator=True)`, followed by
`StandardScaler` **only** for GLM encodings (scaling is irrelevant to trees).
Categorical branch: `SimpleImputer(strategy=constant, fill_value="__missing__")` + encoder by mode:

| Mode | Encoder | Used by |
|---|---|---|
| `target` | `sklearn.preprocessing.TargetEncoder` (internal CV smoothing = leakage-safe) | LogisticRegression, TweedieRegressor |
| `native` | `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`; estimator receives categorical feature indices (LightGBM `categorical_feature`, CatBoost `cat_features`) | LightGBM, CatBoost |
| `onehot` | `OneHotEncoder(handle_unknown="ignore")` | XGBoost |

Missingness indicators: numeric via `add_indicator=True`; categorical via the explicit
`__missing__` category (equivalent signal, keeps encoder output dtype-clean).
`get_feature_names_out` is exposed; a longest-prefix mapper groups engineered/encoded names back to
their raw source column for SHAP aggregation. Every fitted object is instantiated fresh per fit with
the global seed; nothing is fitted outside `fit` calls on training folds/splits.

## 6. Modeling (src/models)

### 6.1 Benchmark matrix

| Model | Task | Encoding | Loss / objective |
|---|---|---|---|
| `logistic_regression` | frequency (claim probability) | target | LogLoss, natural priors + exposure weights |
| `lightgbm_freq` | frequency | native | binary logloss |
| `xgboost_freq` | frequency | onehot | `binary:logistic` |
| `catboost_freq` | frequency | native | Logloss |
| `tweedie_glm` | pure premium (expected claim cost) | target | Tweedie deviance, power 1.5 |
| `lightgbm_tweedie` | pure premium | native | `objective=tweedie`, power 1.5 |
| `xgboost_tweedie` | pure premium | onehot | `reg:tweedie`, power 1.5 |
| `catboost_tweedie` | pure premium | native | `Tweedie:variance_power=1.5` |

Pure-premium target: `claim_amount` when available (OpenML path); else
`claim_status x expected_severity`.

### 6.2 Weighting rules (approved adjustments)

- Every estimator is fitted with `sample_weight = exposure`.
- **Frequency classifiers: natural class priors.** No `scale_pos_weight`, no class-weight
  resampling, no SMOTE — anywhere. This keeps `P(claim)` calibrated so that
  `expected_pure_premium = claim_probability x expected_severity` is on the right scale.
- Pure-premium Tweedie models weight by `exposure` as well (deviance per exposure-year).

### 6.3 Tuning (tune.py)

- Optuna TPE sampler with `seed = 42`; per-model search spaces and `n_trials` from config
  (bounded budget: GLMs 15, GBMs 8-12).
- **Manual 5-fold stratified CV** on the train split (skips sklearn param-routing pitfalls and
  threads `sample_weight` through every fold). Optional seeded tuning subsample caps CV rows for
  wall-clock control; the final fit always uses the full train+val.
- Objective: mean fold **PR-AUC** (frequency), mean fold **Tweedie deviance** (pure premium).
- Best params persisted to `artifacts/reports/optuna_<model>.json`.

### 6.4 Training flow (train.py)

1. Load config; seed everything from `seed: 42`.
2. Ingest -> domain features -> stratified split (indices persisted).
3. Per model: tune (6.3) -> **5-fold OOF predictions on train+val** with best params ->
   risk-tier thresholds (6.5) -> final preprocessor fit on train+val -> estimator refit on
   train+val -> bundle saved.
4. Champion = best frequency model by OOF PR-AUC on train+val; `champion.joblib` saved.
5. Persist `train_summary.json`, `run_metadata.json` (seed, library versions, split hash,
   timestamps), `input_defaults.json` (train+val medians/modes + ranges for the app form),
   `schema.json` (raw input columns, dtypes, allowed categories, ranges, aliases
   `subscription_length -> exposure`).

### 6.5 Risk tiers (approved adjustment 3)

From **out-of-fold** validation scores of the model itself (5-fold stratified OOF over train+val
with best params; each row scored by a model that did not see it):

- `low_max` = P25 of OOF scores, `high_min` = P75 of OOF scores.
- Tier assignment: score <= `low_max` -> Low; score >= `high_min` -> High; else Medium.
- Thresholds and the OOF score summary are stored in each bundle's metadata (auditable); the app is
  stateless and simply applies the stored cutoffs.

## 7. Artifact contract

| Path | Content |
|---|---|
| `artifacts/models/bundle_<model>.joblib` | ClaimsModelBundle: preprocessor, estimator, metadata (model kind, encoding, feature columns, column groups, severity assumption, OOF tier thresholds, OOF score summary, seed) |
| `artifacts/models/champion.joblib` | copy of the winning frequency bundle |
| `artifacts/models/champion_name.txt` | champion model name |
| `artifacts/reports/data_split.json` | split indices + SHA-256 split hash + class rates per split |
| `artifacts/reports/train_summary.json` | tuning + OOF results per model (val metrics, best params) |
| `artifacts/reports/metrics.json` | full test metrics per model (section 8) |
| `artifacts/reports/model_comparison.json` | benchmark table (all models, test metrics; drives diagnostics tab) |
| `artifacts/reports/run_metadata.json` | seed, library versions, dataset fingerprint, timestamps |
| `artifacts/reports/input_defaults.json` | per-feature default (median/mode) + min/max for the app form |
| `artifacts/reports/schema.json` | batch-scoring schema: required columns, dtypes, allowed categories, aliases |
| `artifacts/reports/optuna_<model>.json` | best params + CV scores per model |
| `artifacts/reports/shap_global.json` | mean abs SHAP per raw feature (champion) |
| `artifacts/plots/calibration.png` | calibration curve per frequency model (10 equal-frequency bins) |
| `artifacts/plots/lift.png` | cumulative lift by decile (champion) |
| `artifacts/plots/roc_pr.png` | ROC + PR curves (all frequency models) |
| `artifacts/plots/shap_summary.png` | SHAP beeswarm (champion) |
| `artifacts/plots/shap_importance.png` | mean abs SHAP bar chart (champion) |
| `artifacts/explanations/shap_local_reference.png` | local SHAP waterfall for a reference test policy |

## 8. Evaluation (src/evaluation)

Computed per model on the **untouched test split** (exposure-weighted where meaningful):

| Metric | Definition |
|---|---|
| PR-AUC | `sklearn.metrics.average_precision_score` |
| ROC-AUC | `roc_auc_score` |
| Brier score | mean squared error of calibrated probability (frequency) |
| Calibration | 10 equal-frequency bins: predicted vs observed -> `calibration.png` |
| Normalized Gini | `gini(y, pred) / gini(y, y)` (Lorenz/rank-based; 1.0 for a perfect ranking) |
| Cumulative lift | observed rate in top-k deciles / overall rate -> `lift.png` |
| Tweedie deviance | power 1.5 (config), on claim cost (`claim_status x expected_severity` locally; `claim_amount` on OpenML), exposure-weighted |

SHAP (`explain.py`): TreeExplainer for LightGBM/XGBoost/CatBoost; LinearExplainer for GLMs with a
small stored background sample and a coefficient-based fallback. Global outputs: beeswarm + bar
importance + `shap_global.json` (aggregated to raw source columns). Local output: waterfall for a
reference policy. Matplotlib `Agg` backend; clean minimal styling; no emoji in figure text.

## 9. Inference API (src/models/predict.py)

```python
class ClaimsModelBundle:
    model_kind: "frequency" | "pure_pure_premium"  # frequency | pure_premium
    def predict(self, df: pd.DataFrame) -> pd.DataFrame
        # columns: claim_probability (frequency models),
        #          expected_pure_premium (probability x severity; or model output for pure-premium),
        #          risk_tier ("Low" | "Medium" | "High" via stored OOF thresholds),
        #          claim_frequency_per_exposure (probability / exposure)
    def validate_input(self, df: pd.DataFrame) -> ValidationReport
        # ok, errors[], warnings[], n_rows; errors name the column and the fix
    def explain_local(self, row: pd.DataFrame) -> pd.DataFrame
        # feature, contribution (SHAP), aggregated to raw source columns, sorted by |contribution|
    def save(self, path) / @classmethod load(self, path)   # joblib
```

`predict` pipeline: validate -> `build_domain_features` -> select stored feature columns ->
`preprocessor.transform` -> estimator -> assemble outputs -> apply stored tier thresholds. Row
order and row count are preserved exactly. Pure-premium bundles report model-implied expected cost
and leave `claim_probability` NaN (the app uses the champion frequency bundle for probability).

## 10. Streamlit application (app/streamlit_app.py)

**Design rules.** Minimal, high-contrast, clear typography; compact sidebar; 3 focused tabs;
**strictly zero emojis** in titles, tabs, labels, status messages, and charts — enforced by
`tests/test_app_strings.py` (scans app sources for emoji code points).

- **Sidebar**: champion model summary, severity assumption input (default from config; drives
  `expected_pure_premium`), risk-tier threshold display, artifact regeneration timestamp.
- **Loading**: `st.cache_resource` for the bundle + reports; missing artifacts produce an
  actionable message ("Run `make train` first"), never a stack trace.
- **Tab 1 — Single Policy Simulator**: compact `st.form` (policyholder / vehicle / safety groups)
  with defaults and bounds from `input_defaults.json` + `schema.json`; policy duration
  (`subscription_length`) is an editable input that moves the prediction (approved adjustment 2).
  On submit: KPI cards via `st.metric` — **Claim Probability**, **Risk Tier** (Low/Medium/High),
  **Expected Pure Premium** — plus a local SHAP driver chart (horizontal bar of top contributions)
  from `explain_local`.
- **Tab 2 — Batch Scoring**: CSV `st.file_uploader` -> `validate_input` schema report (missing
  columns, invalid categories, per-column issues) -> scored preview with risk tiers ->
  `st.download_button` for the scored CSV; row order/count preserved.
- **Tab 3 — Model Diagnostics**: model comparison table (`model_comparison.json`), calibration /
  lift / ROC-PR plots from `artifacts/plots/`, global SHAP importance, run metadata (seed,
  library versions, split hash).

## 11. Configuration (configs/config.yaml)

Sections (single source of truth; every knob below is config-driven):

- `seed: 42` — global; propagated to splits, Optuna sampler, estimators, OOF folds.
- `data`: `raw_path`, `fallback: openml`, OpenML dataset names, `expected_severity: 3000.0`,
  `drop_zero_exposure: true`, target/exposure/id/severity column names, alias mapping.
- `split`: `train: 0.70`, `val: 0.15`, `test: 0.15`, `stratify_target: claim_status`.
- `features`: domain-ratio toggle, engineered column names, column-type overrides, exclusions.
- `models.<name>`: `task` (frequency | pure_premium), `family` (glm | lightgbm | xgboost |
  catboost), `encoding` (target | native | onehot), `params`, `search_space` (float/int/
  categorical with log flags), `n_trials`.
- `tuning`: `cv_folds: 5`, `subsample_rows` (seeded cap for CV wall-clock).
- `metrics`: `tweedie_power: 1.5`, `calibration_bins: 10`, `lift_deciles: 10`.
- `risk_tiers`: `quantiles: [0.25, 0.75]` (OOF-derived cutoffs; approved adjustment 3).
- `app`: defaults source, KPI formatting, page title.

## 12. Determinism and leakage guarantees

- Split before any fitting; stratified on `claim_status`; indices persisted with a SHA-256 hash.
- Every transformer is fitted on training data only (per CV fold during tuning; train+val for the
  final fit); test data is touched exactly once, by `evaluation/metrics.py` and `explain.py`.
- All stochastic components derive from `seed: 42` (split, Optuna TPE, estimator random_state,
  OOF folds, tuning subsample); LightGBM uses deterministic training flags.
- Frequency probabilities are calibrated by construction: natural priors + exposure weights only
  (approved adjustment 1).
- Tests assert each guarantee independently (see PLAN.md section 6).
