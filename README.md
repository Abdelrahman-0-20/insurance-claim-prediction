# Insurance Claims Prediction

A modular, production-grade Python repository for motor insurance claims prediction:
config-driven training, strict leakage prevention, actuarial evaluation
(PR-AUC, ROC-AUC, calibration/Brier, Normalized Gini, cumulative lift, Tweedie deviance),
SHAP explainability, and a clean three-tab Streamlit application.

- Python 3.11+ (tested on 3.12), `seed = 42` everywhere
- Interpretable GLMs benchmarked against LightGBM, XGBoost, and CatBoost
- Exposure-weighted actuarial fitting, Optuna hyperparameter tuning, out-of-fold risk tiers
- Zero emojis anywhere in the application UI

## Quickstart

```bash
make setup      # creates .venv, installs dependencies, copies the dataset to data/raw/claims.csv
make test       # pytest: determinism, zero leakage, inference contract, actuarial metrics
make train      # tunes and trains all configured models, writes artifacts/models + reports
make evaluate   # test-split metrics, plots, and global/local SHAP reports
make app        # streamlit run app/streamlit_app.py
```

`make setup` expects `Insurance claims data.csv` in the repository root (it copies it to
`data/raw/claims.csv`). If the file is absent, the pipeline auto-fetches the OpenML
`freMTPL2freq` + `freMTPL2sev` benchmark datasets instead (network required).

If a training run is interrupted, continue where it stopped (models whose bundle and tuning
report already exist on disk are skipped):

```bash
.venv/bin/python -m src.models.train --config configs/config.yaml --resume
```

## Repository structure

```
configs/config.yaml          single source of truth (seed, data, split, models, tuning, metrics)
pyproject.toml               packaging + runtime/dev dependencies
Makefile                     setup | train | evaluate | test | app | clean
docs/PLAN.md                 implementation plan (approved)
docs/ARCHITECTURE.md         system architecture and contracts
data/raw/claims.csv          raw dataset (copied by make setup)
src/
  config.py                  typed config loading, deterministic seeding
  data/ingest.py             local CSV loader + OpenML fallback -> canonical schema
  data/split.py              stratified train/val/test indices (fits nothing)
  features/build_features.py domain ratios + ColumnTransformer factory (3 encoding modes)
  models/tune.py             estimator builders, exposure-weighted CV, Optuna tuning
  models/train.py            end-to-end training CLI, artifact persistence
  models/predict.py          ClaimsModelBundle inference wrapper
  evaluation/metrics.py      actuarial + classification metrics, plots, evaluation CLI
  evaluation/explain.py      SHAP helpers (per-family explainers), report CLI
app/streamlit_app.py         3-tab Streamlit application
tests/                       pytest suite (44 tests)
artifacts/                   models, JSON reports, plots, SHAP outputs
```

## Data

| Property | Value |
|---|---|
| Source | `Insurance claims data.csv` (58,592 policies x 41 columns) |
| Target | `claim_status` (binary), claim rate 6.4% |
| Exposure | `subscription_length` (0-14 policy-years; 276 zero-exposure rows dropped) |
| Features | policyholder (age, region, density), vehicle (age, segment, model, engine, dimensions), safety equipment flags |
| Missing values | none in the local source |

**Assumptions**

1. The provided CSV is the canonical `data/raw/claims.csv`; OpenML `freMTPL2freq` +
   `freMTPL2sev` is the automatic fallback with the same canonical schema.
2. `subscription_length` serves as both an input feature and the exposure weight; rows with
   zero exposure are dropped at ingest (logged and asserted).
3. The local source has no claim amounts, so
   **expected pure premium = claim probability x expected severity**
   (`data.expected_severity`, default 3,000; editable in the app sidebar). On the OpenML
   path, severity targets come from actual `ClaimAmount`s.

## Methodology

### Leakage prevention

- The stratified train/val/test split (70/15/15, stratified on `claim_status`, seed 42) is
  computed **before any transformer is fitted**; the split hash is persisted to
  `artifacts/reports/data_split.json`.
- Every preprocessor is fitted on training rows only (per fold during tuning; train+val for
  the final refit). The test split is touched exactly once, by evaluation.
- Domain ratios are pure functions with no fitted state.
- Tests prove that fitted statistics equal train-only medians, that the target/weight columns
  never appear among transformed features, and that `split.py` fits nothing.

### Feature engineering

Parsed mechanics (`torque_nm`, `power_bhp`) and guarded domain ratios:
driver-to-vehicle age ratio `customer_age / (vehicle_age + 1)`, specific power
`power_bhp / displacement`, weight-to-power, and the inverse age ratio. Divisions are
epsilon-guarded (zero denominators become missing values, handled by imputers with
missingness indicators).

### Preprocessing

A `ColumnTransformer` with two pipeline branches (median imputer + optional scaling for
numerics; constant imputer + encoder for categoricals), one encoding mode per model family:

| Mode | Encoder | Used by |
|---|---|---|
| `target` | `TargetEncoder` (internal CV smoothing, leakage-safe) | LogisticRegression, TweedieRegressor |
| `native` | `OrdinalEncoder` (unknown -> -1) + estimator categorical indices | LightGBM, CatBoost |
| `onehot` | `OneHotEncoder(handle_unknown="ignore")` | XGBoost |

### Weighting and calibration (approved rules)

- Every estimator is fitted with `sample_weight = exposure`.
- **Frequency classifiers use natural class priors**: no `scale_pos_weight`, no
  `class_weight`, no SMOTE anywhere (enforced by a source-scanning test). This keeps
  probabilities calibrated so the pure premium is on the right scale.
- Pure-premium models (Tweedie objective, power 1.5) are exposure-weighted as well.

### Models and tuning

| Model | Task | Objective |
|---|---|---|
| LogisticRegression | frequency | log loss |
| TweedieRegressor | pure premium | Tweedie deviance (power 1.5) |
| LightGBM | frequency + pure premium | binary logloss / tweedie |
| XGBoost | frequency + pure premium | binary:logistic / reg:tweedie |
| CatBoost | frequency + pure premium | Logloss / Tweedie:variance_power=1.5 |

Optuna (TPE, seed 42) tunes each model with a manual 5-fold weighted stratified CV on a
seeded tuning subsample; the objective is mean fold PR-AUC (frequency) or mean Tweedie
deviance (pure premium). Best parameters are refitted on train+val, and the champion is the
frequency model with the best **out-of-fold PR-AUC** on train+val.

### Risk tiers

Low/Medium/High cutoffs are the P25/P75 quantiles of the model's **out-of-fold**
predictions on train+val (each row scored by a model that never saw it), persisted in the
bundle - never recomputed from in-sample scores or from the input batch.

### Evaluation

All metrics are computed on the untouched test split, exposure-weighted where meaningful:

| Metric | Definition |
|---|---|
| PR-AUC | average precision on claim status |
| ROC-AUC | area under the ROC curve |
| Brier / calibration | mean squared probability error; 10 equal-frequency bins |
| Normalized Gini | Lorenz-curve Gini divided by the perfect-ranking Gini (1.0 = perfect, -1.0 = inverted) |
| Cumulative lift | observed claim rate in the top-k risk deciles / overall rate |
| Tweedie deviance | power 1.5 on claim cost (`claim_status x severity` locally) |

### Explainability

SHAP with per-family explainers (TreeExplainer for the boosted trees, LinearExplainer for the
GLMs with a coefficient-based fallback). Global importance is aggregated back to raw source
columns; local explanations show one policy's drivers. Outputs:
`artifacts/plots/shap_summary.png`, `shap_importance.png`,
`artifacts/reports/shap_global.json`, `artifacts/explanations/shap_local_reference.png`.

## Benchmark results

Test split (untouched, exposure-weighted), seed 42:

| Model | Task | PR-AUC | ROC-AUC | Brier | Norm. Gini | Tweedie dev. |
|---|---|---|---|---|---|---|
| `xgboost_freq` **(champion)** | frequency | 0.1035 | 0.5964 | 0.0702 | 0.1927 | 86.0170 |
| `lightgbm_freq` | frequency | 0.1017 | 0.5977 | 0.0702 | 0.1953 | 85.9579 |
| `catboost_freq` | frequency | 0.1003 | 0.5948 | 0.0705 | 0.1895 | 86.3213 |
| `logistic_regression` | frequency | 0.0925 | 0.5690 | 0.0705 | 0.1380 | 86.8225 |
| `catboost_tweedie` | pure premium | 0.1064 | 0.5988 | - | 0.1976 | 59.3386 |
| `lightgbm_tweedie` | pure premium | 0.1015 | 0.5943 | - | 0.1886 | 56.1030 |
| `xgboost_tweedie` | pure premium | 0.0995 | 0.5924 | - | 0.1848 | 58.4668 |
| `tweedie_glm` | pure premium | 0.0905 | 0.5658 | - | 0.1315 | 59.9463 |

Conditions: 8,748-row untouched test split, exposure-weighted metrics, seed 42
(split hash `6a03cee8a7da7592`), Python 3.12.3, scikit-learn 1.9.1, LightGBM 4.7.0,
XGBoost 3.4.1, CatBoost 1.2.10, Optuna 5.1.0, SHAP 0.54.1. Brier/calibration apply to
frequency models only; pure-premium models are judged by Tweedie deviance (lower is better)
and by ranking metrics on their expected-cost scores.

Champion selection: highest out-of-fold PR-AUC on train+val.

Note on signal strength: this portfolio has weak inherent signal (a 6.4% claim rate with
near-random claim labels), so absolute discriminative values are modest across all model
families - the benchmark's value is in the honest, calibrated comparison.

## Streamlit application

```bash
make app     # streamlit run app/streamlit_app.py
```

If port 8501 is already in use on your machine, pick another one:

```bash
.venv/bin/streamlit run app/streamlit_app.py --server.port 8505
```

### Deploying to Streamlit Community Cloud

The repository ships a `requirements.txt` (Streamlit Cloud's dependency file) and the
committed `artifacts/`, so a deployment needs no training step:

1. Push the repository to GitHub (already set up for
   `Abdelrahman-0-20/insurance-claim-prediction`).
2. In Streamlit Community Cloud, create an app from that repo with main file
   `app/streamlit_app.py`.
3. If dependency installation ever fails, open "Manage app" to read the installer log;
   `requirements.txt` is the file Streamlit Cloud installs from.

Design: minimal, high-contrast layout, compact sidebar, strictly zero emojis (enforced by
`tests/test_app_strings.py`).

1. **Single Policy Simulator** - form with train-derived defaults for every input;
   KPI cards for claim probability, risk tier (Low/Medium/High), and expected pure
   premium; a local SHAP driver chart for the scored policy.
2. **Batch Scoring** - CSV upload with schema validation (missing columns, unknown
   categories), tier summary, preview, and scored-CSV download; row order and count are
   preserved. A one-row input template can be downloaded.
3. **Model Diagnostics** - model comparison table, calibration/lift/ROC-PR plots, global
   SHAP importance, and run metadata from `artifacts/`.

If artifacts are missing, the app shows an actionable message instead of a stack trace.

## Tests

```bash
make test    # 46 tests
```

| Suite | What it proves |
|---|---|
| `test_split` | determinism, stratification, disjointness |
| `test_leakage` | train-only fitted statistics, target never in features, split fits nothing |
| `test_features` | deterministic domain features, ratio guards, preprocessor determinism, unknown-category safety |
| `test_weights_calibration` | exposure-only weights, natural priors (no scale_pos_weight/class_weight/SMOTE) |
| `test_predict` | inference contract: columns, probability range, pure premium = probability x severity, tier partition, row order, actionable validation errors, save/load roundtrip |
| `test_risk_tiers` | OOF-derived P25/P75 cutoffs drive tier assignment |
| `test_metrics` | perfect/random/degenerate metric correctness |
| `test_ingest` | local normalization + OpenML fallback join/aggregation |
| `test_app_strings` | zero emojis in any app source |
| `test_app_smoke` | the app executes headlessly (Streamlit AppTest): 3 tabs render, form submission produces the KPI cards |

## Reproducibility

Every random component derives from `seed: 42` in `configs/config.yaml`: the split,
Optuna's TPE sampler, estimator `random_state`, CV folds, and the tuning subsample.
`artifacts/reports/run_metadata.json` records library versions, the split hash, and the
training timestamp. Re-running `make train` reproduces identical tuned parameters.
