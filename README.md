# Loan Default Prediction

A production-grade, explainable, and fairness-audited credit-default classification
system. Trains on `global_company.csv` (~100k loan records), serves predictions
through a self-contained inference engine, and ships a full lending-intelligence
layer on top: risk-based pricing, segmentation, anomaly detection, actionable
recourse, and drift monitoring.

The design principle throughout is **leakage-safe by construction**: all cleaning,
imputation, scaling, and encoding live inside an sklearn `Pipeline` fit on training
folds only. Heavy logic lives in `src/`; notebooks stay thin and only orchestrate
and narrate.

---

## Problem

Binary classification — *will a loan default?* The target `Loan Status` is encoded
as `Defaulted = 1` / `Did not default = 0`, with a **~85 / 15 class imbalance**.
Because accuracy is misleading under imbalance, models are optimized and reported on
**F1 / ROC-AUC / PR-AUC** with probability **calibration**, not raw accuracy.

## Results

Best model: **stacking ensemble**, selected by cross-validated ROC-AUC, evaluated on
a held-out test set at a profit/F1-tuned decision threshold of **0.79**.

| Metric | Test | Train | Overfit gap |
|---|---|---|---|
| ROC-AUC | **0.897** | 0.967 | 0.070 |
| PR-AUC | **0.784** | 0.899 | — |
| F1 (tuned threshold) | **0.724** | 0.805 | — |
| Precision / Recall | 0.798 / 0.662 | 0.854 / 0.762 | — |
| Balanced accuracy | 0.816 | 0.870 | — |
| Brier (calibration) | 0.106 | 0.084 | — |
| MCC | 0.684 | 0.775 | — |

**Model leaderboard** (test ROC-AUC, default threshold):

| Model | ROC-AUC | PR-AUC | F1 | Brier |
|---|---|---|---|---|
| Stacking | 0.897 | 0.784 | 0.640 | 0.106 |
| CatBoost | 0.895 | 0.780 | 0.664 | 0.100 |
| Voting (soft) | 0.892 | 0.773 | 0.679 | 0.095 |
| XGBoost | 0.891 | 0.774 | 0.676 | 0.094 |
| Random Forest | 0.891 | 0.770 | 0.710 | 0.073 |
| HistGradientBoosting | 0.888 | 0.764 | 0.668 | 0.100 |
| KNN | 0.877 | 0.732 | 0.408 | 0.081 |
| Logistic Regression | 0.850 | 0.614 | 0.543 | 0.142 |

> Figures and raw metrics are regenerated into `results/` on every training run.

## Design highlights

**Leakage discipline.** Cleaning, imputation, scaling, and encoding are pipeline
steps fit on training folds only. The decision threshold is tuned on
cross-validated out-of-fold predictions — never on the test set.

**Honest data handling.** Drops three verified-duplicate columns (`sex`, `age.1`,
`number_of_defaults.1`), canonicalizes dirty categories (`$USD` → `USD`,
`Data Scintist` → `Data Scientist`), recovers junk numerics (`'69_'` → `69`), and
median / `Unknown`-imputes the rest rather than dropping rows.

**Right encoding for the cardinality.** One-hot for low-cardinality nominals;
**target encoding** for the 157-value `location` (no fake ordinal ordering, no 157
sparse columns).

**Model zoo + ensembles.** Logistic Regression, KNN, Random Forest,
HistGradientBoosting, XGBoost, LightGBM, CatBoost, plus soft-voting and stacking —
all imbalance-aware and tuned with `RandomizedSearchCV` over `StratifiedKFold`.

**Trust and stability.** Train↔test gap (overfit check), ROC/PR curves, probability
calibration (Brier), and a confusion matrix evaluated at the tuned threshold.

**Fairness audit** across `gender` — demographic parity, equal opportunity, and FPR
gaps (`src/fairness.py`).

**Explainability for compliance.** Per-applicant **adverse-action reason codes**
(why an application was declined), with protected attributes excluded from the
reasons (`src/explain.py`).

**Adversarial validation.** Detects features that drift between the training period
and the future (e.g. absolute-time features) and prunes them
(`src/feature_selection.py`).

**Rich feature engineering (~78 features).** Amortization math
(`estimated_monthly_payment`, `interest_cost_est`), affordability ratios (`dti`,
`loan_to_income`, `balance_to_income`), credit-history signals, hand-coded risk
scores, interaction terms, log transforms, and date features (quarter / ISO-week /
day-of-week / period-end flags + **cyclical** sin·cos encodings so December ≈
January).

**Learned group aggregates.** `GroupStatsEncoder` computes per job/location/marital
mean·median·std plus deviation-from-peer, fit on the training fold only —
leakage-safe and single-row-ready at inference time.

**Missingness as signal.** Per-field missing flags plus a `total_missing_fields`
count captured *before* imputation. It is predictive: default rate rises
14% → 23% → 37% as undisclosed fields go 0 → 1 → 2.

### Lending intelligence layer

| Capability | Module | What it does |
|---|---|---|
| Profit optimization & risk-based pricing | `src/profit.py` | Picks the approval threshold that maximizes expected profit; prices each loan by risk |
| Customer segmentation | `src/segmentation.py` | Borrower personas profiled by risk *and* profitability |
| Actionable recourse | `src/recourse.py` | Not just *why* declined but *what would get you approved* (counterfactuals, DiCE-compatible) |
| Fraud / anomaly detection | `src/anomaly.py` | Unsupervised gate for abnormal applications |
| Drift monitoring | `src/monitoring.py` | PSI + Evidently report, surfaced in a Streamlit app |

## Project structure

```
src/         config, data_loader, cleaning, feature_engineering, preprocessing,
             feature_selection (+ adversarial validation), models, train, evaluate,
             fairness, explain, inference, profit, segmentation, anomaly, recourse,
             monitoring
inference/   self-contained serving package (no src dependency):
             config, transforms, compat (module-alias shim), dto, analytics, engine
notebooks/   01_eda → 02_feature_engineering → 03_feature_selection →
             04_model_training → 05_model_evaluation → 06_inference_demo →
             07_segmentation → 08_profit_and_pricing → 09_fraud_anomaly →
             10_recourse → 11_monitoring_drift → 12_inference_engine_demo
app/         streamlit_app.py  (score + explain + recourse demo)
models/      best_model.joblib + model_metadata.json  (produced by training)
results/     figures/  metrics/  reports/             (deliverables)
tests/       pytest sanity checks
```

## Quickstart

```bash
pip install -r requirements.txt

pytest -q                       # sanity checks
python -m src.train --quick     # fast end-to-end run on a 20k subsample
python -m src.train             # full sweep -> models/ + results/

# score new applicants (raw schema) and explain declines
python -m src.inference --input new_applicants.csv --output scored.csv --explain

streamlit run app/streamlit_app.py   # interactive demo (after training)
```

Or work through `notebooks/01..12` in order — `12` is the capstone demo driven by the
portable `inference/` engine.

## Portable inference engine

`inference/` is a standalone serving layer with **zero dependency on `src/`** — copy
the folder, a trained `.joblib`, and its metadata JSON, and it runs anywhere. It
loads a model pickled under `src.*` via a module-alias shim (`compat.py`), accepts
typed DTOs, and returns typed result objects.

```python
from inference import (
    InferenceConfig, LoanInferenceEngine, ApplicantDTO, PredictionOptions,
)

engine = LoanInferenceEngine(InferenceConfig(
    model_path="models/best_model.joblib",
    metadata_path="models/model_metadata.json",
))

res = engine.predict(
    ApplicantDTO(
        loan_amount=52000, salary=2400, outstanding_balance=71000,
        interest_rate=0.24, number_of_defaults=2, remaining_term=66,
        age=27, gender="male", job="Data Analyst", location="Beitbridge",
        marital_status="single", is_employed=True,
        disbursement_date="2023 09 12", applicant_id="APP-1001",
    ),
    PredictionOptions(include_reasons=True, include_recourse=True, include_pricing=True),
)
res.to_dict()   # probability, decision, risk_band, reasons, recourse, pricing

engine.batch_predict(applicants, options)    # list[PredictionResult]
engine.segment_portfolio(df, k=4)            # personas by risk & value
engine.optimize_profit(df, outcomes)         # profit-max threshold + uplift
engine.detect_anomalies(df)                  # fraud / abnormal applications
engine.monitor_drift(reference, current)     # PSI drift report
```

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost ·
imbalanced-learn · Evidently · Streamlit · pytest. See `requirements.txt` for pins.

## Environment notes

- **SHAP is intentionally omitted** — it conflicts with NumPy ≥ 2.4 (numba) in this
  environment. Explainability uses permutation importance (global) plus occlusion
  reason codes (local). Pin `numpy<=2.3` to re-enable SHAP.
- **LightGBM auto-disables** under scikit-learn ≥ 1.6 (it calls the removed
  `force_all_finite` kwarg and crashes at fit). Training is resilient and skips any
  incompatible model; the rest of the zoo still trains. Upgrade LightGBM or pin
  scikit-learn < 1.6 to re-enable it.
- The original exploratory notebook `loan_credit_score_prediction.ipynb` is kept at
  the repo root for reference; the `notebooks/` series supersedes it.
- Fairness gaps are a **signal to investigate**, not an automatic verdict — pair them
  with domain and legal review before deployment.
