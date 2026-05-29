"""Central configuration: paths, seeds, column groups, business settings.

Everything that other modules need to agree on lives here so the pipeline is
reproducible and there is a single source of truth for column names.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets"
RAW_DATA = DATA_DIR / "global_company.csv"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
REPORTS_DIR = RESULTS_DIR / "reports"

for _d in (MODELS_DIR, FIGURES_DIR, METRICS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MODEL_ARTIFACT = MODELS_DIR / "best_model.joblib"
METADATA_ARTIFACT = MODELS_DIR / "model_metadata.json"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #
TARGET_RAW = "Loan Status"          # column name in the raw csv
TARGET = "target"                   # binary 0/1 column we create
POSITIVE_LABEL = "Defaulted"        # maps to 1 (the event we predict)
NEGATIVE_LABEL = "Did not default"  # maps to 0

# --------------------------------------------------------------------------- #
# Columns to discard during cleaning
#   - index / id columns carry no signal
#   - exact-duplicate columns (verified identical): sex==gender,
#     number_of_defaults.1==number_of_defaults, age.1==age
#   - constant columns after canonicalisation (currency, country) carry no signal
# --------------------------------------------------------------------------- #
DROP_COLUMNS = [
    "Unnamed: 0",
    "loan_id",
    "sex",                 # identical to gender
    "number_of_defaults.1",  # identical duplicate
    "age.1",               # identical duplicate
    "currency",            # constant after canonicalisation (USD / $USD)
    "country",             # constant after canonicalisation (all Zimbabwe)
]

RAW_DATE_COLUMN = "disbursemet_date"   # note: typo exists in the source data
DATE_COLUMN = "disbursement_date"      # cleaned name

# --------------------------------------------------------------------------- #
# Feature groups AFTER cleaning + feature engineering.
# These drive the preprocessing ColumnTransformer.
# --------------------------------------------------------------------------- #
BASE_NUMERIC = [
    "loan_amount",
    "outstanding_balance",
    "interest_rate",
    "age",
    "number_of_defaults",
    "remaining_term",
    "salary",
    "is_employed",          # cast to 0/1
]

ENGINEERED_NUMERIC = [
    # --- amortization / cost-of-credit (from loan_amount, interest_rate, term) ---
    "monthly_rate",
    "estimated_monthly_payment",
    "total_repayment_est",
    "interest_cost_est",
    "interest_cost_ratio",
    # --- affordability / indebtedness ---
    "dti",                       # estimated payment / monthly income
    "loan_to_income",
    "balance_to_income",
    "balance_to_loan",
    "balance_minus_loan",
    "interest_burden",
    # --- credit history ---
    "has_prior_default",
    "defaults_per_decade",
    # --- domain flags / scores ---
    "is_high_interest",
    "is_urban",
    "is_stable_job",
    "is_risky_job",
    "job_risk_score",
    "marital_risk_score",
    # --- interaction (cross) features ---
    "amount_x_rate",
    "rate_x_term",
    "dti_x_defaults",
    "lti_x_defaults",
    # --- log transforms (tame heavy right tails) ---
    "log_loan_amount",
    "log_salary",
    "log_outstanding_balance",
    "log_estimated_monthly_payment",
    "log_interest_cost_est",
    "log_loan_to_income",
    # --- calendar (within-year + cyclical). Absolute-time (loan_year,
    #     loan_dayofyear) is computed but EXCLUDED from the model (drift risk). ---
    "loan_month",
    "loan_quarter",
    "loan_week",
    "loan_dayofweek",
    "loan_is_month_end",
    "loan_is_quarter_end",
    "month_sin",
    "month_cos",
    "quarter_sin",
    "quarter_cos",
    "dow_sin",
    "dow_cos",
    # --- missingness (computed BEFORE imputation; see feature_engineering) ---
    "job_missing",
    "location_missing",
    "marital_status_missing",
    "remaining_term_missing",
    "total_missing_fields",
]

# Computed but intentionally NOT fed to the model (absolute-time = drift risk).
DRIFT_PRONE_TIME_FEATURES = ["loan_year", "loan_dayofyear"]

NUMERIC_FEATURES = BASE_NUMERIC + ENGINEERED_NUMERIC

# Engineered categorical buckets (one-hot encoded).
ENGINEERED_CATEGORICAL = ["age_bucket", "rate_segment"]
LOW_CARD_CATEGORICAL = ["gender", "job", "marital_status"] + ENGINEERED_CATEGORICAL
HIGH_CARD_CATEGORICAL = ["location"]                          # target encoded (157 values)

ALL_FEATURES = NUMERIC_FEATURES + LOW_CARD_CATEGORICAL + HIGH_CARD_CATEGORICAL

# --------------------------------------------------------------------------- #
# Learned group-aggregate / peer-residual features.
# Computed by a stateful transformer fit on the TRAINING fold only (leakage-safe),
# unlike the Kaggle concat(train,test) pattern. Each (keys, value) yields
# mean/median/std per group + a deviation-from-group-mean ("peer residual").
# --------------------------------------------------------------------------- #
GROUP_AGG_SPECS = [
    (["job"], "salary"),
    (["job"], "loan_amount"),
    (["job"], "interest_rate"),
    (["location"], "loan_amount"),
    (["location"], "interest_rate"),
    (["marital_status"], "salary"),
]
GROUP_AGG_STATS = ["mean", "median", "std"]

# Raw fields whose per-row missingness we flag (before imputation).
MISSINGNESS_FIELDS = ["job", "location", "marital_status", "remaining_term"]

# --------------------------------------------------------------------------- #
# Fairness: protected attributes we must NOT discriminate on.
# We never feed these to the model as a privileged signal beyond what is
# legally usable, and we audit outcomes across their groups.
# --------------------------------------------------------------------------- #
PROTECTED_ATTRIBUTES = ["gender"]

# --------------------------------------------------------------------------- #
# Human-readable descriptions used for adverse-action (rejection) reason codes.
# --------------------------------------------------------------------------- #
FEATURE_DESCRIPTIONS = {
    "number_of_defaults": "Number of previous loan defaults",
    "outstanding_balance": "Outstanding balance owed",
    "loan_amount": "Requested / current loan amount",
    "interest_rate": "Interest rate on the loan",
    "remaining_term": "Remaining term of the loan",
    "salary": "Declared salary / income",
    "age": "Applicant age",
    "is_employed": "Employment status",
    "loan_to_income": "Loan amount relative to income",
    "debt_to_income": "Outstanding debt relative to income",
    "balance_to_loan": "Balance owed relative to original loan",
    "balance_minus_loan": "Balance owed above original loan amount",
    "interest_burden": "Total interest burden",
    "loan_year": "Year the loan was disbursed",
    "loan_month": "Month the loan was disbursed",
    "loan_quarter": "Quarter the loan was disbursed",
    "loan_week": "ISO week the loan was disbursed",
    "loan_dayofweek": "Day of week the loan was disbursed",
    "gender": "Gender",
    "job": "Occupation",
    "marital_status": "Marital status",
    "location": "Location",
}
