"""Self-contained config for the inference package.

A standalone copy of the constants the feature transforms + engine need, so the
`inference/` package has NO dependency on `src/`. Keep in sync with src/config.py
if you change the feature contract (or retrain — the pickled pipeline carries its
own fitted state regardless).
"""
from __future__ import annotations

RANDOM_STATE = 42

# Target
TARGET = "target"
POSITIVE_LABEL = "Defaulted"

# Cleaning
DROP_COLUMNS = [
    "Unnamed: 0", "loan_id", "sex", "number_of_defaults.1", "age.1",
    "currency", "country",
]
RAW_DATE_COLUMN = "disbursemet_date"
DATE_COLUMN = "disbursement_date"

# Numeric base (raw) columns coerced during cleaning
BASE_NUMERIC = [
    "loan_amount", "outstanding_balance", "interest_rate", "age",
    "number_of_defaults", "remaining_term", "salary", "is_employed",
]

DRIFT_PRONE_TIME_FEATURES = ["loan_year", "loan_dayofyear"]

ENGINEERED_CATEGORICAL = ["age_bucket", "rate_segment"]
LOW_CARD_CATEGORICAL = ["gender", "job", "marital_status"] + ENGINEERED_CATEGORICAL
HIGH_CARD_CATEGORICAL = ["location"]

GROUP_AGG_SPECS = [
    (["job"], "salary"),
    (["job"], "loan_amount"),
    (["job"], "interest_rate"),
    (["location"], "loan_amount"),
    (["location"], "interest_rate"),
    (["marital_status"], "salary"),
]
GROUP_AGG_STATS = ["mean", "median", "std"]

MISSINGNESS_FIELDS = ["job", "location", "marital_status", "remaining_term"]

PROTECTED_ATTRIBUTES = ["gender"]

# Raw fields surfaced as adverse-action reasons / actionable recourse.
EXPLAINABLE_RAW = [
    "number_of_defaults", "outstanding_balance", "loan_amount", "interest_rate",
    "remaining term", "salary", "age", "is_employed", "job", "marital_status",
    "location",
]

FEATURE_DESCRIPTIONS = {
    "number_of_defaults": "Number of previous loan defaults",
    "outstanding_balance": "Outstanding balance owed",
    "loan_amount": "Requested / current loan amount",
    "interest_rate": "Interest rate on the loan",
    "remaining term": "Remaining term of the loan",
    "salary": "Declared salary / income",
    "age": "Applicant age",
    "is_employed": "Employment status",
    "job": "Occupation",
    "marital_status": "Marital status",
    "location": "Location",
}

# Raw schema the model consumes (what a request DTO must materialise).
RAW_INPUT_COLUMNS = [
    "gender", "is_employed", "job", "location", "marital_status",
    "loan_amount", "outstanding_balance", "interest_rate", "number_of_defaults",
    "remaining term", "salary", "age", "disbursemet_date",
]
