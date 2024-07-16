"""Rich, row-wise feature engineering (the proven-style base, adapted to our schema).

ORDER MATTERS (and is enforced by the pipeline):
  cleaning/format-standardisation  ->  THIS (missingness + engineered features)
  ->  learned group aggregates  ->  imputation  ->  scaling/encoding  ->  model

Everything here is purely row-wise and deterministic, so it runs identically at
train and inference time. Features fall into families:
  * amortization / cost-of-credit (monthly payment, total repayment, interest cost)
  * affordability / indebtedness (DTI, loan-to-income, balance ratios)
  * credit history, domain flags & hand-coded risk scores
  * interaction (cross) terms
  * log transforms for heavy tails
  * calendar + cyclical date features
  * MISSINGNESS: per-field flags + a total-missing-fields count, captured BEFORE
    imputation so a model can learn whether *not disclosing* a field predicts default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Domain knowledge (hand-coded, like a credit analyst would).
_STABLE_JOBS = {"teacher", "nurse", "doctor", "accountant", "lawyer", "engineer"}
_RISKY_JOBS = {"data analyst", "software developer", "data scientist"}
_JOB_RISK = {
    "teacher": 1, "nurse": 1, "doctor": 1, "accountant": 2, "lawyer": 2,
    "engineer": 2, "data scientist": 2, "data analyst": 3, "software developer": 3,
}
_MARITAL_RISK = {"married": 1, "single": 2, "divorced": 3, "widowed": 3}
_URBAN = {"harare", "bulawayo"}


def _safe_div(a, b, clip=None):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = (a / b.where(b != 0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return out.clip(upper=clip) if clip is not None else out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ----------------------------------------------------------------- #
    # 1) MISSINGNESS — recorded BEFORE any imputation downstream.
    # ----------------------------------------------------------------- #
    miss_cols = [c for c in config.MISSINGNESS_FIELDS if c in df.columns]
    for col in miss_cols:
        df[f"{col}_missing"] = df[col].isna().astype("float")
    # how many of the meaningful fields are blank for this applicant
    df["total_missing_fields"] = df[miss_cols].isna().sum(axis=1).astype("float") if miss_cols else 0.0

    loan = pd.to_numeric(df.get("loan_amount"), errors="coerce")
    bal = pd.to_numeric(df.get("outstanding_balance"), errors="coerce")
    sal = pd.to_numeric(df.get("salary"), errors="coerce")
    rate = pd.to_numeric(df.get("interest_rate"), errors="coerce")
    term = pd.to_numeric(df.get("remaining_term"), errors="coerce")
    ndef = pd.to_numeric(df.get("number_of_defaults"), errors="coerce")
    age = pd.to_numeric(df.get("age"), errors="coerce")

    # ----------------------------------------------------------------- #
    # 2) Amortization / cost of credit
    #    (interest_rate treated as an annual rate; salary as monthly income)
    # ----------------------------------------------------------------- #
    mrate = rate / 12.0
    df["monthly_rate"] = mrate
    n = term.clip(lower=1)
    growth = np.power(1 + mrate, n)
    pay_amortized = loan * mrate * growth / (growth - 1)
    df["estimated_monthly_payment"] = np.where(mrate.fillna(0) == 0, loan / n, pay_amortized)
    df["total_repayment_est"] = df["estimated_monthly_payment"] * term
    df["interest_cost_est"] = df["total_repayment_est"] - loan
    df["interest_cost_ratio"] = _safe_div(df["interest_cost_est"], loan.clip(lower=1))

    # ----------------------------------------------------------------- #
    # 3) Affordability / indebtedness
    # ----------------------------------------------------------------- #
    df["dti"] = _safe_div(df["estimated_monthly_payment"], sal.clip(lower=1), clip=20)
    df["loan_to_income"] = _safe_div(loan, sal.clip(lower=1), clip=200)
    df["balance_to_income"] = _safe_div(bal, sal.clip(lower=1), clip=200)
    df["balance_to_loan"] = _safe_div(bal, loan)
    df["balance_minus_loan"] = bal - loan
    df["interest_burden"] = rate * bal

    # ----------------------------------------------------------------- #
    # 4) Credit history
    # ----------------------------------------------------------------- #
    df["has_prior_default"] = (ndef > 0).astype("float")
    df["defaults_per_decade"] = _safe_div(ndef, (age / 10).clip(lower=0.1))

    # ----------------------------------------------------------------- #
    # 5) Domain flags & hand-coded risk scores
    # ----------------------------------------------------------------- #
    job_l = df["job"].astype("object").str.lower() if "job" in df.columns else pd.Series(index=df.index, dtype="object")
    loc_l = df["location"].astype("object").str.lower() if "location" in df.columns else pd.Series(index=df.index, dtype="object")
    mar_l = df["marital_status"].astype("object").str.lower() if "marital_status" in df.columns else pd.Series(index=df.index, dtype="object")

    df["is_high_interest"] = (rate > 0.20).astype("float")
    df["is_urban"] = loc_l.isin(_URBAN).astype("float")
    df["is_stable_job"] = job_l.isin(_STABLE_JOBS).astype("float")
    df["is_risky_job"] = job_l.isin(_RISKY_JOBS).astype("float")
    df["job_risk_score"] = job_l.map(_JOB_RISK).fillna(3).astype("float")
    df["marital_risk_score"] = mar_l.map(_MARITAL_RISK).fillna(2).astype("float")

    # ----------------------------------------------------------------- #
    # 6) Interaction (cross) features
    # ----------------------------------------------------------------- #
    df["amount_x_rate"] = loan * rate
    df["rate_x_term"] = rate * term
    df["dti_x_defaults"] = df["dti"] * (ndef + 1)
    df["lti_x_defaults"] = df["loan_to_income"] * (ndef + 1)

    # ----------------------------------------------------------------- #
    # 7) Log transforms (smooth heavy tails)
    # ----------------------------------------------------------------- #
    df["log_loan_amount"] = np.log1p(loan.clip(lower=0))
    df["log_salary"] = np.log1p(sal.clip(lower=0))
    df["log_outstanding_balance"] = np.log1p(bal.clip(lower=0))
    df["log_estimated_monthly_payment"] = np.log1p(df["estimated_monthly_payment"].clip(lower=0))
    df["log_interest_cost_est"] = np.log1p(df["interest_cost_est"].clip(lower=0))
    df["log_loan_to_income"] = np.log1p(df["loan_to_income"].clip(lower=0))

    # ----------------------------------------------------------------- #
    # 8) Categorical buckets (one-hot encoded downstream)
    # ----------------------------------------------------------------- #
    df["age_bucket"] = pd.cut(
        age, bins=[18, 25, 35, 45, 55, 65, 120],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
    ).astype("object")
    df["rate_segment"] = pd.cut(
        rate, bins=[-0.01, 0.15, 0.20, 0.25, 1.0],
        labels=["low", "medium", "high", "very_high"],
    ).astype("object")

    # ----------------------------------------------------------------- #
    # 9) Calendar + cyclical date features
    # ----------------------------------------------------------------- #
    dt = (
        pd.to_datetime(df[config.DATE_COLUMN], errors="coerce")
        if config.DATE_COLUMN in df.columns
        else pd.Series(pd.NaT, index=df.index)
    )
    iso = dt.dt.isocalendar()
    df["loan_year"] = dt.dt.year                  # drift-prone: kept for analysis only
    df["loan_dayofyear"] = dt.dt.dayofyear        # drift-prone: ditto
    df["loan_month"] = dt.dt.month
    df["loan_quarter"] = dt.dt.quarter
    df["loan_week"] = pd.to_numeric(iso["week"], errors="coerce")
    df["loan_dayofweek"] = dt.dt.dayofweek
    df["loan_is_month_end"] = dt.dt.is_month_end.astype("float")
    df["loan_is_quarter_end"] = dt.dt.is_quarter_end.astype("float")
    df["month_sin"] = np.sin(2 * np.pi * df["loan_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["loan_month"] / 12)
    df["quarter_sin"] = np.sin(2 * np.pi * df["loan_quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["loan_quarter"] / 4)
    df["dow_sin"] = np.sin(2 * np.pi * df["loan_dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["loan_dayofweek"] / 7)
    if config.DATE_COLUMN in df.columns:
        df = df.drop(columns=[config.DATE_COLUMN])

    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the modelling columns present (used to coerce raw inference input)."""
    cols = [c for c in config.ALL_FEATURES if c in df.columns]
    return df[cols]
