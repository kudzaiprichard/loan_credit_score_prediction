"""Feature transforms — standalone copies of the training-time logic.

These exist so a model pickled under `src.*` can be unpickled using THIS package
(via inference.compat module aliases) with no dependency on the src/ tree. The
logic is identical to src/cleaning.py, src/feature_engineering.py and the
GroupStatsEncoder / selector in src/preprocessing.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config

# ---- cleaning ------------------------------------------------------------- #
_JOB_MAP = {
    "softwaredeveloper": "Software Developer", "software developer": "Software Developer",
    "data scintist": "Data Scientist", "data scientist": "Data Scientist",
    "data analyst": "Data Analyst", "teacher": "Teacher", "nurse": "Nurse",
    "doctor": "Doctor", "accountant": "Accountant", "lawyer": "Lawyer", "engineer": "Engineer",
}
_BOOL_MAP = {True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0,
             "yes": 1, "no": 0, "Y": 1, "N": 0, 1: 1, 0: 0}


def _canon_job(value):
    if pd.isna(value):
        return np.nan
    return _JOB_MAP.get(str(value).strip().lower(), str(value).strip())


def _canon_marital(value):
    if pd.isna(value):
        return np.nan
    v = str(value).strip()
    return np.nan if v == "" else v.lower()


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "remaining term" in df.columns:
        df = df.rename(columns={"remaining term": "remaining_term"})
    df = df.drop(columns=config.DROP_COLUMNS, errors="ignore")

    if "remaining_term" in df.columns:
        df["remaining_term"] = (
            df["remaining_term"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
            .replace("", np.nan)
        )
    for col in config.BASE_NUMERIC:
        if col != "is_employed" and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "job" in df.columns:
        df["job"] = df["job"].map(_canon_job)
    if "marital_status" in df.columns:
        df["marital_status"] = df["marital_status"].map(_canon_marital)
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype("object").str.strip().str.lower()
    if "location" in df.columns:
        df["location"] = df["location"].astype("object").str.strip()

    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype("float")
        elif col == "is_employed":
            df[col] = df[col].map(_BOOL_MAP).astype("float")

    if config.RAW_DATE_COLUMN in df.columns:
        parsed = pd.to_datetime(df[config.RAW_DATE_COLUMN], format="%Y %m %d", errors="coerce")
        df[config.DATE_COLUMN] = parsed
        df = df.drop(columns=[config.RAW_DATE_COLUMN])
    return df


# ---- feature engineering -------------------------------------------------- #
_STABLE_JOBS = {"teacher", "nurse", "doctor", "accountant", "lawyer", "engineer"}
_RISKY_JOBS = {"data analyst", "software developer", "data scientist"}
_JOB_RISK = {"teacher": 1, "nurse": 1, "doctor": 1, "accountant": 2, "lawyer": 2,
             "engineer": 2, "data scientist": 2, "data analyst": 3, "software developer": 3}
_MARITAL_RISK = {"married": 1, "single": 2, "divorced": 3, "widowed": 3}
_URBAN = {"harare", "bulawayo"}


def _safe_div(a, b, clip=None):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = (a / b.where(b != 0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return out.clip(upper=clip) if clip is not None else out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    miss_cols = [c for c in config.MISSINGNESS_FIELDS if c in df.columns]
    for col in miss_cols:
        df[f"{col}_missing"] = df[col].isna().astype("float")
    df["total_missing_fields"] = df[miss_cols].isna().sum(axis=1).astype("float") if miss_cols else 0.0

    loan = pd.to_numeric(df.get("loan_amount"), errors="coerce")
    bal = pd.to_numeric(df.get("outstanding_balance"), errors="coerce")
    sal = pd.to_numeric(df.get("salary"), errors="coerce")
    rate = pd.to_numeric(df.get("interest_rate"), errors="coerce")
    term = pd.to_numeric(df.get("remaining_term"), errors="coerce")
    ndef = pd.to_numeric(df.get("number_of_defaults"), errors="coerce")
    age = pd.to_numeric(df.get("age"), errors="coerce")

    mrate = rate / 12.0
    df["monthly_rate"] = mrate
    n = term.clip(lower=1)
    growth = np.power(1 + mrate, n)
    pay_amortized = loan * mrate * growth / (growth - 1)
    df["estimated_monthly_payment"] = np.where(mrate.fillna(0) == 0, loan / n, pay_amortized)
    df["total_repayment_est"] = df["estimated_monthly_payment"] * term
    df["interest_cost_est"] = df["total_repayment_est"] - loan
    df["interest_cost_ratio"] = _safe_div(df["interest_cost_est"], loan.clip(lower=1))

    df["dti"] = _safe_div(df["estimated_monthly_payment"], sal.clip(lower=1), clip=20)
    df["loan_to_income"] = _safe_div(loan, sal.clip(lower=1), clip=200)
    df["balance_to_income"] = _safe_div(bal, sal.clip(lower=1), clip=200)
    df["balance_to_loan"] = _safe_div(bal, loan)
    df["balance_minus_loan"] = bal - loan
    df["interest_burden"] = rate * bal

    df["has_prior_default"] = (ndef > 0).astype("float")
    df["defaults_per_decade"] = _safe_div(ndef, (age / 10).clip(lower=0.1))

    job_l = df["job"].astype("object").str.lower() if "job" in df.columns else pd.Series(index=df.index, dtype="object")
    loc_l = df["location"].astype("object").str.lower() if "location" in df.columns else pd.Series(index=df.index, dtype="object")
    mar_l = df["marital_status"].astype("object").str.lower() if "marital_status" in df.columns else pd.Series(index=df.index, dtype="object")

    df["is_high_interest"] = (rate > 0.20).astype("float")
    df["is_urban"] = loc_l.isin(_URBAN).astype("float")
    df["is_stable_job"] = job_l.isin(_STABLE_JOBS).astype("float")
    df["is_risky_job"] = job_l.isin(_RISKY_JOBS).astype("float")
    df["job_risk_score"] = job_l.map(_JOB_RISK).fillna(3).astype("float")
    df["marital_risk_score"] = mar_l.map(_MARITAL_RISK).fillna(2).astype("float")

    df["amount_x_rate"] = loan * rate
    df["rate_x_term"] = rate * term
    df["dti_x_defaults"] = df["dti"] * (ndef + 1)
    df["lti_x_defaults"] = df["loan_to_income"] * (ndef + 1)

    df["log_loan_amount"] = np.log1p(loan.clip(lower=0))
    df["log_salary"] = np.log1p(sal.clip(lower=0))
    df["log_outstanding_balance"] = np.log1p(bal.clip(lower=0))
    df["log_estimated_monthly_payment"] = np.log1p(df["estimated_monthly_payment"].clip(lower=0))
    df["log_interest_cost_est"] = np.log1p(df["interest_cost_est"].clip(lower=0))
    df["log_loan_to_income"] = np.log1p(df["loan_to_income"].clip(lower=0))

    df["age_bucket"] = pd.cut(age, bins=[18, 25, 35, 45, 55, 65, 120],
                              labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]).astype("object")
    df["rate_segment"] = pd.cut(rate, bins=[-0.01, 0.15, 0.20, 0.25, 1.0],
                                labels=["low", "medium", "high", "very_high"]).astype("object")

    dt = (pd.to_datetime(df[config.DATE_COLUMN], errors="coerce")
          if config.DATE_COLUMN in df.columns else pd.Series(pd.NaT, index=df.index))
    iso = dt.dt.isocalendar()
    df["loan_year"] = dt.dt.year
    df["loan_dayofyear"] = dt.dt.dayofyear
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


# ---- learned group aggregates + numeric selector -------------------------- #
class GroupStatsEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, specs=None, stats=None):
        self.specs = specs if specs is not None else config.GROUP_AGG_SPECS
        self.stats = stats if stats is not None else config.GROUP_AGG_STATS

    def fit(self, X, y=None):
        self.maps_ = {}
        self.global_ = {}
        for keys, vcol in self.specs:
            if vcol not in X.columns or not all(k in X.columns for k in keys):
                continue
            filled = X.assign(**{k: X[k].fillna("__NA__") for k in keys})
            grouped = filled.groupby(keys, observed=True)[vcol]
            for stat in self.stats:
                self.maps_[(tuple(keys), vcol, stat)] = grouped.agg(stat)
            self.global_[vcol] = {"mean": float(X[vcol].mean()),
                                  "median": float(X[vcol].median()),
                                  "std": float(X[vcol].std())}
        return self

    def transform(self, X):
        X = X.copy()
        for keys, vcol in self.specs:
            if (tuple(keys), vcol, self.stats[0]) not in self.maps_:
                continue
            key_filled = X[keys].copy()
            for k in keys:
                key_filled[k] = key_filled[k].fillna("__NA__")
            prefix = "_".join(keys)
            for stat in self.stats:
                gm = self.maps_[(tuple(keys), vcol, stat)]
                if len(keys) == 1:
                    vals = key_filled[keys[0]].map(gm)
                else:
                    vals = key_filled.merge(gm.rename("v").reset_index(), on=keys, how="left")["v"]
                vals = pd.to_numeric(vals, errors="coerce").fillna(self.global_[vcol][stat])
                X[f"{prefix}__{vcol}__{stat}"] = vals.to_numpy()
            X[f"{prefix}__{vcol}__dev"] = (
                pd.to_numeric(X[vcol], errors="coerce").to_numpy()
                - X[f"{prefix}__{vcol}__mean"].to_numpy()
            )
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


def select_numeric_columns(df: pd.DataFrame):
    deny = set(config.DRIFT_PRONE_TIME_FEATURES)
    cat = set(config.LOW_CARD_CATEGORICAL) | set(config.HIGH_CARD_CATEGORICAL)
    return [c for c in df.columns
            if c not in deny and c not in cat and pd.api.types.is_numeric_dtype(df[c])]
