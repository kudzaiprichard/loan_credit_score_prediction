"""Deterministic data cleaning.

IMPORTANT design rule for leakage safety:
  * `clean_features` does only ROW-WISE deterministic transforms (no statistics
    learned from data). It is safe to run before the train/test split and is
    embedded inside the model Pipeline so inference re-uses the exact same logic.
  * Anything that LEARNS from data (imputation values, scalers, encoders) lives
    in `preprocessing.py` and is fit on the training fold only.
  * Target creation and row dropping (`prepare_target`) are training-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Known dirty -> canonical mappings discovered during EDA.
_JOB_MAP = {
    "softwaredeveloper": "Software Developer",
    "software developer": "Software Developer",
    "data scintist": "Data Scientist",   # typo in source
    "data scientist": "Data Scientist",
    "data analyst": "Data Analyst",
    "teacher": "Teacher",
    "nurse": "Nurse",
    "doctor": "Doctor",
    "accountant": "Accountant",
    "lawyer": "Lawyer",
    "engineer": "Engineer",
}


def _canon_job(value):
    if pd.isna(value):
        return np.nan
    key = str(value).strip().lower()
    return _JOB_MAP.get(key, str(value).strip())


def _canon_marital(value):
    if pd.isna(value):
        return np.nan
    v = str(value).strip()
    return np.nan if v == "" else v.lower()


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature-only cleaning used at BOTH train and inference time.

    - drops id / duplicate / constant columns
    - canonicalises dirty categorical text (currency/country are dropped as
      constant so we only need job + marital here)
    - parses the (mis-spelt) disbursement date column
    - normalises dtypes / column names

    Never touches the target and never drops rows.
    """
    df = df.copy()

    # Normalise the column with an embedded space so it is a valid identifier.
    if "remaining term" in df.columns:
        df = df.rename(columns={"remaining term": "remaining_term"})

    # Drop id / duplicate / constant columns (tolerant: ignore if absent).
    df = df.drop(columns=config.DROP_COLUMNS, errors="ignore")

    # `remaining_term` is stored as text with stray junk e.g. '69_' -> 69.
    # Strip everything but digits/decimal/sign so we recover the real value
    # instead of throwing the row away.
    if "remaining_term" in df.columns:
        df["remaining_term"] = (
            df["remaining_term"].astype(str)
            .str.replace(r"[^0-9.\-]", "", regex=True)
            .replace("", np.nan)
        )

    # Defensively coerce numeric columns: any remaining non-numeric junk -> NaN,
    # which the median imputer in the pipeline will fill.
    for col in config.BASE_NUMERIC:
        if col == "is_employed":
            continue
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Canonicalise messy categoricals.
    if "job" in df.columns:
        df["job"] = df["job"].map(_canon_job)
    if "marital_status" in df.columns:
        df["marital_status"] = df["marital_status"].map(_canon_marital)
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype("object").str.strip().str.lower()
    if "location" in df.columns:
        df["location"] = df["location"].astype("object").str.strip()

    # Standardise ALL boolean-like columns to numeric 0/1 (consistent format).
    _bool_map = {True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0,
                 "yes": 1, "no": 0, "Y": 1, "N": 0, 1: 1, 0: 0}
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype("float")
        elif col == "is_employed":
            df[col] = df[col].map(_bool_map).astype("float")

    # Parse the date (format e.g. '2022 10 29'); derive year/month, drop raw.
    date_src = config.RAW_DATE_COLUMN
    if date_src in df.columns:
        parsed = pd.to_datetime(df[date_src], format="%Y %m %d", errors="coerce")
        df[config.DATE_COLUMN] = parsed
        df = df.drop(columns=[date_src])

    return df


# Canonical target normalisation: strip, lowercase, collapse whitespace, then map.
# We pin the positive class explicitly (default = 1) rather than using a blind
# LabelEncoder, which would assign labels alphabetically and could silently flip
# which class counts as "default".
_TARGET_MAP = {
    "defaulted": 1, "default": 1, "defaults": 1, "yes": 1, "1": 1,
    "did not default": 0, "did not defaulted": 0, "no default": 0,
    "not defaulted": 0, "no": 0, "0": 0,
}


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """Training-only: normalise the label text, build the binary target, drop
    unusable rows.

    Rows with an unmappable / missing target are dropped — we cannot learn or
    score from them.
    """
    df = df.copy()
    norm = (
        df[config.TARGET_RAW].astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    df[config.TARGET] = norm.map(_TARGET_MAP)

    unknown = norm[df[config.TARGET].isna()].unique()
    if len(unknown):
        print(f"[cleaning] unmapped target values dropped: {list(unknown)[:10]}")
    before = len(df)
    df = df.dropna(subset=[config.TARGET])
    dropped = before - len(df)
    if dropped:
        print(f"[cleaning] dropped {dropped} rows with missing/unknown target")
    df[config.TARGET] = df[config.TARGET].astype(int)
    return df.drop(columns=[config.TARGET_RAW])


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing count + pct, sorted — used in EDA to justify the
    impute-vs-drop decision."""
    n = len(df)
    miss = df.isna().sum()
    out = (
        pd.DataFrame({"missing": miss, "pct": (miss / n * 100).round(2)})
        .query("missing > 0")
        .sort_values("missing", ascending=False)
    )
    return out
