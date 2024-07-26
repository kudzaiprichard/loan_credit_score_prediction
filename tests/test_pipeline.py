"""Fast sanity tests — run with `pytest -q`.

These guard the contract that makes the project trustworthy: cleaning is
deterministic, the pipeline trains on raw rows, and a fitted pipeline produces
sane probabilities + reason codes from raw input.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import config, explain
from src.cleaning import clean_features, prepare_target
from src.data_loader import load_raw
from src.feature_engineering import add_features
from src.preprocessing import make_pipeline


def _sample(n=3000):
    raw = load_raw()
    raw = prepare_target(raw)
    return raw.sample(n, random_state=0)


def test_cleaning_drops_dupes_and_ids():
    raw = load_raw().head(200)
    cleaned = clean_features(raw)
    for col in ["Unnamed: 0", "loan_id", "sex", "number_of_defaults.1", "age.1", "currency", "country"]:
        assert col not in cleaned.columns
    assert "remaining_term" in cleaned.columns


def test_feature_engineering_adds_ratios():
    raw = clean_features(load_raw().head(200))
    fe = add_features(raw)
    for col in config.ENGINEERED_NUMERIC:
        assert col in fe.columns
    assert not np.isinf(fe["balance_to_loan"].dropna()).any()


def test_pipeline_trains_and_predicts_from_raw():
    df = _sample()
    y = df[config.TARGET].to_numpy()
    X = df.drop(columns=[config.TARGET])
    pipe = make_pipeline(LogisticRegression(max_iter=500, class_weight="balanced"))
    pipe.fit(X, y)
    proba = pipe.predict_proba(X.head(50))[:, 1]
    assert proba.shape == (50,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_reason_codes_exclude_protected():
    df = _sample()
    y = df[config.TARGET].to_numpy()
    X = df.drop(columns=[config.TARGET])
    pipe = make_pipeline(LogisticRegression(max_iter=500, class_weight="balanced"))
    pipe.fit(X, y)
    ref = explain.compute_reference_profile(X, y)
    out = explain.local_reason_codes(pipe, X.head(1), ref, top_k=4)
    assert 0 <= out["default_probability"] <= 1
    assert all(r["feature"] not in config.PROTECTED_ATTRIBUTES for r in out["top_reasons"])
