"""Unsupervised anomaly / fraud detection (idea 4).

Orthogonal to default prediction: instead of "will this loan go bad?", we ask
"does this application look *abnormal*?" — impossible salary/loan combinations,
data-entry errors, or potential fraud. Useful as a second gate before approval.

Uses scikit-learn IsolationForest (+ LocalOutlierFactor). If `pyod` is installed
it is exposed too (40+ detectors), but it is optional.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .cleaning import clean_features
from .feature_engineering import add_features


def _matrix(raw: pd.DataFrame):
    feats = add_features(clean_features(raw))
    cols = [c for c in config.NUMERIC_FEATURES if c in feats.columns]
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())])
    return pipe.fit_transform(feats[cols]), cols


def detect(raw: pd.DataFrame, contamination: float = 0.02) -> pd.DataFrame:
    """Flag the most anomalous applications.

    Returns the input with `anomaly_score` (higher = more anomalous) and an
    `is_anomaly` flag for the top `contamination` fraction.
    """
    X, _ = _matrix(raw)
    iso = IsolationForest(
        n_estimators=300, contamination=contamination,
        random_state=config.RANDOM_STATE, n_jobs=-1,
    ).fit(X)
    # IsolationForest: higher decision_function = more normal -> negate for score
    score = -iso.decision_function(X)
    flag = iso.predict(X) == -1
    out = raw.copy().reset_index(drop=True)
    out["anomaly_score"] = np.round(score, 4)
    out["is_anomaly"] = flag
    return out.sort_values("anomaly_score", ascending=False)


def anomaly_summary(flagged: pd.DataFrame) -> pd.DataFrame:
    """Compare flagged vs normal on key fields to explain *why* they look odd."""
    cols = ["loan_amount", "salary", "outstanding_balance", "interest_rate",
            "number_of_defaults", "age"]
    cols = [c for c in cols if c in flagged.columns]
    grp = flagged.groupby("is_anomaly")[cols].mean().round(2)
    grp.index = grp.index.map({True: "anomalous", False: "normal"})
    return grp
