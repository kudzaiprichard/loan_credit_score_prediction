"""Learned preprocessing + full Pipeline assembly.

Pipeline order (each learned step is fit on the TRAINING fold only):
    clean_features        (deterministic: dedup, standardise formats, parse dates)
 -> add_features          (deterministic: missingness, ratios, logs, cyclical, scores)
 -> GroupStatsEncoder     (LEARNED: peer/group mean·median·std + deviations)
 -> ColumnTransformer     (LEARNED: median-impute + scale; Unknown-impute + encode)
 -> estimator

Encoding: One-Hot for low-cardinality nominals (no fake ordering); TargetEncoder
(cross-fitted, leakage-safe) for the 157-value `location`. The numeric branch uses
a dynamic selector so newly engineered/group features are scaled automatically,
while drift-prone absolute-time features are excluded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
    TargetEncoder,
)

from . import config
from .cleaning import clean_features
from .feature_engineering import add_features


# --------------------------------------------------------------------------- #
# Learned group-aggregate / peer-residual encoder
# --------------------------------------------------------------------------- #
class GroupStatsEncoder(BaseEstimator, TransformerMixin):
    """Fit group→stat maps on training data; apply (with a global fallback) at
    transform. This is the leakage-safe, single-row-friendly version of the
    Kaggle concat(train,test) group-aggregate trick.
    """

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
            self.global_[vcol] = {
                "mean": float(X[vcol].mean()),
                "median": float(X[vcol].median()),
                "std": float(X[vcol].std()),
            }
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
                    vals = key_filled.merge(
                        gm.rename("v").reset_index(), on=keys, how="left"
                    )["v"]
                vals = pd.to_numeric(vals, errors="coerce").fillna(self.global_[vcol][stat])
                X[f"{prefix}__{vcol}__{stat}"] = vals.to_numpy()
            # peer residual: how far this applicant is from their group's mean
            X[f"{prefix}__{vcol}__dev"] = (
                pd.to_numeric(X[vcol], errors="coerce").to_numpy()
                - X[f"{prefix}__{vcol}__mean"].to_numpy()
            )
        return X

    def get_feature_names_out(self, input_features=None):  # pragma: no cover
        return input_features


# --------------------------------------------------------------------------- #
# Column selection
# --------------------------------------------------------------------------- #
def select_numeric_columns(df: pd.DataFrame):
    """All numeric columns EXCEPT drift-prone absolute-time features.

    A callable selector means group-aggregate columns (added at fit time) and any
    new engineered numeric are scaled automatically — no config edits needed.
    """
    deny = set(config.DRIFT_PRONE_TIME_FEATURES)
    cat = set(config.LOW_CARD_CATEGORICAL) | set(config.HIGH_CARD_CATEGORICAL)
    return [
        c for c in df.columns
        if c not in deny and c not in cat and pd.api.types.is_numeric_dtype(df[c])
    ]


def build_preprocessor(numeric_impute: str = "median") -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy=numeric_impute)),
        ("scale", StandardScaler()),
    ])
    low_card_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    high_card_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("target_enc", TargetEncoder(random_state=config.RANDOM_STATE)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, select_numeric_columns),
            ("low_cat", low_card_pipe, config.LOW_CARD_CATEGORICAL),
            ("high_cat", high_card_pipe, config.HIGH_CARD_CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_feature_steps():
    """Deterministic clean + engineer steps shared by every model."""
    return [
        ("clean", FunctionTransformer(clean_features, validate=False)),
        ("engineer", FunctionTransformer(add_features, validate=False)),
    ]


def make_pipeline(estimator, numeric_impute: str = "median", group_stats: bool = True) -> Pipeline:
    """Full raw-input -> prediction pipeline for an estimator."""
    steps = make_feature_steps()
    if group_stats:
        steps.append(("group_stats", GroupStatsEncoder()))
    steps.append(("preprocess", build_preprocessor(numeric_impute)))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)
