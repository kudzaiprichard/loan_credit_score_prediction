"""Customer segmentation -> borrower personas (idea 2).

Unsupervised clustering on the engineered numeric features, then each cluster is
profiled by size, default rate AND profitability — so segmentation drives business
action ("grow segment C: low risk + high margin; tighten segment E"), not just
pretty scatter plots.

Uses scikit-learn KMeans (always available). HDBSCAN is used automatically if
installed (density-based, finds irregular clusters + noise).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .cleaning import clean_features
from .feature_engineering import add_features


def _numeric_matrix(raw: pd.DataFrame):
    feats = add_features(clean_features(raw))
    cols = [c for c in config.NUMERIC_FEATURES if c in feats.columns]
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())])
    X = pipe.fit_transform(feats[cols])
    return X, cols, pipe


def choose_k(raw: pd.DataFrame, k_range=range(2, 9), sample=8000) -> pd.DataFrame:
    """Silhouette score per k to pick the number of segments."""
    X, _, _ = _numeric_matrix(raw)
    if len(X) > sample:
        idx = np.random.RandomState(config.RANDOM_STATE).choice(len(X), sample, replace=False)
        X = X[idx]
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(X)
        rows.append({"k": k, "silhouette": float(silhouette_score(X, km.labels_))})
    return pd.DataFrame(rows)


def fit_segments(raw: pd.DataFrame, k: int = 4, method: str = "kmeans"):
    """Fit clusters; return (labels, fitted_estimator, preprocessing_pipe, columns)."""
    X, cols, pipe = _numeric_matrix(raw)
    if method == "hdbscan":
        try:
            import hdbscan
            model = hdbscan.HDBSCAN(min_cluster_size=max(50, len(X) // 100))
            labels = model.fit_predict(X)
            return labels, model, pipe, cols
        except Exception:
            pass  # fall back to kmeans
    model = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(X)
    return model.labels_, model, pipe, cols


def profile_segments(raw: pd.DataFrame, labels, y=None) -> pd.DataFrame:
    """Per-segment business profile: size, financials, default rate, est. margin."""
    df = raw.copy().reset_index(drop=True)
    df["segment"] = labels
    if y is not None:
        df["_default"] = np.asarray(y)
    rev = pd.to_numeric(df["loan_amount"], errors="coerce") * pd.to_numeric(df["interest_rate"], errors="coerce")
    df["_revenue_if_repaid"] = rev

    agg = {
        "loan_amount": "mean",
        "salary": "mean",
        "interest_rate": "mean",
        "number_of_defaults": "mean",
        "_revenue_if_repaid": "mean",
    }
    if "_default" in df.columns:
        agg["_default"] = "mean"
    out = df.groupby("segment").agg(agg)
    out["size"] = df.groupby("segment").size()
    out = out.rename(columns={"_default": "actual_default_rate",
                              "_revenue_if_repaid": "avg_revenue_if_repaid"})
    return out.sort_values("size", ascending=False).round(3)


def pca_2d(raw: pd.DataFrame):
    """2-D PCA projection for plotting segments (no UMAP dependency)."""
    X, _, _ = _numeric_matrix(raw)
    return PCA(n_components=2, random_state=config.RANDOM_STATE).fit_transform(X)
