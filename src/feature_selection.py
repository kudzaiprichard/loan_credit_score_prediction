"""Reusable feature-selection helpers.

Used by notebook 03. Three complementary signals:
  * permutation importance (predictive value, model-agnostic) -> in src.explain
  * correlation pruning (drop redundant near-duplicate numerics)
  * ADVERSARIAL VALIDATION (drop features that drift -> overfit on real data)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

from . import config


def correlated_pairs(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Return numeric feature pairs with |corr| >= threshold (candidates to prune)."""
    num = [c for c in config.NUMERIC_FEATURES if c in df.columns]
    corr = df[num].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = [
        {"feature_a": a, "feature_b": b, "abs_corr": round(float(upper.loc[a, b]), 3)}
        for a in upper.index
        for b in upper.columns
        if pd.notna(upper.loc[a, b]) and upper.loc[a, b] >= threshold
    ]
    return pd.DataFrame(pairs).sort_values("abs_corr", ascending=False) if pairs else pd.DataFrame(
        columns=["feature_a", "feature_b", "abs_corr"]
    )


def select_by_importance(importance_df: pd.DataFrame, min_importance: float = 0.0) -> list:
    """Keep raw inputs whose permutation importance exceeds a floor."""
    keep = importance_df.loc[importance_df["importance"] > min_importance, "feature"]
    return keep.tolist()


# --------------------------------------------------------------------------- #
# Adversarial validation
# --------------------------------------------------------------------------- #
def adversarial_validation(
    X: pd.DataFrame,
    holdout_frac: float = 0.25,
    split: str = "temporal",
    exclude_time: bool = True,
    n_repeats: int = 5,
    max_samples: int = 8000,
) -> dict:
    """Detect features that DRIFT between the training period and 'real'/future data.

    Idea: relabel rows as 0 = "old/train" vs 1 = "new/holdout" and train a
    classifier to tell them apart. If it can (ROC-AUC well above 0.5) the two
    populations differ, and the features the classifier relies on are the ones
    causing covariate shift — they will overfit to the training period and hurt
    real-world performance. Those are pruning candidates.

    split='temporal' uses the disbursement date (the realistic case: you train on
    the past and deploy on the future). split='random' is a sanity check that
    should yield AUC ~0.5.

    `exclude_time=True` blanks the date so the model can't trivially recover the
    split from the calendar features it defines — we want to know whether the
    *other* features drift with time.
    """
    from .preprocessing import make_pipeline

    X = X.reset_index(drop=True).copy()

    # Build the adversarial label.
    if split == "temporal":
        dates = pd.to_datetime(X[config.RAW_DATE_COLUMN], format="%Y %m %d", errors="coerce")
        order = dates.sort_values(kind="mergesort").index
        n_hold = int(len(X) * holdout_frac)
        y_adv = pd.Series(0, index=X.index)
        y_adv.loc[order[-n_hold:]] = 1  # most recent rows = "real/future"
        y_adv = y_adv.to_numpy()
    else:
        rng = np.random.RandomState(config.RANDOM_STATE)
        y_adv = (rng.rand(len(X)) < holdout_frac).astype(int)

    X_adv = X.copy()
    if exclude_time and config.RAW_DATE_COLUMN in X_adv.columns:
        X_adv[config.RAW_DATE_COLUMN] = np.nan  # don't let the split variable leak

    pipe = make_pipeline(
        HistGradientBoostingClassifier(random_state=config.RANDOM_STATE)
    )

    auc = float(
        cross_val_score(pipe, X_adv, y_adv, scoring="roc_auc", cv=3, n_jobs=1).mean()
    )

    # Which raw inputs drive the shift?
    Xi, yi = X_adv, y_adv
    if len(Xi) > max_samples:
        idx = np.random.RandomState(config.RANDOM_STATE).choice(len(Xi), max_samples, replace=False)
        Xi, yi = Xi.iloc[idx], yi[idx]
    pipe.fit(Xi, yi)
    r = permutation_importance(
        pipe, Xi, yi, scoring="roc_auc", n_repeats=n_repeats,
        random_state=config.RANDOM_STATE, n_jobs=1,
    )
    drift = (
        pd.DataFrame({"feature": Xi.columns, "drift_importance": r.importances_mean})
        .sort_values("drift_importance", ascending=False)
        .reset_index(drop=True)
    )

    if auc < 0.55:
        verdict = "No meaningful drift (AUC~0.5): train and future look alike."
    elif auc < 0.7:
        verdict = "Mild drift: monitor the top features below; consider down-weighting."
    else:
        verdict = "Strong drift: prune / re-engineer the top features before trusting CV scores."

    return {
        "adversarial_auc": auc,
        "verdict": verdict,
        "drift_importance": drift,
        "prune_candidates": drift.loc[drift["drift_importance"] > 0.01, "feature"].tolist(),
    }
