"""Explainability: global drivers + per-applicant adverse-action reasons.

Why not SHAP? In this environment SHAP is broken (numba requires NumPy <=2.3,
the env has 2.4). So we use two robust, dependency-light, model-agnostic tools:

  * GLOBAL: sklearn permutation importance — how much ROC-AUC drops when each
    raw input is shuffled.
  * LOCAL (the compliance piece): occlusion / counterfactual reasoning. For a
    declined applicant we replace each input with the value of a *typical
    approved applicant* and measure how much the predicted default probability
    falls. The inputs that most increased the applicant's risk become the
    ranked, plain-English reasons for the decision (ECOA-style reason codes).

Protected attributes (e.g. gender) are explicitly excluded from reason codes so
the explanation can never cite a prohibited basis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import config

# Raw input columns we are willing to surface as decision reasons.
EXPLAINABLE_RAW = [
    "number_of_defaults",
    "outstanding_balance",
    "loan_amount",
    "interest_rate",
    "remaining term",   # raw schema name (note the space)
    "salary",
    "age",
    "is_employed",
    "job",
    "marital_status",
    "location",
]


def compute_reference_profile(raw_df: pd.DataFrame, y) -> dict:
    """A 'typical approved applicant' profile from the non-default training rows.

    numeric -> median, categorical -> mode. Stored in model metadata and used as
    the counterfactual baseline for reason codes.
    """
    approved = raw_df[np.asarray(y) == 0]
    ref = {}
    for col in EXPLAINABLE_RAW:
        if col not in approved.columns:
            continue
        s = approved[col].dropna()
        if s.empty:
            continue
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            ref[col] = float(pd.to_numeric(s, errors="coerce").median())
        else:
            ref[col] = s.mode().iloc[0]
    return ref


def global_importance(pipeline, X_raw, y, n_repeats=5, max_samples=5000, scoring="roc_auc"):
    """Permutation importance over raw inputs (subsampled for speed)."""
    X = X_raw
    yy = np.asarray(y)
    if len(X) > max_samples:
        idx = np.random.RandomState(config.RANDOM_STATE).choice(len(X), max_samples, replace=False)
        X = X.iloc[idx]
        yy = yy[idx]
    r = permutation_importance(
        pipeline, X, yy, scoring=scoring, n_repeats=n_repeats,
        random_state=config.RANDOM_STATE, n_jobs=1,
    )
    out = (
        pd.DataFrame({"feature": X.columns, "importance": r.importances_mean,
                      "std": r.importances_std})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return out


def local_reason_codes(pipeline, raw_row: pd.DataFrame, reference: dict,
                       top_k: int = 4, exclude=None) -> dict:
    """Adverse-action reasons for a single applicant.

    `raw_row` is a 1-row DataFrame in the raw schema. Returns the predicted
    probability and the top features that pushed the applicant toward default.
    """
    exclude = set(exclude or config.PROTECTED_ATTRIBUTES)
    raw_row = raw_row.iloc[[0]].copy()

    base_p = float(pipeline.predict_proba(raw_row)[0, 1])

    contributions = []
    for col, ref_val in reference.items():
        if col in exclude or col not in raw_row.columns:
            continue
        current = raw_row.iloc[0][col]
        # skip if the applicant already matches the reference (no contribution)
        try:
            if pd.notna(current) and current == ref_val:
                continue
        except Exception:
            pass
        cf = raw_row.copy()
        cf.iloc[0, cf.columns.get_loc(col)] = ref_val
        p_cf = float(pipeline.predict_proba(cf)[0, 1])
        delta = base_p - p_cf  # >0 => this input raised the applicant's risk
        if delta > 1e-4:
            contributions.append(
                {
                    "feature": col,
                    "description": config.FEATURE_DESCRIPTIONS.get(col, col),
                    "applicant_value": _clean_val(current),
                    "typical_approved_value": _clean_val(ref_val),
                    "risk_contribution": round(delta, 4),
                }
            )

    contributions.sort(key=lambda d: d["risk_contribution"], reverse=True)
    return {
        "default_probability": round(base_p, 4),
        "top_reasons": contributions[:top_k],
    }


def _clean_val(v):
    if isinstance(v, (np.floating, float)):
        return round(float(v), 2)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v
