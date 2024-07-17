"""Fairness auditing across protected attributes (e.g. gender).

Lending models must not discriminate. We measure, per group:
  * selection (approval/decline) rate            -> demographic parity
  * true-positive rate (recall on defaulters)    -> equal opportunity
  * false-positive rate (good payers flagged)    -> potential harm
and report the max-min gaps. Large gaps are a red flag to investigate, not an
automatic pass/fail — context and legal review matter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def group_metrics(y_true, y_pred, groups) -> pd.DataFrame:
    """Per-group rates. `groups` is an array-like of the protected attribute."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = pd.Series(np.asarray(groups)).reset_index(drop=True)

    rows = []
    for g in sorted(groups.dropna().unique()):
        m = (groups == g).to_numpy()
        yt, yp = y_true[m], y_pred[m]
        if len(yt) == 0:
            continue
        tn, fp, fn, tp = _safe_cm(yt, yp)
        rows.append(
            {
                "group": g,
                "n": int(len(yt)),
                "actual_default_rate": float(yt.mean()),
                "predicted_default_rate": float(yp.mean()),  # selection rate (flagged risky)
                "tpr_recall": tp / (tp + fn) if (tp + fn) else np.nan,
                "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
                "precision": tp / (tp + fp) if (tp + fp) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _safe_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp


def fairness_gaps(group_df: pd.DataFrame) -> dict:
    """Max-min disparities across groups (0 = perfectly equal)."""
    def gap(col):
        vals = group_df[col].dropna()
        return float(vals.max() - vals.min()) if len(vals) else np.nan

    return {
        "demographic_parity_diff": gap("predicted_default_rate"),
        "equal_opportunity_diff": gap("tpr_recall"),
        "fpr_diff": gap("fpr"),
    }


def audit(y_true, y_pred, protected_frame: pd.DataFrame) -> dict:
    """Run the audit for every protected attribute column provided."""
    report = {}
    for attr in protected_frame.columns:
        gdf = group_metrics(y_true, y_pred, protected_frame[attr])
        report[attr] = {
            "by_group": gdf.to_dict(orient="records"),
            "gaps": fairness_gaps(gdf),
        }
    return report
