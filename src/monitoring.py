"""Production drift monitoring (idea 8).

After deployment, the world changes. We watch for it with:
  * PSI (Population Stability Index) per feature — the credit-industry standard.
        PSI < 0.1  : stable
        0.1-0.25   : moderate shift, investigate
        > 0.25     : major shift, retrain
  * An Evidently data-drift report (HTML) if `evidently` is installed.

Pair with adversarial validation (training-time) — together they cover drift
before AND after deployment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .cleaning import clean_features
from .feature_engineering import add_features


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index between a reference and a current sample."""
    expected = pd.to_numeric(pd.Series(expected), errors="coerce").dropna()
    actual = pd.to_numeric(pd.Series(actual), errors="coerce").dropna()
    if expected.empty or actual.empty:
        return np.nan
    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=quantiles)[0] / len(expected)
    a = np.histogram(actual, bins=quantiles)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """PSI for every modelled numeric feature, with a severity label."""
    ref = add_features(clean_features(reference))
    cur = add_features(clean_features(current))
    rows = []
    for col in config.NUMERIC_FEATURES:
        if col in ref.columns and col in cur.columns:
            val = psi(ref[col], cur[col])
            sev = ("stable" if val < 0.1 else "moderate" if val < 0.25 else "major")
            rows.append({"feature": col, "psi": round(val, 4), "severity": sev})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, path=None):
    """Generate an Evidently drift report (HTML) if the library is available."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except Exception:
        try:  # older evidently API
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset
        except Exception as e:  # pragma: no cover
            return f"evidently not available ({e})"
    ref = add_features(clean_features(reference))[config.NUMERIC_FEATURES]
    cur = add_features(clean_features(current))[config.NUMERIC_FEATURES]
    path = path or (config.REPORTS_DIR / "drift_report.html")
    rep = Report(metrics=[DataDriftPreset()])
    try:
        rep.run(reference_data=ref, current_data=cur)
        rep.save_html(str(path))
    except Exception:  # newest API returns a snapshot
        snap = rep.run(reference_data=ref, current_data=cur)
        snap.save_html(str(path))
    return str(path)
