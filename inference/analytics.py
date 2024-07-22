"""Self-contained analytics used by the inference engine.

No `src` imports — reuses this package's `transforms` for feature computation and
the loaded pipeline for scoring. Covers reason codes, recourse, risk-based
pricing, profit optimization, segmentation, anomaly detection and PSI drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .transforms import add_features, clean_features

ACTIONABLE = {
    "outstanding_balance": "decrease", "loan_amount": "decrease",
    "number_of_defaults": "decrease", "remaining term": "decrease",
    "interest_rate": "decrease", "salary": "increase",
}
IMMUTABLE = {"age", "gender", "location", "job", "marital_status"}


# ---- explainability ------------------------------------------------------- #
def reason_codes(pipeline, row: pd.DataFrame, reference: dict, top_k=4, exclude=None) -> dict:
    exclude = set(exclude or config.PROTECTED_ATTRIBUTES)
    row = row.iloc[[0]].copy()
    base_p = float(pipeline.predict_proba(row)[0, 1])
    out = []
    for col, ref_val in reference.items():
        if col in exclude or col not in row.columns:
            continue
        cur = row.iloc[0][col]
        try:
            if pd.notna(cur) and cur == ref_val:
                continue
        except Exception:
            pass
        cf = row.copy()
        cf.iloc[0, cf.columns.get_loc(col)] = ref_val
        delta = base_p - float(pipeline.predict_proba(cf)[0, 1])
        if delta > 1e-4:
            out.append({"feature": col,
                        "description": config.FEATURE_DESCRIPTIONS.get(col, col),
                        "applicant_value": _clean(cur), "typical_approved_value": _clean(ref_val),
                        "risk_contribution": round(delta, 4)})
    out.sort(key=lambda d: d["risk_contribution"], reverse=True)
    return {"default_probability": round(base_p, 4), "top_reasons": out[:top_k]}


def find_recourse(pipeline, row: pd.DataFrame, reference: dict, threshold: float,
                  max_changes=3, steps=10) -> dict:
    row = row.iloc[[0]].copy()
    base_p = float(pipeline.predict_proba(row)[0, 1])
    if base_p < threshold:
        return {"already_approved": True, "default_probability": round(base_p, 4), "changes": []}
    impacts = []
    for feat in ACTIONABLE:
        if feat in IMMUTABLE or feat not in row.columns or feat not in reference:
            continue
        trial = row.copy()
        trial.iloc[0, trial.columns.get_loc(feat)] = reference[feat]
        impacts.append((feat, base_p - float(pipeline.predict_proba(trial)[0, 1])))
    impacts.sort(key=lambda x: x[1], reverse=True)
    applied, working = [], row.copy()
    for feat, _ in impacts:
        if len(applied) >= max_changes:
            break
        start = pd.to_numeric(pd.Series([working.iloc[0][feat]]), errors="coerce").iloc[0]
        target = reference[feat]
        if pd.isna(start):
            start = target
        for frac in np.linspace(1 / steps, 1.0, steps):
            cand = working.copy()
            new_val = start + (target - start) * frac
            cand.iloc[0, cand.columns.get_loc(feat)] = new_val
            p = float(pipeline.predict_proba(cand)[0, 1])
            if p < threshold:
                applied.append({"feature": feat, "description": config.FEATURE_DESCRIPTIONS.get(feat, feat),
                                "from_value": round(float(start), 2), "to_value": round(float(new_val), 2),
                                "resulting_probability": round(p, 4)})
                return {"already_approved": False, "approved_after_changes": True,
                        "default_probability": round(base_p, 4), "changes": applied}
        working.iloc[0, working.columns.get_loc(feat)] = target
        applied.append({"feature": feat, "description": config.FEATURE_DESCRIPTIONS.get(feat, feat),
                        "from_value": round(float(start), 2), "to_value": round(float(target), 2),
                        "resulting_probability": round(float(pipeline.predict_proba(working)[0, 1]), 4)})
    final_p = float(pipeline.predict_proba(working)[0, 1])
    return {"already_approved": False, "approved_after_changes": final_p < threshold,
            "default_probability": round(base_p, 4), "changes": applied}


def recourse_message(result: dict) -> str:
    if result.get("already_approved"):
        return "Application is already in the APPROVE range."
    if not result.get("approved_after_changes"):
        return "No small set of changes flips this decision; the profile is substantially higher risk."
    parts = []
    for c in result["changes"]:
        verb = "reduce" if c["to_value"] < c["from_value"] else "increase"
        parts.append(f"{verb} {c['description']} from {c['from_value']} to {c['to_value']}")
    return "To reach approval: " + "; ".join(parts) + "."


# ---- pricing / profit ----------------------------------------------------- #
def risk_based_price(proba, base_rate=0.12, max_premium=0.20, lgd=0.6):
    proba = np.asarray(proba, dtype=float)
    premium = np.clip(lgd * proba / (1 - proba + 1e-6), 0, max_premium)
    return np.round(base_rate + premium, 4)


def profit_optimization(df, y_true, proba, lgd=0.6, default_threshold=0.5, n=101):
    revenue = pd.to_numeric(df["loan_amount"], errors="coerce") * pd.to_numeric(df["interest_rate"], errors="coerce")
    loss = lgd * pd.to_numeric(df["outstanding_balance"], errors="coerce")
    per_loan = np.where(np.asarray(y_true) == 0, revenue, -loss)
    proba = np.asarray(proba)

    def profit(t):
        return float(np.nansum(per_loan[proba < t]))

    ts = np.linspace(0, 1, n)
    profits = [profit(t) for t in ts]
    best_i = int(np.argmax(profits))
    return {
        "best_threshold": float(ts[best_i]),
        "best_profit": float(profits[best_i]),
        "best_approval_rate": float((proba < ts[best_i]).mean()),
        "profit_at_default_threshold": profit(default_threshold),
        "profit_approve_all": profit(1.0),
        "uplift_vs_approve_all": float(profits[best_i] - profit(1.0)),
        "curve": pd.DataFrame({"threshold": ts, "profit": profits}),
    }


# ---- segmentation / anomaly / drift --------------------------------------- #
def _numeric_matrix(raw: pd.DataFrame):
    feats = add_features(clean_features(raw))
    num = [c for c in feats.columns
           if c not in set(config.DRIFT_PRONE_TIME_FEATURES)
           and pd.api.types.is_numeric_dtype(feats[c])]
    pipe = SkPipeline([("impute", SimpleImputer(strategy="median")),
                       ("scale", StandardScaler())])
    return pipe.fit_transform(feats[num]), num


def segment(raw: pd.DataFrame, k=4):
    X, _ = _numeric_matrix(raw)
    km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(X)
    labels = km.labels_
    df = raw.copy().reset_index(drop=True)
    df["segment"] = labels
    rev = pd.to_numeric(df["loan_amount"], errors="coerce") * pd.to_numeric(df["interest_rate"], errors="coerce")
    df["_rev"] = rev
    prof = (df.groupby("segment")
            .agg(size=("segment", "size"), avg_loan=("loan_amount", "mean"),
                 avg_salary=("salary", "mean"), avg_prior_defaults=("number_of_defaults", "mean"),
                 avg_revenue=("_rev", "mean"))
            .round(2).reset_index())
    return labels.tolist(), prof.to_dict(orient="records")


def detect_anomalies(raw: pd.DataFrame, contamination=0.02):
    X, _ = _numeric_matrix(raw)
    iso = IsolationForest(n_estimators=300, contamination=contamination,
                          random_state=config.RANDOM_STATE, n_jobs=-1).fit(X)
    out = raw.copy().reset_index(drop=True)
    out["anomaly_score"] = np.round(-iso.decision_function(X), 4)
    out["is_anomaly"] = iso.predict(X) == -1
    return out.sort_values("anomaly_score", ascending=False)


def psi(expected, actual, bins=10):
    expected = pd.to_numeric(pd.Series(expected), errors="coerce").dropna()
    actual = pd.to_numeric(pd.Series(actual), errors="coerce").dropna()
    if expected.empty or actual.empty:
        return np.nan
    q = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(q) < 3:
        return 0.0
    q[0], q[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, bins=q)[0] / len(expected), 1e-6, None)
    a = np.clip(np.histogram(actual, bins=q)[0] / len(actual), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame):
    ref = add_features(clean_features(reference))
    cur = add_features(clean_features(current))
    cols = [c for c in ref.columns
            if pd.api.types.is_numeric_dtype(ref[c]) and c in cur.columns
            and c not in set(config.DRIFT_PRONE_TIME_FEATURES)]
    rows = []
    for c in cols:
        v = psi(ref[c], cur[c])
        sev = "stable" if v < 0.1 else "moderate" if v < 0.25 else "major"
        rows.append({"feature": c, "psi": round(v, 4), "severity": sev})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return round(float(v), 2)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v
