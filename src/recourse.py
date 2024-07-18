"""Actionable recourse / counterfactuals (idea 3).

Beyond "why were you declined" (src.explain) we answer "what would get you
approved" — concrete, minimal, *actionable* changes. This is the ethical-AI
differentiator and supports regulatory expectations around recourse.

We search over ACTIONABLE features only (you can pay down balance or shorten the
term; you cannot change age/gender/location), moving each toward a typical
approved applicant until the decision flips to APPROVE.

Native, dependency-light implementation. For a richer, optimisation-based version
consider DiCE (`dice-ml`, from Microsoft Research) — intentionally not required
here so the project runs out-of-the-box.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Features a borrower can plausibly act on, and the helpful direction.
ACTIONABLE = {
    "outstanding_balance": "decrease",
    "loan_amount": "decrease",
    "number_of_defaults": "decrease",
    "remaining term": "decrease",
    "interest_rate": "decrease",
    "salary": "increase",
}
IMMUTABLE = {"age", "gender", "location", "job", "marital_status"}


def _predict(pipeline, row):
    return float(pipeline.predict_proba(row)[0, 1])


def find_recourse(pipeline, row: pd.DataFrame, reference: dict, threshold: float,
                  max_changes: int = 3, steps: int = 10) -> dict:
    """Greedy minimal-change search to flip a DECLINE into an APPROVE.

    Strategy: rank actionable features by how much moving them to the typical
    approved value reduces risk, then apply them one at a time (interpolating in
    `steps`) until predicted default prob < threshold or we run out of budget.
    """
    row = row.iloc[[0]].copy()
    base_p = _predict(pipeline, row)
    if base_p < threshold:
        return {"already_approved": True, "default_probability": round(base_p, 4),
                "changes": []}

    # Score each actionable feature's full-move impact.
    impacts = []
    for feat in ACTIONABLE:
        if feat in IMMUTABLE or feat not in row.columns or feat not in reference:
            continue
        trial = row.copy()
        trial.iloc[0, trial.columns.get_loc(feat)] = reference[feat]
        impacts.append((feat, base_p - _predict(pipeline, trial)))
    impacts.sort(key=lambda x: x[1], reverse=True)

    applied, working = [], row.copy()
    for feat, _ in impacts:
        if len(applied) >= max_changes:
            break
        start = pd.to_numeric(pd.Series([working.iloc[0][feat]]), errors="coerce").iloc[0]
        target = reference[feat]
        if pd.isna(start):
            start = target
        # interpolate from current value toward the approved reference
        for frac in np.linspace(1 / steps, 1.0, steps):
            cand = working.copy()
            new_val = start + (target - start) * frac
            cand.iloc[0, cand.columns.get_loc(feat)] = new_val
            p = _predict(pipeline, cand)
            if p < threshold:
                applied.append({
                    "feature": feat,
                    "description": config.FEATURE_DESCRIPTIONS.get(feat, feat),
                    "from": round(float(start), 2),
                    "to": round(float(new_val), 2),
                    "resulting_probability": round(p, 4),
                })
                return {"already_approved": False, "default_probability": round(base_p, 4),
                        "approved_after_changes": True, "changes": applied}
        # not enough alone: commit the full move and continue with the next feature
        working.iloc[0, working.columns.get_loc(feat)] = target
        applied.append({
            "feature": feat,
            "description": config.FEATURE_DESCRIPTIONS.get(feat, feat),
            "from": round(float(start), 2),
            "to": round(float(target), 2),
            "resulting_probability": round(_predict(pipeline, working), 4),
        })

    final_p = _predict(pipeline, working)
    return {
        "already_approved": False,
        "default_probability": round(base_p, 4),
        "approved_after_changes": final_p < threshold,
        "changes": applied,
    }


def recourse_text(result: dict) -> str:
    if result.get("already_approved"):
        return "Application is already in the APPROVE range."
    if not result.get("approved_after_changes"):
        return ("No small set of changes flips this decision — the profile is "
                "substantially higher risk than typical approved applicants.")
    lines = ["To reach approval, consider:"]
    for c in result["changes"]:
        verb = "reduce" if c["to"] < c["from"] else "increase"
        lines.append(f"  • {verb} {c['description']} from {c['from']} to {c['to']}")
    return "\n".join(lines)
