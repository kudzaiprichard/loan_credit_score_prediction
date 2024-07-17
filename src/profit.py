"""Profit-optimized decisioning & risk-based pricing (idea 1).

The thesis that elevates this project: a lender does not care about F1 — it cares
about *money*. So instead of approving below a 0.5 (or F1-optimal) probability, we
choose the threshold that maximizes **expected portfolio profit**, and we price
each approved loan by its risk (risk-based pricing).

Simple, transparent credit economics (tune the assumptions to your book):
    revenue if repaid  = loan_amount * interest_rate          (interest earned)
    loss if defaults   = LGD * outstanding_balance            (LGD = loss given default)
profit(threshold) = sum over APPROVED loans of (repaid ? revenue : -loss)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _econ(df: pd.DataFrame, lgd: float):
    revenue = pd.to_numeric(df["loan_amount"], errors="coerce") * pd.to_numeric(
        df["interest_rate"], errors="coerce"
    )
    loss = lgd * pd.to_numeric(df["outstanding_balance"], errors="coerce")
    return revenue.to_numpy(), loss.to_numpy()


def profit_at_threshold(df, y_true, proba, threshold, lgd=0.6) -> float:
    """Realized profit if we approve everyone with default prob < threshold."""
    revenue, loss = _econ(df, lgd)
    y_true = np.asarray(y_true)
    approve = np.asarray(proba) < threshold
    per_loan = np.where(y_true == 0, revenue, -loss)
    return float(np.nansum(per_loan[approve]))


def profit_curve(df, y_true, proba, lgd=0.6, n=101) -> pd.DataFrame:
    """Profit / approval-rate / default-rate across the threshold sweep."""
    thresholds = np.linspace(0.0, 1.0, n)
    revenue, loss = _econ(df, lgd)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    per_loan = np.where(y_true == 0, revenue, -loss)
    rows = []
    for t in thresholds:
        approve = proba < t
        n_app = int(approve.sum())
        rows.append(
            {
                "threshold": float(t),
                "approval_rate": n_app / len(proba),
                "profit": float(np.nansum(per_loan[approve])),
                "approved_default_rate": float(y_true[approve].mean()) if n_app else 0.0,
            }
        )
    return pd.DataFrame(rows)


def optimal_threshold(df, y_true, proba, lgd=0.6) -> dict:
    """Profit-maximizing threshold + comparison to approve-all and 0.5."""
    curve = profit_curve(df, y_true, proba, lgd=lgd)
    best = curve.loc[curve["profit"].idxmax()]
    approve_all = profit_at_threshold(df, y_true, proba, 1.0, lgd)
    at_half = profit_at_threshold(df, y_true, proba, 0.5, lgd)
    return {
        "best_threshold": float(best["threshold"]),
        "best_profit": float(best["profit"]),
        "best_approval_rate": float(best["approval_rate"]),
        "profit_approve_all": float(approve_all),
        "profit_at_0.5": float(at_half),
        "uplift_vs_approve_all": float(best["profit"] - approve_all),
        "curve": curve,
    }


def risk_based_price(proba, base_rate=0.12, max_premium=0.20, lgd=0.6) -> np.ndarray:
    """Suggested interest rate per applicant = base + premium that covers expected loss.

    A higher predicted default probability buys a higher rate (or a decline); this
    is how lenders make *more money* from riskier-but-acceptable customers instead
    of rejecting them outright.
    """
    proba = np.asarray(proba, dtype=float)
    premium = np.clip(lgd * proba / (1 - proba + 1e-6), 0, max_premium)
    return np.round(base_rate + premium, 4)
