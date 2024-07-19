"""Inference: score new applicants and explain decisions.

The saved artifact is a full Pipeline (clean -> engineer -> preprocess -> model),
so it consumes RAW applicant rows directly — no preprocessing to re-implement.

Library use:
    from src.inference import LoanScorer
    scorer = LoanScorer()
    scorer.predict(df)                  # probabilities + decisions
    scorer.explain(one_row_df)          # decision + adverse-action reasons

CLI:
    python -m src.inference --input applicants.csv --output scored.csv
    python -m src.inference --input applicants.csv --explain
"""
from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from . import config, explain


class LoanScorer:
    def __init__(self, model_path=None, metadata_path=None):
        self.pipeline = joblib.load(model_path or config.MODEL_ARTIFACT)
        with open(metadata_path or config.METADATA_ARTIFACT) as f:
            self.meta = json.load(f)
        self.threshold = float(self.meta.get("decision_threshold", 0.5))
        self.reference = self.meta.get("reference_profile", {})

    # -- scoring ---------------------------------------------------------- #
    def predict_proba(self, df: pd.DataFrame):
        return self.pipeline.predict_proba(df)[:, 1]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return probability, decision and a risk band per applicant."""
        proba = self.predict_proba(df)
        decision = ["DECLINE" if p >= self.threshold else "APPROVE" for p in proba]
        band = pd.cut(
            proba, bins=[-0.01, 0.15, 0.35, 1.01], labels=["Low", "Medium", "High"]
        )
        out = df.copy()
        out["default_probability"] = proba.round(4)
        out["decision"] = decision
        out["risk_band"] = band
        return out

    # -- explanation (ECOA-style reason codes) ---------------------------- #
    def explain(self, row: pd.DataFrame, top_k: int = 4) -> dict:
        row = row.iloc[[0]] if len(row) else row
        result = explain.local_reason_codes(
            self.pipeline, row, self.reference, top_k=top_k
        )
        result["decision"] = (
            "DECLINE" if result["default_probability"] >= self.threshold else "APPROVE"
        )
        result["threshold"] = self.threshold
        return result

    def explain_text(self, row: pd.DataFrame, top_k: int = 4) -> str:
        r = self.explain(row, top_k=top_k)
        lines = [
            f"Decision: {r['decision']}  "
            f"(estimated default probability {r['default_probability']:.1%}, "
            f"approve below {self.threshold:.1%})",
        ]
        if r["decision"] == "DECLINE" and r["top_reasons"]:
            lines.append("Main reasons:")
            for i, rsn in enumerate(r["top_reasons"], 1):
                lines.append(
                    f"  {i}. {rsn['description']}: your value {rsn['applicant_value']} "
                    f"vs typical approved {rsn['typical_approved_value']}"
                )
        elif r["decision"] == "DECLINE":
            lines.append("  (no single dominant factor; overall profile is higher risk)")
        return "\n".join(lines)


def _cli():
    ap = argparse.ArgumentParser(description="Score loan applicants.")
    ap.add_argument("--input", required=True, help="CSV of applicants (raw schema)")
    ap.add_argument("--output", help="where to write scored CSV")
    ap.add_argument("--explain", action="store_true", help="print reasons for each row")
    args = ap.parse_args()

    scorer = LoanScorer()
    df = pd.read_csv(args.input)
    scored = scorer.predict(df)

    out = args.output or "scored_applicants.csv"
    scored.to_csv(out, index=False)
    print(f"Scored {len(df)} applicants ({scorer.meta['model_name']}) -> {out}")
    print(scored[["default_probability", "decision", "risk_band"]].head(10).to_string())

    if args.explain:
        print("\n--- Reason codes ---")
        for i in range(min(len(df), 10)):
            print(f"\nApplicant #{i}")
            print(scorer.explain_text(df.iloc[[i]]))


if __name__ == "__main__":
    _cli()
