"""LoanInferenceEngine — the configurable, self-contained serving entry point.

Usage:
    from inference import InferenceConfig, LoanInferenceEngine, ApplicantDTO, PredictionOptions

    engine = LoanInferenceEngine(InferenceConfig(model_path="models/best_model.joblib",
                                                 metadata_path="models/model_metadata.json"))
    result = engine.predict(ApplicantDTO(loan_amount=30000, salary=3000, ...),
                            PredictionOptions(include_reasons=True, include_recourse=True,
                                              include_pricing=True))
    result.to_dict()

Design notes:
  * fully self-contained — no `src` imports; loads a model pickled under `src.*`
    via inference.compat module aliases.
  * configurable — point it at any model/metadata; optional separate preprocessing
    artifact is applied before the model if supplied.
  * one prediction entry (`predict`/`batch_predict`) whose response is shaped by
    `PredictionOptions`; plus dedicated portfolio entries (segment / anomalies /
    drift / profit) for the analytics capabilities.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from . import analytics, compat
from .dto import (
    ApplicantDTO,
    PredictionOptions,
    PredictionResult,
    ProfitResult,
    ReasonCode,
    RecourseStep,
    SegmentationResult,
    applicants_to_frame,
)

RISK_BANDS = [(-0.01, 0.15, "Low"), (0.15, 0.35, "Medium"), (0.35, 1.01, "High")]


@dataclass
class InferenceConfig:
    model_path: str = "models/best_model.joblib"
    metadata_path: Optional[str] = "models/model_metadata.json"
    preprocessing_path: Optional[str] = None     # optional separate fitted preprocessor
    threshold: Optional[float] = None             # override metadata threshold
    base_rate: float = 0.12
    lgd: float = 0.6                              # loss given default (pricing/profit)


def _band(p: float) -> str:
    for lo, hi, name in RISK_BANDS:
        if lo < p <= hi:
            return name
    return "High"


class LoanInferenceEngine:
    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        compat.register_aliases()              # make src-pickled models loadable
        self.pipeline = joblib.load(self.config.model_path)
        self.preprocessor = (
            joblib.load(self.config.preprocessing_path)
            if self.config.preprocessing_path else None
        )
        self.metadata = {}
        if self.config.metadata_path and Path(self.config.metadata_path).exists():
            with open(self.config.metadata_path) as f:
                self.metadata = json.load(f)
        self.threshold = float(
            self.config.threshold
            if self.config.threshold is not None
            else self.metadata.get("decision_threshold", 0.5)
        )
        self.reference = self.metadata.get("reference_profile", {})
        self.model_name = self.metadata.get("model_name", "unknown")

    # -- internal ---------------------------------------------------------- #
    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.preprocessor.transform(df) if self.preprocessor is not None else df

    def _proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(self._prep(df))[:, 1]

    def _result(self, raw_row: pd.DataFrame, p: float, aid, options: PredictionOptions) -> PredictionResult:
        decision = "DECLINE" if p >= self.threshold else "APPROVE"
        res = PredictionResult(applicant_id=aid, default_probability=round(float(p), 4),
                               decision=decision, risk_band=_band(p), threshold=self.threshold)
        if options.include_reasons and decision == "DECLINE":
            rc = analytics.reason_codes(self.pipeline, raw_row, self.reference,
                                        top_k=options.top_k_reasons)
            res.reasons = [ReasonCode(**r) for r in rc["top_reasons"]]
        if options.include_recourse and decision == "DECLINE":
            rec = analytics.find_recourse(self.pipeline, raw_row, self.reference, self.threshold)
            res.recourse = [RecourseStep(**c) for c in rec.get("changes", [])]
            res.recourse_message = analytics.recourse_message(rec)
        if options.include_pricing:
            rate = float(analytics.risk_based_price([p], self.config.base_rate, lgd=self.config.lgd)[0])
            res.pricing = {"suggested_interest_rate": rate, "base_rate": self.config.base_rate}
        return res

    # -- prediction API ---------------------------------------------------- #
    def predict(self, applicant, options: Optional[PredictionOptions] = None) -> PredictionResult:
        options = options or PredictionOptions()
        df = applicants_to_frame(applicant)
        aid = df.attrs.get("applicant_ids", [None])[0]
        p = self._proba(df)[0]
        return self._result(df.iloc[[0]], p, aid, options)

    def batch_predict(self, applicants, options: Optional[PredictionOptions] = None) -> list:
        options = options or PredictionOptions()
        df = applicants_to_frame(applicants)
        ids = df.attrs.get("applicant_ids", [None] * len(df))
        proba = self._proba(df)
        return [self._result(df.iloc[[i]], proba[i], ids[i] if i < len(ids) else None, options)
                for i in range(len(df))]

    def predict_proba(self, applicants) -> np.ndarray:
        return self._proba(applicants_to_frame(applicants))

    # -- portfolio analytics (ideas 1,2,4,8) ------------------------------- #
    def segment_portfolio(self, applicants, k: int = 4) -> SegmentationResult:
        df = applicants_to_frame(applicants)
        labels, profile = analytics.segment(df, k=k)
        return SegmentationResult(n_segments=k, profile=profile, labels=labels)

    def detect_anomalies(self, applicants, contamination: float = 0.02) -> pd.DataFrame:
        return analytics.detect_anomalies(applicants_to_frame(applicants), contamination=contamination)

    def monitor_drift(self, reference, current) -> pd.DataFrame:
        return analytics.drift_report(applicants_to_frame(reference), applicants_to_frame(current))

    def optimize_profit(self, applicants, outcomes, default_threshold: Optional[float] = None) -> ProfitResult:
        df = applicants_to_frame(applicants)
        proba = self._proba(df)
        out = analytics.profit_optimization(
            df, np.asarray(outcomes), proba, lgd=self.config.lgd,
            default_threshold=default_threshold if default_threshold is not None else self.threshold,
        )
        out.pop("curve", None)
        return ProfitResult(**out)

    def __repr__(self) -> str:
        return (f"LoanInferenceEngine(model={self.model_name!r}, "
                f"threshold={self.threshold:.3f}, preprocessor={self.preprocessor is not None})")
