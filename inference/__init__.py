"""Self-contained loan inference package.

Portable serving layer for the loan-default model. Has NO dependency on the
project's `src/` tree — copy this folder + a trained `.joblib` + its metadata
json and it works anywhere.

    from inference import (
        InferenceConfig, LoanInferenceEngine,
        ApplicantDTO, PredictionOptions,
    )
"""
from .dto import (
    ApplicantDTO,
    PredictionOptions,
    PredictionResult,
    ProfitResult,
    ReasonCode,
    RecourseStep,
    SegmentationResult,
)
from .engine import InferenceConfig, LoanInferenceEngine

__all__ = [
    "InferenceConfig", "LoanInferenceEngine",
    "ApplicantDTO", "PredictionOptions", "PredictionResult",
    "ReasonCode", "RecourseStep", "SegmentationResult", "ProfitResult",
]
__version__ = "1.0.0"
