"""Data Transfer Objects — the typed request/response contract.

Callers build an `ApplicantDTO` (not a raw dict) and receive typed result objects.
DTOs guarantee the full raw schema is present (missing fields -> None -> imputed),
and every result has `.to_dict()` for JSON/API responses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import pandas as pd

from . import config


@dataclass
class ApplicantDTO:
    """A single loan applicant in business terms (maps to the raw model schema)."""
    loan_amount: Optional[float] = None
    salary: Optional[float] = None
    outstanding_balance: Optional[float] = None
    interest_rate: Optional[float] = None
    number_of_defaults: Optional[float] = None
    remaining_term: Optional[float] = None
    age: Optional[float] = None
    gender: Optional[str] = None
    job: Optional[str] = None
    location: Optional[str] = None
    marital_status: Optional[str] = None
    is_employed: Optional[Any] = None
    disbursement_date: Optional[str] = None     # 'YYYY MM DD'
    applicant_id: Optional[str] = None

    # field name in the DTO -> raw column name expected by the pipeline
    _RAW_MAP = {
        "remaining_term": "remaining term",
        "disbursement_date": "disbursemet_date",
    }

    @classmethod
    def from_dict(cls, d: dict) -> "ApplicantDTO":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_raw_dict(self) -> dict:
        out = {}
        for fname in self.__dataclass_fields__:
            if fname == "applicant_id":
                continue
            raw_name = self._RAW_MAP.get(fname, fname)
            out[raw_name] = getattr(self, fname)
        # ensure every column the model expects exists
        for col in config.RAW_INPUT_COLUMNS:
            out.setdefault(col, None)
        return out


def applicants_to_frame(applicants) -> pd.DataFrame:
    """Accept a single DTO, a list of DTOs, a dict, a list of dicts or a DataFrame."""
    if isinstance(applicants, pd.DataFrame):
        return applicants.reset_index(drop=True)
    if isinstance(applicants, (ApplicantDTO, dict)):
        applicants = [applicants]
    rows, ids = [], []
    for a in applicants:
        dto = a if isinstance(a, ApplicantDTO) else ApplicantDTO.from_dict(a)
        rows.append(dto.to_raw_dict())
        ids.append(dto.applicant_id)
    df = pd.DataFrame(rows)
    df.attrs["applicant_ids"] = ids
    return df


@dataclass
class ReasonCode:
    feature: str
    description: str
    applicant_value: Any
    typical_approved_value: Any
    risk_contribution: float


@dataclass
class RecourseStep:
    feature: str
    description: str
    from_value: Any
    to_value: Any
    resulting_probability: float


@dataclass
class PredictionResult:
    applicant_id: Optional[str]
    default_probability: float
    decision: str                      # APPROVE / DECLINE
    risk_band: str                     # Low / Medium / High
    threshold: float
    reasons: Optional[list] = None             # list[ReasonCode]
    recourse: Optional[list] = None            # list[RecourseStep]
    recourse_message: Optional[str] = None
    pricing: Optional[dict] = None             # {suggested_rate, ...}

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class PredictionOptions:
    include_reasons: bool = True
    include_recourse: bool = False
    include_pricing: bool = False
    top_k_reasons: int = 4


@dataclass
class SegmentationResult:
    n_segments: int
    profile: list                      # list[dict] per-segment stats
    labels: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"n_segments": self.n_segments, "profile": self.profile}


@dataclass
class ProfitResult:
    best_threshold: float
    best_profit: float
    best_approval_rate: float
    profit_at_default_threshold: float
    profit_approve_all: float
    uplift_vs_approve_all: float

    def to_dict(self) -> dict:
        return asdict(self)
