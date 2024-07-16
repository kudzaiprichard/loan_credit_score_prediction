"""Raw data IO. Keep this dumb: read bytes off disk, no transformations."""
from __future__ import annotations

import pandas as pd

from . import config


def load_raw(path=None) -> pd.DataFrame:
    """Load the raw csv exactly as stored (no cleaning)."""
    path = path or config.RAW_DATA
    # engine/dtype kept default so duplicate columns get the pandas `.1` suffix
    return pd.read_csv(path)


def target_distribution(df: pd.DataFrame) -> pd.Series:
    """Convenience: raw target value counts."""
    return df[config.TARGET_RAW].value_counts(dropna=False)
