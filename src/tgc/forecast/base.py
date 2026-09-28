"""Forecaster interface and the uniform output contract (Section 5.4).

Every model writes rows with columns FORECAST_COLUMNS; calibration and decisions read only that.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
QCOLS = [f"q{int(round(q * 100))}" for q in QUANTILES]
FORECAST_COLUMNS = ["series_id", "origin_local", "target_local", "horizon", *QCOLS,
                    "y_true", "model", "model_revision"]


class Forecaster:
    """Base class. ``predict`` returns 24 rows: target_local, horizon, q10..q90.

    ``context_df`` has columns local_time, target (+ covariates, operator_fc...) for the
    ``context_days`` days before the origin; ``future_cov_df`` has local_time and known-future
    columns for the 24 target hours (never the target).
    """
    name: str = "base"
    revision: str = "n/a"
    context_days: int = 28
    gap_check_days: int | None = None  # None -> context_days
    uses_covariates: bool = False

    def fit(self, train_df: pd.DataFrame) -> None:  # no-op for foundation models
        return None

    def predict(self, context_df: pd.DataFrame, future_cov_df: pd.DataFrame, horizon: int = 24,
                quantiles: Sequence[float] = QUANTILES) -> pd.DataFrame:
        raise NotImplementedError

    def predict_batch(self, items: list[tuple[pd.DataFrame, pd.DataFrame]], horizon: int = 24,
                      quantiles: Sequence[float] = QUANTILES) -> list[pd.DataFrame]:
        """Default: loop. GPU models override to batch many origins in one call."""
        return [self.predict(c, f, horizon, quantiles) for c, f in items]


def frame(future_cov_df: pd.DataFrame, qarr: np.ndarray) -> pd.DataFrame:
    """Build a prediction frame from a (24, 9) quantile array."""
    out = pd.DataFrame(np.asarray(qarr, dtype=float), columns=QCOLS)
    out.insert(0, "target_local", future_cov_df["local_time"].values)
    out.insert(1, "horizon", np.arange(1, len(out) + 1))
    return out


def sort_quantiles(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Enforce non-crossing quantiles by sorting; returns the number of rows that crossed."""
    q = df[QCOLS].to_numpy()
    crossed = int((np.diff(q, axis=1) < 0).any(axis=1).sum())
    df = df.copy()
    df[QCOLS] = np.sort(q, axis=1)
    return df, crossed


def validate(df: pd.DataFrame) -> None:
    missing = set(FORECAST_COLUMNS) - set(df.columns)
    assert not missing, f"forecast table missing {missing}"
    assert df["horizon"].between(1, 24).all()
    assert (df["target_local"] >= df["origin_local"]).all()
    assert np.isfinite(df[QCOLS].to_numpy()).all(), "non-finite quantiles"
