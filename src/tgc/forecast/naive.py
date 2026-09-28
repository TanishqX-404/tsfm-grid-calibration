"""Seasonal naive with empirical residual quantiles per hour of day."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tgc.forecast.base import QUANTILES, Forecaster, frame


class SeasonalNaive(Forecaster):
    name = "seasonal_naive"

    def __init__(self, lag_hours: int, residual_days: int = 90):
        self.lag = lag_hours
        self.residual_days = residual_days
        # needs lag + residual window of history
        self.context_days = residual_days + lag_hours // 24
        self.gap_check_days = 28  # extra history may hold gaps; residual quantiles skip NaN

    def predict(self, context_df, future_cov_df, horizon=24, quantiles=QUANTILES):
        y = context_df["target"].to_numpy()
        n = len(y)
        lag = self.lag
        point = y[n - lag: n - lag + horizon]              # same hour, lag earlier
        res_n = self.residual_days * 24
        r = y[n - res_n:] - y[n - res_n - lag: n - lag]    # in-sample naive residuals
        r = r.reshape(self.residual_days, 24)              # rows = days, cols = hour of day
        rq = np.nanquantile(r, quantiles, axis=0).T           # (24, 9)
        return frame(future_cov_df, point[:, None] + rq)
