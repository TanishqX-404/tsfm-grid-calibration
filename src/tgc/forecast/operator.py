"""Balancing-authority day-ahead demand forecast (EIA-930), point only.

All nine quantiles equal the point forecast; conformal layers build intervals around it.
"""
from __future__ import annotations

import numpy as np

from tgc.forecast.base import QUANTILES, Forecaster, frame


class OperatorForecast(Forecaster):
    name = "operator"
    context_days = 1

    def predict(self, context_df, future_cov_df, horizon=24, quantiles=QUANTILES):
        p = future_cov_df["operator_fc"].to_numpy()
        return frame(future_cov_df, np.repeat(p[:, None], len(quantiles), axis=1))
