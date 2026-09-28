"""Forecast, calibration and decision metrics (Section 9). Inputs are already normalized
(capacity for solar/wind, training mean for load)."""
from __future__ import annotations

import numpy as np

from tgc.forecast.base import QCOLS, QUANTILES


def nmae(y, med):
    return float(np.mean(np.abs(y - med)))


def nrmse(y, med):
    return float(np.sqrt(np.mean((y - med) ** 2)))


def mase(y, med, naive_mae_train):
    return nmae(y, med) / naive_mae_train


def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def mean_pinball(y, Q):
    """Average quantile loss over the 0.1..0.9 grid (a CRPS approximation). Q: (n, 9)."""
    return float(np.mean([pinball(y, Q[:, i], t).mean() for i, t in enumerate(QUANTILES)]))


def reliability(y, Q):
    """Observed frequency of y <= q_tau for each tau (reliability diagram)."""
    return {t: float(np.mean(y <= Q[:, i])) for i, t in enumerate(QUANTILES)}


def picp(y, lower, upper):
    return float(np.mean((y >= lower) & (y <= upper)))


def width(lower, upper):
    return float(np.mean(upper - lower))


def winkler(y, lower, upper, alpha):
    pen = (2 / alpha) * (np.maximum(lower - y, 0) + np.maximum(y - upper, 0))
    return float(np.mean(upper - lower + pen))


def worst_hour_gap(y, lower, upper, hour, nominal):
    """Largest |coverage - nominal| across hours of day (percentage points)."""
    gaps = []
    for h in np.unique(hour):
        m = hour == h
        gaps.append(abs(picp(y[m], lower[m], upper[m]) - nominal))
    return 100 * float(np.max(gaps))


def naive_mae(target_series: np.ndarray, lag: int) -> float:
    """In-sample seasonal-naive MAE (MASE denominator), on the training period."""
    y = np.asarray(target_series, float)
    d = np.abs(y[lag:] - y[:-lag])
    return float(np.nanmean(d))


__all__ = ["nmae", "nrmse", "mase", "mean_pinball", "reliability", "picp", "width", "winkler",
           "worst_hour_gap", "naive_mae", "QCOLS"]
