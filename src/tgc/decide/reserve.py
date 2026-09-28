"""Probabilistic reserve sizing (Section 7.1), in normalized units.

Upward reserve covers under-forecast load or generation shortfall. With reserve cost c_R and
shortfall penalty c_S the expected-cost-optimal reserve is the tau* = 1 - c_R/c_S quantile of
the forecast error (newsvendor), so the calibrated one-sided bound at tau* sizes it.
"""
from __future__ import annotations

import numpy as np


def error(y, median, target: str):
    """Positive error = the direction reserve must cover."""
    return y - median if target == "load" else median - y


def reserve_from_bound(bound, median, target: str):
    r = bound - median if target == "load" else median - bound
    return np.maximum(r, 0.0)


def day_costs(R, err, c_R: float, c_S: float, mask=None):
    """Per-day cost terms for (days, 24) arrays; ``mask`` True hours are excluded (solar night)."""
    R = np.where(mask, 0.0, R) if mask is not None else R
    e = np.where(mask, 0.0, err) if mask is not None else err
    short = np.maximum(0.0, e - R)
    oracle_R = np.maximum(e, 0.0)
    hours = (~mask).sum(axis=1) if mask is not None else np.full(len(R), R.shape[1])
    return {
        "reserve": R.sum(axis=1),
        "shortfall": short.sum(axis=1),
        "short_hours": (short > 1e-12).sum(axis=1),
        "hours": hours,
        "cost": c_R * R.sum(axis=1) + c_S * short.sum(axis=1),
        "oracle_cost": c_R * oracle_R.sum(axis=1),
    }


def deterministic_reserve(median, k: float):
    """Reference rule: reserve = a fixed share of the forecast."""
    return np.maximum(k * median, 0.0)
