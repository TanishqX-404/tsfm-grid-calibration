"""Glue: forecast table -> (day x horizon) arrays -> calibrated bounds for every variant."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tgc.calibrate import scores
from tgc.calibrate.aci import aci
from tgc.calibrate.nexcp import nexcp
from tgc.calibrate.pid import pid
from tgc.calibrate.split_cqr import split_cqr
from tgc.forecast.base import QCOLS, QUANTILES

VARIANTS = ("native", "split_cqr", "nexcp", "aci", "pid")


@dataclass
class Grid:
    days: pd.DatetimeIndex        # forecast days (rows)
    y: np.ndarray                 # (T, 24)
    q: dict                       # quantile column -> (T, 24)
    target_local: np.ndarray      # (T, 24) timestamps


def to_grid(fc: pd.DataFrame, days: pd.DatetimeIndex | None = None) -> Grid:
    """Pivot a forecast table to day x horizon arrays; missing days become NaN rows."""
    fc = fc.sort_values(["origin_local", "horizon"])
    if days is None:
        days = pd.date_range(fc["origin_local"].min(), fc["origin_local"].max(), freq="D")

    def piv(col):
        return fc.pivot(index="origin_local", columns="horizon", values=col).reindex(index=days, columns=range(1, 25))

    y = piv("y_true").to_numpy(float)
    q = {c: piv(c).to_numpy(float) for c in QCOLS}
    tl = np.asarray(days.values[:, None] + (np.arange(24) * np.timedelta64(1, "h"))[None, :])
    return Grid(days, y, q, tl)


def nearest_qcol(level: float) -> str:
    i = int(np.argmin([abs(t - level) for t in QUANTILES]))
    return QCOLS[i]


def base_and_scores(g: Grid, level: float, side: str):
    """Base bounds and the score matrix for a two-sided level or a one-sided bound.

    side 'two': [q10, q90]; 'upper': nearest native quantile to ``level``;
    'lower': nearest native quantile to 1 - ``level``.
    """
    if side == "two":
        lo, hi = g.q["q10"], g.q["q90"]
        return lo, hi, scores.two_sided(lo, hi, g.y)
    if side == "upper":
        hi = g.q[nearest_qcol(level)]
        return None, hi, scores.upper(hi, g.y)
    lo = g.q[nearest_qcol(1 - level)]
    return lo, None, scores.lower(lo, g.y)


def correction(S: np.ndarray, alpha: float, variant: str, params: dict) -> tuple[np.ndarray, dict]:
    if variant == "native":
        return np.zeros_like(S), {"capped": 0}
    if variant == "split_cqr":
        return split_cqr(S, alpha, params["window_days"])
    if variant == "nexcp":
        return nexcp(S, alpha, params["rho"], params.get("window_days"))
    if variant == "aci":
        return aci(S, alpha, params["gamma"], params["window_days"])
    if variant == "pid":
        return pid(S, alpha, params["eta"], params["window_days"])
    raise ValueError(variant)


def calibrate(g: Grid, level: float, side: str, variant: str, params: dict):
    """Return (lower, upper, info) arrays (T, 24); one-sided bounds have +-inf on the open side."""
    lo, hi, S = base_and_scores(g, level, side)
    Q, info = correction(S, 1 - level, variant, params)
    lower = lo - Q if lo is not None else np.full_like(Q, -np.inf)
    upper = hi + Q if hi is not None else np.full_like(Q, np.inf)
    if lo is not None and hi is not None:
        # a negative correction can invert the interval; collapse it to the midpoint
        mid = (lower + upper) / 2
        inv = lower > upper
        lower, upper = np.where(inv, mid, lower), np.where(inv, mid, upper)
    return lower, upper, info


def winkler(lower, upper, y, alpha):
    """Interval score: width + (2/alpha) x miss distance."""
    return (upper - lower) + (2 / alpha) * (np.maximum(lower - y, 0) + np.maximum(y - upper, 0))


def pinball(bound, y, tau):
    d = y - bound
    return np.maximum(tau * d, (tau - 1) * d)
