"""Weighted CQR (NexCP): recent days weigh more, weight rho^age."""
from __future__ import annotations

import numpy as np

from tgc.calibrate.scores import weighted_quantile


def nexcp(S: np.ndarray, alpha: float, rho: float = 0.99, window_days: int | None = None):
    T, H = S.shape
    Q = np.full((T, H), np.nan)
    inf = 0
    for t in range(1, T):
        lo = 0 if window_days is None else max(0, t - window_days)
        win = S[lo:t]
        w = rho ** np.arange(len(win))[::-1]  # age 0 for day t-1
        q = weighted_quantile(win, w, 1 - alpha)
        bad = np.isinf(q)
        if bad.any():  # unattainable with few scores: cap at the largest score seen
            inf += int(bad.sum())
            q = np.where(bad, np.nanmax(win, axis=0), q)
        Q[t] = q
    return Q, {"capped": inf}
