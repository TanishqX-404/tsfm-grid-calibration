"""Sliding-window split CQR: one constant per horizon from the last W days' scores."""
from __future__ import annotations

import numpy as np

from tgc.calibrate.scores import conformal_quantile


def split_cqr(S: np.ndarray, alpha: float, window_days: int = 90, lookahead: int = 0):
    """``lookahead`` > 0 deliberately leaks future scores (tests only)."""
    T, H = S.shape
    Q = np.full((T, H), np.nan)
    capped = 0
    for t in range(T):
        end = min(t + lookahead, T)
        win = S[max(0, end - window_days):end]
        if len(win) == 0:
            continue
        Q[t], c = conformal_quantile(win, 1 - alpha)
        capped += int(c.sum())
    return Q, {"capped": capped}
