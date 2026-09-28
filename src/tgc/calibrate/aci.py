"""Adaptive conformal inference (Gibbs & Candes, 2021) on top of sliding split CQR.

alpha_{t+1} = alpha_t + gamma (alpha - err_t), per horizon. When alpha_t <= 0 the interval is
infinite in theory; it is capped at the largest calibration score seen and counted.
"""
from __future__ import annotations

import numpy as np

from tgc.calibrate.scores import conformal_quantile


def aci(S: np.ndarray, alpha: float, gamma: float = 0.005, window_days: int = 90):
    T, H = S.shape
    Q = np.full((T, H), np.nan)
    a = np.full(H, alpha)
    alphas = np.full((T, H), np.nan)
    n_inf = 0
    for t in range(T):
        win = S[max(0, t - window_days):t]
        if len(win):
            level = 1 - a
            q, capped = conformal_quantile(win, np.minimum(level, 1.0))
            inf = a <= 0
            if inf.any():
                q = np.where(inf, np.nanmax(S[:t], axis=0), q)  # cap at the largest score seen
                n_inf += int(inf.sum())
            Q[t] = q
        alphas[t] = a
        s = S[t]
        ok = ~np.isnan(s) & ~np.isnan(Q[t])
        err = (s > Q[t]).astype(float)
        a = np.where(ok, a + gamma * (alpha - err), a)
    return Q, {"capped": n_inf, "alpha_path": alphas}
