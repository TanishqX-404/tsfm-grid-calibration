"""Conformal PID control, quantile-tracking (P) term (Angelopoulos et al., 2023).

q_{t+1} = q_t + eta (err_t - alpha), with eta = eta_frac x (score range in the warm-up window).
Before ``window_days`` scores exist the sliding split-CQR quantile is used; tracking starts
from that quantile.
"""
from __future__ import annotations

import numpy as np

from tgc.calibrate.scores import conformal_quantile


def pid(S: np.ndarray, alpha: float, eta_frac: float = 0.01, window_days: int = 90):
    T, H = S.shape
    Q = np.full((T, H), np.nan)
    q = np.full(H, np.nan)
    eta = np.full(H, np.nan)
    seen = np.zeros(H, dtype=int)
    for t in range(T):
        tracking = seen >= window_days
        if (~tracking).any() and t > 0:
            init, _ = conformal_quantile(S[max(0, t - window_days):t], 1 - alpha)
            win = S[max(0, t - window_days):t]
            rng = np.nanmax(win, axis=0) - np.nanmin(win, axis=0) if np.isfinite(win).any() else np.full(H, np.nan)
            q = np.where(tracking, q, init)
            eta = np.where(tracking, eta, eta_frac * rng)
        Q[t] = q
        s = S[t]
        ok = ~np.isnan(s) & ~np.isnan(q)
        err = (s > q).astype(float)
        q = np.where(ok & tracking, q + eta * (err - alpha), q)
        seen += ~np.isnan(s)
    return Q, {"capped": 0}
