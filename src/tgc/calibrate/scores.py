"""Conformity scores and conformal quantiles.

Arrays are (T days, H horizons). NaN rows mark days without a forecast/outcome.
A calibrator maps a score matrix S to a correction matrix Q where Q[t] depends only on S[:t]
(scores of days strictly before t): outcomes of day D are known only after D ends.
"""
from __future__ import annotations

import numpy as np


def two_sided(lo, hi, y):
    """CQR score (Romano et al., 2019): positive when y is outside [lo, hi]."""
    return np.maximum(lo - y, y - hi)


def upper(hi, y):
    """One-sided upper-bound score: bound = hi + Q covers y when y - hi <= Q."""
    return y - hi


def lower(lo, y):
    """One-sided lower-bound score: bound = lo - Q covers y when lo - y <= Q."""
    return lo - y


def conformal_quantile(window: np.ndarray, level) -> tuple[np.ndarray, np.ndarray]:
    """Per-column ceil(level (n+1))-th smallest non-NaN score.

    ``window``: (n, H); ``level``: scalar or (H,). Returns (Q, capped) where ``capped`` marks
    columns whose finite-sample quantile would be +inf (k > n); these are capped at the largest
    score seen. Columns with no scores give NaN.
    """
    window = np.atleast_2d(window)
    H = window.shape[1]
    level = np.broadcast_to(np.asarray(level, dtype=float), (H,))
    srt = np.sort(window, axis=0)                   # NaN last
    n = np.sum(~np.isnan(window), axis=0)
    k = np.ceil(level * (n + 1)).astype(int)
    capped = k > n
    k = np.clip(k, 1, np.maximum(n, 1))
    q = srt[k - 1, np.arange(H)] if len(srt) else np.full(H, np.nan)
    q = np.where(n > 0, q, np.nan)
    low = level <= 0                                # empty interval requested: smallest score
    if low.any():
        q = np.where(low, np.nanmin(np.where(np.isnan(window), np.inf, window), axis=0), q)
        q = np.where(n > 0, q, np.nan)
    return q, capped & (n > 0) & ~low


def weighted_quantile(window: np.ndarray, weights: np.ndarray, level: float) -> np.ndarray:
    """NexCP weighted quantile (Barber et al., 2023) with a point mass of weight 1 at +inf
    for the test point; weights for NaN scores are dropped. Returns +inf when unattainable."""
    n, H = window.shape
    out = np.full(H, np.nan)
    for h in range(H):
        s = window[:, h]
        ok = ~np.isnan(s)
        if not ok.any():
            continue
        s, w = s[ok], weights[ok]
        order = np.argsort(s)
        s, w = s[order], w[order]
        cw = np.cumsum(w) / (w.sum() + 1.0)
        idx = np.searchsorted(cw, level - 1e-12)
        out[h] = s[idx] if idx < len(s) else np.inf
    return out
