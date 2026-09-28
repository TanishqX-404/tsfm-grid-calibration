"""Statistical tests (Section 8.3)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def diebold_mariano(loss_a, loss_b, h: int = 1):
    """DM test with the Harvey-Leybourne-Newbold small-sample correction.

    Returns (statistic, two-sided p). Negative statistic: model A has lower loss.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[~np.isnan(d)]
    T = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[: T - k] - dbar)) / T for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / T
    if var <= 0:
        return 0.0, 1.0
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = hln * dm
    p = 2 * stats.t.sf(abs(stat), df=T - 1)
    return float(stat), float(p)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def kupiec_pof(miss, alpha: float) -> float:
    """Kupiec POF test p-value. ``miss``: 1 where the outcome fell outside; ``alpha``: nominal miss rate."""
    miss = np.asarray(miss, int)
    n, x = len(miss), int(miss.sum())
    if n == 0:
        return np.nan
    phat = x / n

    def ll(p):
        p = min(max(p, 1e-12), 1 - 1e-12)
        return (n - x) * np.log(1 - p) + x * np.log(p)

    lr = -2 * (ll(alpha) - ll(phat))
    return float(stats.chi2.sf(lr, 1))


def christoffersen_ind(miss) -> float:
    """Christoffersen (1998) independence test p-value on a 0/1 miss sequence."""
    m = np.asarray(miss, int)
    a, b = m[:-1], m[1:]
    n00, n01 = np.sum((a == 0) & (b == 0)), np.sum((a == 0) & (b == 1))
    n10, n11 = np.sum((a == 1) & (b == 0)), np.sum((a == 1) & (b == 1))
    n0, n1 = n00 + n01, n10 + n11
    if n0 == 0 or n1 == 0 or (n01 + n11) == 0:
        return np.nan
    p01, p11 = n01 / n0, n11 / n1
    p = (n01 + n11) / (n0 + n1)

    def xlogy(k, q):
        return 0.0 if k == 0 else k * np.log(max(q, 1e-12))

    l0 = xlogy(n00, 1 - p) + xlogy(n01, p) + xlogy(n10, 1 - p) + xlogy(n11, p)
    l1 = xlogy(n00, 1 - p01) + xlogy(n01, p01) + xlogy(n10, 1 - p11) + xlogy(n11, p11)
    return float(stats.chi2.sf(-2 * (l0 - l1), 1))


def block_bootstrap_ci(x, block: int = 7, n: int = 1000, seed: int = 0, level: float = 0.95):
    """Moving-block bootstrap CI for the mean of a daily series (NaNs dropped)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    T = len(x)
    if T == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, max(T - block + 1, 1), size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :T]
    means = x[np.minimum(idx, T - 1)].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))
