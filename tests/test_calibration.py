"""Section 6.3 sanity tests and the no-look-ahead tests for every conformal variant."""
import numpy as np
import pytest

from tgc.calibrate import core, scores
from tgc.calibrate.aci import aci
from tgc.calibrate.nexcp import nexcp
from tgc.calibrate.pid import pid
from tgc.calibrate.split_cqr import split_cqr

Z90 = 1.2815515655446004  # N(0,1) 0.9 quantile

VARIANTS = {
    "split_cqr": lambda S, a: split_cqr(S, a, 90),
    "nexcp": lambda S, a: nexcp(S, a, 0.99),
    "aci": lambda S, a: aci(S, a, 0.01, 90),
    "pid": lambda S, a: pid(S, a, 0.01, 90),
}


def coverage(S, Q, rows=slice(None)):
    s, q = S[rows], Q[rows]
    ok = ~np.isnan(q)
    return float(np.mean(s[ok] <= q[ok]))


def gaussian(T, H, lo, hi, seed=0, shift_at=None, shift=0.0):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((T, H))
    if shift_at is not None:
        y[shift_at:] += shift
    return scores.two_sided(np.full((T, H), lo), np.full((T, H), hi), y)


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("level", [0.8, 0.9])
def test_gaussian_nominal_coverage(variant, level):
    """1. Known quantile model: every variant within +-1 pp of nominal over 5,000 days."""
    S = gaussian(5000, 2, -Z90, Z90, seed=1)
    Q, _ = VARIANTS[variant](S, 1 - level)
    assert abs(coverage(S, Q, slice(100, None)) - level) <= 0.01


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("level", [0.8, 0.9])
def test_too_narrow_quantiles_are_widened(variant, level):
    """3. Deliberately too-narrow quantiles: all conformal variants restore nominal coverage."""
    S = gaussian(5000, 2, -0.5, 0.5, seed=2)
    Q, _ = VARIANTS[variant](S, 1 - level)
    native = float(np.mean(S[100:] <= 0))
    assert native < 0.5
    assert abs(coverage(S, Q, slice(100, None)) - level) <= 0.01


def test_mean_shift_split_undercovers_adaptive_recovers():
    """2. Mean shift halfway: split CQR under-covers right after it; ACI and PID recover within
    about 30 days."""
    T, s = 2000, 1000
    S = gaussian(T, 24, -Z90, Z90, seed=3, shift_at=s, shift=2.0)
    level = 0.9
    Qs, _ = split_cqr(S, 1 - level, 90)
    assert coverage(S, Qs, slice(s, s + 7)) < level - 0.15
    for fn in (lambda S, a: aci(S, a, 0.05, 90), lambda S, a: pid(S, a, 0.05, 90)):
        Q, _ = fn(S, 1 - level)
        assert abs(coverage(S, Q, slice(s + 30, s + 130)) - level) <= 0.04


def test_aci_caps_infinite_intervals():
    S = gaussian(400, 1, -Z90, Z90, seed=4)
    S[200:220] += 10  # a burst of misses drives alpha_t below zero with a large gamma
    Q, info = aci(S, 0.1, 0.5, 90)
    assert info["capped"] > 0 and np.isfinite(Q[1:]).all()


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_no_look_ahead(variant):
    """Q[t] must depend only on S[:t]: perturbing S[t0:] leaves Q[:t0+1] unchanged."""
    rng = np.random.default_rng(5)
    S = rng.standard_normal((300, 24))
    Q, _ = VARIANTS[variant](S, 0.1)
    for t0 in (120, 200, 299):
        S2 = S.copy()
        S2[t0:] = 50 + rng.standard_normal(S2[t0:].shape)
        Q2, _ = VARIANTS[variant](S2, 0.1)
        np.testing.assert_array_equal(Q[: t0 + 1], Q2[: t0 + 1])


def _spiky(T=600, H=24, seed=6):
    rng = np.random.default_rng(seed)
    S = np.abs(rng.standard_normal((T, H)))
    spikes = np.arange(150, T, 45)
    S[spikes] += 20.0 * np.arange(1, len(spikes) + 1)[:, None]  # each shock beats all earlier ones
    return S, spikes


def test_lookahead_injection_is_detected():
    """The leakage tests fail as designed when leakage is injected (lookahead=1)."""
    rng = np.random.default_rng(7)
    S = rng.standard_normal((300, 24))
    Q, _ = split_cqr(S, 0.1, 90, lookahead=1)
    S2 = S.copy()
    S2[200:] = 50
    Q2, _ = split_cqr(S2, 0.1, 90, lookahead=1)
    assert not np.array_equal(Q[:201], Q2[:201])     # invariance check catches it

    S, spikes = _spiky()
    honest, _ = split_cqr(S, 0.01, 90)
    leaky, _ = split_cqr(S, 0.01, 90, lookahead=1)
    # a shifted (leaky) calibrator "covers" unforeseeable shocks; the honest one cannot
    assert coverage(S[spikes], honest[spikes]) == 0.0
    assert coverage(S[spikes], leaky[spikes]) > 0.5


def test_conformal_quantile_rank():
    w = np.arange(1, 10, dtype=float)[:, None]          # 1..9, n = 9
    q, capped = scores.conformal_quantile(w, 0.8)      # ceil(0.8 * 10) = 8th smallest
    assert q[0] == 8 and not capped[0]
    q, capped = scores.conformal_quantile(w, 0.95)     # ceil(9.5) = 10 > 9 -> capped at max
    assert q[0] == 9 and capped[0]


def test_one_sided_bounds_via_core():
    rng = np.random.default_rng(8)
    T = 3000
    y = rng.standard_normal((T, 24))
    import pandas as pd
    days = pd.date_range("2020-01-01", periods=T, freq="D")
    qs = {c: np.full((T, 24), v) for c, v in zip(core.QCOLS, np.linspace(-0.5, 0.5, 9))}
    g = core.Grid(days, y, qs, np.zeros((T, 24)))
    for side, lvl in (("upper", 0.95), ("lower", 0.95)):
        lo, up, _ = core.calibrate(g, lvl, side, "split_cqr", {"window_days": 90})
        cov = np.mean(y[100:] <= up[100:]) if side == "upper" else np.mean(y[100:] >= lo[100:])
        assert abs(cov - lvl) < 0.01
