import numpy as np
import pytest

from tgc.evaluate import metrics as M
from tgc.evaluate import tests_stat as S


def test_metrics_toy_reference_values():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    med = np.array([1.5, 2.0, 2.0, 5.0])
    assert M.nmae(y, med) == pytest.approx((0.5 + 0 + 1 + 1) / 4)
    assert M.nrmse(y, med) == pytest.approx(np.sqrt((0.25 + 0 + 1 + 1) / 4))
    lo, up = np.array([0.5, 1.5, 3.5, 3.0]), np.array([1.5, 2.5, 4.0, 5.0])
    assert M.picp(y, lo, up) == pytest.approx(0.75)
    assert M.width(lo, up) == pytest.approx((1 + 1 + 0.5 + 2) / 4)
    # Winkler at alpha = 0.2: y=3 misses [3.5,4] by 0.5 -> penalty 10 * 0.5 = 5
    assert M.winkler(y, lo, up, 0.2) == pytest.approx((1 + 1 + 0.5 + 5 + 2) / 4)
    # pinball at tau = 0.9 for y=1, q=0.5: 0.9 * 0.5
    assert M.pinball(np.array([1.0]), np.array([0.5]), 0.9)[0] == pytest.approx(0.45)
    assert M.naive_mae(np.array([1, 2, 3, 1, 2, 3.0]), 3) == 0.0


def test_mean_pinball_perfect_is_zero():
    y = np.linspace(0, 1, 50)
    assert M.mean_pinball(y, np.repeat(y[:, None], 9, axis=1)) == 0.0


def test_dm_identical_losses_p_close_to_one():
    rng = np.random.default_rng(0)
    a = rng.gamma(2, 1, 300)
    stat, p = S.diebold_mariano(a, a.copy())
    assert p == pytest.approx(1.0)


def test_dm_detects_better_model():
    rng = np.random.default_rng(1)
    a = rng.gamma(2, 1, 300)
    stat, p = S.diebold_mariano(a, a + 0.5 + rng.normal(0, 0.1, 300))
    assert stat < 0 and p < 1e-6


def test_holm():
    np.testing.assert_allclose(S.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_kupiec_and_christoffersen():
    rng = np.random.default_rng(2)
    miss = (rng.random(5000) < 0.1).astype(int)
    assert S.kupiec_pof(miss, 0.1) > 0.01
    assert S.kupiec_pof(miss, 0.3) < 1e-6
    assert S.christoffersen_ind(miss) > 0.01
    clustered = np.repeat((rng.random(500) < 0.1).astype(int), 10)
    assert S.christoffersen_ind(clustered) < 1e-6


def test_block_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(3)
    x = rng.normal(1.0, 1.0, 300)
    lo, hi = S.block_bootstrap_ci(x, 7, 1000, 0)
    assert lo < x.mean() < hi and hi - lo < 0.5
