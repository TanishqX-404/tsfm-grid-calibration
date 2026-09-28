import numpy as np
import pytest

from tgc.decide import battery as bat
from tgc.decide import reserve as res

CFG = {"price_offpeak": 1.0, "price_peak": 2.0, "peak_hours": [16, 17, 18, 19, 20]}


def test_newsvendor_tau():
    for ratio, tau in [(10, 0.9), (20, 0.95), (50, 0.98), (100, 0.99)]:
        assert 1 - 1 / ratio == pytest.approx(tau)


def test_reserve_perfect_forecast_zero_regret():
    rng = np.random.default_rng(0)
    y = rng.uniform(0.5, 1.5, (5, 24))
    med = y - rng.normal(0, 0.05, y.shape)
    err = res.error(y, med, "load")
    R = res.reserve_from_bound(med + np.maximum(err, 0), med, "load")  # bound = the truth
    c = res.day_costs(R, err, 1.0, 20.0)
    np.testing.assert_allclose(c["cost"] - c["oracle_cost"], 0.0, atol=1e-12)
    assert c["short_hours"].sum() == 0


def test_reserve_hand_computed_two_days():
    """Two days x 2 hours, load, c_R = 1, c_S = 20."""
    y = np.array([[1.00, 1.20], [0.90, 1.10]])
    med = np.array([[1.05, 1.00], [0.90, 1.00]])
    upper = np.array([[1.10, 1.15], [0.95, 1.20]])
    err = res.error(y, med, "load")                  # [[-.05, .20], [0, .10]]
    R = res.reserve_from_bound(upper, med, "load")   # [[.05, .15], [.05, .20]]
    c = res.day_costs(R, err, 1.0, 20.0)
    # day 1: reserve .20, shortfall max(0,.20-.15)=.05 -> cost .20 + 20*.05 = 1.20; oracle .20
    # day 2: reserve .25, shortfall 0 -> cost .25; oracle .10
    np.testing.assert_allclose(c["reserve"], [0.20, 0.25], atol=1e-12)
    np.testing.assert_allclose(c["shortfall"], [0.05, 0.0], atol=1e-12)
    np.testing.assert_allclose(c["cost"], [1.20, 0.25], atol=1e-6)
    np.testing.assert_allclose(c["oracle_cost"], [0.20, 0.10], atol=1e-6)
    # generation: reserve covers median - y
    errg = res.error(y, med, "solar")
    Rg = res.reserve_from_bound(np.array([[1.0, 0.9], [0.8, 1.0]]), med, "solar")
    np.testing.assert_allclose(Rg, [[0.05, 0.10], [0.10, 0.0]], atol=1e-12)
    np.testing.assert_allclose(errg, -err, atol=1e-12)


def test_reserve_night_mask():
    R = np.ones((1, 24))
    err = np.ones((1, 24)) * 2
    mask = np.zeros((1, 24), bool)
    mask[0, :12] = True
    c = res.day_costs(R, err, 1.0, 10.0, mask)
    assert c["hours"][0] == 12 and c["reserve"][0] == 12


def solar_day(scale=0.9):
    h = np.arange(24)
    return np.clip(scale * np.sin((h - 6) / 12 * np.pi), 0, None)


@pytest.mark.parametrize("y", [solar_day(), solar_day(0.5), 0.3 + 0.2 * np.cos(np.arange(24) / 3)])
def test_battery_perfect_forecast_equals_oracle(y):
    lam = bat.price_profile(CFG)
    bp = bat.BatteryParams()
    lp = bat.DayAheadLP(lam, bp)
    P, c, d = lp.solve(y)
    r = bat.realize(P, y, lam, bp)
    orc = bat.oracle_profit(lp, y)
    assert r["profit"] == pytest.approx(orc, abs=1e-6)
    assert r["short_hours"] == 0
    assert r["profit"] / orc == pytest.approx(1.0, abs=1e-6)


def test_battery_shifts_energy_to_peak():
    lam = bat.price_profile(CFG)
    y = solar_day()
    lp = bat.DayAheadLP(lam, bat.BatteryParams())
    P, c, d = lp.solve(y)
    # s_0 = s_24 = E/2 and no sun after the peak: at most E/2 of headroom is cycled
    assert d[16:21].sum() > 0.4 and c[:16].sum() > 0.4
    assert lam @ P > lam @ y  # storage adds value over selling the raw profile


def test_battery_overcommit_is_penalized():
    lam = bat.price_profile(CFG)
    bp = bat.BatteryParams()
    lp = bat.DayAheadLP(lam, bp)
    y = solar_day(0.5)
    P, _, _ = lp.solve(solar_day(0.9))       # plan on a far too optimistic bound
    r = bat.realize(P, y, lam, bp)
    assert r["short_hours"] > 0 and r["profit"] < bat.oracle_profit(lp, y)
