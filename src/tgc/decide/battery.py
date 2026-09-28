"""Battery firming of a solar or wind plant (Section 7.2), normalized capacity 1.

Day-ahead: robust LP on the calibrated lower bound L_h (cvxpy + HiGHS)
    max sum_h lambda_h P_h
    s.t. P_h <= L_h + d_h - c_h,  s_h = s_{h-1} + sqrt(eta) c_h - d_h / sqrt(eta),
         0 <= s_h <= E, 0 <= c_h <= min(p, L_h), 0 <= d_h <= p, s_0 = s_24 = E/2
Real time: a greedy rule covers deviations from P_h with the battery; leftover shortfall pays
penalty_mult x lambda_h, surplus is sold at surplus_mult x lambda_h.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BatteryParams:
    power: float = 0.5
    energy: float = 1.0
    round_trip_eff: float = 0.9
    penalty_mult: float = 3.0
    surplus_mult: float = 0.5
    simul_penalty: float = 1e-6

    @classmethod
    def from_config(cls, cfg: dict) -> "BatteryParams":
        return cls(**{k: cfg[k] for k in cls.__dataclass_fields__ if k in cfg})


def price_profile(cfg: dict, hours: int = 24) -> np.ndarray:
    lam = np.full(hours, float(cfg["price_offpeak"]))
    lam[list(cfg["peak_hours"])] = float(cfg["price_peak"])
    return lam


class DayAheadLP:
    """Parametrized LP compiled once and re-solved for each day's L (DPP)."""

    def __init__(self, lam: np.ndarray, bp: BatteryParams):
        import cvxpy as cp
        H = len(lam)
        self.lam = lam
        se = np.sqrt(bp.round_trip_eff)
        self.L = cp.Parameter(H, nonneg=True)
        P, c, d = cp.Variable(H), cp.Variable(H, nonneg=True), cp.Variable(H, nonneg=True)
        s = cp.Variable(H + 1)
        cons = [P <= self.L + d - c, c <= bp.power, c <= self.L, d <= bp.power,
                s[1:] == s[:-1] + se * c - d / se, s >= 0, s <= bp.energy,
                s[0] == bp.energy / 2, s[H] == bp.energy / 2]
        obj = cp.Maximize(lam @ P - bp.simul_penalty * cp.sum(c + d))
        self.prob = cp.Problem(obj, cons)
        self.P, self.c, self.d = P, c, d

    def solve(self, L: np.ndarray):
        self.L.value = np.maximum(np.asarray(L, float), 0.0)
        self.prob.solve(solver="HIGHS")
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"LP status {self.prob.status}")
        return self.P.value.copy(), self.c.value.copy(), self.d.value.copy()


def realize(P, y, lam, bp: BatteryParams):
    """Greedy real-time dispatch against outcomes y; starts at s = E/2."""
    se = np.sqrt(bp.round_trip_eff)
    s = bp.energy / 2
    revenue = surplus_rev = penalty = short_energy = 0.0
    short_hours = 0
    for h in range(len(P)):
        dev = y[h] - P[h]
        revenue += lam[h] * P[h]
        if dev >= 0:
            ch = min(dev, bp.power, (bp.energy - s) / se)
            s += se * ch
            surplus_rev += bp.surplus_mult * lam[h] * (dev - ch)
        else:
            need = -dev
            dis = min(need, bp.power, s * se)
            s -= dis / se
            short = need - dis
            if short > 1e-9:
                short_hours += 1
                short_energy += short
                penalty += bp.penalty_mult * lam[h] * short
    return {"profit": revenue + surplus_rev - penalty, "revenue": revenue, "surplus_revenue": surplus_rev,
            "penalty": penalty, "short_energy": short_energy, "short_hours": short_hours,
            "committed": float(np.sum(P)), "end_soc": s}


def oracle_profit(lp: DayAheadLP, y) -> float:
    P, _, _ = lp.solve(y)
    return float(lp.lam @ P)
