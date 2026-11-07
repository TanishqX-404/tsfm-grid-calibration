"""Reserve cost of native vs calibrated bounds at equal quantile levels (reviewer request R1.2).

    python -m tgc.evaluate.same_level_cost

At tau* = 0.90 the native q0.9 (q0.1 for solar/wind) exists, so native and calibrated bounds are
compared at the same level; at 0.95 the native bound stays at 0.9. Uses the saved test decision
metrics; the calibration variant per model and series is the one selected on validation at that tau*.
Writes out/metrics/test/same_level_cost.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tgc import config

MAIN = ["seasonal_naive", "lgbm-s0", "nhits-s0", "patchtst-s0", "chronos2", "moirai2", "timesfm25", "tirex"]
VARIANTS = ["native", "split_cqr", "nexcp", "aci", "pid"]


def main():
    mdir = config.path(config.load("evaluate")["out_dir"])
    res = mdir if (mdir / "test" / "decisions_reserve.csv").exists() else config.path("results/metrics")
    t = pd.read_csv(res / "test" / "decisions_reserve.csv")
    v = pd.read_csv(res / "val" / "decisions_reserve.csv")
    rows = []
    for tau in (0.9, 0.95):
        vv = v[np.isclose(v.tau, tau) & v.variant.isin(VARIANTS) & v.model.isin(MAIN)]
        sel = vv.loc[vv.groupby(["model", "series_id"]).regret.idxmin(), ["model", "series_id", "variant"]]
        tt = t[np.isclose(t.tau, tau) & t.model.isin(MAIN)]
        cal = sel.merge(tt, on=["model", "series_id", "variant"])
        nat = tt[tt.variant == "native"]
        for m in MAIN:
            c, n = cal[cal.model == m], nat[nat.model == m]
            rows.append({"tau": tau, "model": m, "cost_native": n.cost.mean(), "cost_calibrated": c.cost.mean(),
                         "gain_pct": 100 * (1 - c.cost.mean() / n.cost.mean()),
                         "shortfall_native_pct": 100 * n.shortfall_freq.mean(),
                         "shortfall_calibrated_pct": 100 * c.shortfall_freq.mean(),
                         "series_kept_native": int((sel[sel.model == m].variant == "native").sum())})
    out = mdir / "test"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(out / "same_level_cost.csv", index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
