"""Reserve cost with Chronos-2's native 0.95 bound (reviewer request R1.2), test period.

    python -m tgc.evaluate.chronos2_tail_cost --allow-test

Reads out/forecasts_tail/chronos2 (from tgc.forecast.chronos2_tail), checks that its q10/q50/q90
equal the main Chronos-2 forecasts, and sizes reserve at tau* = 0.95 from the native q95 (load) or
q05 (solar, wind). Compares with the native-q0.9 and validation-selected calibrated rows of
decisions_reserve.csv. Writes out/metrics/test/chronos2_tail_cost.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tgc import config, splits
from tgc.calibrate.run import load_forecasts
from tgc.data import io
from tgc.decide import reserve as res

VARIANTS = ["native", "split_cqr", "nexcp", "aci", "pid"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args(argv)
    period = splits.get_period("test", allow_test=args.allow_test)
    dc = config.load("decisions")["reserve"]
    tail = config.path(config.load("models")["out_dir"]).parent / "forecasts_tail" / "chronos2"
    mdir = config.path(config.load("evaluate")["out_dir"])
    resdir = mdir if (mdir / "test" / "decisions_reserve.csv").exists() else config.path("results/metrics")
    t = pd.read_csv(resdir / "test" / "decisions_reserve.csv")
    v = pd.read_csv(resdir / "val" / "decisions_reserve.csv")
    t, v = (x[(x.model == "chronos2") & np.isclose(x.tau, 0.95) & x.variant.isin(VARIANTS)] for x in (t, v))
    sel = v.loc[v.groupby("series_id").regret.idxmin()].set_index("series_id").variant
    rows = []
    for sid in config.series_ids():
        target = config.split_series(sid)[1]
        f = pd.read_parquet(tail / f"{sid}.parquet")
        f = f[period.contains(f["origin_local"])]
        main_fc = load_forecasts(config.path(config.load("models")["out_dir"]) / "chronos2", sid)
        m = f.merge(main_fc[["origin_local", "horizon", "q10", "q50", "q90"]], on=["origin_local", "horizon"], suffixes=("", "_main"))
        maxdiff = float(max(np.abs(m[c] - m[c + "_main"]).max() for c in ("q10", "q50", "q90")))
        night = io.load_series(sid).set_index("local_time")["is_night"]
        f = f.assign(is_night=night.reindex(f["target_local"]).fillna(False).to_numpy() & (target == "solar"))
        days = sorted(f["origin_local"].unique())
        w = lambda c: f.pivot(index="origin_local", columns="horizon", values=c).reindex(index=days, columns=range(1, 25)).to_numpy(float)
        y, med = w("y_true"), w("q50")
        bound = w("q95") if target == "load" else w("q05")
        mask = w("is_night") > 0.5
        cst = res.day_costs(res.reserve_from_bound(bound, med, target), res.error(y, med, target), dc["c_R"], dc["c_R"] * 20, mask)
        ok = ~np.isnan(y).any(axis=1)
        short_freq = cst["short_hours"][ok].sum() / cst["hours"][ok].sum()
        tt = t[t.series_id == sid].set_index("variant")
        rows.append({"series_id": sid, "n_days": int(ok.sum()), "max_abs_diff_vs_main": maxdiff,
                     "cost_native_q95": cst["cost"][ok].mean(), "cost_native_q90": tt.loc["native", "cost"],
                     "cost_calibrated": tt.loc[sel[sid], "cost"], "variant": sel[sid],
                     "shortfall_native_q95": short_freq, "shortfall_native_q90": tt.loc["native", "shortfall_freq"],
                     "shortfall_calibrated": tt.loc[sel[sid], "shortfall_freq"]})
    df = pd.DataFrame(rows)
    out = mdir / "test"
    out.mkdir(parents=True, exist_ok=True)
    df.round(6).to_csv(out / "chronos2_tail_cost.csv", index=False)
    print(df.round(4).to_string(index=False))
    print(df.mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
