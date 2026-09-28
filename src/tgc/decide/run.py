"""Decision CLI: reserve sizing and battery firming for every model x calibration variant.

    python -m tgc.decide.run --phase val                      # reserve on validation (tunes k)
    python -m tgc.decide.run --phase test --allow-test        # final run

Writes out/decisions/{reserve,battery}/{model}_{variant}_{series}.parquet (one row per day and
level) plus deterministic-rule reserve files ({model}_deterministic_{series}).
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from tgc import config, splits
from tgc.decide import battery as bat
from tgc.decide import reserve as res

log = logging.getLogger("tgc.decide")


def wide(df: pd.DataFrame, col: str, days) -> np.ndarray:
    return df.pivot(index="origin_local", columns="horizon", values=col).reindex(index=days, columns=range(1, 25)).to_numpy(float)


def load_cal(model, variant, sid):
    p = config.path(config.load("calibration")["out_dir"]) / model / variant / f"{sid}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def reserve_rows(cal: pd.DataFrame, sid: str, days, dc: dict) -> pd.DataFrame:
    target = config.split_series(sid)[1]
    side = "upper" if target == "load" else "lower"
    base = cal[(cal["side"] == side)]
    out = []
    for ratio in dc["cost_ratios"]:
        tau = 1 - 1 / ratio
        d = base[np.isclose(base["level"], tau)]
        if d.empty:
            continue
        y, med = wide(d, "y_true", days), wide(d, "median", days)
        bound = wide(d, "upper" if side == "upper" else "lower", days)
        night = wide(d.assign(n=d["is_night"].astype(float)), "n", days) > 0.5
        err = res.error(y, med, target)
        R = res.reserve_from_bound(bound, med, target)
        cst = res.day_costs(R, err, dc["c_R"], dc["c_R"] * ratio, night)
        valid = ~np.isnan(y).any(axis=1)
        f = pd.DataFrame({"day": days, "ratio": ratio, "tau": tau, **cst})[valid]
        f["regret"] = f["cost"] - f["oracle_cost"]
        out.append(f)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def deterministic_rows(cal: pd.DataFrame, sid: str, days_val, days_eval, dc: dict) -> tuple[pd.DataFrame, dict]:
    """Reserve = k x forecast, with k per cost ratio chosen on validation days only."""
    splits.assert_before_test(pd.Series(days_val), "deterministic k tuning")
    target = config.split_series(sid)[1]
    d = cal[(cal["side"] == "two") & np.isclose(cal["level"], 0.9)]
    out, ks = [], {}
    for ratio in dc["cost_ratios"]:
        best = None
        for k in dc["deterministic_grid"]:
            y, med = wide(d, "y_true", days_val), wide(d, "median", days_val)
            night = wide(d.assign(n=d["is_night"].astype(float)), "n", days_val) > 0.5
            c = res.day_costs(res.deterministic_reserve(med, k), res.error(y, med, target),
                              dc["c_R"], dc["c_R"] * ratio, night)["cost"]
            m = np.nanmean(c)
            if best is None or m < best[1]:
                best = (k, m)
        k = best[0]
        ks[ratio] = k
        y, med = wide(d, "y_true", days_eval), wide(d, "median", days_eval)
        night = wide(d.assign(n=d["is_night"].astype(float)), "n", days_eval) > 0.5
        cst = res.day_costs(res.deterministic_reserve(med, k), res.error(y, med, target),
                            dc["c_R"], dc["c_R"] * ratio, night)
        valid = ~np.isnan(y).any(axis=1)
        f = pd.DataFrame({"day": days_eval, "ratio": ratio, "tau": 1 - 1 / ratio, "k": k, **cst})[valid]
        f["regret"] = f["cost"] - f["oracle_cost"]
        out.append(f)
    return pd.concat(out, ignore_index=True), ks


def battery_rows(cal: pd.DataFrame, sid: str, days, bc: dict, lp, oracle: dict) -> pd.DataFrame:
    bp = bat.BatteryParams.from_config(bc)
    lam = lp.lam
    d0 = cal[cal["side"] == "lower"]
    out = []
    for level in bc["levels"]:
        d = d0[np.isclose(d0["level"], level)]
        if d.empty:
            continue
        y, L = wide(d, "y_true", days), wide(d, "lower", days)
        for i, day in enumerate(days):
            if np.isnan(y[i]).any() or np.isnan(L[i]).any():
                continue
            P, _, _ = lp.solve(np.clip(L[i], 0, None))
            r = bat.realize(P, y[i], lam, bp)
            key = pd.Timestamp(day)
            if key not in oracle:
                oracle[key] = bat.oracle_profit(lp, np.clip(y[i], 0, None))
            out.append({"day": day, "level": level, **r, "oracle_profit": oracle[key]})
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="val", choices=["val", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--series", nargs="*")
    ap.add_argument("--problems", nargs="*", default=["reserve", "battery"])
    ap.add_argument("--dry-run", action="store_true", help="3 days only")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dc = config.load("decisions")
    cc = config.load("calibration")
    days = splits.origin_days(args.phase, allow_test=args.allow_test)
    days_val = splits.origin_days("val")
    if args.dry_run:
        days = days[:3]
    root = config.path(cc["out_dir"])
    models = args.models or sorted(p.name for p in root.iterdir() if p.is_dir())
    series = args.series or config.series_ids()
    out_root = config.path(dc["out_dir"]) / args.phase
    lam = bat.price_profile(dc["battery"])
    lp = bat.DayAheadLP(lam, bat.BatteryParams.from_config(dc["battery"]))
    oracle_cache: dict = {}
    for m in models:
        variants = args.variants or sorted(p.name for p in (root / m).iterdir() if p.is_dir())
        for sid in series:
            target = config.split_series(sid)[1]
            orc = oracle_cache.setdefault(sid, {})
            for v in variants:
                cal = load_cal(m, v, sid)
                if cal is None:
                    continue
                if "reserve" in args.problems:
                    r = reserve_rows(cal, sid, days, dc["reserve"])
                    dest = out_root / "reserve" / f"{m}_{v}_{sid}.parquet"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    r.to_parquet(dest, index=False)
                    if v == "native":  # deterministic rule depends only on the model's median
                        det, ks = deterministic_rows(cal, sid, days_val, days, dc["reserve"])
                        det.to_parquet(out_root / "reserve" / f"{m}_deterministic_{sid}.parquet", index=False)
                if "battery" in args.problems and target in ("solar", "wind"):
                    b = battery_rows(cal, sid, days, dc["battery"], lp, orc)
                    dest = out_root / "battery" / f"{m}_{v}_{sid}.parquet"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    b.to_parquet(dest, index=False)
                log.info("decided %s/%s/%s", m, v, sid)


if __name__ == "__main__":
    main()
