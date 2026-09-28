"""Calibration CLI.

    python -m tgc.calibrate.run --tune            # choose ACI gamma / PID eta on validation only
    python -m tgc.calibrate.run                   # calibrate every model in out/forecasts
    python -m tgc.calibrate.run --window 30       # ablation: split/NexCP/ACI window W

Writes out/calibrated/{model}/{variant}/{series}.parquet with columns series_id, origin_local,
target_local, horizon, level, side, lower, upper, lower_raw, upper_raw, median, y_true, is_night.
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
import yaml

from tgc import config, splits
from tgc.calibrate import core
from tgc.data import io

log = logging.getLogger("tgc.calibrate")


def targets_of(series_id):
    return config.split_series(series_id)[1]


def specs(series_id: str) -> list[tuple[float, str]]:
    """(level, side) pairs: two-sided levels + one-sided bounds (upper for load, lower otherwise)."""
    cc = config.load("calibration")
    side = "upper" if targets_of(series_id) == "load" else "lower"
    return [(l, "two") for l in cc["levels"]] + [(l, side) for l in cc["one_sided_levels"]]


def spec_key(level, side):
    return f"{side}@{level:g}"


def load_forecasts(model_dir, series_id) -> pd.DataFrame | None:
    p = model_dir / series_id
    files = sorted(p.glob("*.parquet"))
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def night_mask(series_id, grid: core.Grid) -> np.ndarray:
    df = io.load_series(series_id).set_index("local_time")["is_night"]
    return df.reindex(pd.DatetimeIndex(grid.target_local.ravel())).fillna(False).to_numpy().reshape(grid.y.shape)


def variant_params(variant, tuned, window=None) -> dict:
    cc = config.load("calibration")
    if variant == "native":
        return {}
    p = dict(cc[variant])
    p.pop("gamma_grid", None)
    p.pop("eta_grid", None)
    if window is not None:
        p["window_days"] = window
    if variant == "aci":
        p["gamma"] = tuned.get("aci_gamma", cc["aci"]["gamma_grid"][1])
    if variant == "pid":
        p["eta"] = tuned.get("pid_eta", cc["pid"]["eta_grid"][1])
    return p


def score_for_tuning(lower, upper, y, level, side, mask):
    if side == "two":
        s = core.winkler(lower, upper, y, 1 - level)
    elif side == "upper":
        s = core.pinball(upper, y, level)
    else:
        s = core.pinball(lower, y, 1 - level)
    s = np.where(mask, np.nan, s)
    return float(np.nanmean(s))


def tune(models: list[str], series: list[str]) -> dict:
    """Pick ACI gamma and PID eta per model x series x spec on the validation period only."""
    cc = config.load("calibration")
    val = splits.get_period("val")
    warm = pd.Timestamp(cc["warmup_start"])
    root = config.path(cc["in_dir"])
    out: dict = {}
    for m in models:
        for sid in series:
            fc = load_forecasts(root / m, sid)
            if fc is None:
                continue
            fc = fc[(fc["origin_local"] >= warm) & (fc["origin_local"] < val.end_exclusive)]
            splits.assert_before_test(fc["target_local"], "calibration tuning")
            g = core.to_grid(fc, pd.date_range(warm, val.end, freq="D"))
            night = night_mask(sid, g) if targets_of(sid) == "solar" else np.zeros_like(g.y, bool)
            res = {}
            for level, side in specs(sid):
                r = {}
                for variant, key, grid_key in (("aci", "aci_gamma", "gamma_grid"), ("pid", "pid_eta", "eta_grid")):
                    W = cc[variant]["window_days"]
                    evalmask = night.copy()
                    evalmask[:W] = True  # score only after the warm-up window
                    best = None
                    for v in cc[variant][grid_key]:
                        lo, up, _ = core.calibrate(g, level, side, variant, variant_params(variant, {key: v}))
                        s = score_for_tuning(lo, up, g.y, level, side, evalmask)
                        if best is None or s < best[1]:
                            best = (v, s)
                    r[key] = float(best[0])
                res[spec_key(level, side)] = r
            out.setdefault(m, {})[sid] = res
            log.info("tuned %s/%s", m, sid)
    return out


def clip_bounds(lower, upper, lo_clip, hi_clip):
    """Physical bounds: load >= 0; solar/wind in [0, capacity]. Open (infinite) sides stay open."""
    lower, upper = lower.copy(), upper.copy()
    for a in (lower, upper):
        f = np.isfinite(a)
        a[f] = np.maximum(a[f], lo_clip)
        if hi_clip is not None:
            a[f] = np.minimum(a[f], hi_clip)
    return lower, upper


def calibrate_series(model_dir, sid, tuned, variants, window=None, suffix=""):
    cc = config.load("calibration")
    fc = load_forecasts(model_dir, sid)
    if fc is None:
        return
    warm = pd.Timestamp(cc["warmup_start"])
    fc = fc[fc["origin_local"] >= warm]
    g = core.to_grid(fc, pd.date_range(warm, fc["origin_local"].max(), freq="D"))
    night = night_mask(sid, g)
    target = targets_of(sid)
    clip = cc["clip"]
    lo_clip = clip["load_lower"] if target == "load" else clip["gen_lower"]
    hi_clip = None if target == "load" else clip["gen_upper"]
    base = pd.DataFrame({
        "series_id": sid,
        "origin_local": np.repeat(g.days.values, 24),
        "target_local": g.target_local.ravel(),
        "horizon": np.tile(np.arange(1, 25), len(g.days)),
        "median": g.q["q50"].ravel(),
        "y_true": g.y.ravel(),
        "is_night": night.ravel(),
    })
    meta = {}
    for variant in variants:
        rows = []
        for level, side in specs(sid):
            t = tuned.get(spec_key(level, side), {})
            lo, up, info = core.calibrate(g, level, side, variant, variant_params(variant, t, window))
            meta[f"{variant}:{spec_key(level, side)}"] = {"capped": info["capped"],
                                                          **{k: v for k, v in t.items()}}
            d = base.copy()
            d["level"], d["side"] = level, side
            d["lower_raw"], d["upper_raw"] = lo.ravel(), up.ravel()
            d["lower"], d["upper"] = clip_bounds(lo.ravel(), up.ravel(), lo_clip, hi_clip)
            rows.append(d.dropna(subset=["y_true", "median"]))
        out = pd.concat(rows, ignore_index=True)
        dest = config.path(cc["out_dir"]) / model_dir.name / f"{variant}{suffix}" / f"{sid}.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(dest, index=False)
    mp = config.path(cc["out_dir"]) / model_dir.name / f"_meta{suffix}_{sid}.json"
    mp.write_text(json.dumps(meta, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", help="default: every directory in out/forecasts")
    ap.add_argument("--series", nargs="*")
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--window", type=int, default=None, help="ablation window W (days)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cc = config.load("calibration")
    root = config.path(cc["in_dir"])
    models = args.models or sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
    series = args.series or config.series_ids()
    if args.dry_run:
        for m in models:
            print(f"calibrate {m}: {series} x {args.variants or cc['variants']}")
        return
    tuned_path = config.path(cc["tuned_path"])
    if args.tune:
        new = tune(models, series)
        old = yaml.safe_load(tuned_path.read_text()) if tuned_path.exists() else {}
        for m, v in new.items():
            old.setdefault(m, {}).update(v)
        tuned_path.parent.mkdir(parents=True, exist_ok=True)
        tuned_path.write_text(yaml.safe_dump(old, sort_keys=True))
        return
    tuned = yaml.safe_load(tuned_path.read_text()) if tuned_path.exists() else {}
    variants = args.variants or cc["variants"]
    if args.window is not None:
        variants = [v for v in variants if v in ("split_cqr", "nexcp", "aci")]
    suffix = f"_w{args.window}" if args.window is not None else ""
    for m in models:
        for sid in series:
            if not tuned.get(m, {}).get(sid):
                log.warning("%s/%s: no tuned gamma/eta; run --tune first (using defaults)", m, sid)
            calibrate_series(root / m, sid, tuned.get(m, {}).get(sid, {}), variants, args.window, suffix)
            log.info("calibrated %s/%s", m, sid)


if __name__ == "__main__":
    main()
