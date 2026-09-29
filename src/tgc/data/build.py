"""Build tidy per-region Parquet: data/clean/{region}.parquet (Section 4.4 cleaning rules).

Columns: timestamp_utc, local_time, local_hour, series_id, target (normalized), target_mw,
scale, is_imputed, is_night, operator_fc (load only, normalized), covariates...
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from tgc import config, splits
from tgc.data import eia930, solar, weather

log = logging.getLogger(__name__)


def fill_short_gaps(s: pd.Series, max_gap: int) -> tuple[pd.Series, int]:
    """Linearly interpolate NaN runs of length <= max_gap; longer runs stay NaN."""
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    fillable = isna & (run_len <= max_gap)
    interp = s.interpolate(method="linear", limit_area="inside")
    out = s.where(~fillable, interp)
    return out, int(fillable.sum())


def trailing_capacity(local_time: pd.Series, mw: pd.Series, window_days: int) -> pd.Series:
    """Capacity proxy: max over the previous ``window_days`` complete days (causal, per day)."""
    day = local_time.dt.floor("D")
    dmax = mw.groupby(day.values).max()
    cap = dmax.shift(1).rolling(window_days, min_periods=1).max()
    return pd.Series(day.map(cap).values, index=mw.index)


def remove_spikes(s: pd.Series, ratio: float) -> tuple[pd.Series, int]:
    """Flag isolated spikes (> ratio x trailing 7-day median, or <= 0) in load as missing."""
    med = s.shift(1).rolling(24 * 7, min_periods=24).median()
    bad = (s > ratio * med) | (s < med / ratio) | (s <= 0)
    return s.mask(bad), int(bad.sum())


def build_region(region: str) -> tuple[pd.DataFrame, dict]:
    cfg = config.load("data")
    rc = cfg["regions"][region]
    cl = cfg["cleaning"]
    raw = eia930.load_region(region)
    off = pd.Timedelta(hours=rc["utc_offset_hours"])
    t0, t1 = pd.Timestamp(cfg["raw_start"]), pd.Timestamp(cfg["raw_end"])
    grid = pd.date_range(t0, t1 - pd.Timedelta(hours=1), freq="h")  # local standard
    raw["local_time"] = raw["timestamp_utc"] + off
    raw = raw.drop_duplicates("local_time").set_index("local_time").reindex(grid)
    raw.index.name = "local_time"
    raw["timestamp_utc"] = raw.index - off
    raw = raw.reset_index()

    train = splits.get_period("train")
    frames, report = [], {}
    for target in rc["targets"]:
        sid = f"{region}_{target}"
        rep = {}
        mw = raw[f"{target}_mw"].astype(float)
        imp = raw[f"{target}_imputed"].fillna(False).astype(bool)
        rep["missing_raw"] = int(mw.isna().sum())
        if target == "load":
            mw, rep["spikes_removed"] = remove_spikes(mw, cl["outlier_ratio"])
        else:
            rep["negative_clipped"] = int((mw < 0).sum())
            mw = mw.clip(lower=0)
        mw, rep["interpolated"] = fill_short_gaps(mw, cl["max_interp_gap_hours"])
        rep["missing_after_fill"] = int(mw.isna().sum())

        if target == "load":
            in_train = train.contains(raw["local_time"])
            splits.assert_before_test(raw.loc[in_train, "local_time"], "load scale")
            scale = pd.Series(float(mw[in_train].mean()), index=mw.index)
            night = np.zeros(len(mw), dtype=bool)
        else:
            scale = trailing_capacity(raw["local_time"], mw, cl["capacity_window_days"])
            if target == "solar":
                night = solar.empirical_night(raw["local_time"], mw, scale)
                pts = cfg["weather"]["points"][region]["solar"]
                geo = solar.elevation_night(raw["timestamp_utc"], pts, cl["night_elevation_deg"])
                rep["night_hours"] = int(night.sum())
                # share of generation that a pure elevation mask would have zeroed
                rep["energy_share_in_geometric_night"] = float(mw[geo].sum() / mw.sum())
                mw = mw.where(~night | mw.isna(), 0.0)
            else:
                night = np.zeros(len(mw), dtype=bool)
        df = pd.DataFrame({
            "timestamp_utc": raw["timestamp_utc"],
            "local_time": raw["local_time"],
            "local_hour": raw["local_time"].dt.hour,
            "series_id": sid,
            "target": mw / scale,
            "target_mw": mw,
            "scale": scale,
            "is_imputed": imp,
            "is_night": night,
        })
        if target == "load":
            df["operator_fc"] = raw["operator_fc_mw"] / scale
        wx = weather.load(region, target)
        if wx is not None:
            wx = wx.set_index("timestamp_utc")
            cols = weather.covariate_columns(target)
            df = df.join(wx[cols], on="timestamp_utc")
            for c in cols:
                df[c], _ = fill_short_gaps(df[c], cl["max_interp_gap_hours"])
            rep["weather"] = True
        else:
            rep["weather"] = False
        report[sid] = rep
        frames.append(df)
    return pd.concat(frames, ignore_index=True), report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regions", nargs="*")
    ap.add_argument("--download", action="store_true", help="download raw EIA-930 first")
    ap.add_argument("--weather", action="store_true", help="download Open-Meteo covariates first")
    ap.add_argument("--dry-run", action="store_true", help="print what would be built")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = config.load("data")
    regions = args.regions or list(cfg["regions"])
    if args.dry_run:
        for r in regions:
            print(f"build data/clean/{r}.parquet")
        return
    if args.download:
        eia930.download()
    if args.weather:
        weather.download(regions)
    out_dir = config.path(cfg["paths"]["clean_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    full = {}
    for r in regions:
        df, rep = build_region(r)
        df.to_parquet(out_dir / f"{r}.parquet", index=False)
        full.update(rep)
        log.info("wrote %s: %d rows", out_dir / f"{r}.parquet", len(df))
    rdir = config.path(cfg["paths"]["report_dir"])
    rdir.mkdir(parents=True, exist_ok=True)
    src = config.path(cfg["paths"]["raw_dir"]) / "_source.json"
    if src.exists():
        full["_source"] = json.loads(src.read_text())
    (rdir / "build_report.json").write_text(json.dumps(full, indent=2))
    (out_dir / "_source.json").write_text(json.dumps(full.get("_source", {}), indent=2))


if __name__ == "__main__":
    main()
