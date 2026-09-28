"""Open-Meteo weather covariates.

Test/validation periods use the *Historical Forecast* API (archived NWP runs), never
reanalysis, so covariates carry only information a day-ahead forecaster had. ERA5
(Historical Weather API) is used only before ``archived_forecast_start``.
Values are averaged over the configured grid points with the configured weights and
stored per region/target in data/weather/{region}_{target}.parquet (UTC).
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from tgc import config

log = logging.getLogger(__name__)


def _fetch(url: str, lat: float, lon: float, start: str, end: str, variables: list[str]) -> pd.DataFrame:
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "hourly": ",".join(variables), "timezone": "UTC", "wind_speed_unit": "ms",
    })
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{url}?{q}", timeout=120) as r:
                js = json.load(r)
            break
        except Exception as e:  # network hiccup / rate limit
            if attempt == 4:
                raise
            log.warning("retry %s (%s)", attempt + 1, e)
            time.sleep(2 ** (attempt + 1))
    h = js["hourly"]
    df = pd.DataFrame({v: h[v] for v in variables})
    df.index = pd.to_datetime(h["time"])
    return df


def _chunks(start: pd.Timestamp, end: pd.Timestamp, months: int = 12):
    s = start
    while s < end:
        e = min(s + pd.DateOffset(months=months), end)
        yield s, e - pd.Timedelta(days=1)
        s = e


def download(regions=None, force: bool = False) -> None:
    cfg = config.load("data")
    w = cfg["weather"]
    out_dir = config.path(cfg["paths"]["weather_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = pd.Timestamp(cfg["raw_start"])
    t1 = pd.Timestamp(cfg["raw_end"])
    split = pd.Timestamp(w["archived_forecast_start"])
    for region, per_target in w["points"].items():
        if regions and region not in regions:
            continue
        for target, pts in per_target.items():
            dest = out_dir / f"{region}_{target}.parquet"
            if dest.exists() and not force:
                continue
            variables = w["variables"][target]
            frames = []
            for url, a, b in [(w["reanalysis_url"], t0, split), (w["historical_forecast_url"], split, t1)]:
                if a >= b:
                    continue
                per_point = []
                for lat, lon, wt in pts:
                    parts = [_fetch(url, lat, lon, s.date().isoformat(), e.date().isoformat(), variables)
                             for s, e in _chunks(a, b)]
                    per_point.append((pd.concat(parts), wt))
                tot = sum(wt for _, wt in per_point)
                avg = sum(df * wt for df, wt in per_point) / tot
                avg["weather_source"] = "era5" if url == w["reanalysis_url"] else "archived_nwp"
                frames.append(avg)
            df = pd.concat(frames).sort_index()
            df = df[~df.index.duplicated(keep="last")]
            df.index.name = "timestamp_utc"
            df.reset_index().to_parquet(dest)
            log.info("wrote %s (%d rows)", dest, len(df))


def covariate_columns(target: str) -> list[str]:
    return list(config.load("data")["weather"]["variables"][target])


def load(region: str, target: str) -> pd.DataFrame | None:
    cfg = config.load("data")
    p = config.path(cfg["paths"]["weather_dir"]) / f"{region}_{target}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)
