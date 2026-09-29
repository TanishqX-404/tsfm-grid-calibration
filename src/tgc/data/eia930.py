"""Download and extract EIA-930 hourly series from PUDL's pre-cleaned Parquet tables."""
from __future__ import annotations

import logging
import urllib.request

import pandas as pd

from tgc import config

log = logging.getLogger(__name__)


def raw_path(table_key: str):
    cfg = config.load("data")
    return config.path(cfg["paths"]["raw_dir"]) / f"{cfg['pudl']['tables'][table_key]}.parquet"


def download(force: bool = False) -> None:
    cfg = config.load("data")
    for key, table in cfg["pudl"]["tables"].items():
        dest = raw_path(key)
        if dest.exists() and not force:
            log.info("exists: %s", dest)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{cfg['pudl']['base_url']}/{table}.parquet"
        log.info("downloading %s", url)
        tmp = dest.with_suffix(".part")
        _, headers = urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
        record_source(key, url, headers.get("Last-Modified"))


def record_source(key: str, url: str, last_modified: str | None) -> None:
    """Keep the PUDL nightly version (Last-Modified) of each raw table for the data report."""
    import json
    p = raw_path(key).parent / "_source.json"
    meta = json.loads(p.read_text()) if p.exists() else {}
    meta[key] = {"url": url, "last_modified": last_modified}
    p.write_text(json.dumps(meta, indent=2))


def load_region(region: str) -> pd.DataFrame:
    """Hourly UTC frame for one balancing authority.

    Columns: timestamp_utc, load_mw, load_imputed, operator_fc_mw, solar_mw, solar_imputed,
    wind_mw, wind_imputed. Uses the EIA 'adjusted' columns; ``*_imputed`` marks hours EIA imputed.
    Solar and wind sum the plain, with- and without-integrated-storage categories (EIA split the
    category in 2024).
    """
    cfg = config.load("data")
    t0, t1 = pd.Timestamp(cfg["raw_start"]), pd.Timestamp(cfg["raw_end"])
    # widen by a day so the local-time window is fully covered
    f = [("balancing_authority_code_eia", "==", region),
         ("datetime_utc", ">=", t0 - pd.Timedelta(days=1)), ("datetime_utc", "<", t1 + pd.Timedelta(days=1))]
    ops = pd.read_parquet(raw_path("operations"), filters=f)
    ops = ops.groupby("datetime_utc").agg(
        load_mw=("demand_adjusted_mwh", "first"),
        load_imputed=("demand_imputed_eia_mwh", lambda s: s.notna().any()),
        operator_fc_mw=("demand_forecast_mwh", "first"),
    )
    gen = pd.read_parquet(raw_path("generation"), filters=f)
    gen["src"] = gen["generation_energy_source"].astype(str)
    out = ops
    for tech in ("solar", "wind"):
        g = gen[gen["src"].isin([tech, f"{tech}_w_integrated_battery_storage",
                                 f"{tech}_wo_integrated_battery_storage"])]
        agg = g.groupby("datetime_utc").agg(
            mw=("net_generation_adjusted_mwh", lambda s: s.sum(min_count=1)),
            imp=("net_generation_imputed_eia_mwh", lambda s: s.notna().any()),
        )
        out = out.join(agg.rename(columns={"mw": f"{tech}_mw", "imp": f"{tech}_imputed"}), how="outer")
    out.index.name = "timestamp_utc"
    return out.reset_index()
