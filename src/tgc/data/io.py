"""Readers for the clean per-region tables."""
from __future__ import annotations

import pandas as pd

from tgc import config


def load_series(series_id: str) -> pd.DataFrame:
    region, _ = config.split_series(series_id)
    p = config.path(config.load("data")["paths"]["clean_dir"]) / f"{region}.parquet"
    df = pd.read_parquet(p)
    df = df[df["series_id"] == series_id].sort_values("local_time").reset_index(drop=True)
    return df.dropna(axis=1, how="all")
