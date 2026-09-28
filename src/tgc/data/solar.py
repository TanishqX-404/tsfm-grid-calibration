"""Solar geometry (NOAA approximation) and the night mask."""
from __future__ import annotations

import numpy as np
import pandas as pd


def solar_elevation(ts_utc: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """Solar elevation angle in degrees at UTC timestamps (NOAA low-precision formulas)."""
    ts = pd.DatetimeIndex(ts_utc)
    doy = ts.dayofyear.values
    hour = ts.hour.values + ts.minute.values / 60.0
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = hour * 60 + eqtime + 4 * lon  # true solar time, minutes
    ha = np.deg2rad(tst / 4 - 180)
    phi = np.deg2rad(lat)
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    return 90 - np.rad2deg(np.arccos(np.clip(cosz, -1, 1)))


def elevation_night(ts_utc: pd.DatetimeIndex, points, threshold_deg: float = 0.0) -> np.ndarray:
    """Night if the mid-hour elevation is below ``threshold_deg`` at every point."""
    mid = pd.DatetimeIndex(ts_utc) + pd.Timedelta(minutes=30)
    el = np.max([solar_elevation(mid, p[0], p[1]) for p in points], axis=0)
    return el < threshold_deg


def empirical_night(local_time: pd.Series, value: pd.Series, scale: pd.Series,
                    window_days: int = 28, frac: float = 0.01) -> np.ndarray:
    """Causal, data-driven night mask.

    Hour-of-day h is night on day D when, over the previous ``window_days`` days, generation at
    hour h never exceeded ``frac`` x capacity. Robust to the timestamp shifts present in some
    EIA-930 submissions (CISO 2022-2025), where a geometric mask would zero real generation.
    """
    df = pd.DataFrame({"t": local_time.values, "v": value.values, "s": scale.values})
    df["day"] = df["t"].dt.floor("D")
    df["h"] = df["t"].dt.hour
    wide = df.pivot_table(index="day", columns="h", values="v", aggfunc="first")
    prev_max = wide.shift(1).rolling(window_days, min_periods=7).max()
    m = prev_max.stack(future_stack=True).rename("pm").reset_index()
    df = df.merge(m, on=["day", "h"], how="left")
    night = (df["pm"] <= frac * df["s"]).fillna(False).values
    return night
