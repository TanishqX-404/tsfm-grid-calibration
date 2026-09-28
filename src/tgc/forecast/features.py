"""Calendar and lag features for the tabular baseline. Lags >= 24 h are known at issue time
for every hour of day D, so the same features serve training and rolling-origin prediction."""
from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

_HOL = None


def holidays() -> pd.DatetimeIndex:
    global _HOL
    if _HOL is None:
        _HOL = USFederalHolidayCalendar().holidays("2015-01-01", "2030-12-31")
    return _HOL


def make_features(df: pd.DataFrame, lags, covariates=()) -> pd.DataFrame:
    """``df``: hourly rows (complete grid) with local_time, target, covariates."""
    df = df.sort_values("local_time").reset_index(drop=True)
    X = pd.DataFrame(index=df.index)
    for L in lags:
        X[f"lag{L}"] = df["target"].shift(L)
    t = df["local_time"]
    X["hour"] = t.dt.hour
    X["weekday"] = t.dt.weekday
    X["month"] = t.dt.month
    X["holiday"] = t.dt.floor("D").isin(holidays()).astype(int)
    for c in covariates:
        X[c] = df[c].values
    return X
