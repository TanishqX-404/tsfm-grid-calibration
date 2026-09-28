"""Train / validation / test boundaries: the only source of dates in the code base.

All timestamps are *local standard time* (fixed UTC offset, no DST), stored as naive
pandas Timestamps. Day D's 24 targets are hours 00:00..23:00 of D (hour-beginning), so
horizon h = local hour + 1. The forecast for D is issued at the end of D-1 and its context
is the ``context_days`` days before D.

Fitting and tuning code must call :func:`assert_before_test` on its data; test dates can
only be obtained with ``allow_test=True``, which only the final run scripts pass.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from tgc import config


class LeakageError(RuntimeError):
    """Raised when code that must not see test data receives it."""


@dataclass(frozen=True)
class Period:
    name: str
    start: pd.Timestamp  # first day, 00:00
    end: pd.Timestamp    # last day, 00:00 (inclusive)

    @property
    def end_exclusive(self) -> pd.Timestamp:
        return self.end + pd.Timedelta(days=1)

    def days(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, self.end, freq="D")

    def contains(self, ts) -> pd.Series | bool:
        return (ts >= self.start) & (ts < self.end_exclusive)


def _periods() -> dict[str, Period]:
    s = config.load("data")["splits"]
    return {k: Period(k, pd.Timestamp(v[0]), pd.Timestamp(v[1])) for k, v in s.items()}


def get_period(name: str, allow_test: bool = False) -> Period:
    if name == "test" and not allow_test:
        raise LeakageError("test period requested without allow_test=True")
    if name == "all":
        p = _periods()
        if not allow_test:
            raise LeakageError("'all' includes the test period; pass allow_test=True")
        return Period("all", p["val"].start, p["test"].end)
    return _periods()[name]


def test_start() -> pd.Timestamp:
    return _periods()["test"].start


def val_start() -> pd.Timestamp:
    return _periods()["val"].start


def assert_before(ts, boundary: pd.Timestamp, what: str = "data") -> None:
    ts = pd.to_datetime(pd.Series(ts)) if not isinstance(ts, pd.Timestamp) else pd.Series([ts])
    if len(ts) and ts.max() >= boundary:
        raise LeakageError(f"{what} contains {ts.max()} >= boundary {boundary}")


def assert_before_test(ts, what: str = "data") -> None:
    """Guard for every fit/tune entry point: no timestamp may reach the test period."""
    assert_before(ts, test_start(), what)


def origin_days(phase: str, allow_test: bool = False) -> pd.DatetimeIndex:
    """Forecast days for a phase: 'val', 'test', or 'all' (= val + test)."""
    return get_period(phase, allow_test=allow_test).days()


def context_window(day: pd.Timestamp, context_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) of the context for forecast day ``day`` (end == day 00:00)."""
    return day - pd.Timedelta(days=context_days), day
