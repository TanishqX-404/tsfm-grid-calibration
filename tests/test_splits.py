import numpy as np
import pandas as pd
import pytest

from tgc import backtest, splits


def synthetic_series(start="2024-10-01", days=200, gap_at=None, gap_len=0):
    t = pd.date_range(start, periods=days * 24, freq="h")
    y = np.sin(np.arange(len(t)) * 2 * np.pi / 24) + 2
    df = pd.DataFrame({"local_time": t, "target": y, "series_id": "X_load"})
    if gap_at is not None:
        i = t.get_loc(pd.Timestamp(gap_at))
        df.loc[i:i + gap_len - 1, "target"] = np.nan
    return df


def test_periods_are_ordered_and_disjoint():
    tr, va = splits.get_period("train"), splits.get_period("val")
    te = splits.get_period("test", allow_test=True)
    assert tr.end < va.start and va.end < te.start
    assert len(te.days()) == 304


def test_test_period_requires_explicit_permission():
    with pytest.raises(splits.LeakageError):
        splits.get_period("test")
    with pytest.raises(splits.LeakageError):
        splits.origin_days("all")


def test_tuning_guard_rejects_test_dates():
    splits.assert_before_test(pd.Series(pd.to_datetime(["2025-10-31 23:00"])))
    with pytest.raises(splits.LeakageError):
        splits.assert_before_test(pd.Series(pd.to_datetime(["2025-10-31 23:00", "2025-11-01 00:00"])))


def test_no_context_reaches_origin():
    df = synthetic_series()
    days = pd.date_range("2024-11-15", "2025-01-10", freq="D")
    n = 0
    for o in backtest.rolling_origins(df, days, 28):
        assert o.context["local_time"].max() < o.day
        assert o.context["local_time"].max() == o.day - pd.Timedelta(hours=1)
        assert len(o.context) == 28 * 24
        assert o.future["local_time"].min() == o.day and len(o.future) == 24
        n += 1
    assert n == len(days)


def test_long_gaps_drop_days_and_are_logged():
    df = synthetic_series(gap_at="2024-12-01 05:00", gap_len=5)
    days = pd.date_range("2024-11-20", "2025-01-10", freq="D")
    drops = []
    kept = [o.day for o in backtest.rolling_origins(df, days, 28, drops)]
    assert pd.Timestamp("2024-12-01") not in kept            # target gap
    assert pd.Timestamp("2024-12-20") not in kept            # context gap
    assert pd.Timestamp("2024-12-31") in kept                # gap left the 28-day window
    assert {d["reason"] for d in drops} == {"context_gap", "target_gap"}
    assert len(kept) + len(drops) == len(days)
