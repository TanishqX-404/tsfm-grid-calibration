"""Rolling-origin generator: one origin per day, context strictly before the origin."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

from tgc import splits


@dataclass
class Origin:
    series_id: str
    day: pd.Timestamp            # forecast day D (local standard, 00:00)
    context: pd.DataFrame        # rows with local_time in [D - context_days, D)
    future: pd.DataFrame         # the 24 rows of day D; ``target`` is the truth (never passed to models)


def rolling_origins(df: pd.DataFrame, days, context_days: int, drop_log: list | None = None,
                    check_days: int | None = None) -> Iterator[Origin]:
    """Yield origins for one series.

    ``df``: tidy rows of one series indexed by a complete hourly ``local_time`` grid.
    Days whose context or target contain NaN (gaps longer than the interpolation limit)
    are skipped and recorded in ``drop_log``. Only the last ``check_days`` (default: all) of the
    context are gap-checked, so a baseline that reads extra history (seasonal naive's residual
    window) drops exactly the same days as every other model.
    """
    df = df.sort_values("local_time").set_index("local_time", drop=False)
    sid = df["series_id"].iloc[0]
    tgt = df["target"]
    for day in pd.DatetimeIndex(days):
        c0, c1 = splits.context_window(day, context_days)
        ctx = df.loc[c0: c1 - pd.Timedelta(hours=1)]
        fut = df.loc[day: day + pd.Timedelta(hours=23)]
        reason = None
        if len(ctx) != context_days * 24 or len(fut) != 24:
            reason = "incomplete"
        elif ctx["target"].iloc[-(check_days or context_days) * 24:].isna().any():
            reason = "context_gap"
        elif fut["target"].isna().any():
            reason = "target_gap"
        if reason:
            if drop_log is not None:
                drop_log.append({"series_id": sid, "day": day, "reason": reason})
            continue
        assert ctx["local_time"].max() < day  # no look-ahead, by construction
        yield Origin(sid, day, ctx.reset_index(drop=True), fut.reset_index(drop=True))
