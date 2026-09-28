"""Data report (M1 acceptance): row counts, gaps, dropped forecast days, integrity checks,
and a one-week spot plot per series."""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tgc import backtest, config, splits
from tgc.data import io


def gap_runs(s: pd.Series) -> list[int]:
    isna = s.isna()
    rid = (isna != isna.shift()).cumsum()
    return [int(n) for n in isna.groupby(rid).sum() if n > 0]


def series_report(sid: str, allow_test: bool) -> dict:
    df = io.load_series(sid)
    ctx_days = config.load("models")["context_days"]
    rep = {
        "rows": len(df),
        "first": str(df["local_time"].min()), "last": str(df["local_time"].max()),
        "duplicate_timestamps": int(df["timestamp_utc"].duplicated().sum()),
        "utc_monotonic": bool(df["timestamp_utc"].is_monotonic_increasing),
        "hourly_regular": bool((df["timestamp_utc"].diff().dropna() == pd.Timedelta(hours=1)).all()),
        "missing_hours": int(df["target"].isna().sum()),
        "gap_runs": gap_runs(df["target"])[:20],
        "imputed_hours": int(df["is_imputed"].sum()),
        "night_hours": int(df["is_night"].sum()),
    }
    for phase in ["train", "val"] + (["test"] if allow_test else []):
        p = splits.get_period(phase, allow_test=allow_test)
        drops: list = []
        n = sum(1 for _ in backtest.rolling_origins(df, p.days(), ctx_days, drops))
        rep[f"{phase}_origins"] = n
        rep[f"{phase}_dropped_days"] = len(drops)
        m = p.contains(df["local_time"])
        rep[f"{phase}_imputed_share"] = float(df.loc[m, "is_imputed"].mean())
    return rep


def spot_plot(sids, out, week_start="2025-06-02"):
    fig, axes = plt.subplots(len(sids), 1, figsize=(8, 1.6 * len(sids)), sharex=True)
    t0 = pd.Timestamp(week_start)
    for ax, sid in zip(np.atleast_1d(axes), sids):
        df = io.load_series(sid)
        w = df[(df.local_time >= t0) & (df.local_time < t0 + pd.Timedelta(days=7))]
        ax.plot(w.local_time, w.target, lw=1)
        if "operator_fc" in w:
            ax.plot(w.local_time, w.operator_fc, lw=0.8, ls="--")
        ax.set_ylabel(sid, fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args(argv)
    cfg = config.load("data")
    rdir = config.path(cfg["paths"]["report_dir"])
    rdir.mkdir(parents=True, exist_ok=True)
    sids = config.series_ids()
    rep = {sid: series_report(sid, args.allow_test) for sid in sids}
    (rdir / "data_report.json").write_text(json.dumps(rep, indent=2))
    tab = pd.DataFrame(rep).T
    cols = [c for c in tab.columns if c not in ("gap_runs", "first", "last")]
    print(tab[cols].to_string())
    spot_plot(sids, rdir / "spot_week.png")
    ok = all(r["duplicate_timestamps"] == 0 and r["utc_monotonic"] and r["hourly_regular"] for r in rep.values())
    print("integrity:", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
