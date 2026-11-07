"""Accuracy by lead-time group (reviewer request R2), test period.

    python -m tgc.evaluate.horizon_accuracy --allow-test

The forecast is issued at midnight, so horizon h is a lead time of h hours; hours 13-24 are the
closest to a market day-ahead lead time. LightGBM and seasonal naive use lags >= 24 h only, whereas
the TSFMs, N-HiTS and PatchTST see the context up to midnight. Writes
out/metrics/test/horizon_accuracy.csv (nMAE per model x series x horizon group; solar daytime only).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tgc import config, splits
from tgc.calibrate.run import load_forecasts
from tgc.data import io

MODELS = ["seasonal_naive", "lgbm-s0", "nhits-s0", "patchtst-s0", "chronos2", "moirai2", "timesfm25", "tirex", "operator"]
GROUPS = {"h1-6": (1, 6), "h1-12": (1, 12), "h13-24": (13, 24), "all": (1, 24)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args(argv)
    period = splits.get_period("test", allow_test=args.allow_test)
    root = config.path(config.load("models")["out_dir"])
    rows = []
    for m in MODELS:
        for sid in config.series_ids():
            fc = load_forecasts(root / m, sid)
            if fc is None:
                continue
            fc = fc[period.contains(fc["origin_local"])]
            if sid.endswith("solar"):
                night = io.load_series(sid).set_index("local_time")["is_night"]
                fc = fc[~night.reindex(fc["target_local"]).fillna(False).to_numpy()]
            for g, (lo, hi) in GROUPS.items():
                x = fc[fc["horizon"].between(lo, hi)]
                rows.append({"model": m, "series_id": sid, "group": g, "n": len(x),
                             "nmae": float(np.mean(np.abs(x["y_true"] - x["q50"])))})
    out = config.path(config.load("evaluate")["out_dir"]) / "test"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(6).to_csv(out / "horizon_accuracy.csv", index=False)
    print("wrote", out / "horizon_accuracy.csv")


if __name__ == "__main__":
    main()
