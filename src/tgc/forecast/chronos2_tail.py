"""Chronos-2 with its native tail quantiles (reviewer request R1.2), test period only.

    python -m tgc.forecast.chronos2_tail --allow-test

Same pinned checkpoint, 28-day context and dropped days as the main run, but requests the levels
0.05, 0.1, 0.5, 0.9, 0.95. Records the checkpoint's own training quantile grid, so the paper can
state whether 0.05/0.95 are native or interpolated. Writes out/forecasts_tail/chronos2/{series}.parquet
and out/forecasts_tail/chronos2/_meta.json (no change to the main forecast tables).
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from tgc import backtest, config, envlog, splits
from tgc.data import io
from tgc.forecast._fm import device

LEVELS = [0.05, 0.1, 0.5, 0.9, 0.95]
COLS = ["q05", "q10", "q50", "q90", "q95"]
log = logging.getLogger("tgc.chronos2_tail")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from chronos import Chronos2Pipeline
    mc = config.load("models")["models"]["chronos2"]
    ctx_days = config.load("models")["context_days"]
    pipe = Chronos2Pipeline.from_pretrained(mc["checkpoint"], revision=mc["revision"], device_map=device())
    train_q = [float(q) for q in pipe.quantiles]
    meta = {"checkpoint": mc["checkpoint"], "revision": mc["revision"], "training_quantiles": train_q,
            "requested": LEVELS, "requested_are_native": bool(set(LEVELS) <= set(round(q, 6) for q in train_q))}
    log.info("training quantiles: %s", train_q)
    days = splits.origin_days("test", allow_test=args.allow_test)
    if args.dry_run:
        days = days[:3]
    out = config.path(config.load("models")["out_dir"]).parent / "forecasts_tail" / "chronos2"
    out.mkdir(parents=True, exist_ok=True)
    for sid in config.series_ids():
        df = io.load_series(sid)
        origins = list(backtest.rolling_origins(df, days, ctx_days))
        rows = []
        for i in range(0, len(origins), args.batch_size):
            chunk = origins[i:i + args.batch_size]
            ctx = [o.context["target"].to_numpy(dtype=np.float32) for o in chunk]
            q, _ = pipe.predict_quantiles(ctx, prediction_length=24, quantile_levels=LEVELS,
                                          batch_size=args.batch_size)
            for o, t in zip(chunk, q):
                arr = np.sort(t.squeeze(0).float().cpu().numpy(), axis=1)  # (24, 5), non-crossing
                f = pd.DataFrame(arr, columns=COLS)
                f.insert(0, "series_id", sid)
                f.insert(1, "origin_local", o.day)
                f.insert(2, "target_local", o.future["local_time"].to_numpy())
                f.insert(3, "horizon", np.arange(1, 25))
                f["y_true"] = o.future["target"].to_numpy()
                rows.append(f)
        pd.concat(rows, ignore_index=True).to_parquet(out / f"{sid}.parquet", index=False)
        log.info("%s: %d origins", sid, len(origins))
    (out / "_meta.json").write_text(json.dumps(meta, indent=2))
    envlog.write(out / "_env.json", {"model": "chronos2_tail"})


if __name__ == "__main__":
    main()
