"""Rolling-origin forecasting CLI with chunked, resumable writes.

    python -m tgc.forecast.run --model chronos2 --phase all --allow-test
    python -m tgc.forecast.run --model lgbm --seed 0 --phase val
    python -m tgc.forecast.run --model tirex --dry-run          # 3 days per series

Output: out/forecasts/{name}/{series}/{yyyymm}.parquet in the Section 5.4 schema.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

from tgc import backtest, config, envlog, splits
from tgc.data import io
from tgc.forecast import registry
from tgc.forecast.base import FORECAST_COLUMNS, QCOLS, sort_quantiles, validate

log = logging.getLogger("tgc.forecast")


def out_root(dry_run: bool):
    root = config.path(config.load("models")["out_dir"])
    return root.parent / "dry_run" / root.name if dry_run else root


def fit_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Training rows (plus one week of lag warm-up) - never beyond the training period."""
    tr = splits.get_period("train")
    m = (df["local_time"] >= tr.start - pd.Timedelta(days=7)) & (df["local_time"] < tr.end_exclusive)
    out = df[m].copy()
    splits.assert_before_test(out["local_time"], "fit frame")
    return out


def run_series(model_key, sid, days, name, seed, covariates, context_days, dry_run, force, batch_days):
    df = io.load_series(sid)
    fc = registry.build(model_key, sid, seed=seed, covariates=covariates, context_days=context_days)
    if covariates:
        missing = [c for c in fc.covariates if c not in df.columns]
        assert not missing, f"{sid}: covariates {missing} not in clean data (run make data with --weather)"
    out_dir = out_root(dry_run) / name / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    months = pd.Series(pd.DatetimeIndex(days)).dt.to_period("M")
    todo = {str(m): pd.DatetimeIndex(days)[(months == m).values] for m in months.unique()}
    todo = {m: d for m, d in todo.items()
            if force or not (out_dir / f"{m.replace('-', '')}.parquet").exists()}
    if not todo:
        log.info("%s/%s: all chunks exist, skipping", name, sid)
        return
    t_fit = time.time()
    fc.fit(fit_frame(df))
    t_fit = time.time() - t_fit
    fut_cols = ["local_time"] + [c for c in ("operator_fc", *getattr(fc, "covariates", [])) if c in df.columns]
    drops, n_cross, t_pred, n_orig = [], 0, 0.0, 0
    for month, mdays in todo.items():
        check = fc.gap_check_days or fc.context_days
        origins = list(backtest.rolling_origins(df, mdays, fc.context_days, drops, check_days=check))
        if model_key == "operator":
            ok = [o for o in origins if o.future["operator_fc"].notna().all()]
            drops += [{"series_id": sid, "day": o.day, "reason": "no_operator_fc"} for o in origins if o not in ok]
            origins = ok
        if not origins:
            continue
        rows = []
        for i in range(0, len(origins), batch_days):
            chunk = origins[i:i + batch_days]
            items = [(o.context, o.future[fut_cols]) for o in chunk]
            t0 = time.time()
            preds = fc.predict_batch(items)
            t_pred += time.time() - t0
            for o, p in zip(chunk, preds):
                p.insert(0, "series_id", sid)
                p.insert(1, "origin_local", o.day)
                p["y_true"] = o.future["target"].to_numpy()
                p["model"] = name
                p["model_revision"] = str(fc.revision)
                rows.append(p)
        out = pd.concat(rows, ignore_index=True)
        out, c = sort_quantiles(out)
        n_cross += c
        out = out[FORECAST_COLUMNS]
        validate(out)
        out.to_parquet(out_dir / f"{month.replace('-', '')}.parquet", index=False)
        n_orig += len(origins)
        log.info("%s/%s %s: %d origins", name, sid, month, len(origins))
    meta = {"fit_seconds": t_fit, "predict_seconds": t_pred, "origins": n_orig,
            "crossing_rows_sorted": n_cross, "dropped": [{**d, "day": str(d["day"])} for d in drops]}
    mp = out_dir / "_meta.json"
    old = json.loads(mp.read_text()) if mp.exists() else {}
    for k in ("fit_seconds", "predict_seconds", "origins", "crossing_rows_sorted"):
        meta[k] += old.get(k, 0) if k != "fit_seconds" else 0
    mp.write_text(json.dumps(meta, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=registry.model_names())
    ap.add_argument("--series", nargs="*", help="default: all series the model applies to")
    ap.add_argument("--phase", default="val", choices=["val", "test", "all"])
    ap.add_argument("--allow-test", action="store_true", help="required for phase test/all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--covariates", action="store_true")
    ap.add_argument("--context-days", type=int, default=None)
    ap.add_argument("--batch-days", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true", help="3 origins per series into out/dry_run")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    days = splits.origin_days(args.phase, allow_test=args.allow_test)
    if args.dry_run:
        days = days[: config.load("models")["dry_run_days"]]
    name = registry.output_name(args.model, args.seed, args.covariates, args.context_days)
    sids = args.series or [s for s in config.series_ids() if registry.applies_to(args.model, s)]
    envlog.write(out_root(args.dry_run).parent / "env" / f"{name}.json",
                 {"model": args.model, "phase": args.phase, "series": sids})
    for sid in sids:
        run_series(args.model, sid, days, name, args.seed, args.covariates, args.context_days,
                   args.dry_run, args.force, args.batch_days)


if __name__ == "__main__":
    main()
