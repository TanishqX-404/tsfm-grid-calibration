"""Tune LightGBM hyper-parameters on the validation period (Optuna, fixed trial budget).

Writes configs/tuned/lgbm_{series}[_cov].json. Never reads test data.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

from tgc import config, splits
from tgc.data import io, weather
from tgc.forecast import lgbm
from tgc.forecast.run import fit_frame

log = logging.getLogger("tgc.tune")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", nargs="*")
    ap.add_argument("--covariates", action="store_true")
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="2 trials, no file written")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    mc = config.load("models")["models"]["lgbm"]
    n_trials = 2 if args.dry_run else (args.n_trials or mc["n_trials"])
    val = splits.get_period("val")
    for sid in args.series or config.series_ids():
        out = lgbm.tuned_params_path(sid, args.covariates)
        if out.exists() and not (args.force or args.dry_run):
            log.info("%s exists, skipping", out)
            continue
        df = io.load_series(sid)
        _, target = config.split_series(sid)
        cov = weather.covariate_columns(target) if args.covariates else []
        va = df[val.contains(df["local_time"])]
        t0 = time.time()
        best, score = lgbm.tune(fit_frame(df), va, mc["lags"], cov, mc["fixed"], n_trials,
                                mc["tune_quantiles"])
        log.info("%s: val pinball %.5f in %.0fs  %s", sid, score, time.time() - t0, best)
        if not args.dry_run:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"params": best, "val_pinball": score, "n_trials": n_trials,
                                       "tune_quantiles": mc["tune_quantiles"]}, indent=2))


if __name__ == "__main__":
    main()
