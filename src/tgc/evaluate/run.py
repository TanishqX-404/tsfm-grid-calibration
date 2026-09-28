"""Evaluation CLI: metrics, statistical tests and decision summaries for one phase.

    python -m tgc.evaluate.run --phase val
    python -m tgc.evaluate.run --phase test --allow-test

Writes long-format CSVs to out/metrics/{phase}/: accuracy, reliability, calibration,
decisions_reserve, decisions_battery, dm_tests, compute.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging

import numpy as np
import pandas as pd

from tgc import config, splits
from tgc.data import io
from tgc.evaluate import metrics as M
from tgc.evaluate import tests_stat as S
from tgc.forecast.base import QCOLS, QUANTILES

log = logging.getLogger("tgc.evaluate")


def load_dir(p):
    files = sorted(p.glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else None


def in_phase(df, col, period):
    return df[period.contains(df[col])]


def night_lookup(sid):
    return io.load_series(sid).set_index("local_time")["is_night"]


def mase_denominator(sid):
    """In-sample seasonal-naive MAE on the training period (lag as in the naive baseline)."""
    df = io.load_series(sid)
    tr = splits.get_period("train")
    y = df.loc[tr.contains(df["local_time"]), "target"].to_numpy()
    lag = config.load("models")["models"]["seasonal_naive"]["lag_hours"][config.split_series(sid)[1]]
    return M.naive_mae(y, lag)


def accuracy(period):
    root = config.path(config.load("models")["out_dir"])
    acc, rel, daily = [], [], {}
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for sdir in sorted(p for p in mdir.iterdir() if p.is_dir()):
            sid = sdir.name
            fc = load_dir(sdir)
            if fc is None:
                continue
            fc = in_phase(fc, "origin_local", period)
            if fc.empty:
                continue
            if sid.endswith("solar"):
                fc = fc[~night_lookup(sid).reindex(fc["target_local"]).fillna(False).to_numpy()]
            y, Q = fc["y_true"].to_numpy(), fc[QCOLS].to_numpy()
            med = fc["q50"].to_numpy()
            row = {"model": mdir.name, "series_id": sid, "n_days": fc["origin_local"].nunique(),
                   "nmae": M.nmae(y, med), "nrmse": M.nrmse(y, med),
                   "mase": M.mase(y, med, mase_denominator(sid)), "pinball": M.mean_pinball(y, Q),
                   "native80_coverage": M.picp(y, fc["q10"].to_numpy(), fc["q90"].to_numpy()),
                   "native80_width": M.width(fc["q10"].to_numpy(), fc["q90"].to_numpy())}
            acc.append(row)
            for t, f in M.reliability(y, Q).items():
                rel.append({"model": mdir.name, "series_id": sid, "tau": t, "observed": f})
            pl = np.mean([M.pinball(y, Q[:, i], t) for i, t in enumerate(QUANTILES)], axis=0)
            daily[(mdir.name, sid)] = pd.Series(pl).groupby(fc["origin_local"].to_numpy()).mean()
    return pd.DataFrame(acc), pd.DataFrame(rel), daily


def dm_tests(daily: dict):
    rows = []
    sids = sorted({s for _, s in daily})
    for sid in sids:
        models = sorted(m for m, s in daily if s == sid)
        pairs = list(itertools.combinations(models, 2))
        res = []
        for a, b in pairs:
            la, lb = daily[(a, sid)], daily[(b, sid)]
            common = la.index.intersection(lb.index)
            stat, p = S.diebold_mariano(la[common].to_numpy(), lb[common].to_numpy())
            res.append({"series_id": sid, "model_a": a, "model_b": b, "n_days": len(common), "dm_stat": stat, "p": p})
        if res:
            adj = S.holm([r["p"] for r in res])
            for r, pa in zip(res, adj):
                r["p_holm"] = pa
            rows += res
    return pd.DataFrame(rows)


def calibration(period):
    root = config.path(config.load("calibration")["out_dir"])
    rows = []
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for vdir in sorted(p for p in mdir.iterdir() if p.is_dir()):
            for f in sorted(vdir.glob("*.parquet")):
                sid = f.stem
                d = in_phase(pd.read_parquet(f), "origin_local", period)
                if sid.endswith("solar"):
                    d = d[~d["is_night"]]
                for (level, side), g in d.groupby(["level", "side"]):
                    y, lo, up = g["y_true"].to_numpy(), g["lower"].to_numpy(), g["upper"].to_numpy()
                    miss = ((y < lo) | (y > up)).astype(int)
                    cov = 1 - miss.mean()
                    r = {"model": mdir.name, "variant": vdir.name, "series_id": sid, "level": level, "side": side,
                         "coverage": cov, "gap_pp": 100 * (cov - level),
                         "coverage_unclipped": 1 - ((y < g["lower_raw"]) | (y > g["upper_raw"])).mean(),
                         "kupiec_p": S.kupiec_pof(miss, 1 - level)}
                    # independence on each horizon's daily miss sequence; median p across horizons
                    ps = [S.christoffersen_ind(gg.sort_values("origin_local").pipe(
                        lambda x: ((x.y_true < x.lower) | (x.y_true > x.upper)).astype(int).to_numpy()))
                        for _, gg in g.groupby("horizon")]
                    r["christoffersen_p_median"] = float(np.nanmedian(ps)) if np.isfinite(ps).any() else np.nan
                    if side == "two":
                        hours = g["horizon"].to_numpy()
                        r.update(width=M.width(lo, up), winkler=M.winkler(y, lo, up, 1 - level),
                                 worst_hour_gap_pp=M.worst_hour_gap(y, lo, up, hours, level))
                    else:
                        tau = level if side == "upper" else 1 - level
                        b = up if side == "upper" else lo
                        r.update(pinball=float(M.pinball(y, b, tau).mean()),
                                 distance=float(np.mean(np.abs(b - g["median"].to_numpy()))))
                    rows.append(r)
    return pd.DataFrame(rows)


def decisions(period, phase_dir):
    ev = config.load("evaluate")
    bs = ev["bootstrap"]
    rrows, brows = [], []
    for f in sorted((phase_dir / "reserve").glob("*.parquet")):
        d = pd.read_parquet(f)
        if d.empty:
            continue
        d = in_phase(d, "day", period)
        stem = f.stem
        sid = "_".join(stem.split("_")[-2:])
        model, variant = split_model_variant(stem, sid)
        for ratio, g in d.groupby("ratio"):
            lo, hi = S.block_bootstrap_ci(g["regret"].to_numpy(), bs["block_days"], bs["n_resamples"], bs["seed"])
            rrows.append({"model": model, "variant": variant, "series_id": sid, "ratio": ratio,
                          "tau": g["tau"].iloc[0], "n_days": len(g), "cost": g["cost"].mean(),
                          "regret": g["regret"].mean(), "regret_ci_lo": lo, "regret_ci_hi": hi,
                          "oracle_cost": g["oracle_cost"].mean(), "reserve": g["reserve"].mean(),
                          "shortfall_freq": g["short_hours"].sum() / g["hours"].sum(),
                          "ens": g["shortfall"].mean()})
    for f in sorted((phase_dir / "battery").glob("*.parquet")):
        d = pd.read_parquet(f)
        if d.empty:
            continue
        d = in_phase(d, "day", period)
        stem = f.stem
        sid = "_".join(stem.split("_")[-2:])
        model, variant = split_model_variant(stem, sid)
        for level, g in d.groupby("level"):
            hrs = 24 * len(g)
            brows.append({"model": model, "variant": variant, "series_id": sid, "level": level,
                          "n_days": len(g), "profit": g["profit"].mean(), "oracle_profit": g["oracle_profit"].mean(),
                          "pct_oracle": 100 * g["profit"].sum() / g["oracle_profit"].sum(),
                          "shortfall_freq": g["short_hours"].sum() / hrs, "short_energy": g["short_energy"].mean(),
                          "committed": g["committed"].mean()})
    return pd.DataFrame(rrows), pd.DataFrame(brows)


def split_model_variant(stem: str, sid: str):
    variants = sorted({p.name for p in config.path(config.load("calibration")["out_dir"]).glob("*/*") if p.is_dir()}
                      | {"deterministic"}, key=len, reverse=True)
    for v in variants:
        if stem.endswith(f"_{v}_{sid}"):
            return stem[: -len(f"_{v}_{sid}")], v
    raise ValueError(stem)


def compute_table():
    root = config.path(config.load("models")["out_dir"])
    rows = []
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        env = root.parent / "env" / f"{mdir.name}.json"
        gpu = json.loads(env.read_text()).get("gpu") if env.exists() else None
        for meta in sorted(mdir.glob("*/_meta.json")):
            m = json.loads(meta.read_text())
            rows.append({"model": mdir.name, "series_id": meta.parent.name, "gpu": gpu,
                         "fit_seconds": m["fit_seconds"], "predict_seconds": m["predict_seconds"],
                         "origins": m["origins"], "dropped_days": len(m["dropped"]),
                         "crossing_rows_sorted": m["crossing_rows_sorted"]})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="val", choices=["val", "test"])
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    period = splits.get_period(args.phase, allow_test=args.allow_test)
    out = config.path(config.load("evaluate")["out_dir"]) / args.phase
    out.mkdir(parents=True, exist_ok=True)
    acc, rel, daily = accuracy(period)
    acc.round(6).to_csv(out / "accuracy.csv", index=False)
    rel.round(6).to_csv(out / "reliability.csv", index=False)
    dm_tests(daily).round(6).to_csv(out / "dm_tests.csv", index=False)
    log.info("accuracy: %d rows", len(acc))
    if config.path(config.load("calibration")["out_dir"]).exists():
        calibration(period).round(6).to_csv(out / "calibration.csv", index=False)
    dec_dir = config.path(config.load("decisions")["out_dir"]) / args.phase
    if dec_dir.exists():
        r, b = decisions(period, dec_dir)
        r.round(6).to_csv(out / "decisions_reserve.csv", index=False)
        b.round(6).to_csv(out / "decisions_battery.csv", index=False)
    compute_table().round(3).to_csv(out / "compute.csv", index=False)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
