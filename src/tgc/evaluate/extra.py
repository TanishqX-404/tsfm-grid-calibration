"""Supplementary analyses (Section 8.3 / 9 follow-ups), from saved outputs only.

    python -m tgc.evaluate.extra --phase test --allow-test

Writes to out/metrics/{phase}/:
  reserve_diffs.csv       paired moving-block bootstrap CIs for daily reserve cost and regret
                          differences (model vs model; calibrated vs native), per series and pooled
  seed_spread.csv         reserve regret mean / sd over seeds for trained baselines
  imputed_sensitivity.csv accuracy and 90% coverage with and without EIA-imputed hours
  aci_capping.csv         share of test days on which ACI's alpha_t <= 0 (interval capped)

Every selection (calibration variant per model and series) is taken from the *validation*
decision results, so nothing here is chosen on the evaluated period.
"""
from __future__ import annotations

import argparse
import itertools
import logging

import numpy as np
import pandas as pd
import yaml

from tgc import config, splits
from tgc.calibrate import core
from tgc.calibrate.run import load_forecasts, spec_key, variant_params
from tgc.data import io
from tgc.evaluate import metrics as M
from tgc.evaluate import tests_stat as S
from tgc.plots.style import family, primary

log = logging.getLogger("tgc.extra")
VARIANTS = ("native", "split_cqr", "nexcp", "aci", "pid")


def paired_ci(diff: pd.Series, bs: dict):
    lo, hi = S.block_bootstrap_ci(diff.to_numpy(), bs["block_days"], bs["n_resamples"], bs["seed"])
    return float(diff.mean()), lo, hi


def selected_variants(tau: float) -> pd.DataFrame:
    """Per (model, series): the calibration variant with the lowest *validation* regret at tau."""
    v = pd.read_csv(config.path(config.load("evaluate")["out_dir"]) / "val" / "decisions_reserve.csv")
    v = v[np.isclose(v["tau"], tau) & v["variant"].isin(VARIANTS)]
    return v.loc[v.groupby(["model", "series_id"])["regret"].idxmin(), ["model", "series_id", "variant"]]


def daily(phase_dir, model, variant, sid, tau, period):
    p = phase_dir / "reserve" / f"{model}_{variant}_{sid}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    d = d[np.isclose(d["tau"], tau) & period.contains(d["day"])]
    return d.set_index("day")[["cost", "regret"]]


def reserve_diffs(period, phase_dir, tau, bs):
    sel = selected_variants(tau)
    sel = sel[sel["model"].map(primary) & (sel["model"] != "operator")]
    series = sorted(sel["series_id"].unique())
    runs = {(r.model, r.series_id): daily(phase_dir, r.model, r.variant, r.series_id, tau, period)
            for r in sel.itertuples()}
    rows = []
    models = sorted(sel["model"].unique())
    for a, b in itertools.permutations(models, 2):
        if a > b:
            continue
        per = {}
        for sid in series:
            da, db = runs.get((a, sid)), runs.get((b, sid))
            if da is None or db is None:
                continue
            per[sid] = da.join(db, lsuffix="_a", rsuffix="_b", how="inner")
        for metric in ("cost", "regret"):
            pooled = []
            for sid, j in per.items():
                diff = j[f"{metric}_a"] - j[f"{metric}_b"]
                m, lo, hi = paired_ci(diff, bs)
                rows.append({"comparison": "model", "a": a, "b": b, "series_id": sid, "metric": metric,
                             "n_days": len(diff), "mean_diff": m, "ci_lo": lo, "ci_hi": hi})
                pooled.append(diff.rename(sid))
            if pooled:  # mean over series of the daily difference (normalized units, comparable)
                diff = pd.concat(pooled, axis=1).mean(axis=1)
                m, lo, hi = paired_ci(diff, bs)
                rows.append({"comparison": "model", "a": a, "b": b, "series_id": "pooled", "metric": metric,
                             "n_days": len(diff), "mean_diff": m, "ci_lo": lo, "ci_hi": hi})
    # calibrated (validation-selected variant) vs native, per model
    for model in models:
        pooled = []
        for sid in series:
            v = sel[(sel["model"] == model) & (sel["series_id"] == sid)]
            if v.empty:
                continue
            dc = runs.get((model, sid))
            dn = daily(phase_dir, model, "native", sid, tau, period)
            if dc is None or dn is None:
                continue
            j = dc.join(dn, lsuffix="_cal", rsuffix="_nat", how="inner")
            diff = j["cost_cal"] - j["cost_nat"]
            m, lo, hi = paired_ci(diff, bs)
            rows.append({"comparison": "calibrated_vs_native", "a": f"{model}:{v['variant'].iloc[0]}", "b": f"{model}:native",
                         "series_id": sid, "metric": "cost", "n_days": len(diff), "mean_diff": m, "ci_lo": lo, "ci_hi": hi})
            pooled.append(diff.rename(sid))
        if pooled:
            diff = pd.concat(pooled, axis=1).mean(axis=1)
            m, lo, hi = paired_ci(diff, bs)
            rows.append({"comparison": "calibrated_vs_native", "a": f"{model}:selected", "b": f"{model}:native",
                         "series_id": "pooled", "metric": "cost", "n_days": len(diff), "mean_diff": m, "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(rows)
    out["significant"] = (out["ci_lo"] > 0) | (out["ci_hi"] < 0)
    return out


def seed_spread(period, phase_dir, tau):
    v = pd.read_csv(config.path(config.load("evaluate")["out_dir"]) / "val" / "decisions_reserve.csv")
    v = v[np.isclose(v["tau"], tau) & v["variant"].isin(VARIANTS) & v["model"].str.contains(r"-s\d")]
    sel = v.loc[v.groupby(["model", "series_id"])["regret"].idxmin(), ["model", "series_id", "variant"]]
    rows = []
    for r in sel.itertuples():
        d = daily(phase_dir, r.model, r.variant, r.series_id, tau, period)
        if d is not None:
            rows.append({"family": family(r.model), "model": r.model, "series_id": r.series_id,
                         "cost": d["cost"].mean(), "regret": d["regret"].mean()})
    d = pd.DataFrame(rows)
    return d.groupby(["family", "series_id"]).agg(n_seeds=("model", "nunique"), regret_mean=("regret", "mean"),
                                                  regret_sd=("regret", "std"), cost_mean=("cost", "mean"),
                                                  cost_sd=("cost", "std")).reset_index()


def imputed_sensitivity(period):
    fc_root = config.path(config.load("models")["out_dir"])
    cal_root = config.path(config.load("calibration")["out_dir"])
    rows = []
    for sid in config.series_ids():
        s = io.load_series(sid).set_index("local_time")
        for mdir in sorted(p for p in fc_root.iterdir() if p.is_dir() and primary(p.name)):
            fc = load_forecasts(mdir, sid)
            if fc is None:
                continue
            fc = fc[period.contains(fc["origin_local"])]
            if fc.empty:
                continue
            imp = s["is_imputed"].reindex(fc["target_local"]).fillna(False).to_numpy()
            night = s["is_night"].reindex(fc["target_local"]).fillna(False).to_numpy()
            keep = ~night
            r = {"model": mdir.name, "series_id": sid, "imputed_hours": int((imp & keep).sum()),
                 "nmae_all": M.nmae(fc["y_true"].to_numpy()[keep], fc["q50"].to_numpy()[keep]),
                 "nmae_no_imputed": M.nmae(fc["y_true"].to_numpy()[keep & ~imp], fc["q50"].to_numpy()[keep & ~imp])}
            cp = cal_root / mdir.name / "aci" / f"{sid}.parquet"
            if cp.exists():
                c = pd.read_parquet(cp)
                c = c[(c["side"] == "two") & np.isclose(c["level"], 0.9) & period.contains(c["origin_local"])
                      & ~c["is_night"] & c["lower"].notna()]
                ci = s["is_imputed"].reindex(c["target_local"]).fillna(False).to_numpy()
                hit = ((c["y_true"] >= c["lower"]) & (c["y_true"] <= c["upper"])).to_numpy()
                r.update(aci90_cov_all=hit.mean(), aci90_cov_no_imputed=hit[~ci].mean())
            rows.append(r)
    return pd.DataFrame(rows)


def aci_capping(period, level=0.9):
    cc = config.load("calibration")
    tuned_p = config.path(cc["tuned_path"])
    tuned = yaml.safe_load(tuned_p.read_text()) if tuned_p.exists() else {}
    root = config.path(cc["in_dir"])
    warm = pd.Timestamp(cc["warmup_start"])
    rows = []
    for mdir in sorted(p for p in root.iterdir() if p.is_dir() and primary(p.name)):
        for sid in config.series_ids():
            fc = load_forecasts(mdir, sid)
            if fc is None:
                continue
            fc = fc[fc["origin_local"] >= warm]
            g = core.to_grid(fc, pd.date_range(warm, fc["origin_local"].max(), freq="D"))
            t = tuned.get(mdir.name, {}).get(sid, {}).get(spec_key(level, "two"), {})
            _, _, S_ = core.base_and_scores(g, level, "two")
            _, info = core.correction(S_, 1 - level, "aci", variant_params("aci", t))
            a = info["alpha_path"]
            m = period.contains(pd.Series(g.days)).to_numpy()
            rows.append({"model": mdir.name, "series_id": sid, "gamma": t.get("aci_gamma"),
                         "capped_share": float(np.mean(a[m] <= 0)), "alpha_min": float(np.nanmin(a[m])),
                         "alpha_max": float(np.nanmax(a[m]))})
    return pd.DataFrame(rows)


FMS = ("chronos2", "moirai2", "timesfm25", "tirex")


def context_ablation(period, phase_dir, tau, contexts=(7, 28, 90)):
    """Foundation models at 7/28/90 days of context, scored on the days every run covers
    (a longer context drops more days around data gaps). Calibration fixed to ACI."""
    fc_root = config.path(config.load("models")["out_dir"])
    cal_root = config.path(config.load("calibration")["out_dir"])
    base = config.load("models")["context_days"]
    name = lambda m, c: m if c == base else f"{m}-ctx{c}"
    rows = []
    for sid in config.series_ids():
        runs = {(m, c): load_forecasts(fc_root / name(m, c), sid) for m in FMS for c in contexts}
        if any(v is None for v in runs.values()):
            continue
        runs = {k: v[period.contains(v["origin_local"])] for k, v in runs.items()}
        common = set.intersection(*(set(v["origin_local"]) for v in runs.values()))
        night = io.load_series(sid).set_index("local_time")["is_night"]
        for (m, c), fc in runs.items():
            fc = fc[fc["origin_local"].isin(common)]
            fc = fc[~night.reindex(fc["target_local"]).fillna(False).to_numpy()]
            r = {"model": m, "context_days": c, "series_id": sid, "n_days": len(common),
                 "nmae": M.nmae(fc["y_true"].to_numpy(), fc["q50"].to_numpy()),
                 "pinball": M.mean_pinball(fc["y_true"].to_numpy(), fc[list(M.QCOLS)].to_numpy()),
                 "native80_coverage": M.picp(fc["y_true"].to_numpy(), fc["q10"].to_numpy(), fc["q90"].to_numpy())}
            cp = cal_root / name(m, c) / "aci" / f"{sid}.parquet"
            if cp.exists():
                cal = pd.read_parquet(cp)
                cal = cal[(cal["side"] == "two") & np.isclose(cal["level"], 0.9) & cal["origin_local"].isin(common)
                          & ~cal["is_night"]]
                r["aci90_coverage"] = M.picp(cal["y_true"].to_numpy(), cal["lower"].to_numpy(), cal["upper"].to_numpy())
                r["aci90_width"] = M.width(cal["lower"].to_numpy(), cal["upper"].to_numpy())
            d = daily(phase_dir, name(m, c), "aci", sid, tau, period)
            if d is not None:
                d = d[d.index.isin(common)]
                r["reserve_cost_aci"] = float(d["cost"].mean())
            rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="val", choices=["val", "test"])
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ev = config.load("evaluate")
    period = splits.get_period(args.phase, allow_test=args.allow_test)
    out = config.path(ev["out_dir"]) / args.phase
    phase_dir = config.path(config.load("decisions")["out_dir"]) / args.phase
    tau = ev["headline_tau"]
    reserve_diffs(period, phase_dir, tau, ev["bootstrap"]).round(6).to_csv(out / "reserve_diffs.csv", index=False)
    log.info("reserve_diffs done")
    seed_spread(period, phase_dir, tau).round(6).to_csv(out / "seed_spread.csv", index=False)
    imputed_sensitivity(period).round(6).to_csv(out / "imputed_sensitivity.csv", index=False)
    log.info("imputed done")
    aci_capping(period).round(6).to_csv(out / "aci_capping.csv", index=False)
    context_ablation(period, phase_dir, tau).round(6).to_csv(out / "context_ablation.csv", index=False)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
