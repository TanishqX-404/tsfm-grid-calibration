"""Tables T1-T6 (Section 10.2) as CSV and LaTeX, from saved outputs only (deterministic).

    python -m tgc.plots.tables --phase test
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
import yaml

from tgc import config, splits
from tgc.plots.style import LABELS, VLABELS, family, primary

log = logging.getLogger("tgc.tables")


def write(df: pd.DataFrame, out, name, bold_min=None):
    out.mkdir(parents=True, exist_ok=True)
    df = df.round(4)
    df.to_csv(out / f"{name}.csv", index=False, float_format="%.4f", lineterminator="\n")
    tex = df.copy()
    for col in bold_min or []:
        if col in tex:
            best = tex[col].min()
            tex[col] = [f"\\textbf{{{v:.4f}}}" if v == best else f"{v:.4f}" for v in tex[col]]
    (out / f"{name}.tex").write_text(tex.to_latex(index=False, escape=False, float_format="%.4f"))


def t1_data(out, phase):
    rep_path = config.path(config.load("data")["paths"]["report_dir"]) / "data_report.json"
    if not rep_path.exists():
        return
    rep = json.loads(rep_path.read_text())
    from tgc.data import io
    from tgc import splits
    rows = []
    for sid, r in rep.items():
        region, target = config.split_series(sid)
        df = io.load_series(sid)
        tr = splits.get_period("train")
        n_orig = sum(r.get(f"{p}_origins", 0) + r.get(f"{p}_dropped_days", 0) for p in ("train", "val", "test"))
        n_drop = sum(r.get(f"{p}_dropped_days", 0) for p in ("train", "val", "test"))
        rows.append({"series": sid, "region": config.load("data")["regions"][region]["name"], "target": target,
                     "first": r["first"][:10], "last": r["last"][:10],
                     "scale_MW": float(df.loc[tr.contains(df["local_time"]), "scale"].mean()),
                     "dropped_day_share": n_drop / max(n_orig, 1), "imputed_hours": r["imputed_hours"],
                     **{f"{p}_dropped_days": r.get(f"{p}_dropped_days") for p in ("train", "val", "test")},
                     "test_origins": r.get("test_origins")})
    write(pd.DataFrame(rows), out, "T1_data")


def t2_accuracy(acc, dm, out):
    d = acc[acc["model"].map(primary)].copy()
    d["model"] = d["model"].map(lambda m: LABELS.get(family(m), m))
    # mean +- sd over seeds for trained baselines
    seeds = acc[acc["model"].str.contains(r"-s\d") & ~acc["model"].str.contains("-cov|-ctx")].copy()
    if not seeds.empty:
        seeds["fam"] = seeds["model"].map(family)
        sd = seeds.groupby(["fam", "series_id"])["nmae"].std().rename("nmae_sd_seeds").reset_index()
        sd["model"] = sd["fam"].map(lambda f: LABELS.get(f, f))
        d = d.merge(sd[["model", "series_id", "nmae_sd_seeds"]], on=["model", "series_id"], how="left")
    cols = ["model", "series_id", "nmae", "pinball", "native80_coverage"] + (["nmae_sd_seeds"] if "nmae_sd_seeds" in d else [])
    d = d[cols].sort_values(["series_id", "nmae"])
    if dm is not None and not dm.empty:
        # mark models significantly worse (Holm p < 0.05) than the best-pinball model per series
        best = acc[acc["model"].map(primary)].sort_values("pinball").groupby("series_id").head(1).set_index("series_id")["model"]
        sig = {}
        for _, r in dm.iterrows():
            b = best.get(r["series_id"])
            other = r["model_b"] if r["model_a"] == b else r["model_a"] if r["model_b"] == b else None
            if other is not None:
                sig[(r["series_id"], LABELS.get(family(other), other))] = r["p_holm"] < 0.05
        d["dm_worse_than_best"] = [sig.get((s, m), False) for s, m in zip(d["series_id"], d["model"])]
    write(d, out, "T2_accuracy")


def t3_decisions(res, bat, out, tau, res_sel=None, bat_sel=None, seeds=None, period=None):
    """Per model: calibration variant (reserve) and variant + interval level (battery) are
    *selected on validation* (``res_sel``/``bat_sel``) and reported on the evaluated period, so the
    test table carries no post-hoc choice. Also writes the full battery table per level."""
    if res is None or res.empty:
        return
    res_sel = res if res_sel is None else res_sel
    bat_sel = bat if bat_sel is None else bat_sel

    def pick(df, keys, col, how):
        idx = df.groupby(keys)[col].idxmin() if how == "min" else df.groupby(keys)[col].idxmax()
        return df.loc[idx]

    r_sel = res_sel[res_sel["model"].map(primary) & np.isclose(res_sel["tau"], tau)
                    & ~res_sel["variant"].str.contains("_w") & (res_sel["variant"] != "deterministic")]
    chosen = pick(r_sel, ["model", "series_id"], "regret", "min")[["model", "series_id", "variant"]]
    r = res[np.isclose(res["tau"], tau)]
    best = chosen.merge(r, on=["model", "series_id", "variant"], how="left")
    det = r[r["variant"] == "deterministic"][["model", "series_id", "regret"]].rename(columns={"regret": "regret_deterministic"})
    best = best.merge(det, on=["model", "series_id"], how="left")
    best = best[["model", "series_id", "variant", "cost", "regret", "regret_ci_lo", "regret_ci_hi", "shortfall_freq",
                 "regret_deterministic"]]
    # illustrative dollars: normalized MWh/day x mean scale (MW) x one stated reserve price
    usd = config.load("decisions")["reserve"]["illustrative_prices"]["c_R_usd_per_mwh"]
    best["cost_usd_per_day"] = best["cost"] * best["series_id"].map(_mean_scale) * usd
    if seeds is not None and not seeds.empty:
        sd = seeds.rename(columns={"regret_sd": "regret_sd_seeds"})[["family", "series_id", "regret_sd_seeds"]]
        best = best.assign(family=best["model"].map(family)).merge(sd, on=["family", "series_id"], how="left").drop(columns="family")
    if bat is not None and not bat.empty:
        b_sel = bat_sel[bat_sel["model"].map(primary) & ~bat_sel["variant"].str.contains("_w")]
        bc = pick(b_sel, ["model", "series_id"], "pct_oracle", "max")[["model", "series_id", "variant", "level"]]
        bb = bc.merge(bat, on=["model", "series_id", "variant", "level"], how="left")
        bb = bb[["model", "series_id", "variant", "level", "pct_oracle", "shortfall_freq"]].rename(columns={
            "variant": "battery_variant", "level": "battery_level", "shortfall_freq": "battery_shortfall_freq"})
        best = best.merge(bb, on=["model", "series_id"], how="left")
        best["battery_variant"] = best["battery_variant"].map(lambda v: VLABELS.get(v, v) if isinstance(v, str) else v)
    best["model"] = best["model"].map(lambda m: LABELS.get(family(m), m))
    best["variant"] = best["variant"].map(lambda v: VLABELS.get(v, v))
    write(best.sort_values(["series_id", "regret"]), out, "T3_decisions")


_SCALE = {}


def _mean_scale(sid):
    """Mean normalizer (MW) of a series over the evaluated period, for illustrative dollars."""
    if sid not in _SCALE:
        from tgc.data import io
        df = io.load_series(sid)
        m = _PERIOD.contains(df["local_time"]) if _PERIOD is not None else slice(None)
        _SCALE[sid] = float(df.loc[m, "scale"].mean())
    return _SCALE[sid]


_PERIOD = None


def t7_reserve_diffs(diffs, out):
    """Pooled paired-bootstrap differences in daily reserve cost: every model vs the cheapest, and
    each model's validation-selected calibration vs its native intervals."""
    if diffs is None or diffs.empty:
        return
    p = diffs[(diffs["series_id"] == "pooled") & (diffs["metric"] == "cost")].copy()
    m = p[p["comparison"] == "model"]
    means = {}
    for r in m.itertuples():
        means.setdefault(r.a, []).append(r.mean_diff)
        means.setdefault(r.b, []).append(-r.mean_diff)
    ref = min(means, key=lambda k: np.mean(means[k]))
    rows = []
    for r in m.itertuples():
        if ref in (r.a, r.b):
            other, sign = (r.b, 1) if r.a == ref else (r.a, -1)
            lo, hi = sorted((sign * r.ci_lo, sign * r.ci_hi))
            rows.append({"comparison": f"{LABELS.get(family(ref), ref)} - {LABELS.get(family(other), other)}",
                         "mean_diff": sign * r.mean_diff, "ci_lo": lo, "ci_hi": hi, "significant": r.significant})
    for r in p[p["comparison"] == "calibrated_vs_native"].itertuples():
        mdl = r.a.split(":")[0]
        rows.append({"comparison": f"{LABELS.get(family(mdl), mdl)}: calibrated - native",
                     "mean_diff": r.mean_diff, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "significant": r.significant})
    write(pd.DataFrame(rows), out, "T7_reserve_cost_differences")


def t4_calibration(cal, out):
    d = cal[(cal["side"] == "two") & np.isclose(cal["level"], 0.9)]
    cols = ["model", "variant", "series_id", "coverage", "gap_pp", "width", "winkler", "worst_hour_gap_pp",
            "kupiec_p", "christoffersen_p_median", "coverage_unclipped"]
    write(d[cols].sort_values(["series_id", "model", "variant"]), out, "T4_calibration")


def t5_leakage(out):
    a = yaml.safe_load(config.path("configs/leakage_audit.yaml").read_text())
    rows = [{"model": LABELS.get(m, m), **v, "released_before_test": str(v["release"])[:7] < a["test_start"][:7]}
            for m, v in a["models"].items()]
    write(pd.DataFrame(rows), out, "T5_leakage")


def t6_compute(comp, out):
    if comp is None or comp.empty:
        return
    g = comp.groupby("model").agg(gpu=("gpu", "first"), fit_seconds=("fit_seconds", "sum"),
                                  predict_seconds=("predict_seconds", "sum"), origins=("origins", "sum"))
    g["seconds_per_series_304_origins"] = g["predict_seconds"] / g["origins"] * 304
    write(g.reset_index(), out, "T6_compute")


def read(p):
    return pd.read_csv(p) if p.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", default="val", choices=["val", "test"])
    args = ap.parse_args(argv)
    ev = config.load("evaluate")
    mdir = config.path(ev["out_dir"]) / args.phase
    out = config.path(ev["tables_dir"]) / args.phase
    t1_data(out, args.phase)
    acc, dm = read(mdir / "accuracy.csv"), read(mdir / "dm_tests.csv")
    if acc is not None:
        t2_accuracy(acc, dm, out)
    vdir = config.path(ev["out_dir"]) / "val"  # selections are always made on validation
    global _PERIOD
    _PERIOD = splits.get_period(args.phase, allow_test=True)
    t3_decisions(read(mdir / "decisions_reserve.csv"), read(mdir / "decisions_battery.csv"), out, ev["headline_tau"],
                 read(vdir / "decisions_reserve.csv"), read(vdir / "decisions_battery.csv"), read(mdir / "seed_spread.csv"))
    t7_reserve_diffs(read(mdir / "reserve_diffs.csv"), out)
    cal = read(mdir / "calibration.csv")
    if cal is not None:
        t4_calibration(cal, out)
    t5_leakage(out)
    t6_compute(read(mdir / "compute.csv"), out)


if __name__ == "__main__":
    main()
