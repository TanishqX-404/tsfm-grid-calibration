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

from tgc import config
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
                     "dropped_day_share": n_drop / max(n_orig, 1), "imputed_hours": r["imputed_hours"]})
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


def t3_decisions(res, bat, out, tau, res_sel=None, bat_sel=None):
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
    best = best[["model", "series_id", "variant", "regret", "regret_ci_lo", "regret_ci_hi", "shortfall_freq",
                 "regret_deterministic"]]
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
    t3_decisions(read(mdir / "decisions_reserve.csv"), read(mdir / "decisions_battery.csv"), out, ev["headline_tau"],
                 read(vdir / "decisions_reserve.csv"), read(vdir / "decisions_battery.csv"))
    cal = read(mdir / "calibration.csv")
    if cal is not None:
        t4_calibration(cal, out)
    t5_leakage(out)
    t6_compute(read(mdir / "compute.csv"), out)


if __name__ == "__main__":
    main()
