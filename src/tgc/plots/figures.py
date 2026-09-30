"""Figures F1-F9 (Section 10.1), rebuilt from saved outputs only.

    python -m tgc.plots.figures --phase test
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from tgc import config, splits
from tgc.plots import style
from tgc.plots.style import COLORS, LABELS, MARKERS, VLABELS, family, primary, plt

log = logging.getLogger("tgc.figures")


def target_of(sid):
    return config.split_series(sid)[1]


def f1_pipeline(out):
    fig, ax = plt.subplots(figsize=(style.WIDTH, 0.9))
    stages = ["EIA-930\n+ weather", "Forecasters\n(4 TSFM + baselines)", "Calibration\n(5 variants)",
              "Decisions\n(reserve, battery)", "Evaluation"]
    for i, s in enumerate(stages):
        x = i * 1.0
        ax.add_patch(plt.Rectangle((x, 0), 0.82, 0.8, fc="#f2f2f2", ec="#333333", lw=0.6))
        ax.text(x + 0.41, 0.4, s, ha="center", va="center", fontsize=5.5)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + 1.0, 0.4), xytext=(x + 0.82, 0.4), arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlim(-0.05, len(stages) - 0.1)
    ax.set_ylim(-0.05, 0.85)
    ax.axis("off")
    style.save(fig, out / "F1_pipeline")


def f2_coverage_width(cal, out, level=0.9):
    d = cal[(cal["side"] == "two") & np.isclose(cal["level"], level) & cal["model"].map(primary)
            & ~cal["variant"].str.contains("_w")]
    if d.empty:
        return
    d = d.assign(target=d["series_id"].map(target_of), fam=d["model"].map(family))
    g = d.groupby(["target", "fam", "variant"])[["coverage", "width"]].mean().reset_index()
    targets = [t for t in ("load", "solar", "wind") if t in set(g["target"])]
    fig, axes = plt.subplots(1, len(targets), figsize=(style.WIDTH * 2, 2.0), squeeze=False)
    for ax, t in zip(axes[0], targets):
        for _, r in g[g["target"] == t].iterrows():
            ax.scatter(r["width"], r["coverage"], c=COLORS.get(r["fam"], "k"), marker=MARKERS.get(r["variant"], "o"),
                       s=18, edgecolors="none")
        ax.axhline(level, ls="--", lw=0.7, c="k")
        ax.set_xlabel(f"{int(level * 100)}% width (normalized)")
        ax.set_title(t, fontsize=8)
    axes[0][0].set_ylabel("Empirical coverage")
    _legend(fig, g["fam"].unique(), g["variant"].unique())
    style.save(fig, out / "F2_coverage_width")


def _legend(fig, fams, variants):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", c=COLORS[f], label=LABELS.get(f, f)) for f in style.ORDER if f in set(fams)]
    h += [Line2D([], [], ls="", marker=MARKERS[v], c="grey", label=VLABELS.get(v, v)) for v in MARKERS if v in set(variants)]
    fig.tight_layout()
    fig.legend(handles=h, loc="upper center", ncol=min(len(h), 6), frameon=False, bbox_to_anchor=(0.5, 0.0))


def f3_battery_frontier(bat, out):
    d = bat[bat["model"].map(primary) & ~bat["variant"].str.contains("_w")]
    if d.empty:
        return
    d = d.assign(fam=d["model"].map(family))
    g = d.groupby(["fam", "variant", "level"])[["shortfall_freq", "pct_oracle"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(style.WIDTH, 2.6))
    for (f, v), gg in g.groupby(["fam", "variant"]):
        gg = gg.sort_values("level")
        ax.plot(gg["shortfall_freq"], gg["pct_oracle"], c=COLORS.get(f, "k"), lw=0.5, alpha=0.7)
        ax.scatter(gg["shortfall_freq"], gg["pct_oracle"], c=COLORS.get(f, "k"), marker=MARKERS.get(v, "o"),
                   s=8, edgecolors="none")
    ax.set_xlabel("Shortfall frequency (share of hours)")
    ax.set_ylabel("Profit (% of oracle)")
    _legend(fig, g["fam"].unique(), g["variant"].unique())
    style.save(fig, out / "F3_battery_frontier")


def f4_rank_slope(acc, res, out, tau=0.95):
    a = acc[acc["model"].map(primary)].assign(fam=lambda x: x["model"].map(family))
    r = res[res["model"].map(primary) & np.isclose(res["tau"], tau) & (res["variant"] != "deterministic")]
    if a.empty or r.empty:
        return
    r = r.assign(fam=r["model"].map(family)).groupby(["fam", "series_id"])["regret"].min().reset_index()
    # only models forecasting every series (the operator covers load only)
    fams = sorted(f for f in set(a["fam"]) & set(r["fam"]) if a[a["fam"] == f]["series_id"].nunique() == a["series_id"].nunique())
    ra = a[a["fam"].isin(fams)].pivot(index="series_id", columns="fam", values="nmae").rank(axis=1).mean()
    rr = r[r["fam"].isin(fams)].pivot(index="series_id", columns="fam", values="regret").rank(axis=1).mean()
    fig, ax = plt.subplots(figsize=(style.WIDTH, 2.4))
    for f in fams:
        ax.plot([0, 1], [ra[f], rr[f]], c=COLORS.get(f, "k"), marker="o", ms=3, lw=1)
    for x, vals, ha in ((-0.04, ra, "right"), (1.04, rr, "left")):
        for f, y in _spread(vals[fams]).items():
            ax.text(x, y, LABELS.get(f, f), ha=ha, va="center", fontsize=6)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1], ["Rank by nMAE", f"Rank by regret (tau*={tau})"])
    ax.invert_yaxis()
    ax.set_ylabel("Mean rank across series")
    style.save(fig, out / "F4_rank_slope")


def _spread(vals, gap=0.28):
    """Label positions: keep order, push apart labels closer than ``gap`` (rank units)."""
    v = vals.sort_values()
    out, last = {}, -np.inf
    for f, y in v.items():
        y = max(y, last + gap)
        out[f] = last = y
    return pd.Series(out)


def f5_reliability(rel, out):
    d = rel[rel["model"].map(primary)].assign(fam=lambda x: x["model"].map(family),
                                              target=lambda x: x["series_id"].map(target_of))
    d = d[d["fam"] != "operator"]
    if d.empty:
        return
    targets = [t for t in ("load", "solar", "wind") if t in set(d["target"])]
    fig, axes = plt.subplots(1, len(targets), figsize=(style.WIDTH * 2, 2.0), squeeze=False)
    for ax, t in zip(axes[0], targets):
        g = d[d["target"] == t].groupby(["fam", "tau"])["observed"].mean().reset_index()
        for f, gg in g.groupby("fam"):
            ax.plot(gg["tau"], gg["observed"], c=COLORS.get(f, "k"), marker="o", ms=2, lw=0.8, label=LABELS.get(f, f))
        ax.plot([0, 1], [0, 1], ls="--", c="k", lw=0.6)
        ax.set_xlabel("Nominal quantile level")
        ax.set_title(t, fontsize=8)
    axes[0][0].set_ylabel("Observed frequency")
    axes[0][-1].legend(frameon=False, fontsize=6)
    style.save(fig, out / "F5_reliability")


def _cal_file(model, variant, sid):
    return config.path(config.load("calibration")["out_dir"]) / model / variant / f"{sid}.parquet"


def pick_model(cal, prefer=("chronos2", "timesfm25", "tirex", "moirai2", "lgbm-s0", "seasonal_naive")):
    have = set(cal["model"]) if cal is not None else set()
    return next((m for m in prefer if m in have), None)


def f6_rolling_coverage(model, period, out, sid="ERCO_wind", level=0.9, days=30):
    fig, ax = plt.subplots(figsize=(style.WIDTH, 1.8))
    any_ = False
    for v in ("native", "split_cqr", "aci"):
        p = _cal_file(model, v, sid)
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        d = d[(d["side"] == "two") & np.isclose(d["level"], level) & period.contains(d["origin_local"])]
        hit = ((d["y_true"] >= d["lower"]) & (d["y_true"] <= d["upper"])).groupby(d["origin_local"]).mean()
        roll = hit.rolling(days, min_periods=days).mean()
        ax.plot(roll.index, roll.values, marker=MARKERS[v], markevery=30, ms=3, lw=0.8, label=VLABELS[v],
                c=COLORS.get(family(model), "k"), alpha={"native": 0.4, "split_cqr": 0.7, "aci": 1.0}[v])
        any_ = True
    if not any_:
        plt.close(fig)
        return
    ax.axhline(level, ls="--", c="k", lw=0.6)
    ax.set_ylabel(f"{days}-day rolling coverage")
    ax.legend(frameon=False, fontsize=6)
    fig.autofmt_xdate()
    style.save(fig, out / "F6_rolling_coverage")


def f7_hour_heatmap(cal_rows_fn, model, period, out, sid="ERCO_wind", level=0.9):
    rows, names = [], []
    for v in ("native", "split_cqr", "nexcp", "aci", "pid"):
        p = _cal_file(model, v, sid)
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        d = d[(d["side"] == "two") & np.isclose(d["level"], level) & period.contains(d["origin_local"])]
        cov = ((d["y_true"] >= d["lower"]) & (d["y_true"] <= d["upper"])).groupby(d["horizon"]).mean()
        rows.append(100 * (cov.reindex(range(1, 25)).to_numpy() - level))
        names.append(VLABELS[v])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(style.WIDTH, 1.4))
    lim = max(10, np.nanmax(np.abs(rows)))
    im = ax.imshow(np.array(rows), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(len(names)), names)
    ax.set_xticks(range(0, 24, 3), [str(h) for h in range(0, 24, 3)])
    ax.set_xlabel("Hour of day (local standard time)")
    fig.colorbar(im, ax=ax, label="Coverage gap (pp)")
    style.save(fig, out / "F7_hour_heatmap")


def f8_fan(model, period, out, sid="ERCO_wind", variant="split_cqr"):
    p = _cal_file(model, variant, sid)
    if not p.exists():
        return
    d = pd.read_parquet(p)
    d = d[(d["side"] == "two") & period.contains(d["origin_local"])]
    if d.empty:
        return
    t0 = d["origin_local"].min() + pd.Timedelta(days=60)
    d = d[(d["target_local"] >= t0) & (d["target_local"] < t0 + pd.Timedelta(days=7))]
    fig, ax = plt.subplots(figsize=(style.WIDTH, 1.6))
    c = COLORS.get(family(model), "k")
    for lvl, a in ((0.9, 0.2), (0.8, 0.35)):
        g = d[np.isclose(d["level"], lvl)].sort_values("target_local")
        ax.fill_between(g["target_local"], g["lower"], g["upper"], color=c, alpha=a, lw=0, label=f"{int(lvl * 100)}%")
    g = d[np.isclose(d["level"], 0.9)].sort_values("target_local")
    ax.plot(g["target_local"], g["median"], c=c, lw=0.8, label="Median")
    ax.plot(g["target_local"], g["y_true"], c="k", lw=0.8, label="Observed")
    ax.set_ylabel("Wind (normalized)")
    ax.legend(frameon=False, fontsize=6, ncol=4)
    fig.autofmt_xdate()
    style.save(fig, out / "F8_fan")


def f9_ablations(acc, cal, out, ctx_tab=None):
    panels = []
    if acc is not None and acc["model"].str.contains("-cov").any():
        panels.append("cov")
    if ctx_tab is not None and not ctx_tab.empty:
        panels.append("ctx")
    if cal is not None and cal["variant"].str.contains("_w").any():
        panels.append("win")
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(style.WIDTH * len(panels) / 1.5, 2.0), squeeze=False)
    fig.subplots_adjust(wspace=0.45)
    for ax, p in zip(axes[0], panels):
        if p == "cov":
            d = acc.assign(fam=acc["model"].map(family), cov=acc["model"].str.contains("-cov"),
                           target=acc["series_id"].map(target_of))
            d = d[d["model"].str.contains("-cov") | d["model"].map(primary)]
            g = d.groupby(["fam", "target", "cov"])["nmae"].mean().unstack("cov").dropna()
            for (f, t), r in g.iterrows():
                ax.plot([0, 1], [r[False], r[True]], c=COLORS.get(f, "k"), marker="o", ms=2, lw=0.7)
            ax.set_xticks([0, 1], ["no weather", "weather"])
            ax.set_ylabel("nMAE")
        elif p == "ctx":  # common-day table from evaluate.extra
            g = ctx_tab.groupby(["model", "context_days"])["nmae"].mean().reset_index()
            for f, gg in g.groupby("model"):
                ax.plot(gg["context_days"], gg["nmae"], c=COLORS.get(f, "k"), marker="o", ms=2, lw=0.7,
                        label=LABELS.get(f, f))
            ax.set_xscale("log")
            ax.set_xticks([7, 28, 90], ["7", "28", "90"])
            ax.minorticks_off()
            ax.set_xlabel("Context (days)")
            ax.set_ylabel("nMAE (mean over series)")
            ax.legend(frameon=False, fontsize=5)
        else:
            d = cal[(cal["side"] == "two") & np.isclose(cal["level"], 0.9) & cal["model"].map(primary)]
            d = d.assign(W=d["variant"].str.extract(r"_w(\d+)")[0].astype(float), base=d["variant"].str.replace(r"_w\d+", "", regex=True))
            d.loc[d["W"].isna(), "W"] = config.load("calibration")["split_cqr"]["window_days"]
            # mean coverage is ~nominal for every W; the window matters for hour-of-day coverage
            # NexCP excluded: its default weights all past days, so it has no W = 90 point
            for base, mk in (("split_cqr", "s"), ("aci", "^")):
                g = d[d["base"] == base].groupby("W")["worst_hour_gap_pp"].mean()
                if len(g) > 1:
                    ax.plot(g.index, g.values, marker=mk, ms=3, c="k", lw=0.7, label=VLABELS[base])
            ax.set_xticks([30, 90, 180])
            ax.set_xlabel("Calibration window W (days)")
            ax.set_ylabel("Worst-hour coverage gap (pp)")
            ax.legend(frameon=False, fontsize=5)
    style.save(fig, out / "F9_ablations")


def read(path):
    return pd.read_csv(path) if path.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", default="val", choices=["val", "test"])
    ap.add_argument("--allow-test", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    style.setup()
    ev = config.load("evaluate")
    period = splits.get_period(args.phase, allow_test=args.allow_test)
    mdir = config.path(ev["out_dir"]) / args.phase
    out = config.path(ev["figures_dir"]) / args.phase
    acc, rel = read(mdir / "accuracy.csv"), read(mdir / "reliability.csv")
    cal, res, bat = read(mdir / "calibration.csv"), read(mdir / "decisions_reserve.csv"), read(mdir / "decisions_battery.csv")
    f1_pipeline(out)
    if cal is not None:
        f2_coverage_width(cal, out)
    if bat is not None and not bat.empty:
        f3_battery_frontier(bat, out)
    if acc is not None and res is not None and not res.empty:
        f4_rank_slope(acc, res, out, ev["headline_tau"])
    if rel is not None:
        f5_reliability(rel, out)
    m = pick_model(cal)
    if m:
        f6_rolling_coverage(m, period, out, days=ev["rolling_coverage_days"])
        f7_hour_heatmap(None, m, period, out)
        f8_fan(m, period, out)
    f9_ablations(acc, cal, out, read(mdir / "context_ablation.csv"))
    log.info("figures in %s", out)


if __name__ == "__main__":
    main()
