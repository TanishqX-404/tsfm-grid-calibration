"""Generate the paper's tables and every number quoted in the text from results/ (test period).

    python paper/make_tables.py      # writes paper/generated/*.tex and copies figures

Nothing in main.tex is typed by hand: numbers come from \\newcommand macros in numbers.tex.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
M = RES / "metrics" / "test"
T = RES / "tables" / "test"
OUT = ROOT / "paper" / "generated"
FIG = ROOT / "paper" / "figures"

MAIN = ["seasonal_naive", "lgbm-s0", "nhits-s0", "patchtst-s0", "chronos2", "moirai2", "timesfm25", "tirex"]
FMS = ["chronos2", "moirai2", "timesfm25", "tirex"]
NAME = {"seasonal_naive": "Seasonal naive", "lgbm-s0": "LightGBM", "nhits-s0": "N-HiTS", "patchtst-s0": "PatchTST",
        "chronos2": "Chronos-2", "moirai2": "Moirai-2.0", "timesfm25": "TimesFM-2.5", "tirex": "TiRex",
        "operator": "Operator (BA)"}
LABEL2KEY = {v: k for k, v in NAME.items()} | {"LightGBM": "lgbm-s0", "N-HiTS": "nhits-s0", "PatchTST": "patchtst-s0"}
SERIES = ["ERCO_load", "ERCO_solar", "ERCO_wind", "CISO_load", "CISO_solar", "CISO_wind"]
SHORT = {"ERCO_load": "ERCOT L", "ERCO_solar": "ERCOT S", "ERCO_wind": "ERCOT W",
         "CISO_load": "CAISO L", "CISO_solar": "CAISO S", "CISO_wind": "CAISO W"}
VARS = ["native", "split_cqr", "nexcp", "aci", "pid"]
VNAME = {"native": "Native", "split_cqr": "Split CQR", "nexcp": "NexCP", "aci": "ACI", "pid": "PID"}

N: dict[str, str] = {}


def num(key, value, fmt="{:.3f}"):
    N[key] = fmt.format(value) if not isinstance(value, str) else value


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def tab_accuracy():
    a = pd.read_csv(M / "accuracy.csv")
    dm = pd.read_csv(M / "dm_tests.csv")
    models = MAIN + ["operator"]
    piv = a[a.model.isin(models)].pivot(index="model", columns="series_id", values="nmae").reindex(models)[SERIES]
    cov = a[a.model.isin(models)].groupby("model").native80_coverage.mean().reindex(models)
    pin = a[a.model.isin(MAIN)].pivot(index="model", columns="series_id", values="pinball")
    # DM (pinball) vs the best-pinball model per series; Holm within the main-model family only
    dm = dm[dm.model_a.isin(MAIN) & dm.model_b.isin(MAIN)].copy()
    dm["p_holm_main"] = dm.groupby("series_id")["p"].transform(lambda p: holm(p.to_numpy()))
    tie = {}
    for sid in SERIES:
        best = pin[sid].idxmin()
        for m in MAIN:
            if m == best:
                tie[(m, sid)] = True
                continue
            r = dm[(dm.series_id == sid) & (((dm.model_a == best) & (dm.model_b == m)) | ((dm.model_b == best) & (dm.model_a == m)))]
            tie[(m, sid)] = bool(len(r)) and r.p_holm_main.iloc[0] >= 0.05
    lines = [r"\begin{tabular}{l" + "c" * len(SERIES) + "c}", r"\toprule",
             "Model & " + " & ".join(SHORT[s] for s in SERIES) + r" & Cov$_{80}$ \\", r"\midrule"]
    for m in models:
        cells = []
        for s in SERIES:
            v = piv.loc[m, s]
            if np.isnan(v):
                cells.append("--")
                continue
            txt = f"{v:.3f}"
            if m in MAIN and v == piv.loc[MAIN, s].min():
                txt = r"\textbf{" + txt + "}"
            if m in MAIN and tie.get((m, s)):
                txt = r"\underline{" + txt + "}"
            cells.append(txt)
        if m == "operator":
            lines.append(r"\midrule")
        covtxt = "--" if m == "operator" else f"{cov[m]:.2f}"  # a point forecast has no interval
        lines.append(f"{NAME[m]} & " + " & ".join(cells) + f" & {covtxt}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_accuracy.tex").write_text("\n".join(lines) + "\n")
    # numbers
    num("nmaeChronosErcoWind", piv.loc["chronos2", "ERCO_wind"])
    num("nmaeLgbmErcoWind", piv.loc["lgbm-s0", "ERCO_wind"])
    num("nmaeNaiveErcoWind", piv.loc["seasonal_naive", "ERCO_wind"])
    num("nmaeOperatorErcoLoad", piv.loc["operator", "ERCO_load"])
    num("nmaeBestFmErcoLoad", piv.loc[FMS, "ERCO_load"].min())
    num("nmaeLgbmCisoSolar", piv.loc["lgbm-s0", "CISO_solar"])
    num("nmaeBestFmCisoSolar", piv.loc[FMS, "CISO_solar"].min())
    fm_mean, lg_mean = piv.loc[FMS, ["ERCO_wind", "CISO_wind"]].mean().mean(), piv.loc["lgbm-s0", ["ERCO_wind", "CISO_wind"]].mean()
    num("windGainPct", 100 * (1 - fm_mean / lg_mean), "{:.0f}")
    sig = sum(1 for s in SERIES if not tie.get(("lgbm-s0", s)) and pin.loc["chronos2", s] < pin.loc["lgbm-s0", s])
    # direct chronos2 vs lgbm DM (Holm, main family)
    d = dm[((dm.model_a == "chronos2") & (dm.model_b == "lgbm-s0")) | ((dm.model_b == "chronos2") & (dm.model_a == "lgbm-s0"))]
    wins = int(((d.p_holm_main < 0.05) & (np.sign(d.dm_stat) == np.where(d.model_a == "chronos2", -1, 1))).sum())
    num("chronosBeatsLgbmSeries", str(wins))
    top = [m for m in ["chronos2", "moirai2", "tirex"]]
    pairs = dm[dm.model_a.isin(top) & dm.model_b.isin(top)]
    num("topThreeSigPairs", str(int((pairs.p_holm_main < 0.05).sum())))
    num("topThreePairsTotal", str(len(pairs)))
    num("nativeCovMinAll", cov[MAIN].min(), "{:.2f}")
    num("nativeCovMaxAll", cov[MAIN].max(), "{:.2f}")


def tab_calibration():
    c = pd.read_csv(M / "calibration.csv")
    c = c[c.model.isin(MAIN) & (c.side == "two") & c.variant.isin(VARS)]

    def agg(d):
        return d.groupby("variant").agg(covg=("coverage", "mean"), lo=("coverage", "min"), hi=("coverage", "max"),
                                        w=("width", "mean"), wink=("winkler", "mean"), wh=("worst_hour_gap_pp", "mean"),
                                        kup=("kupiec_p", lambda p: 100 * (p > 0.05).mean()),
                                        chr=("christoffersen_p_median", lambda p: 100 * (p > 0.05).mean()))
    g80 = agg(c[np.isclose(c.level, 0.8)]).reindex(VARS)
    g90 = agg(c[np.isclose(c.level, 0.9) & (c.variant != "native")]).reindex(VARS[1:])
    lines = [r"\begin{tabular}{lccccccc}", r"\toprule",
             r"Variant & Cov. & Range & Width & IS & Worst & Kup. & Chr. \\",
             r" & & & & & hr (pp) & (\%) & (\%) \\", r"\midrule",
             r"\multicolumn{8}{l}{\emph{Target 80\% (native band $[\hat q_{0.1},\hat q_{0.9}]$ available)}} \\"]
    for v, r in g80.iterrows():
        lines.append(f"{VNAME[v]} & {r.covg:.3f} & {r.lo:.2f}--{r.hi:.2f} & {r.w:.3f} & {r.wink:.3f} & {r.wh:.1f} & {r.kup:.0f} & {r.chr:.0f}" + r" \\")
    lines += [r"\midrule", r"\multicolumn{8}{l}{\emph{Target 90\% (no native interval)}} \\"]
    for v, r in g90.iterrows():
        lines.append(f"{VNAME[v]} & {r.covg:.3f} & {r.lo:.2f}--{r.hi:.2f} & {r.w:.3f} & {r.wink:.3f} & {r.wh:.1f} & {r.kup:.0f} & {r.chr:.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_calibration.tex").write_text("\n".join(lines) + "\n")
    conf = VARS[1:]
    num("nativeEightyCov", g80.loc["native", "covg"], "{:.3f}")
    num("confEightyCovLo", g80.loc[conf, "covg"].min(), "{:.3f}")
    num("confEightyCovHi", g80.loc[conf, "covg"].max(), "{:.3f}")
    num("eightyWidthIncreasePct", 100 * (g80.loc[conf, "w"].mean() / g80.loc["native", "w"] - 1), "{:.0f}")
    num("nativeEightyIS", g80.loc["native", "wink"], "{:.3f}")
    num("confEightyISLo", g80.loc[conf, "wink"].min(), "{:.3f}")
    num("confEightyISHi", g80.loc[conf, "wink"].max(), "{:.3f}")
    num("nativeEightyWorstHour", g80.loc["native", "wh"], "{:.1f}")
    num("confEightyWorstHourLo", g80.loc[conf, "wh"].min(), "{:.1f}")
    num("confEightyWorstHourHi", g80.loc[conf, "wh"].max(), "{:.1f}")
    num("nativeEightyKupPass", g80.loc["native", "kup"], "{:.0f}")
    num("confNinetyCovLo", g90.covg.min(), "{:.3f}")
    num("confNinetyCovHi", g90.covg.max(), "{:.3f}")
    num("confNinetyRangeLo", g90.lo.min(), "{:.2f}")
    num("confNinetyRangeHi", g90.hi.max(), "{:.2f}")
    n80 = c[np.isclose(c.level, 0.8) & (c.variant == "native")]
    num("nativeEightyUnder", 100 * (n80.coverage < 0.8).mean(), "{:.0f}")
    num("nativeEightyWorstHourMax", n80.worst_hour_gap_pp.max(), "{:.0f}")
    cap = pd.read_csv(M / "aci_capping.csv")
    cap = cap[cap.model.isin(MAIN)]
    num("aciCapMeanPct", 100 * cap.capped_share.mean(), "{:.1f}")
    num("aciCapMaxPct", 100 * cap.capped_share.max(), "{:.1f}")
    imp = pd.read_csv(M / "imputed_sensitivity.csv")
    num("imputedCovDiffPp", 100 * (imp.aci90_cov_all - imp.aci90_cov_no_imputed).abs().max(), "{:.1f}")
    cw = pd.read_csv(M / "calibration.csv")
    cw = cw[cw.model.isin(MAIN) & (cw.side == "two") & np.isclose(cw.level, 0.9)]
    for tag, v in (("Thirty", "split_cqr_w30"), ("Ninety", "split_cqr"), ("OneEighty", "split_cqr_w180")):
        num(f"worstHourW{tag}", cw[cw.variant == v].worst_hour_gap_pp.mean(), "{:.1f}")


def val_selected(tau=0.95):
    v = pd.read_csv(RES / "metrics" / "val" / "decisions_reserve.csv")
    v = v[np.isclose(v.tau, tau) & v.variant.isin(VARS)]
    return v.loc[v.groupby(["model", "series_id"]).regret.idxmin(), ["model", "series_id", "variant"]]


def tab_decisions():
    tau = 0.95
    r = pd.read_csv(M / "decisions_reserve.csv")
    r = r[np.isclose(r.tau, tau)]
    sel = val_selected(tau)
    sel = sel[sel.model.isin(MAIN)]
    per = sel.merge(r, on=["model", "series_id", "variant"])          # all days of each series
    cost = per.pivot(index="model", columns="series_id", values="cost").reindex(MAIN)[SERIES]
    pooled = pd.read_csv(M / "pooled_costs.csv").set_index("model").reindex(MAIN)
    diffs = pd.read_csv(M / "reserve_diffs.csv")
    dcost = diffs[(diffs.metric == "cost") & (diffs.comparison == "model")]
    b = pd.read_csv(T / "T3_decisions.csv")
    b["key"] = b.model.map(LABEL2KEY)
    bat = b[b.key.isin(MAIN)].groupby("key").pct_oracle.mean()

    def not_sig_vs(best, m, sid):
        if m == best:
            return True
        d = dcost[(dcost.series_id == sid) & (((dcost.a == best) & (dcost.b == m)) | ((dcost.a == m) & (dcost.b == best)))]
        return bool(len(d)) and not bool(d.significant.iloc[0])

    order = pooled.sort_values("cost_selected").index
    lines = [r"\begin{tabular}{l" + "c" * len(SERIES) + "cccc}", r"\toprule",
             r" & \multicolumn{6}{c}{Reserve cost, validation-selected calibration} & \multicolumn{3}{c}{Mean (common days)} & Battery \\",
             r"\cmidrule(lr){2-7}\cmidrule(lr){8-10}",
             "Model & " + " & ".join(SHORT[s] for s in SERIES) + r" & Cal. & Native & Fixed & (\% orc.) \\", r"\midrule"]
    for m in order:
        cells = []
        for sid in SERIES:
            best = cost[sid].idxmin()
            txt = f"{cost.loc[m, sid]:.2f}"
            if m == best:
                txt = r"\textbf{" + txt + "}"
            if not_sig_vs(best, m, sid):
                txt = r"\underline{" + txt + "}"
            cells.append(txt)
        pc = pooled.loc[m]
        mean_txt = f"{pc.cost_selected:.2f}"
        if m == order[0]:
            mean_txt = r"\textbf{" + mean_txt + "}"
        if not_sig_vs(order[0], m, "pooled"):
            mean_txt = r"\underline{" + mean_txt + "}"
        bt = f"{bat[m]:.1f}"
        if bat[m] == bat.max():
            bt = r"\textbf{" + bt + "}"
        lines.append(f"{NAME[m]} & " + " & ".join(cells) + f" & {mean_txt} & {pc.cost_native:.2f} & {pc.cost_deterministic:.2f} & {bt}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_decisions.tex").write_text("\n".join(lines) + "\n")

    best = order[0]
    num("costBestModel", NAME[best], "{}")
    num("costBest", pooled.loc[best, "cost_selected"], "{:.2f}")
    num("commonDays", str(int(pooled.n_days.iloc[0])))
    pdiff = dcost[dcost.series_id == "pooled"]

    def pair(m):
        d = pdiff[((pdiff.a == best) & (pdiff.b == m)) | ((pdiff.a == m) & (pdiff.b == best))].iloc[0]
        sgn = -1 if d.a == best else 1   # express as cost(m) - cost(best)
        lo, hi = sorted((sgn * d.ci_lo, sgn * d.ci_hi))
        return sgn * d.mean_diff, lo, hi, bool(d.significant)
    rivals = [m for m in order[1:]]
    nsig = sum(pair(m)[3] for m in rivals)
    num("nRivals", str(len(rivals)))
    num("nRivalsSig", str(nsig))
    for k, m in (("Second", order[1]), ("Third", order[2])):
        dm, lo, hi, _ = pair(m)
        num(f"rival{k}", NAME[m], "{}")
        num(f"rival{k}Diff", dm, "{:.2f}")
        num(f"rival{k}Lo", lo, "{:.2f}")
        num(f"rival{k}Hi", hi, "{:.2f}")
    cv = diffs[(diffs.comparison == "calibrated_vs_native") & (diffs.series_id == "pooled")]
    num("calNativeLo", -cv.mean_diff.max(), "{:.2f}")
    num("calNativeHi", -cv.mean_diff.min(), "{:.2f}")
    num("calNativeSig", str(int(cv.significant.sum())))
    num("calNativeN", str(len(cv)))
    red = 100 * (1 - pooled.cost_selected / pooled.cost_deterministic)
    num("detReductionLo", red.min(), "{:.0f}")
    num("detReductionHi", red.max(), "{:.0f}")
    num("batFmLo", bat[FMS].min(), "{:.1f}")
    num("batFmHi", bat[FMS].max(), "{:.1f}")
    num("batBestModel", NAME[bat.idxmax()], "{}")
    num("batLgbm", bat["lgbm-s0"], "{:.1f}")
    num("batNaive", bat["seasonal_naive"], "{:.1f}")
    bb = pd.read_csv(M / "decisions_battery.csv")
    bb = bb[bb.model.isin(MAIN) & bb.variant.isin(VARS)]
    spread = bb.groupby(["model", "series_id", "variant"]).pct_oracle.max().groupby(["model", "series_id"]).agg(lambda x: x.max() - x.min())
    num("batVariantSpread", spread.max(), "{:.1f}")
    op = b[b.model == "Operator"]
    num("opRegretErco", op[op.series_id == "ERCO_load"].regret.iloc[0], "{:.2f}")
    num("bestRegretErcoLoad", b[(b.series_id == "ERCO_load") & b.key.isin(MAIN)].regret.min(), "{:.2f}")
    num("opNativeRegret", r[(r.model == "operator") & (r.variant == "native")].regret.mean(), "{:.1f}")
    sf = per[per.model.isin(MAIN)].shortfall_freq
    num("shortfallLo", 100 * sf.min(), "{:.1f}")
    num("shortfallHi", 100 * sf.max(), "{:.1f}")
    num("shortfallMean", 100 * sf.mean(), "{:.1f}")

    # H3: rank by nMAE vs by cost vs by pinball loss of the tau* bound (same selected variant)
    a = pd.read_csv(M / "accuracy.csv")
    cal = pd.read_csv(M / "calibration.csv")
    cal = cal[np.isclose(cal.level, tau) & cal.side.isin(["upper", "lower"])]
    pb = sel.merge(cal, on=["model", "series_id", "variant"])[["model", "series_id", "pinball"]]
    rho_mae, rho_pin, disagree, disagree_sig = [], [], 0, 0
    for sid in SERIES:
        aa = a[(a.series_id == sid) & a.model.isin(MAIN)].set_index("model").nmae.rank()
        cc = cost[sid].rank()
        pp = pb[pb.series_id == sid].set_index("model").pinball.reindex(MAIN).rank()
        rho_mae.append(aa.corr(cc, method="spearman"))
        rho_pin.append(pp.corr(cc, method="spearman"))
        if aa.idxmin() != cc.idxmin():
            disagree += 1
            if not not_sig_vs(cc.idxmin(), aa.idxmin(), sid):
                disagree_sig += 1
    num("rhoLo", min(rho_mae), "{:.2f}")
    num("rhoHi", max(rho_mae), "{:.2f}")
    num("rhoPinLo", min(rho_pin), "{:.2f}")
    num("rhoPinHi", max(rho_pin), "{:.2f}")
    num("nTopDisagree", str(disagree))
    num("nTopDisagreeSig", str(disagree_sig))
    num("seedSdMax", pd.read_csv(M / "seed_spread.csv").regret_sd.max(), "{:.2f}")


def tab_checkpoints():
    import yaml
    a = yaml.safe_load((ROOT / "configs" / "leakage_audit.yaml").read_text())
    arxiv = {"chronos2": "2510.15821", "timesfm25": "2310.10688", "moirai2": "2511.11698", "tirex": "2505.23719"}
    lines = [r"\begin{tabular}{lcc}", r"\toprule", r"Model & Weights committed & Report (arXiv) \\", r"\midrule"]
    for m in ("tirex", "moirai2", "timesfm25", "chronos2"):
        lines.append(f"{NAME[m]} & {a['models'][m]['release']} & {arxiv[m]}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_checkpoints.tex").write_text("\n".join(lines) + "\n")


def ablation():
    d = pd.read_csv(M / "context_ablation.csv")
    g = d.groupby("context_days")[["nmae", "aci90_width", "reserve_cost_aci"]].mean()
    num("ctxNmaeSeven", g.loc[7, "nmae"], "{:.3f}")
    num("ctxNmaeTwentyEight", g.loc[28, "nmae"], "{:.3f}")
    num("ctxNmaeNinety", g.loc[90, "nmae"], "{:.3f}")
    num("ctxWidthGainPct", 100 * (1 - g.loc[90, "aci90_width"] / g.loc[28, "aci90_width"]), "{:.0f}")
    num("ctxCostSeven", g.loc[7, "reserve_cost_aci"], "{:.2f}")
    num("ctxCostTwentyEight", g.loc[28, "reserve_cost_aci"], "{:.2f}")
    num("ctxCostNinety", g.loc[90, "reserve_cost_aci"], "{:.2f}")
    num("ctxCommonDaysErco", str(int(d[d.series_id == "ERCO_load"].n_days.iloc[0])))


def data_numbers():
    t1 = pd.read_csv(T / "T1_data.csv")
    num("ercotDropped", str(int(t1[t1.series == "ERCO_load"].test_dropped_days.iloc[0])))
    num("testDays", "304")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    tab_accuracy()
    tab_calibration()
    tab_decisions()
    tab_checkpoints()
    ablation()
    data_numbers()
    lines = ["% generated by paper/make_tables.py -- do not edit"]
    for k, v in sorted(N.items()):
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
    for f in ("F1_pipeline", "F2_coverage_width", "F3_battery_frontier", "F4_rank_slope", "F9_ablations"):
        shutil.copy(RES / "figures" / "test" / f"{f}.pdf", FIG / f"{f}.pdf")
    for k, v in sorted(N.items()):
        print(f"{k:24s} {v}")


if __name__ == "__main__":
    main()
