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
        lines.append(f"{NAME[m]} & " + " & ".join(cells) + f" & {cov[m]:.2f}" + r" \\")
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
    sd = pd.read_csv(M / "seed_spread.csv")
    num("seedSdMax", sd.regret_sd.max(), "{:.2f}")


def tab_calibration():
    c = pd.read_csv(M / "calibration.csv")
    c = c[c.model.isin(MAIN) & (c.side == "two") & np.isclose(c.level, 0.9) & c.variant.isin(VARS)]
    g = c.groupby("variant").agg(covg=("coverage", "mean"), lo=("coverage", "min"), hi=("coverage", "max"),
                                 wh=("worst_hour_gap_pp", "mean"), w=("width", "mean"),
                                 kup=("kupiec_p", lambda p: 100 * (p > 0.05).mean()),
                                 chr=("christoffersen_p_median", lambda p: 100 * (p > 0.05).mean())).reindex(VARS)
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Variant & Cov. & Range & Worst hr & Width & Kup. & Chr. \\",
             r" & & & (pp) & & (\%) & (\%) \\", r"\midrule"]
    for v, r in g.iterrows():
        lines.append(f"{VNAME[v]} & {r.covg:.3f} & {r.lo:.2f}--{r.hi:.2f} & {r.wh:.1f} & {r.w:.3f} & {r.kup:.0f} & {r.chr:.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_calibration.tex").write_text("\n".join(lines) + "\n")
    num("nativeCovNinety", g.loc["native", "covg"], "{:.3f}")
    num("nativeWorstHour", g.loc["native", "wh"], "{:.0f}")
    num("confWorstHourLo", g.loc[VARS[1:], "wh"].min(), "{:.1f}")
    num("confWorstHourHi", g.loc[VARS[1:], "wh"].max(), "{:.1f}")
    num("confCovLo", g.loc[VARS[1:], "lo"].min(), "{:.3f}")
    num("confCovHi", g.loc[VARS[1:], "hi"].max(), "{:.3f}")
    num("confKupLo", g.loc[VARS[1:], "kup"].min(), "{:.0f}")
    num("confKupHi", g.loc[VARS[1:], "kup"].max(), "{:.0f}")
    num("widthIncreasePct", 100 * (g.loc[VARS[1:], "w"].mean() / g.loc["native", "w"] - 1), "{:.0f}")
    n80 = pd.read_csv(M / "calibration.csv")
    n80 = n80[n80.model.isin(MAIN) & (n80.side == "two") & np.isclose(n80.level, 0.8) & (n80.variant == "native")]
    num("nativeEightyKupPass", 100 * (n80.kupiec_p > 0.05).mean(), "{:.0f}")
    num("nativeEightyUnder", 100 * (n80.coverage < 0.8).mean(), "{:.0f}")
    num("nativeEightyWorstHour", n80.worst_hour_gap_pp.mean(), "{:.0f}")
    num("nativeEightyWorstHourMax", n80.worst_hour_gap_pp.max(), "{:.0f}")
    cap = pd.read_csv(M / "aci_capping.csv")
    cap = cap[cap.model.isin(MAIN)]
    num("aciCapMeanPct", 100 * cap.capped_share.mean(), "{:.1f}")
    num("aciCapMaxPct", 100 * cap.capped_share.max(), "{:.1f}")
    imp = pd.read_csv(M / "imputed_sensitivity.csv")
    num("imputedCovDiffPp", 100 * (imp.aci90_cov_all - imp.aci90_cov_no_imputed).abs().max(), "{:.1f}")
    # calibration window ablation (split CQR; W = 30 / 90 / 180)
    cw = pd.read_csv(M / "calibration.csv")
    cw = cw[cw.model.isin(MAIN) & (cw.side == "two") & np.isclose(cw.level, 0.9)]
    for tag, v in (("Thirty", "split_cqr_w30"), ("Ninety", "split_cqr"), ("OneEighty", "split_cqr_w180")):
        num(f"worstHourW{tag}", cw[cw.variant == v].worst_hour_gap_pp.mean(), "{:.1f}")


def tab_decisions():
    t3 = pd.read_csv(T / "T3_decisions.csv")
    t3["key"] = t3.model.map(LABEL2KEY)
    r = pd.read_csv(M / "decisions_reserve.csv")
    r = r[np.isclose(r.tau, 0.95)]
    nat = r[r.variant == "native"].groupby("model").cost.mean()
    det = r[r.variant == "deterministic"].groupby("model").cost.mean()
    sel = t3[t3.key.isin(MAIN)].groupby("key").agg(cost=("cost", "mean"), regret=("regret", "mean"),
                                                   sf=("shortfall_freq", "mean"), bat=("pct_oracle", "mean"))
    order = sel.sort_values("cost").index
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Model & Cal. & Native & Fixed & Short. & Battery \\",
             r" & cost & cost & rule & (\%) & (\% orc.) \\", r"\midrule"]
    for m in order:
        s = sel.loc[m]
        cost = f"{s.cost:.2f}"
        if m == order[0]:
            cost = r"\textbf{" + cost + "}"
        bat = f"{s.bat:.1f}"
        if s.bat == sel.bat.max():
            bat = r"\textbf{" + bat + "}"
        lines.append(f"{NAME[m]} & {cost} & {nat[m]:.2f} & {det[m]:.2f} & {100 * s.sf:.1f} & {bat}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tab_decisions.tex").write_text("\n".join(lines) + "\n")
    num("costBest", sel.cost.min(), "{:.2f}")
    num("costBestModel", NAME[order[0]], "{}")
    red = 100 * (1 - sel.cost / det.reindex(sel.index))
    num("detReductionLo", red.min(), "{:.0f}")
    num("detReductionHi", red.max(), "{:.0f}")
    num("batChronos", sel.loc["chronos2", "bat"], "{:.1f}")
    num("batLgbm", sel.loc["lgbm-s0", "bat"], "{:.1f}")
    num("batNaive", sel.loc["seasonal_naive", "bat"], "{:.1f}")
    num("batFmLo", sel.loc[FMS, "bat"].min(), "{:.1f}")
    num("batFmHi", sel.loc[FMS, "bat"].max(), "{:.1f}")
    num("batBestModel", NAME[sel.bat.idxmax()], "{}")
    # operator (load only)
    op = t3[t3.model == "Operator"]
    num("opRegretErco", op[op.series_id == "ERCO_load"].regret.iloc[0], "{:.2f}")
    fm_erco = t3[(t3.series_id == "ERCO_load") & t3.key.isin(MAIN)].regret.min()
    num("bestRegretErcoLoad", fm_erco, "{:.2f}")
    opnat = r[(r.model == "operator") & (r.variant == "native")].regret.mean()
    num("opNativeRegret", opnat, "{:.1f}")
    # battery variant sensitivity
    b = pd.read_csv(M / "decisions_battery.csv")
    b = b[b.model.isin(MAIN) & b.variant.isin(VARS)]
    spread = b.groupby(["model", "series_id", "variant"]).pct_oracle.max().groupby(["model", "series_id"]).agg(lambda x: x.max() - x.min())
    num("batVariantSpread", spread.max(), "{:.1f}")
    # USD illustration, ERCOT wind, best model
    w = t3[(t3.series_id == "ERCO_wind") & t3.key.isin(MAIN)].sort_values("cost").iloc[0]
    num("usdErcoWind", w.cost_usd_per_day / 1e6, "{:.1f}")
    # H3 rank agreement
    a = pd.read_csv(M / "accuracy.csv")
    rho = []
    disagree = []
    for sid in SERIES:
        aa = a[(a.series_id == sid) & a.model.isin(MAIN)].set_index("model").nmae.rank()
        rr = t3[(t3.series_id == sid) & t3.key.isin(MAIN)].set_index("key").cost.rank()
        cmn = aa.index.intersection(rr.index)
        rho.append(aa[cmn].corr(rr[cmn], method="spearman"))
        if aa[cmn].idxmin() != rr[cmn].idxmin():
            disagree.append(sid)
    num("rhoLo", min(rho), "{:.2f}")
    num("rhoHi", max(rho), "{:.2f}")
    num("nTopDisagree", str(len(disagree)))
    # paired bootstrap (T7)
    t7 = pd.read_csv(T / "T7_reserve_cost_differences.csv")
    mv = t7[t7.comparison.str.contains(" - ") & ~t7.comparison.str.contains(":")]
    closest = mv.sort_values("mean_diff", ascending=False).iloc[0]
    num("closestRival", closest.comparison.split(" - ")[1], "{}")
    num("closestDiff", -closest.mean_diff, "{:.2f}")
    num("closestLo", -closest.ci_hi, "{:.2f}")
    num("closestHi", -closest.ci_lo, "{:.2f}")
    num("nRivalsSig", str(int(mv.significant.sum())))
    num("nRivals", str(len(mv)))
    cv = t7[t7.comparison.str.contains("calibrated - native")]
    num("calNativeLo", -cv.mean_diff.max(), "{:.2f}")
    num("calNativeHi", -cv.mean_diff.min(), "{:.2f}")
    num("calNativeSig", str(int(cv.significant.sum())))
    num("calNativeN", str(len(cv)))


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
