"""Block-bootstrap confidence intervals for empirical coverage (reviewer request R1.3).

    python -m tgc.evaluate.coverage_bootstrap --allow-test

For each model x series x calibration variant x level (two-sided), resample whole blocks of
consecutive test days (moving-block bootstrap) and form a 95% CI for coverage = hits / hours
(solar: daytime hours only). Block lengths 3, 7, 14, 28 days test sensitivity to dependence.
Also a pooled CI per variant: the per-day average over all model-series pairs, resampled by day.
Writes out/metrics/test/coverage_bootstrap{,_pooled}.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tgc import config, splits

MAIN = ["seasonal_naive", "lgbm-s0", "nhits-s0", "patchtst-s0", "chronos2", "moirai2", "timesfm25", "tirex"]
VARIANTS = ["native", "split_cqr", "nexcp", "aci", "pid"]
BLOCKS = (3, 7, 14, 28)


def block_indices(T, block, n, rng):
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n, nb))
    return (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :T]


def daily_hits(cal: pd.DataFrame) -> pd.DataFrame:
    hit = (cal["y_true"] >= cal["lower"]) & (cal["y_true"] <= cal["upper"])
    return pd.DataFrame({"hits": hit.groupby(cal["origin_local"]).sum(), "hours": hit.groupby(cal["origin_local"]).size()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args(argv)
    period = splits.get_period("test", allow_test=args.allow_test)
    root = config.path(config.load("calibration")["out_dir"])
    rng = np.random.default_rng(0)
    rows, pooled_days = [], {}
    for model in MAIN:
        for variant in VARIANTS:
            for sid in config.series_ids():
                p = root / model / variant / f"{sid}.parquet"
                if not p.exists():
                    continue
                c = pd.read_parquet(p)
                c = c[(c["side"] == "two") & period.contains(c["origin_local"]) & c["lower"].notna()]
                if sid.endswith("solar"):
                    c = c[~c["is_night"]]
                for level in (0.8, 0.9):
                    if variant == "native" and level == 0.9:
                        continue  # no native 90% interval
                    d = daily_hits(c[np.isclose(c["level"], level)]).sort_index()
                    pooled_days.setdefault((variant, level), []).append((d["hits"] / d["hours"]).rename(f"{model}|{sid}"))
                    h, n = d["hits"].to_numpy(float), d["hours"].to_numpy(float)
                    cov = h.sum() / n.sum()
                    for b in BLOCKS:
                        idx = block_indices(len(d), b, args.n, rng)
                        boot = h[idx].sum(axis=1) / n[idx].sum(axis=1)
                        lo, hi = np.quantile(boot, [0.025, 0.975])
                        rows.append({"model": model, "variant": variant, "series_id": sid, "level": level,
                                     "block_days": b, "n_days": len(d), "coverage": cov, "ci_lo": lo, "ci_hi": hi,
                                     "gap_pp": 100 * (cov - level), "contains_nominal": lo <= level <= hi})
    out = config.path(config.load("evaluate")["out_dir"]) / "test"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(6).to_csv(out / "coverage_bootstrap.csv", index=False)
    prow = []
    for (variant, level), cols in pooled_days.items():
        daily = pd.concat(cols, axis=1, sort=True).mean(axis=1).dropna().to_numpy()  # average coverage per day
        for b in BLOCKS:
            idx = block_indices(len(daily), b, args.n, rng)
            boot = daily[idx].mean(axis=1)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            prow.append({"variant": variant, "level": level, "block_days": b, "n_pairs": len(cols),
                         "coverage": daily.mean(), "ci_lo": lo, "ci_hi": hi, "contains_nominal": lo <= level <= hi})
    pd.DataFrame(prow).round(6).to_csv(out / "coverage_bootstrap_pooled.csv", index=False)
    print("wrote", out / "coverage_bootstrap.csv")


if __name__ == "__main__":
    main()
