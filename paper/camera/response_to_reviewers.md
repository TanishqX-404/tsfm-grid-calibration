# Response to reviewers — PES IM 2027

We thank both reviewers. All changes fit within the 5-page limit. Some text elsewhere was shortened to make room; no results changed. New analyses use only the frozen test forecasts; nothing was retrained or re-tuned. Code: `src/tgc/evaluate/{same_level_cost,coverage_bootstrap,horizon_accuracy}.py`, outputs in `results/metrics/test/`.

## Reviewer 1

**R1.1 — Fine-tuned TSFMs.** We agree this is relevant. Fine-tuning four models on six series, with its own validation protocol, is outside the scope of a camera-ready revision. We now state it as a limitation (Sec. V): all TSFMs are zero-shot, and fine-tuning (which Chronos-2 supports) may change both accuracy and native calibration. It is left for future work.

**R1.2 — Quantile settings and the same-level cost comparison.**

*Checkpoint and quantile settings.* The checkpoints and revisions are pinned (Table I; revision hashes in the repository). The common output grid is 0.1,…,0.9 because the released TimesFM-2.5 and TiRex heads return only these levels. Chronos-2 also predicts 0.05 and 0.95 natively, but we queried it at the same nine levels so that every model enters calibration and decisions through one table. Sec. II now says this explicitly.

*Cost comparison at the same target quantile level.* We added a comparison at τ\* = 0.90, where every model's native q0.9 (q0.1 for solar and wind) exists. On average, the validation-selected calibration lowers reserve cost by only 1.7%:

| Model | Gain at τ\* = 0.90 (%) | Series kept native |
|---|---|---|
| N-HiTS | 7.9 | 0 of 6 |
| PatchTST | 3.4 | 3 of 6 |
| Seasonal naive | 1.2 | 2 of 6 |
| TimesFM-2.5 | 0.6 | 4 of 6 |
| Chronos-2 | 0.2 | 5 of 6 |
| LightGBM | 0 | 6 of 6 |
| Moirai-2.0 | 0 | 6 of 6 |
| TiRex | 0 | 6 of 6 |

"Series kept native" counts the series on which validation selected the native bound, so no calibration was applied there. At τ\* = 0.95 the average gain is 11.2% (4.2–24.3%). Most of the 0.95 gain therefore comes from reaching a level the native grid lacks, not from correcting a miscalibrated native quantile. Sec. IV (RQ3) now states this.

*Chronos-2 with its native 0.95 quantile.* The pinned checkpoint was trained on 21 levels (0.01, 0.05, 0.1, …, 0.95, 0.99), so 0.95 is native. We ran it once more on the test period with the same checkpoint and context, requesting 0.05/0.1/0.5/0.9/0.95. Its q0.1/q0.5/q0.9 agree with the main run to within 0.004 normalized units, which is GPU numerics. We then sized reserve at τ\* = 0.95 from the native q0.95 (load) or q0.05 (solar, wind):

| Chronos-2 reserve bound | Cost per day (mean over series) | Shortfall hours |
|---|---|---|
| Native q0.95 | 3.97 | 4.8% (1.8–7.2% by series) |
| Validation-selected calibration | 3.98 | 5.3% (4.5–6.1%) |
| Native q0.9 | 4.23 | 10.3% |

So, for the TSFM that offers the right level, its native tail quantile is as cheap as the calibrated bound on average. Calibration adds per-series coverage control (the native shortfall ranges from 1.8% on ERCOT solar to 7.2% on ERCOT load). Sec. II and Sec. IV (RQ3) now report this.

**R1.3 — Bootstrap CIs for coverage, and sensitivity to block length.** We added moving-block bootstrap 95% CIs for empirical coverage (2000 resamples) with blocks of 3, 7, 14 and 28 days (Sec. IV, RQ2).

*Pooled over the 48 model–series pairs:*
- The native 80% band covers 0.781.
- Its CI excludes 0.80 at every block length: 0.770–0.792 with 3-day blocks and 0.763–0.799 with 28-day blocks.
- Every conformal CI, at 80% and at 90%, contains its nominal level at every block length.

*Per pair*, the share of CIs that contain the nominal level:

| Intervals | 3-day blocks | 7-day | 14-day | 28-day |
|---|---|---|---|---|
| Native 80% | 54% | 62% | 62% | 67% |
| Conformal 80% | 98–100% | 98–100% | 98–100% | 94–98% |
| Conformal 90% | 92–100% at every block length | | | |

The Kupiec test (15% pass for native intervals) therefore overstates native failure because misses are dependent; the text now says so. The qualitative conclusions do not change: native intervals under-cover on average and per hour of day, and conformal intervals reach nominal coverage. Results are insensitive to block length beyond the expected widening of the CIs.

## Reviewer 2

**Does the 28-day, history-only, midnight-issue protocol bias the TSFM-vs-baseline comparison?** Partly. A new paragraph in Sec. V ("Protocol bias") discusses it.

LightGBM and seasonal naive use lags of at least 24 h. The TSFMs, N-HiTS and PatchTST see the context up to the issue time. With a midnight issue, this favours the latter at short lead times, which are close to nowcasts. We split test nMAE by lead time. The table gives the best TSFM's nMAE advantage, as the mean over the six series:

| Comparison | Hours 1–6 | Hours 1–12 | Hours 13–24 | All hours |
|---|---|---|---|---|
| vs LightGBM | 57% | 27% | 4.3% | 14.3% |
| vs best trained model | 12.7% | 8.0% | 1.7% | 4.0% |
| vs seasonal naive | 71.5% | 42.8% | 25.7% | 33.6% |

- ERCOT wind: 29.9% lower than LightGBM overall, but 7.9% lower at hours 13–24.
- At hours 13–24, the lead times nearest a market day-ahead forecast, a TSFM still has the lowest nMAE on 5 of 6 series (LightGBM is best on CAISO solar).

We therefore present the RQ1 margins as upper bounds for a market-close issue time and qualify the 30% ERCOT-wind figure in Sec. IV. The calibration and decision findings do not depend on this, because every calibrator is fit per hour of day and native vs. calibrated comparisons are made within each model. The 28-day context is examined in the existing ablation: a 90-day context gives nearly the same nMAE (0.076 vs. 0.077) at equal reserve cost.
