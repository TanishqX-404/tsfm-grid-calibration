# Results snapshot

Generated from saved outputs with the frozen configuration (commit `1934fc0`, "v-frozen").
`val/` = validation (2025-01-01 to 2025-10-31, used for all tuning and selection);
`test/` = the single held-out test run (2025-11-01 to 2026-08-31).

- `metrics/{phase}/*.csv`: long-format metrics (accuracy, reliability, calibration, reserve and
  battery decisions, Diebold-Mariano tests, compute).
- `tables/{phase}/T1-T6`: paper tables (CSV + LaTeX). In `T3_decisions`, each model's calibration
  variant and battery interval level are selected on validation and reported on the phase shown.
- `tables/{phase}/T7_reserve_cost_differences`: pooled paired moving-block bootstrap (7-day blocks,
  1,000 resamples) CIs of daily reserve-cost differences: each model vs the cheapest, and each model's
  validation-selected calibration vs its native intervals. Per-series rows in `metrics/{phase}/reserve_diffs.csv`.
- `metrics/{phase}/seed_spread.csv`, `imputed_sensitivity.csv`, `aci_capping.csv`: seed variability,
  results with/without EIA-imputed hours, and how often ACI's alpha_t fell to <= 0 (interval capped).
- `figures/{phase}/F1-F8`: paper figures (PDF + PNG).

Illustrative dollars (T3 `cost_usd_per_day`) = normalized cost x mean series scale (MW) x one stated
reserve price ($10/MWh, `configs/decisions.yaml`); not a reproduction of any market's rules.

Notes: ERCOT test series drop 30 of 304 days (EIA-930 is missing 2025-12-05, which also falls in
the 28-day context of the next 29 days); the same days are dropped for every model. The context-length
ablation (F9) is added once its runs are available.
