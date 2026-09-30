# Results snapshot

Generated from saved outputs with the frozen configuration (commit `1934fc0`, "v-frozen").
`val/` = validation (2025-01-01 to 2025-10-31, used for all tuning and selection);
`test/` = the single held-out test run (2025-11-01 to 2026-08-31).

- `metrics/{phase}/*.csv`: long-format metrics (accuracy, reliability, calibration, reserve and
  battery decisions, Diebold-Mariano tests, compute).
- `tables/{phase}/T1-T6`: paper tables (CSV + LaTeX). In `T3_decisions`, each model's calibration
  variant and battery interval level are selected on validation and reported on the phase shown.
- `figures/{phase}/F1-F8`: paper figures (PDF + PNG).

Notes: ERCOT test series drop 30 of 304 days (EIA-930 is missing 2025-12-05, which also falls in
the 28-day context of the next 29 days); the same days are dropped for every model. The context-length
ablation (F9) is added once its runs are available.
