# tsfm-grid-calibration

Calibrated foundation-model forecasting for grid decision: do zero-shot time-series foundation
models (Chronos-2, TimesFM-2.5, Moirai-2.0, TiRex) give uncertainty good enough to size reserves and
schedule storage, and what does a conformal layer buy an operator?

Day-ahead (24 h, hourly) load, solar and wind for ERCOT and CAISO from EIA-930, a five-variant
calibration layer (native, split CQR, NexCP, ACI, conformal PID), and two decision problems
(probabilistic reserve sizing, battery firming LP).

## Layout

```
configs/            data, models, calibration, decisions, evaluate (+ tuned/ from validation)
src/tgc/splits.py   train/val/test boundaries: the only source of dates; test needs allow_test
src/tgc/data/       EIA-930 via PUDL Parquet, Open-Meteo weather, cleaning, data report
src/tgc/forecast/   Forecaster interface, baselines, TSFM wrappers, rolling-origin runner
src/tgc/calibrate/  scores + hand-written split CQR / NexCP / ACI / PID
src/tgc/decide/     reserve sizing and battery LP (cvxpy + HiGHS)
src/tgc/evaluate/   metrics, Diebold-Mariano (HLN), Kupiec, Christoffersen, block bootstrap
src/tgc/plots/      figures F1-F9 and tables T1-T6
notebooks/          kaggle_forecast.ipynb (GPU runs)
```

## Setup (CPU)

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest -q
```

## Pipeline

Every stage reads and writes Parquet with fixed schemas; every CLI has `--dry-run` and skips
finished chunks.

```bash
make data                         # download EIA-930 (PUDL), clean, data report
python -m tgc.data.build --weather   # optional: Open-Meteo covariates (needs network access)
make tune                         # LightGBM hyper-parameters on validation (Optuna, 30 trials)
make forecast PHASE=val           # seasonal naive, operator, LightGBM x 3 seeds (CPU)
# foundation models + N-HiTS/PatchTST: notebooks/kaggle_forecast.ipynb on a GPU,
# then copy out/forecasts/* back
make calibrate-tune calibrate     # ACI gamma / PID eta chosen on validation only
make decide eval figures PHASE=val
make freeze                       # commit tuned configs, tag v-frozen
make forecast decide eval figures PHASE=test ALLOW=--allow-test   # the single test run
```

Output contracts:

| File | Columns |
| --- | --- |
| `data/clean/{region}.parquet` | timestamp_utc, local_time, local_hour, series_id, target, target_mw, scale, is_imputed, is_night, operator_fc, covariates |
| `out/forecasts/{model}/{series}/{yyyymm}.parquet` | series_id, origin_local, target_local, horizon, q10..q90, y_true, model, model_revision |
| `out/calibrated/{model}/{variant}/{series}.parquet` | series_id, origin_local, target_local, horizon, level, side, lower, upper, lower_raw, upper_raw, median, y_true, is_night |
| `out/decisions/{phase}/{reserve,battery}/{model}_{variant}_{series}.parquet` | one row per day and level |
| `out/metrics/{phase}/*.csv` | long format, one row per model x variant x series |

## Conventions

- Times are local standard time (fixed UTC offset, no DST): ERCOT UTC-6, CAISO UTC-8. The
  forecast for day D uses the 28 days before D and predicts hours 00-23 of D, so horizon = hour + 1.
- Targets are normalized: solar/wind by a trailing 90-day max (capacity proxy; EIA-860M not wired),
  load by the training-period mean.
- Days whose 28-day context or target contain a gap longer than 3 h are dropped for every model and
  logged in `_meta.json`.
- Coverage is empirical, not guaranteed (energy series are not exchangeable); only ACI carries a
  long-run average coverage bound.

## Data notes (found while building M1)

- **CISO timestamps.** The CAISO solar profile in EIA-930 drifts about 1.5 h later from 2022-Q3 to
  2025 while ERCOT stays stable. A pure solar-elevation night mask would zero real generation, so
  night hours are defined causally from data (no output at that hour over the previous 28 days).
- **CISO demand definition.** Since 2022, EIA-930 CISO demand diverges increasingly from CAISO's
  own day-ahead forecast (normalized MAE 2% -> 10%), with a midday plateau consistent with storage
  charging counted as demand. The operator-forecast row for CISO is therefore not a like-for-like
  benchmark; ERCOT is clean (~2.5%).
- **Solar/wind categories.** EIA split solar and wind into with/without integrated storage in 2024;
  the builder sums all variants.
