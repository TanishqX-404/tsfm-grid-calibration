"""LightGBM quantile regression: one model per quantile, Optuna-tuned on validation."""
from __future__ import annotations

import json
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from tgc import config, splits
from tgc.forecast.base import QUANTILES, Forecaster, frame
from tgc.forecast.features import make_features

log = logging.getLogger(__name__)


def pinball(y, q, tau):
    d = y - q
    return np.mean(np.maximum(tau * d, (tau - 1) * d))


def _fit_one(X, y, tau, params, seed):
    p = dict(params)
    p.update(objective="quantile", alpha=tau, random_state=seed, verbose=-1,
             bagging_seed=seed, feature_fraction_seed=seed, n_jobs=p.get("n_jobs", 0))
    m = lgb.LGBMRegressor(**p)
    m.fit(X, y)
    return m


def tune(train_df, val_df, lags, covariates, fixed, n_trials, taus, seed=0):
    """Optuna search on validation pinball loss. Neither frame may touch the test period."""
    import optuna
    splits.assert_before_test(train_df["local_time"], "lgbm train")
    splits.assert_before_test(val_df["local_time"], "lgbm validation")
    full = pd.concat([train_df, val_df]).drop_duplicates("local_time").sort_values("local_time")
    X = make_features(full, lags, covariates)
    y = full["target"].reset_index(drop=True)
    tr = full["local_time"].isin(train_df["local_time"]).values & X.notna().all(axis=1).values & y.notna().values
    va = full["local_time"].isin(val_df["local_time"]).values & X.notna().all(axis=1).values & y.notna().values

    def objective(trial):
        params = dict(fixed)
        params.update(
            num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            n_estimators=trial.suggest_int("n_estimators", 100, 1000, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            subsample_freq=1,
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
        losses = [pinball(y[va].values, _fit_one(X[tr], y[tr], t, params, seed).predict(X[va]), t) for t in taus]
        return float(np.mean(losses))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = dict(fixed)
    best.update(study.best_params, subsample_freq=1)
    return best, float(study.best_value)


class LGBMQuantile(Forecaster):
    name = "lgbm"
    context_days = 28

    def __init__(self, series_id: str, seed: int = 0, lags=(24, 48, 168), covariates=(),
                 params: dict | None = None):
        self.series_id = series_id
        self.seed = seed
        self.lags = list(lags)
        self.covariates = list(covariates)
        self.uses_covariates = bool(covariates)
        self.params = params or {}
        self.models = {}
        self.name = f"lgbm-s{seed}" + ("-cov" if covariates else "")
        self.revision = f"lgbm-{lgb.__version__}"

    def fit(self, train_df):
        splits.assert_before_test(train_df["local_time"], "lgbm fit")
        X = make_features(train_df, self.lags, self.covariates)
        y = train_df.sort_values("local_time")["target"].reset_index(drop=True)
        ok = X.notna().all(axis=1) & y.notna()
        for tau in QUANTILES:
            self.models[tau] = _fit_one(X[ok], y[ok], tau, self.params, self.seed)

    def predict(self, context_df, future_cov_df, horizon=24, quantiles=QUANTILES):
        return self.predict_batch([(context_df, future_cov_df)], horizon, quantiles)[0]

    def predict_batch(self, items, horizon=24, quantiles=QUANTILES):
        Xs = []
        for ctx, fut in items:
            f = fut.copy()
            f["target"] = np.nan  # truth is never visible
            both = pd.concat([ctx, f], ignore_index=True)
            Xs.append(make_features(both, self.lags, self.covariates).iloc[-horizon:])
        X = pd.concat(Xs, ignore_index=True)
        assert X[[f"lag{L}" for L in self.lags]].notna().all().all(), "lag feature reached into day D"
        Q = np.column_stack([self.models[t].predict(X) for t in quantiles])
        return [frame(fut, Q[i * horizon:(i + 1) * horizon]) for i, (_, fut) in enumerate(items)]


def tuned_params_path(series_id: str, covariates: bool):
    d = config.path(config.load("models")["models"]["lgbm"]["tuned_dir"])
    return d / f"lgbm_{series_id}{'_cov' if covariates else ''}.json"


def load_params(series_id: str, covariates: bool) -> dict:
    p = tuned_params_path(series_id, covariates)
    if p.exists():
        return json.loads(p.read_text())["params"]
    log.warning("no tuned params for %s; using fixed defaults", series_id)
    return dict(config.load("models")["models"]["lgbm"]["fixed"])
