"""N-HiTS and PatchTST via neuralforecast, trained once on the training period (GPU)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tgc import splits
from tgc.forecast.base import QUANTILES, Forecaster, frame


class NeuralForecaster(Forecaster):
    def __init__(self, arch: str, seed: int = 0, context_days: int = 28, max_steps: int = 1000,
                 batch_size: int = 32, learning_rate: float = 1e-3, scaler_type: str = "robust",
                 covariates=()):
        self.arch = arch
        self.seed = seed
        self.context_days = context_days
        self.kw = dict(max_steps=max_steps, batch_size=batch_size, learning_rate=learning_rate,
                       scaler_type=scaler_type)
        self.covariates = list(covariates)
        self.uses_covariates = bool(covariates)
        self.name = f"{arch.lower()}-s{seed}" + ("-cov" if covariates else "")
        self.nf = None

    def _model(self):
        from neuralforecast.losses.pytorch import MQLoss
        from neuralforecast import models as M
        cls = getattr(M, self.arch)
        extra = {"futr_exog_list": self.covariates} if self.covariates else {}
        return cls(h=24, input_size=self.context_days * 24, loss=MQLoss(quantiles=list(QUANTILES)),
                   random_seed=self.seed, enable_progress_bar=False, **self.kw, **extra)

    def fit(self, train_df):
        from neuralforecast import NeuralForecast
        import neuralforecast
        splits.assert_before_test(train_df["local_time"], "neural fit")
        self.revision = f"neuralforecast-{neuralforecast.__version__}"
        df = train_df.rename(columns={"local_time": "ds", "target": "y"})
        df["unique_id"] = "s"
        df = df[["unique_id", "ds", "y", *self.covariates]].dropna(subset=["y"])
        self.nf = NeuralForecast(models=[self._model()], freq="h")
        self.nf.fit(df)

    def predict(self, context_df, future_cov_df, horizon=24, quantiles=QUANTILES):
        return self.predict_batch([(context_df, future_cov_df)], horizon, quantiles)[0]

    def predict_batch(self, items, horizon=24, quantiles=QUANTILES):
        ctx, fut = [], []
        for i, (c, f) in enumerate(items):
            c = c.rename(columns={"local_time": "ds", "target": "y"})[["ds", "y", *self.covariates]].copy()
            c["unique_id"] = str(i)
            ctx.append(c)
            f = f.rename(columns={"local_time": "ds"})[["ds", *self.covariates]].copy()
            f["unique_id"] = str(i)
            fut.append(f)
        kw = {"futr_df": pd.concat(fut)} if self.covariates else {}
        p = self.nf.predict(df=pd.concat(ctx), **kw)
        if "unique_id" not in p.columns:
            p = p.reset_index()
        qcols = [_column_for(p.columns, q) for q in quantiles]
        out = []
        for i, (_, f) in enumerate(items):
            g = p[p["unique_id"] == str(i)].sort_values("ds")
            out.append(frame(f, g[qcols].to_numpy()[:horizon]))
        return out


def _column_for(columns, q: float) -> str:
    """Map a quantile to neuralforecast's MQLoss output column (-lo-X / -median / -hi-X)."""
    best, err = None, 1.0
    for c in columns:
        if c.endswith("-median"):
            level = 0.5
        elif "-lo-" in c:
            level = (100 - float(c.rsplit("-lo-", 1)[1])) / 200
        elif "-hi-" in c:
            level = (100 + float(c.rsplit("-hi-", 1)[1])) / 200
        else:
            continue
        if abs(level - q) < err:
            best, err = c, abs(level - q)
    assert best is not None and err < 1e-6, f"no output column for quantile {q}"
    return best
