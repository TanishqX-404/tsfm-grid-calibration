"""Shared helpers for foundation-model wrappers (GPU; run through the Kaggle notebook)."""
from __future__ import annotations

import numpy as np

from tgc.forecast.base import QUANTILES, Forecaster, frame


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class BatchedFM(Forecaster):
    """Foundation models: no fitting, only the ``context_days`` window, batched inference."""

    def __init__(self, checkpoint: str, revision: str = "main", batch_size: int = 64,
                 context_days: int = 28, covariates=()):
        self.checkpoint = checkpoint
        self.revision = revision
        self.batch_size = batch_size
        self.context_days = context_days
        self.covariates = list(covariates)
        self.uses_covariates = bool(covariates)
        self._model = None

    def _load(self):
        raise NotImplementedError

    def _forecast(self, contexts: list[np.ndarray], past_cov: list[dict], fut_cov: list[dict],
                  horizon: int, quantiles) -> np.ndarray:
        """Return an array (B, horizon, len(quantiles))."""
        raise NotImplementedError

    def predict(self, context_df, future_cov_df, horizon=24, quantiles=QUANTILES):
        return self.predict_batch([(context_df, future_cov_df)], horizon, quantiles)[0]

    def predict_batch(self, items, horizon=24, quantiles=QUANTILES):
        if self._model is None:
            self._model = self._load()
        ctx = [c["target"].to_numpy(dtype=np.float32)[-self.context_days * 24:] for c, _ in items]
        pc = [{k: c[k].to_numpy(dtype=np.float32)[-self.context_days * 24:] for k in self.covariates} for c, _ in items]
        fc = [{k: f[k].to_numpy(dtype=np.float32) for k in self.covariates} for _, f in items]
        Q = self._forecast(ctx, pc, fc, horizon, list(quantiles))
        assert Q.shape == (len(items), horizon, len(quantiles)), Q.shape
        return [frame(f, Q[i]) for i, (_, f) in enumerate(items)]
