"""Chronos-2 (amazon/chronos-2): native past and known-future covariates."""
from __future__ import annotations

import numpy as np

from tgc.forecast._fm import BatchedFM, device


class Chronos2(BatchedFM):
    name = "chronos2"

    def _load(self):
        from chronos import Chronos2Pipeline
        return Chronos2Pipeline.from_pretrained(self.checkpoint, revision=self.revision, device_map=device())

    def _forecast(self, contexts, past_cov, fut_cov, horizon, quantiles):
        if self.covariates:
            inputs = [{"target": c, "past_covariates": p, "future_covariates": f}
                      for c, p, f in zip(contexts, past_cov, fut_cov)]
        else:
            inputs = [c for c in contexts]
        q, _ = self._model.predict_quantiles(inputs, prediction_length=horizon, quantile_levels=quantiles,
                                             batch_size=self.batch_size)
        # each element: (n_variates=1, horizon, n_quantiles)
        return np.stack([t.squeeze(0).float().cpu().numpy() for t in q])
