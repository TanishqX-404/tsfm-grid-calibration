"""Moirai-2.0-R-small (Salesforce/moirai-2.0-R-small), univariate."""
from __future__ import annotations

import numpy as np

from tgc.forecast._fm import BatchedFM, device
from tgc.forecast.base import QUANTILES


class Moirai2(BatchedFM):
    name = "moirai2"

    def _load(self):
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        mod = Moirai2Module.from_pretrained(self.checkpoint, revision=self.revision)
        m = Moirai2Forecast(module=mod, prediction_length=24, context_length=self.context_days * 24,
                            target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        return m.to(device()).eval()

    def _forecast(self, contexts, past_cov, fut_cov, horizon, quantiles):
        levels = list(self._model.module.quantile_levels)
        idx = [levels.index(q) for q in quantiles]
        out = []
        for i in range(0, len(contexts), self.batch_size):
            arr = self._model.predict(past_target=[c[:, None] for c in contexts[i:i + self.batch_size]])
            arr = np.asarray(arr)  # (B, n_quantiles, horizon[, 1])
            if arr.ndim == 4:
                arr = arr[..., 0]
            out.append(np.transpose(arr[:, idx, :horizon], (0, 2, 1)))
        return np.concatenate(out)
