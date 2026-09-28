"""TiRex (NX-AI/TiRex), xLSTM-based, univariate."""
from __future__ import annotations

import numpy as np

from tgc.forecast._fm import BatchedFM, device
from tgc.forecast.base import QUANTILES


class TiRex(BatchedFM):
    name = "tirex"

    def _load(self):
        from tirex import load_model
        return load_model(self.checkpoint, device=device(), hf_kwargs={"revision": self.revision})

    def _forecast(self, contexts, past_cov, fut_cov, horizon, quantiles):
        import torch
        assert tuple(quantiles) == QUANTILES, "TiRex returns the fixed 0.1..0.9 grid"
        q, _ = self._model.forecast(context=torch.tensor(np.stack(contexts)), prediction_length=horizon,
                                    output_type="numpy", batch_size=self.batch_size)
        return np.asarray(q)[:, :horizon, :]
