"""TimesFM-2.5 (google/timesfm-2.5-200m-pytorch). Covariates through XReg."""
from __future__ import annotations

import numpy as np

from tgc.forecast._fm import BatchedFM
from tgc.forecast.base import QUANTILES


class TimesFM25(BatchedFM):
    name = "timesfm25"

    def _load(self):
        import timesfm
        import torch
        torch.set_float32_matmul_precision("high")
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.checkpoint, revision=self.revision)
        m.compile(timesfm.ForecastConfig(
            # at least the requested context (multiple of the 32-step patch), so the 90-day ablation
            # is not silently truncated
            max_context=max(1024, -(-self.context_days * 24 // 32) * 32), max_horizon=256, normalize_inputs=True,
            use_continuous_quantile_head=True, fix_quantile_crossing=True,
            per_core_batch_size=self.batch_size, return_backcast=bool(self.covariates)))
        return m

    def _forecast(self, contexts, past_cov, fut_cov, horizon, quantiles):
        assert tuple(quantiles) == QUANTILES, "TimesFM-2.5 returns the fixed 0.1..0.9 grid"
        if self.covariates:
            dyn = {k: [np.concatenate([p[k], f[k]]) for p, f in zip(past_cov, fut_cov)] for k in self.covariates}
            _, outs = self._model.forecast_with_covariates(
                inputs=[c.astype(np.float32) for c in contexts], dynamic_numerical_covariates=dyn,
                xreg_mode="xreg + timesfm")
            # returns (point_outputs, quantile_outputs); each quantile output is (horizon, 10)
            q = np.stack([np.asarray(o)[:horizon] for o in outs])
        else:
            _, q = self._model.forecast(horizon=horizon, inputs=list(contexts))  # list copy: forecast pads in place
        q = np.asarray(q)[:, :horizon]
        return q[..., 1:10]  # slot 0 is the mean; 1..9 are 0.1..0.9
