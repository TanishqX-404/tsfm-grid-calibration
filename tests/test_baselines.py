"""M3 acceptance checks on real data (skipped when data/clean or outputs are absent)."""
import numpy as np
import pandas as pd
import pytest

from tgc import backtest, config, splits
from tgc.evaluate import metrics as M
from tgc.forecast.base import QCOLS
from tgc.forecast.naive import SeasonalNaive
from tests.conftest import requires_data


@requires_data
@pytest.mark.parametrize("sid,lag", [("ERCO_load", 168), ("CISO_wind", 24)])
def test_seasonal_naive_mase_is_one_on_training(sid, lag):
    """The naive point forecast y[t - lag], scored over the training period against the MASE
    denominator (in-sample naive MAE on training), gives MASE = 1 by construction."""
    from tgc.data import io
    from tgc.evaluate.run import mase_denominator
    df = io.load_series(sid)
    tr = splits.get_period("train")
    ys, ps = [], []
    for o in backtest.rolling_origins(df, tr.days()[7:], 28):
        ys.append(o.future["target"].to_numpy())
        ps.append(o.context["target"].to_numpy()[-lag:][:24])   # same hour, ``lag`` earlier
    y, point = np.concatenate(ys), np.concatenate(ps)
    assert M.mase(y, point, mase_denominator(sid)) == pytest.approx(1.0, abs=0.01)


@requires_data
def test_naive_quantiles_non_crossing():
    from tgc.data import io
    df = io.load_series("ERCO_solar")
    fc = SeasonalNaive(24, 90)
    o = next(backtest.rolling_origins(df, splits.get_period("val").days()[:1], fc.context_days, check_days=28))
    q = fc.predict(o.context, o.future[["local_time"]])[QCOLS].to_numpy()
    assert (np.diff(q, axis=1) >= -1e-12).all()


def _val_accuracy():
    p = config.path(config.load("evaluate")["out_dir"]) / "val" / "accuracy.csv"
    return pd.read_csv(p) if p.exists() else None


@pytest.mark.skipif(_val_accuracy() is None, reason="needs out/metrics/val/accuracy.csv")
def test_lgbm_beats_naive_on_validation():
    a = _val_accuracy()
    for sid in config.series_ids():
        naive = a[(a.model == "seasonal_naive") & (a.series_id == sid)]["pinball"]
        lg = a[(a.model == "lgbm-s0") & (a.series_id == sid)]["pinball"]
        if len(naive) and len(lg):
            assert lg.iloc[0] < naive.iloc[0], sid


@pytest.mark.skipif(_val_accuracy() is None, reason="needs forecasts")
def test_forecast_outputs_non_crossing():
    root = config.path(config.load("models")["out_dir"])
    for f in list(root.glob("*/*/*.parquet"))[:50]:
        q = pd.read_parquet(f)[QCOLS].to_numpy()
        assert (np.diff(q, axis=1) >= -1e-12).all(), f
