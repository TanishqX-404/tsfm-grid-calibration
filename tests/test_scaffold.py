from tgc import config


def test_configs_load():
    for name in ("data", "models", "calibration", "decisions", "evaluate"):
        assert isinstance(config.load(name), dict)


def test_series_ids():
    assert config.series_ids() == ["ERCO_load", "ERCO_solar", "ERCO_wind", "CISO_load", "CISO_solar", "CISO_wind"]


def test_neural_quantile_column_mapping():
    from tgc.forecast.base import QUANTILES
    from tgc.forecast.neural import _column_for
    cols = ["unique_id", "ds", "NHITS-median"] + [f"NHITS-lo-{100 - 200 * q:.1f}" for q in QUANTILES if q < 0.5] \
        + [f"NHITS-hi-{100 - 200 * (1 - q):.1f}" for q in QUANTILES if q > 0.5]
    got = [_column_for(cols, q) for q in QUANTILES]
    assert got == ["NHITS-lo-80.0", "NHITS-lo-60.0", "NHITS-lo-40.0", "NHITS-lo-20.0", "NHITS-median",
                   "NHITS-hi-20.0", "NHITS-hi-40.0", "NHITS-hi-60.0", "NHITS-hi-80.0"]
