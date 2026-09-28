from tgc import config


def test_configs_load():
    for name in ("data", "models", "calibration", "decisions", "evaluate"):
        assert isinstance(config.load(name), dict)


def test_series_ids():
    assert config.series_ids() == ["ERCO_load", "ERCO_solar", "ERCO_wind", "CISO_load", "CISO_solar", "CISO_wind"]
