import pytest

from tgc import config


@pytest.fixture(autouse=True)
def _clear_config_cache():
    config.load.cache_clear()
    yield


def has_clean_data() -> bool:
    return (config.path(config.load("data")["paths"]["clean_dir"]) / "ERCO.parquet").exists()


requires_data = pytest.mark.skipif(not has_clean_data(), reason="needs data/clean (run `make data`)")
