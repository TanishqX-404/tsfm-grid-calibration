"""Build forecasters from configs/models.yaml."""
from __future__ import annotations

from tgc import config
from tgc.data import weather


def model_names() -> list[str]:
    return list(config.load("models")["models"])


def applies_to(model: str, series_id: str) -> bool:
    mc = config.load("models")["models"][model]
    _, target = config.split_series(series_id)
    return target in mc.get("targets", [target])


def build(model: str, series_id: str, seed: int = 0, covariates: bool = False, context_days: int | None = None):
    cfg = config.load("models")
    mc = cfg["models"][model]
    _, target = config.split_series(series_id)
    cov = weather.covariate_columns(target) if covariates else []
    ctx = context_days or cfg["context_days"]
    kind = mc["kind"]
    if kind == "naive":
        from tgc.forecast.naive import SeasonalNaive
        return SeasonalNaive(mc["lag_hours"][target], mc["residual_days"])
    if kind == "operator":
        from tgc.forecast.operator import OperatorForecast
        return OperatorForecast()
    if kind == "lgbm":
        from tgc.forecast.lgbm import LGBMQuantile, load_params
        return LGBMQuantile(series_id, seed, mc["lags"], cov, load_params(series_id, bool(cov)))
    if kind == "neural":
        from tgc.forecast.neural import NeuralForecaster
        return NeuralForecaster(mc["arch"], seed, ctx, mc["max_steps"], mc["batch_size"],
                                mc["learning_rate"], mc["scaler_type"], cov,
                                mc.get("extra_by_target", {}).get(target, mc.get("extra")))
    fm = dict(checkpoint=mc["checkpoint"], revision=mc["revision"], batch_size=mc["batch_size"],
              context_days=ctx)
    if kind == "chronos2":
        from tgc.forecast.chronos2 import Chronos2
        return Chronos2(covariates=cov, **fm)
    if kind == "timesfm25":
        from tgc.forecast.timesfm25 import TimesFM25
        return TimesFM25(covariates=cov, **fm)
    if kind == "moirai2":
        assert not cov, "Moirai-2 is run univariate"
        from tgc.forecast.moirai2 import Moirai2
        return Moirai2(**fm)
    if kind == "tirex":
        assert not cov, "TiRex is univariate"
        from tgc.forecast.tirex import TiRex
        return TiRex(**fm)
    raise ValueError(kind)


def output_name(model: str, seed: int, covariates: bool, context_days: int | None) -> str:
    """Directory name under out/forecasts: model[-s{seed}][-cov][-ctx{N}]."""
    mc = config.load("models")["models"][model]
    name = model
    if "seeds" in mc:
        name += f"-s{seed}"
    if covariates:
        name += "-cov"
    if context_days and context_days != config.load("models")["context_days"]:
        name += f"-ctx{context_days}"
    return name
