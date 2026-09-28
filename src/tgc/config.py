"""YAML config loading. Every setting lives in configs/*.yaml; code holds no dates or paths."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("TGC_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = Path(os.environ.get("TGC_CONFIG_DIR", ROOT / "configs"))


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    """Load configs/{name}.yaml (cached)."""
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def path(p: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def series_ids(regions: list[str] | None = None) -> list[str]:
    cfg = load("data")
    out = []
    for reg, rc in cfg["regions"].items():
        if regions and reg not in regions:
            continue
        out += [f"{reg}_{t}" for t in rc["targets"]]
    return out


def split_series(series_id: str) -> tuple[str, str]:
    region, target = series_id.rsplit("_", 1)
    return region, target
