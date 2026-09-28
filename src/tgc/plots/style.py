"""Fixed colour per model and marker per calibration variant (Okabe-Ito, colour-blind safe)."""
from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {
    "chronos2": "#E69F00", "timesfm25": "#56B4E9", "moirai2": "#009E73", "tirex": "#CC79A7",
    "lgbm": "#0072B2", "nhits": "#D55E00", "patchtst": "#F0E442", "seasonal_naive": "#999999",
    "operator": "#000000",
}
LABELS = {
    "chronos2": "Chronos-2", "timesfm25": "TimesFM-2.5", "moirai2": "Moirai-2.0", "tirex": "TiRex",
    "lgbm": "LightGBM", "nhits": "N-HiTS", "patchtst": "PatchTST", "seasonal_naive": "Seasonal naive",
    "operator": "Operator",
}
MARKERS = {"native": "o", "split_cqr": "s", "nexcp": "D", "aci": "^", "pid": "v", "deterministic": "x"}
VLABELS = {"native": "Native", "split_cqr": "Split CQR", "nexcp": "NexCP", "aci": "ACI", "pid": "PID",
           "deterministic": "Deterministic"}
ORDER = list(COLORS)
WIDTH = 3.5  # IEEE single column, inches


def family(model: str) -> str:
    """'lgbm-s0-cov' -> 'lgbm'."""
    return re.split(r"-(s\d+|cov|ctx\d+)", model)[0]


def primary(model: str) -> bool:
    """Main-comparison runs: seed 0 (or unseeded), univariate, default context."""
    return ("-s" not in model or "-s0" in model) and "-cov" not in model and "-ctx" not in model


def setup():
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.dpi": 300,
    })


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
