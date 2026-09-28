"""Record the software/hardware environment next to outputs (Section 11.6)."""
from __future__ import annotations

import importlib
import json
import platform
import subprocess
from datetime import datetime, timezone

PACKAGES = ["numpy", "pandas", "pyarrow", "lightgbm", "torch", "chronos", "timesfm", "uni2ts",
            "tirex", "neuralforecast", "cvxpy", "scipy"]


def snapshot(extra: dict | None = None) -> dict:
    env = {"time_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
           "platform": platform.platform(), "packages": {}}
    for p in PACKAGES:
        try:
            env["packages"][p] = getattr(importlib.import_module(p), "__version__", "installed")
        except Exception:
            env["packages"][p] = None
    try:
        import torch
        env["cuda"] = torch.version.cuda
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        env["gpu"] = None
    try:
        env["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                            "--format=csv,noheader"], capture_output=True, text=True,
                                           timeout=10).stdout.strip()
    except Exception:
        env["nvidia_smi"] = None
    try:
        env["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    env.update(extra or {})
    return env


def write(path, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot(extra), indent=2))
