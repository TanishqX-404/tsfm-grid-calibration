"""M4 check: two identical dry runs must give identical outputs.

    python -m tgc.forecast.determinism --model chronos2
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from tgc import config
from tgc.forecast import registry
from tgc.forecast.run import main as run_main
from tgc.forecast.run import out_root


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    name = registry.output_name(args.model, args.seed, False, None)
    frames = []
    for _ in range(2):
        run_main(["--model", args.model, "--seed", str(args.seed), "--dry-run", "--force"])
        d = out_root(True) / name
        frames.append(pd.concat([pd.read_parquet(f) for f in sorted(d.glob("*/*.parquet"))], ignore_index=True))
    same = frames[0].equals(frames[1])
    diff = (frames[0].select_dtypes("number") - frames[1].select_dtypes("number")).abs().max().max()
    print(f"{name}: identical={same} max_abs_diff={diff}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
