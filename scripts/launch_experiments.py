#!/usr/bin/env python3
"""
Simple launcher to run multiple Hydra experiments with different hyperparameters.

Edit GRID below to set the search space. By default runs are executed sequentially.
Example:
    python scripts/launch_experiments.py --dry-run
    python scripts/launch_experiments.py --max-runs 5
"""

from itertools import product
import argparse
import shlex
import subprocess
from typing import Iterable, List


# Base command to run (Hydra overrides will be appended)
BASE_CMD = "dora run"

# Hyperparameter grid: keys are Hydra override strings, values are lists to sweep.
GRID = {
    "train.group_random_pct": [0.2, 0.5, 0.7]
}


def build_runs() -> Iterable[List[str]]:
    keys = list(GRID.keys())
    value_lists = [GRID[k] for k in keys]
    for values in product(*value_lists):
        overrides = [f"{k}={v}" for k, v in zip(keys, values)]
        yield overrides


def launch(cmd: List[str], dry_run: bool) -> int:
    print("Launching:", " ".join(cmd))
    if dry_run:
        return 0
    proc = subprocess.run(cmd)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Launch multiple Hydra experiments.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit how many runs to launch.")
    parser.add_argument("--base-cmd", type=str, default=BASE_CMD, help="Base command before overrides.")
    args = parser.parse_args()

    base_parts = shlex.split(args.base_cmd)
    for idx, overrides in enumerate(build_runs(), start=1):
        if args.max_runs is not None and idx > args.max_runs:
            break
        cmd = base_parts + overrides
        code = launch(cmd, args.dry_run)
        if code != 0:
            print(f"Run {idx} failed with exit code {code}, stopping.")
            break


if __name__ == "__main__":
    main()
