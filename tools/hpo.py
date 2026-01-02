from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def _format_override_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, dict)):
        return yaml.safe_dump(v, default_flow_style=True).strip()
    return str(v)


def _override_args(overrides: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for k, v in overrides.items():
        args.extend(["--override", f"{k}={_format_override_value(v)}"])
    return args


def _expand_grid(params: dict[str, Any]) -> list[dict[str, Any]]:
    keys = list(params.keys())
    values = []
    for k in keys:
        v = params[k]
        if not isinstance(v, list):
            raise ValueError(f"Grid param {k!r} must be a list, got {type(v).__name__}")
        values.append(v)
    combos = []
    for prod in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, prod, strict=False)})
    return combos


def _sample_random(params: dict[str, Any], *, n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    runs = []
    for _ in range(n):
        overrides: dict[str, Any] = {}
        for k, spec in params.items():
            if isinstance(spec, list):
                overrides[k] = rng.choice(spec)
                continue
            if isinstance(spec, dict):
                ptype = spec.get("type", "float")
                if ptype == "int":
                    low = int(spec["min"])
                    high = int(spec["max"])
                    step = int(spec.get("step", 1))
                    overrides[k] = rng.randrange(low, high + 1, step)
                else:
                    low = float(spec["min"])
                    high = float(spec["max"])
                    if spec.get("log", False):
                        overrides[k] = math.exp(rng.uniform(math.log(low), math.log(high)))
                    else:
                        overrides[k] = rng.uniform(low, high)
                continue
            overrides[k] = spec
        runs.append(overrides)
    return runs


def _read_last_metrics(metrics_csv: Path) -> dict[str, Any]:
    if not metrics_csv.exists():
        return {}
    with metrics_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows[-1] if rows else {}


def main() -> None:
    p = argparse.ArgumentParser(description="Simple HPO runner (grid/random) without external deps.")
    p.add_argument("--config", required=True, help="Training config YAML.")
    p.add_argument("--space", required=True, help="Search space YAML.")
    p.add_argument("--output_dir", default="runs_hpo", help="Separate output dir for HPO runs.")
    p.add_argument("--max_runs", type=int, default=None, help="Optional cap on number of runs.")
    p.add_argument("--dry_run", action="store_true", help="Print commands only.")
    args = p.parse_args()

    space = yaml.safe_load(Path(args.space).read_text(encoding="utf-8")) or {}
    mode = str(space.get("mode", "grid")).lower()
    params = space.get("params", {}) or {}
    fixed = space.get("fixed_overrides", {}) or {}
    prefix = str(space.get("run_name_prefix", "hpo"))
    seed = int(space.get("seed", 42))
    num_samples = int(space.get("num_samples", 10))

    if mode == "grid":
        runs = _expand_grid(params)
    elif mode == "random":
        runs = _sample_random(params, n=num_samples, seed=seed)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if args.max_runs is not None:
        runs = runs[: int(args.max_runs)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "hpo_results.csv"

    for i, overrides in enumerate(runs, start=1):
        run_name = f"{prefix}_{i:03d}"
        all_overrides = dict(fixed)
        all_overrides.update(overrides)
        all_overrides.setdefault("output_dir", str(output_dir))
        all_overrides.setdefault("run_name", run_name)
        all_overrides.setdefault("train.overwrite", True)

        cmd = [
            sys.executable,
            "-m",
            "brats24.cli",
            "train",
            "--config",
            args.config,
        ] + _override_args(all_overrides)

        if args.dry_run:
            print(" ".join(cmd))
            continue

        print(f"[HPO] Running {run_name} ({i}/{len(runs)})")
        status = "ok"
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            status = f"fail({e.returncode})"

        metrics = _read_last_metrics(output_dir / run_name / "metrics" / "metrics.csv")
        row = {"run_name": run_name, "status": status, **overrides, **metrics}
        _append_csv(results_csv, row)


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
