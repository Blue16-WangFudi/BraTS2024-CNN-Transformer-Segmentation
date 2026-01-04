from __future__ import annotations

import argparse
import json
from pathlib import Path

from brats24.engine.evaluator import evaluate
from brats24.utils.config import load_config


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    p.add_argument("--config", required=True, help="Path to a YAML config file.")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config with dotted keys, e.g. split.max_val_cases=5 (repeatable).",
    )
    p.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Run directory (defaults to deriving from --ckpt if possible).",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default="best.pt",
        help="Checkpoint path or name within <run_dir>/checkpoints/.",
    )
    p.add_argument("--data_dir", type=str, default=None, help="Override data_root.")
    p.add_argument("--device", type=str, default="cuda", help="cuda|cpu|cuda:0")
    return p.parse_args()


def _infer_run_dir(run_dir: str | None, ckpt: str) -> Path:
    if run_dir is not None:
        return Path(run_dir)
    ckpt_path = Path(ckpt)
    if ckpt_path.exists() and ckpt_path.parent.name == "checkpoints":
        return ckpt_path.parent.parent
    raise SystemExit("Provide --run_dir, or pass an existing --ckpt inside a run's checkpoints/ directory.")


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config, overrides=args.override)
    if args.data_dir:
        cfg["data_root"] = args.data_dir

    run_dir = _infer_run_dir(args.run_dir, args.ckpt)
    metrics = evaluate(cfg, run_dir=run_dir, ckpt=args.ckpt, device=args.device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
