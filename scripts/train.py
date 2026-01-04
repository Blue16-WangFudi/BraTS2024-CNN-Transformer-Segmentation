from __future__ import annotations

import argparse

from brats24.engine.trainer import train
from brats24.utils.config import load_config


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a BraTS 2024 segmentation model.")
    p.add_argument("--config", required=True, help="Path to a YAML config file.")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config with dotted keys, e.g. train.epochs=2 (repeatable).",
    )
    p.add_argument("--data_dir", type=str, default=None, help="Override data_root.")
    p.add_argument("--device", type=str, default="cuda", help="cuda|cpu|cuda:0")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config, overrides=args.override)
    if args.data_dir:
        cfg["data_root"] = args.data_dir
    run_dir = train(cfg, device=args.device)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
