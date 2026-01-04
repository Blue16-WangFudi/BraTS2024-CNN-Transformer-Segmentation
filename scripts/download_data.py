from __future__ import annotations

import argparse
import os
from pathlib import Path

from brats24.data.scan import scan_brats_dataset
from brats24.utils.config import load_config


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BraTS 2024 data instructions and directory checks (no download).")
    p.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Dataset root directory (defaults to $DATASET_DIR or ./brats2024-small-dataset).",
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Config file used for modality patterns and seg_pattern.",
    )
    p.add_argument("--check", action="store_true", help="Scan the directory and validate basic structure.")
    p.add_argument(
        "--expect_labels",
        action="store_true",
        help="Require that each case contains a segmentation label file (training set).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir or os.environ.get("DATASET_DIR", "brats2024-small-dataset")

    print("BraTS 2024 dataset acquisition:")
    print("- Register and download from Kaggle: https://www.kaggle.com/competitions/brats2024")
    print(f"- Set DATASET_DIR to the extracted dataset root, or place it at: {data_dir}")
    print()

    if not args.check:
        return

    root = Path(data_dir)
    if not root.exists():
        raise SystemExit(f"Dataset directory not found: {root}")

    cfg = load_config(args.config)
    modalities = list(cfg["modalities"].keys())
    modality_patterns = {k: v["patterns"] for k, v in cfg["modalities"].items()}
    seg_pattern = cfg.get("seg_pattern", "seg")

    report = scan_brats_dataset(
        data_root=root,
        modalities=modalities,
        modality_patterns=modality_patterns,
        seg_pattern=seg_pattern,
        require_label=bool(args.expect_labels),
    )
    total = int(report.get("total_case_dirs", 0))
    usable = sum(1 for c in report.get("cases", []) if c.get("usable", False))
    print(f"Found {total} case directories; usable={usable}.")

    if args.expect_labels and usable == 0:
        raise SystemExit("No usable labeled cases found. Check that you're pointing at the training set.")


if __name__ == "__main__":
    main()
