from __future__ import annotations

import argparse
from pathlib import Path

from brats24.data.scan import build_datalist
from brats24.data.splits import make_random_split
from brats24.utils.config import load_config
from brats24.utils.io import as_path, save_json


def main() -> None:
    p = argparse.ArgumentParser(description="Create fixed splits (random) with a k-fold skeleton.")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True, help="Output JSON path.")
    args = p.parse_args()

    cfg = load_config(args.config)
    data_root = as_path(cfg["data_root"])
    modalities = list(cfg["modalities"].keys())
    modality_patterns = {k: v["patterns"] for k, v in cfg["modalities"].items()}
    seg_pattern = cfg.get("seg_pattern", "seg")

    datalist = build_datalist(
        data_root=data_root,
        modalities=modalities,
        modality_patterns=modality_patterns,
        seg_pattern=seg_pattern,
    )
    split = make_random_split(
        datalist,
        seed=int(cfg["split"]["seed"]),
        train_frac=float(cfg["split"]["train_frac"]),
        val_frac=float(cfg["split"]["val_frac"]),
        max_train_cases=cfg["split"].get("max_train_cases"),
        max_val_cases=cfg["split"].get("max_val_cases"),
    )

    out = {"train": split.train, "val": split.val, "kfold": {"enabled": False, "k": None}}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, out)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

