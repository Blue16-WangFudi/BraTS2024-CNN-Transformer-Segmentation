from __future__ import annotations

from pathlib import Path
from typing import Any

from brats24.data.scan import build_datalist
from brats24.data.transforms import build_val_transforms
from brats24.utils.io import as_path
from brats24.utils.visualization import save_case_mid_slices_png


def vis_samples(cfg: dict[str, Any], *, out_dir: str | None = None) -> Path:
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
    if not datalist:
        raise RuntimeError("No usable cases found for visualization.")

    t = build_val_transforms(cfg)
    sample = t(datalist[0])

    run_name = cfg.get("run_name", "vis")
    default_out = Path(cfg.get("output_dir", "runs")) / run_name / "figures"
    out_path = Path(out_dir) if out_dir is not None else default_out
    out_path.mkdir(parents=True, exist_ok=True)

    png = out_path / "dataset_sample_mid_slices.png"
    save_case_mid_slices_png(sample["image"], sample["label"], png)
    return png

