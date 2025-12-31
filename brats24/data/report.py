from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from brats24.data.scan import scan_brats_dataset
from brats24.utils.io import as_path


def generate_dataset_report(cfg: dict[str, Any]) -> dict[str, Any]:
    data_root = as_path(cfg["data_root"])
    modalities = list(cfg["modalities"].keys())
    modality_patterns = {k: v["patterns"] for k, v in cfg["modalities"].items()}
    seg_pattern = cfg.get("seg_pattern", "seg")

    scan = scan_brats_dataset(
        data_root=data_root,
        modalities=modalities,
        modality_patterns=modality_patterns,
        seg_pattern=seg_pattern,
    )

    missing = Counter()
    multi_match = defaultdict(list)
    for rec in scan["cases"]:
        for k, v in rec.get("missing", {}).items():
            missing[k] += v
        for k, v in rec.get("multi_match", {}).items():
            if v:
                multi_match[k].append({"case_id": rec["case_id"], "matches": v})

    label_values_sample = []
    usable_cases = [c for c in scan["cases"] if c.get("usable", False)]
    skipped_cases = [c["case_id"] for c in scan["cases"] if not c.get("usable", False)]
    for c in usable_cases[:10]:
        try:
            seg_path = Path(c["label"])
            arr = np.asanyarray(nib.load(seg_path).dataobj)
            uniq = np.unique(arr.astype(np.int64, copy=False))
            label_values_sample.append({"case_id": c["case_id"], "unique_values": uniq.tolist()})
        except Exception as e:  # noqa: BLE001
            label_values_sample.append({"case_id": c["case_id"], "error": repr(e)})

    return {
        "data_root": str(data_root),
        "total_case_dirs": scan["total_case_dirs"],
        "usable_cases": len(usable_cases),
        "skipped_cases": scan["total_case_dirs"] - len(usable_cases),
        "skipped_case_ids": skipped_cases,
        "missing_file_stats": dict(missing),
        "multi_match_examples": dict(multi_match),
        "label_unique_values_sample_first10": label_values_sample,
    }
