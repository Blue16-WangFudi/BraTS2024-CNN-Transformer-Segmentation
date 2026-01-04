from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _test_imports() -> None:
    import brats24  # noqa: F401


def _test_scan_case_dir_unlabeled_ok() -> None:
    from brats24.data.scan import scan_case_dir

    modalities = ["t1", "t1ce", "t2", "flair"]
    modality_patterns = {
        "t1": ["t1n", "t1w", "t1"],
        "t1ce": ["t1ce", "t1c"],
        "t2": ["t2w", "t2"],
        "flair": ["t2f", "flair"],
    }
    pat_re = {mod: [re.compile(p, flags=re.IGNORECASE) for p in pats] for mod, pats in modality_patterns.items()}
    seg_re = re.compile("seg", flags=re.IGNORECASE)

    with tempfile.TemporaryDirectory() as tmp:
        case_dir = Path(tmp)
        for name in ["case-t1n.nii.gz", "case-t1c.nii.gz", "case-t2w.nii.gz", "case-t2f.nii.gz"]:
            (case_dir / name).write_bytes(b"")

        unlabeled = scan_case_dir(
            case_dir=case_dir,
            modalities=modalities,
            pat_re=pat_re,
            seg_re=seg_re,
            require_label=False,
        )
        _assert(unlabeled["usable"] is True, f"Expected usable unlabeled case, got: {unlabeled}")
        _assert(unlabeled["has_label"] is False, "Expected has_label=False when no seg file exists.")

        labeled_required = scan_case_dir(
            case_dir=case_dir,
            modalities=modalities,
            pat_re=pat_re,
            seg_re=seg_re,
            require_label=True,
        )
        _assert(labeled_required["usable"] is False, "Expected usable=False when labels are required but missing.")


def _test_model_forward() -> None:
    import torch

    from brats24.engine.modeling import build_model

    cfg = {
        "model_name": "cnn_transformer_unet",
        "num_classes": 4,
        "train": {"patch_size": [32, 32, 32]},
        "model": {
            "base_channels": 4,
            "modality_dropout_p": 0.0,
            "transformer": {
                "num_layers": 1,
                "num_heads": 4,
                "dim_feedforward": 64,
                "dropout": 0.0,
                "pos_encoding": "learned",
            },
        },
        "enable_modality_gate": False,
        "enable_transformer": True,
        "fusion_mode": "serial",
    }

    model = build_model(cfg).eval()
    x = torch.randn(1, 4, 32, 32, 32)
    with torch.inference_mode():
        y = model(x)
    _assert(tuple(y.shape) == (1, 4, 32, 32, 32), f"Unexpected output shape: {tuple(y.shape)}")


def main() -> None:
    _test_imports()
    _test_scan_case_dir_unlabeled_ok()
    _test_model_forward()
    print("smoke_test: ok")


if __name__ == "__main__":
    main()
