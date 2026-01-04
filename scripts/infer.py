from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from brats24.data.scan import scan_case_dir
from brats24.data.transforms import build_infer_transforms
from brats24.engine.modeling import build_model
from brats24.utils.config import load_config
from brats24.utils.io import load_torch
from brats24.utils.postprocess import postprocess_pred_label


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference on a single case.")
    p.add_argument("--config", required=True, help="Path to a YAML config file.")
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config with dotted keys, e.g. infer.overlap=0.75 (repeatable).",
    )
    p.add_argument(
        "--ckpt",
        required=True,
        help="Checkpoint file (.pt/.pth) produced by training (expects a dict with key 'model').",
    )
    p.add_argument(
        "--input",
        required=True,
        help="Either a case directory (with 4 modality NIfTI files) or a 4D NIfTI file (modalities stacked).",
    )
    p.add_argument("--output", required=True, help="Output NIfTI path (e.g., pred.nii.gz).")
    p.add_argument("--device", type=str, default="cuda", help="cuda|cpu|cuda:0")
    p.add_argument("--out_json", type=str, default=None, help="Optional JSON sidecar path.")
    return p.parse_args()


def _load_input(cfg: dict[str, Any], input_path: Path) -> tuple[torch.Tensor, Path]:
    t = build_infer_transforms(cfg)
    if input_path.is_dir():
        modalities = list(cfg["modalities"].keys())
        modality_patterns = {k: v["patterns"] for k, v in cfg["modalities"].items()}
        seg_pattern = cfg.get("seg_pattern", "seg")
        import re

        pat_re = {mod: [re.compile(p, flags=re.IGNORECASE) for p in pats] for mod, pats in modality_patterns.items()}
        seg_re = re.compile(seg_pattern, flags=re.IGNORECASE)
        case = scan_case_dir(
            case_dir=input_path,
            modalities=modalities,
            pat_re=pat_re,
            seg_re=seg_re,
            require_label=False,
        )
        if not case.get("usable", False):
            raise RuntimeError(f"Input case is missing modalities: {case.get('missing')}")
        sample = t({"image": case["image"]})
        ref_path = Path(case["image"][0])
        img = sample["image"]
    else:
        sample = t({"image": str(input_path)})
        ref_path = input_path
        img = sample["image"]

    if img.ndim != 4 or img.shape[0] != 4:
        raise ValueError(f"Expected image as (4,D,H,W) after transforms, got {tuple(img.shape)}")
    return img, ref_path


@torch.inference_mode()
def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config, overrides=args.override)

    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    state = load_torch(Path(args.ckpt), map_location=str(device))
    model.load_state_dict(state["model"])
    model.eval()

    img, ref_path = _load_input(cfg, Path(args.input))
    img_b = img.unsqueeze(0).to(device)

    roi_size = tuple(int(x) for x in cfg["train"]["patch_size"])
    sw_batch_size = int(cfg.get("infer", {}).get("sw_batch_size", 1))
    overlap = float(cfg.get("infer", {}).get("overlap", 0.5))
    logits = sliding_window_inference(
        img_b,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
    )
    pred_label = postprocess_pred_label(cfg, logits.argmax(dim=1, keepdim=True)).detach().cpu().numpy().astype(np.int16)[0, 0]

    ref_img = nib.load(str(ref_path))
    pred_nii = nib.Nifti1Image(pred_label, affine=ref_img.affine, header=ref_img.header)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(pred_nii, str(out_path))

    out: dict[str, Any] = {"input": str(args.input), "output": str(out_path)}
    if args.out_json:
        jpath = Path(args.out_json)
        jpath.parent.mkdir(parents=True, exist_ok=True)
        jpath.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
