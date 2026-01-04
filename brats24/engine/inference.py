from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from brats24.data.scan import scan_case_dir
from brats24.data.transforms import build_infer_transforms, build_val_transforms
from brats24.engine.modeling import build_model
from brats24.utils.io import as_path, load_torch, save_json
from brats24.utils.postprocess import postprocess_pred_label
from brats24.utils.visualization import save_pred_overlay_png


@torch.inference_mode()
def infer(
    cfg: dict[str, Any],
    *,
    run_dir: Path,
    ckpt: str = "best.pt",
    case_id: str | None = None,
    case_dir: str | None = None,
    device: str = "cuda",
    out_dir: str | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    ckpt_path = Path(ckpt)
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / ckpt

    model = build_model(cfg).to(device)
    state = load_torch(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    data_root = as_path(cfg["data_root"])
    if case_dir is not None:
        case_path = Path(case_dir)
    elif case_id is not None:
        case_path = data_root / case_id
    else:
        raise ValueError("Provide either case_id or case_dir.")
    if not case_path.exists():
        raise FileNotFoundError(str(case_path))

    modalities = list(cfg["modalities"].keys())
    modality_patterns = {k: v["patterns"] for k, v in cfg["modalities"].items()}
    seg_pattern = cfg.get("seg_pattern", "seg")
    pat_re = {mod: [re.compile(p, flags=re.IGNORECASE) for p in pats] for mod, pats in modality_patterns.items()}
    seg_re = re.compile(seg_pattern, flags=re.IGNORECASE)
    case = scan_case_dir(case_dir=case_path, modalities=modalities, pat_re=pat_re, seg_re=seg_re, require_label=False)
    if not case.get("usable", False):
        raise RuntimeError(f"Case not usable: {case['case_id']} (missing={case.get('missing')})")

    has_label = case.get("label") is not None
    item = {"image": case["image"], "label": case.get("label")}
    if has_label:
        sample = build_val_transforms(cfg)(item)
        label = sample["label"]
    else:
        sample = build_infer_transforms(cfg)({"image": case["image"]})
        label = None

    img = sample["image"].unsqueeze(0).to(device)
    roi_size = tuple(int(x) for x in cfg["train"]["patch_size"])
    sw_batch_size = int(cfg.get("infer", {}).get("sw_batch_size", 1))
    overlap = float(cfg.get("infer", {}).get("overlap", 0.5))
    logits = sliding_window_inference(img, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=model, overlap=overlap)
    pred_label = postprocess_pred_label(cfg, logits.argmax(dim=1, keepdim=True)).detach().cpu().numpy().astype(np.int16)[0, 0]

    out_root = Path(out_dir) if out_dir is not None else (run_dir / "predictions")
    out_root.mkdir(parents=True, exist_ok=True)

    ref_img = nib.load(case["image"][0])
    pred_nii = nib.Nifti1Image(pred_label, affine=ref_img.affine, header=ref_img.header)
    pred_path = out_root / f"{case['case_id']}_pred.nii.gz"
    nib.save(pred_nii, pred_path)

    out: dict[str, Any] = {"case_id": case["case_id"], "pred_path": str(pred_path)}
    if label is not None:
        png = out_root / f"{case['case_id']}_pred_vs_gt.png"
        save_pred_overlay_png(sample["image"], label, torch.from_numpy(pred_label).unsqueeze(0), png)
        out["overlay_png"] = str(png)

    save_json(out_root / f"{case['case_id']}_infer.json", out)
    return out
