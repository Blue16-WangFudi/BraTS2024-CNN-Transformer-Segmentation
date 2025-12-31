from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.data import Dataset
from torch.utils.data import DataLoader

from brats24.data.scan import build_datalist
from brats24.data.splits import make_random_split
from brats24.data.transforms import build_val_transforms
from brats24.engine.modeling import build_model
from brats24.utils.io import as_path, load_torch, save_json
from brats24.utils.postprocess import postprocess_pred_label
from brats24.utils.regions import compute_region_dice


@torch.inference_mode()
def evaluate(cfg: dict[str, Any], *, run_dir: Path, ckpt: str, device: str = "cuda") -> dict[str, Any]:
    run_dir = Path(run_dir)
    model = build_model(cfg).to(device)
    ckpt_path = Path(ckpt)
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / ckpt
    state = load_torch(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

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
    val_list = split.val
    if not val_list:
        raise RuntimeError("Validation split is empty; cannot evaluate.")

    val_ds = Dataset(val_list, transform=build_val_transforms(cfg))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    num_classes = int(cfg["num_classes"])
    dice_metric = DiceMetric(include_background=False, reduction="none")

    roi_size = tuple(int(x) for x in cfg["train"]["patch_size"])
    sw_batch_size = int(cfg.get("infer", {}).get("sw_batch_size", 1))
    overlap = float(cfg.get("infer", {}).get("overlap", 0.5))

    dices = []
    region_dices = []
    for batch in val_loader:
        img = batch["image"].to(device)
        lbl = batch["label"].to(device)

        logits = sliding_window_inference(img, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=model, overlap=overlap)
        pred_label = postprocess_pred_label(cfg, logits.argmax(dim=1, keepdim=True))
        y_pred = F.one_hot(pred_label.squeeze(1).long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        y_true = F.one_hot(lbl.squeeze(1).long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

        dice = dice_metric(y_pred=y_pred, y=y_true)
        dices.append(dice.detach().cpu())

        if cfg.get("loss", {}).get("regions"):
            region_dices.append(compute_region_dice(cfg, pred_logits=logits.detach(), target=lbl.detach()).cpu())

    dices_t = torch.cat(dices, dim=0)
    mean_per_class = dices_t.mean(dim=0).tolist()
    mean_dice = float(torch.nanmean(dices_t).item())

    out: dict[str, Any] = {"val_mean_dice": mean_dice, "val_per_class_dice": mean_per_class}
    if region_dices:
        rd = torch.stack(region_dices, dim=0).mean(dim=0)
        out["val_region_names"] = list(cfg["loss"]["regions"].keys())
        out["val_region_dice"] = rd.tolist()

    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "metrics" / "eval_metrics.json", out)

    return out
