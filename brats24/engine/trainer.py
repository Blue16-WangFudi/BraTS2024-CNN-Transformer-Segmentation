from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from monai.data import Dataset, list_data_collate
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from brats24.data.report import generate_dataset_report
from brats24.data.scan import build_datalist
from brats24.data.splits import make_random_split
from brats24.data.transforms import build_train_transforms, build_val_transforms
from brats24.engine.modeling import build_model
from brats24.losses.dice_ce import dice_ce_loss
from brats24.losses.region_aux import region_aux_dice_loss
from brats24.utils.config import resolved_run_dir
from brats24.utils.env_dump import dump_environment
from brats24.utils.io import as_path, load_torch, save_csv_row, save_json, save_torch
from brats24.utils.postprocess import postprocess_pred_label
from brats24.utils.regions import compute_region_dice
from brats24.utils.seed import seed_everything
from brats24.utils.tb_export import export_scalars_to_png
from brats24.utils.visualization import save_case_mid_slices_png, save_pred_overlay_png


def train(cfg: dict[str, Any], *, device: str = "cuda") -> Path:
    seed_everything(int(cfg.get("seed", 42)))

    run_dir = resolved_run_dir(cfg)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    dump_environment(run_dir / "artifacts" / "env")

    dataset_report = generate_dataset_report(cfg)
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    save_json(Path("artifacts") / "dataset_report.json", dataset_report)
    save_json(run_dir / "artifacts" / "dataset_report.json", dataset_report)

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
        raise RuntimeError(f"No usable cases found under data_root={data_root}")

    split = make_random_split(
        datalist,
        seed=int(cfg["split"]["seed"]),
        train_frac=float(cfg["split"]["train_frac"]),
        val_frac=float(cfg["split"]["val_frac"]),
        max_train_cases=cfg["split"].get("max_train_cases"),
        max_val_cases=cfg["split"].get("max_val_cases"),
    )
    train_list, val_list = split.train, split.val
    if not train_list or not val_list:
        raise RuntimeError("Empty train/val split; adjust split settings.")

    save_json(run_dir / "artifacts" / "splits.json", {"train": train_list, "val": val_list})

    train_ds = Dataset(train_list, transform=build_train_transforms(cfg))
    val_ds = Dataset(val_list, transform=build_val_transforms(cfg))

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"]["num_workers"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    amp_enabled = bool(cfg["train"].get("amp", True)) and ("cuda" in device) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    num_classes = int(cfg["num_classes"])
    dice_metric = DiceMetric(include_background=False, reduction="none")

    tb = SummaryWriter(log_dir=str(run_dir / "tb"))
    best_metric = -1.0
    best_epoch = -1

    _save_dataset_sample_figure(cfg, train_list[0], run_dir / "figures" / "dataset_sample_mid_slices.png")

    epochs = int(cfg["train"]["epochs"])
    roi_size = tuple(int(x) for x in cfg["train"]["patch_size"])
    sw_batch_size = int(cfg.get("infer", {}).get("sw_batch_size", 1))
    overlap = float(cfg.get("infer", {}).get("overlap", 0.5))

    metrics_csv = run_dir / "metrics" / "metrics.csv"

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"train {epoch}/{epochs}", leave=False)
        for batch in pbar:
            img = batch["image"].to(device)
            lbl = batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(img)
                loss = dice_ce_loss(
                    logits,
                    lbl,
                    dice_weight=float(cfg["loss"]["dice_weight"]),
                    ce_weight=float(cfg["loss"]["ce_weight"]),
                )
                if cfg.get("enable_region_aux", False) and cfg.get("loss", {}).get("regions"):
                    loss = loss + float(cfg["loss"]["region_aux_weight"]) * region_aux_dice_loss(cfg, logits, lbl)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.detach().item())
            steps += 1
            pbar.set_postfix(loss=f"{running / max(steps,1):.4f}")

        train_loss = running / max(steps, 1)
        tb.add_scalar("loss/train", train_loss, epoch)

        model.eval()
        val_losses = []
        dices = []
        region_dices = []

        with torch.inference_mode():
            for batch in tqdm(val_loader, desc=f"val {epoch}/{epochs}", leave=False):
                img = batch["image"].to(device)
                lbl = batch["label"].to(device)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    logits = sliding_window_inference(
                        img,
                        roi_size=roi_size,
                        sw_batch_size=sw_batch_size,
                        predictor=model,
                        overlap=overlap,
                    )

                vloss = dice_ce_loss(
                    logits,
                    lbl,
                    dice_weight=float(cfg["loss"]["dice_weight"]),
                    ce_weight=float(cfg["loss"]["ce_weight"]),
                )
                val_losses.append(vloss.detach().cpu())

                pred_label = postprocess_pred_label(cfg, logits.argmax(dim=1, keepdim=True))
                y_pred = F.one_hot(pred_label.squeeze(1).long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
                y_true = F.one_hot(lbl.squeeze(1).long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
                dice = dice_metric(y_pred=y_pred, y=y_true)
                dices.append(dice.detach().cpu())

                if cfg.get("loss", {}).get("regions"):
                    region_dices.append(compute_region_dice(cfg, pred_logits=logits.detach(), target=lbl.detach()).cpu())

        val_loss = float(torch.stack(val_losses).mean().item()) if val_losses else float("nan")
        dices_t = torch.cat(dices, dim=0) if dices else torch.empty((0, num_classes - 1))
        per_class = dices_t.mean(dim=0).tolist() if dices else []
        mean_dice = float(torch.nanmean(dices_t).item()) if dices else float("nan")

        tb.add_scalar("loss/val", val_loss, epoch)
        tb.add_scalar("dice/val_mean", mean_dice, epoch)
        for i, d in enumerate(per_class, start=1):
            tb.add_scalar(f"dice/class_{i}", float(d), epoch)

        rd = None
        if region_dices:
            rd = torch.stack(region_dices, dim=0).mean(dim=0)
            for name, v in zip(cfg["loss"]["regions"].keys(), rd.tolist(), strict=False):
                tb.add_scalar(f"dice_region/{name}", float(v), epoch)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mean_dice": mean_dice,
            "sec_per_epoch": time.time() - t0,
        }
        for i, d in enumerate(per_class, start=1):
            row[f"val_dice_class_{i}"] = float(d)
        if rd is not None:
            for name, v in zip(cfg["loss"]["regions"].keys(), rd.tolist(), strict=False):
                row[f"val_region_dice_{name}"] = float(v)

        save_csv_row(metrics_csv, row)

        save_torch(run_dir / "checkpoints" / "last.pt", {"model": model.state_dict(), "epoch": epoch, "cfg": cfg})
        if mean_dice > best_metric:
            best_metric = mean_dice
            best_epoch = epoch
            save_torch(run_dir / "checkpoints" / "best.pt", {"model": model.state_dict(), "epoch": epoch, "cfg": cfg})

    tb.flush()
    tb.close()

    save_json(run_dir / "artifacts" / "best.json", {"best_epoch": best_epoch, "best_val_mean_dice": best_metric})

    export_scalars_to_png(run_dir / "tb", run_dir / "figures")
    _save_prediction_figure(cfg, run_dir, device=device)

    return run_dir


def _save_dataset_sample_figure(cfg: dict[str, Any], sample_item: dict[str, Any], out_png: Path) -> None:
    t = build_val_transforms(cfg)
    out = t(sample_item)
    save_case_mid_slices_png(out["image"], out["label"], out_png)


def _save_prediction_figure(cfg: dict[str, Any], run_dir: Path, *, device: str) -> None:
    model = build_model(cfg).to(device)
    state = load_torch(run_dir / "checkpoints" / "best.pt", map_location=device)
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
    if not split.val:
        return

    ds = Dataset(split.val[:1], transform=build_val_transforms(cfg))
    sample = ds[0]
    img = sample["image"].unsqueeze(0).to(device)
    lbl = sample["label"]
    roi_size = tuple(int(x) for x in cfg["train"]["patch_size"])
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=("cuda" in device) and torch.cuda.is_available()):
            logits = sliding_window_inference(
                img,
                roi_size=roi_size,
                sw_batch_size=int(cfg.get("infer", {}).get("sw_batch_size", 1)),
                predictor=model,
                overlap=float(cfg.get("infer", {}).get("overlap", 0.5)),
            )
        pred = postprocess_pred_label(cfg, logits.argmax(dim=1, keepdim=True)).squeeze(0)
    save_pred_overlay_png(sample["image"], lbl, pred, run_dir / "figures" / "pred_vs_gt_overlay.png")
