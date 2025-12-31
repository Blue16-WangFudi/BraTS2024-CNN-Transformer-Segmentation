from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_case_mid_slices_png(image: torch.Tensor, label: torch.Tensor, out_path: Path) -> None:
    img = _to_numpy(image)  # (C, D, H, W)
    lbl = _to_numpy(label)  # (1, D, H, W)
    if img.ndim != 4 or lbl.ndim != 4:
        raise ValueError(f"Expected image (C,D,H,W) and label (1,D,H,W), got {img.shape} {lbl.shape}")

    c, d, _, _ = img.shape
    z = d // 2
    fig, axes = plt.subplots(1, c, figsize=(4 * c, 4))
    if c == 1:
        axes = [axes]
    for i in range(c):
        axes[i].imshow(img[i, z], cmap="gray")
        axes[i].imshow(lbl[0, z], cmap="jet", alpha=0.25, vmin=0)
        axes[i].set_title(f"modality_{i}")
        axes[i].axis("off")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_pred_overlay_png(image: torch.Tensor, label: torch.Tensor, pred: torch.Tensor, out_path: Path) -> None:
    img = _to_numpy(image)
    lbl = _to_numpy(label)
    prd = _to_numpy(pred)
    z = img.shape[1] // 2
    base = img[0, z]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(base, cmap="gray")
    ax.imshow(lbl[0, z], cmap="jet", alpha=0.25, vmin=0)
    ax.imshow(prd[0, z], cmap="spring", alpha=0.25, vmin=0)
    ax.set_title("GT (jet) vs Pred (spring)")
    ax.axis("off")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

