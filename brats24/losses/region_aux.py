from __future__ import annotations

from typing import Any

import torch


def _soft_dice_binary(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(2, pred.ndim))
    intersect = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersect + eps) / (denom + eps)
    return 1.0 - dice.mean()


def region_aux_dice_loss(cfg: dict[str, Any], logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    regions: dict[str, list[int]] = cfg["loss"]["regions"]
    probs = torch.softmax(logits, dim=1)
    target_idx = target.long().squeeze(1)

    losses = []
    for labels in regions.values():
        labels_t = torch.as_tensor(labels, device=probs.device, dtype=torch.long)
        if labels_t.numel() == 0:
            continue
        region_prob = probs.index_select(dim=1, index=labels_t).sum(dim=1, keepdim=True)
        region_gt = torch.isin(target_idx, labels_t).unsqueeze(1).float()
        losses.append(_soft_dice_binary(region_prob, region_gt))

    if not losses:
        return torch.zeros((), device=logits.device)
    return torch.stack(losses).mean()

