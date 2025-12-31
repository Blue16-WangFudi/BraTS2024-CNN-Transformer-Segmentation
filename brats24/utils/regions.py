from __future__ import annotations

from typing import Any

import torch


def compute_region_dice(cfg: dict[str, Any], *, pred_logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    regions: dict[str, list[int]] = cfg.get("loss", {}).get("regions") or {}
    if not regions:
        return torch.empty((0,), device=pred_logits.device)

    probs = torch.softmax(pred_logits, dim=1)
    target_idx = target.long().squeeze(1)

    dices = []
    for labels in regions.values():
        labels_t = torch.as_tensor(labels, device=pred_logits.device, dtype=torch.long)
        if labels_t.numel() == 0:
            dices.append(torch.tensor(float("nan"), device=pred_logits.device))
            continue
        pred_r = probs.index_select(dim=1, index=labels_t).sum(dim=1)
        gt_r = torch.isin(target_idx, labels_t).float()
        dims = tuple(range(1, pred_r.ndim))
        intersect = (pred_r * gt_r).sum(dim=dims)
        denom = pred_r.sum(dim=dims) + gt_r.sum(dim=dims)
        dice = (2 * intersect + eps) / (denom + eps)
        dices.append(dice.mean())

    return torch.stack(dices)

