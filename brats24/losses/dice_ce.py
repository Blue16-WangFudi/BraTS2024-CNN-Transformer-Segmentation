from __future__ import annotations

import torch
import torch.nn.functional as F


def _soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    target_1h = F.one_hot(target.long().squeeze(1), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    dims = tuple(range(2, probs.ndim))
    intersect = (probs * target_1h).sum(dim=dims)
    denom = probs.sum(dim=dims) + target_1h.sum(dim=dims)

    dice = (2 * intersect + eps) / (denom + eps)
    dice_fg = dice[:, 1:]
    return 1.0 - dice_fg.mean()


def dice_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    dice_weight: float = 1.0,
    ce_weight: float = 1.0,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target.long().squeeze(1))
    dice = _soft_dice_loss(logits, target)
    return float(dice_weight) * dice + float(ce_weight) * ce

