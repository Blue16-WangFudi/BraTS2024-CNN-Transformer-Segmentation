from __future__ import annotations

import torch.nn as nn


def summarize_model(model: nn.Module) -> str:
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"params={params:,} trainable={trainable:,}"

