from __future__ import annotations

import torch
import torch.nn as nn


class ModalityGate(nn.Module):
    def __init__(self, num_modalities: int, *, hidden: int = 32):
        super().__init__()
        self.num_modalities = int(num_modalities)
        self.mlp = nn.Sequential(
            nn.Linear(self.num_modalities, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.num_modalities),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected x as (B, M, D, H, W), got {tuple(x.shape)}")
        b, m, *_ = x.shape
        if m != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {m}")
        stats = x.mean(dim=(2, 3, 4))
        w = self.mlp(stats).view(b, m, 1, 1, 1)
        return x * w


def apply_modality_dropout(x: torch.Tensor, *, p: float, training: bool) -> torch.Tensor:
    if (not training) or p <= 0:
        return x
    if x.ndim != 5:
        return x
    b, m, _, _, _ = x.shape
    if m <= 1:
        return x
    do_drop = torch.rand((b,), device=x.device) < float(p)
    if not do_drop.any():
        return x
    idx = torch.randint(low=0, high=m, size=(int(do_drop.sum().item()),), device=x.device)
    x = x.clone()
    x[do_drop, idx] = 0
    return x
