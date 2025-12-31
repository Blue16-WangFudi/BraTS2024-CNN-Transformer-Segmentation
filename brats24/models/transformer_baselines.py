from __future__ import annotations

from typing import Any

import torch.nn as nn


class UNETRBaseline(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        from monai.networks.nets import UNETR

        self.net = UNETR(
            in_channels=4,
            out_channels=int(cfg["num_classes"]),
            img_size=tuple(int(x) for x in cfg["train"]["patch_size"]),
            feature_size=int(cfg.get("model", {}).get("base_channels", 16)),
            hidden_size=256,
            mlp_dim=512,
            num_heads=4,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
        )

    def forward(self, x):
        return self.net(x)


class SwinUNETRBaseline(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        from monai.networks.nets import SwinUNETR

        self.net = SwinUNETR(
            img_size=tuple(int(x) for x in cfg["train"]["patch_size"]),
            in_channels=4,
            out_channels=int(cfg["num_classes"]),
            feature_size=int(cfg.get("model", {}).get("base_channels", 24)),
            use_checkpoint=False,
        )

    def forward(self, x):
        return self.net(x)

