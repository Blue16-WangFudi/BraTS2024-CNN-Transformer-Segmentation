from __future__ import annotations

from typing import Any

import torch.nn as nn
from monai.networks.nets import UNet


class UNet3D(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        in_channels = 4
        out_channels = int(cfg["num_classes"])
        base = int(cfg.get("model", {}).get("base_channels", 32))
        channels = (base, base * 2, base * 4, base * 8, base * 16)
        strides = (2, 2, 2, 2)
        self.net = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=2,
            norm="INSTANCE",
        )

    def forward(self, x):
        return self.net(x)

