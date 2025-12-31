from __future__ import annotations

from typing import Any

import torch.nn as nn


def build_model(cfg: dict[str, Any]) -> nn.Module:
    name = cfg["model_name"]
    if name == "unet3d":
        from brats24.models.unet3d import UNet3D

        return UNet3D(cfg)
    if name == "unetr":
        from brats24.models.transformer_baselines import UNETRBaseline

        return UNETRBaseline(cfg)
    if name == "swinunetr":
        from brats24.models.transformer_baselines import SwinUNETRBaseline

        return SwinUNETRBaseline(cfg)
    if name == "cnn_transformer_unet":
        from brats24.models.cnn_transformer_unet import CNNTransformerUNet

        return CNNTransformerUNet(cfg)

    raise ValueError(f"Unknown model_name={name!r}")

