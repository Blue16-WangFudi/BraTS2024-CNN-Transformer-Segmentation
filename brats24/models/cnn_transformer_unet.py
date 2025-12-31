from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from monai.networks.blocks import Convolution

from brats24.models.fusion import EmbeddedTransformerBlock, ParallelFusionPlaceholder, SerialBottleneckTransformer
from brats24.models.modality_gate import ModalityGate, apply_modality_dropout


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            Convolution(
                spatial_dims=3,
                in_channels=in_ch,
                out_channels=out_ch,
                strides=1,
                kernel_size=3,
                padding=1,
                act="PRELU",
                norm="INSTANCE",
            ),
            Convolution(
                spatial_dims=3,
                in_channels=out_ch,
                out_channels=out_ch,
                strides=1,
                kernel_size=3,
                padding=1,
                act="PRELU",
                norm="INSTANCE",
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=2), _ConvBlock(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class _Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = _ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            x = torch.nn.functional.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class CNNTransformerUNet(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        base = int(cfg.get("model", {}).get("base_channels", 24))
        in_ch = 4
        out_ch = int(cfg["num_classes"])

        self.enable_modality_gate = bool(cfg.get("enable_modality_gate", True))
        self.modality_dropout_p = float(cfg.get("model", {}).get("modality_dropout_p", 0.0))
        self.enable_transformer = bool(cfg.get("enable_transformer", True))
        self.fusion_mode = str(cfg.get("fusion_mode", "serial")).lower()

        self.gate = ModalityGate(in_ch) if self.enable_modality_gate else nn.Identity()

        self.inc = _ConvBlock(in_ch, base)
        self.down1 = _Down(base, base * 2)
        self.down2 = _Down(base * 2, base * 4)
        self.down3 = _Down(base * 4, base * 8)
        self.down4 = _Down(base * 8, base * 16)

        ps = cfg.get("train", {}).get("patch_size", [96, 96, 96])
        pd, ph, pw = (int(ps[0]), int(ps[1]), int(ps[2]))
        serial_max_tokens = max(1, (pd // 16) * (ph // 16) * (pw // 16))
        embedded_max_tokens = max(1, (pd // 8) * (ph // 8) * (pw // 8))

        tcfg = cfg.get("model", {}).get("transformer", {})
        if self.enable_transformer:
            self.serial_tf = SerialBottleneckTransformer(
                channels=base * 16,
                max_tokens=serial_max_tokens,
                num_layers=int(tcfg.get("num_layers", 1)),
                num_heads=int(tcfg.get("num_heads", 4)),
                dim_feedforward=int(tcfg.get("dim_feedforward", 256)),
                dropout=float(tcfg.get("dropout", 0.1)),
                pos_encoding=str(tcfg.get("pos_encoding", "learned")),
            )
            self.embedded_tf = EmbeddedTransformerBlock(
                channels=base * 8,
                max_tokens=embedded_max_tokens,
                num_layers=int(tcfg.get("num_layers", 1)),
                num_heads=int(tcfg.get("num_heads", 4)),
                dim_feedforward=int(tcfg.get("dim_feedforward", 256)),
                dropout=float(tcfg.get("dropout", 0.1)),
                pos_encoding=str(tcfg.get("pos_encoding", "learned")),
            )
        else:
            self.serial_tf = nn.Identity()
            self.embedded_tf = nn.Identity()
        self.parallel_placeholder = ParallelFusionPlaceholder()

        self.up1 = _Up(base * 16, base * 8, base * 8)
        self.up2 = _Up(base * 8, base * 4, base * 4)
        self.up3 = _Up(base * 4, base * 2, base * 2)
        self.up4 = _Up(base * 2, base, base)
        self.outc = nn.Conv3d(base, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.modality_dropout_p > 0:
            x = apply_modality_dropout(x, p=self.modality_dropout_p, training=self.training)
        x = self.gate(x) if self.enable_modality_gate else x

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.fusion_mode == "serial":
            if self.enable_transformer:
                x5 = self.serial_tf(x5)
        elif self.fusion_mode == "embedded":
            if self.enable_transformer:
                x4 = self.embedded_tf(x4)
        elif self.fusion_mode == "parallel":
            return self.parallel_placeholder(x, x5)
        else:
            raise ValueError(f"Unknown fusion_mode={self.fusion_mode!r}")

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
