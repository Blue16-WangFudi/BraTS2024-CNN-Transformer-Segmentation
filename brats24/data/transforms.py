from __future__ import annotations

from typing import Any

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)


class LabelRemapd:
    def __init__(self, keys: list[str], mapping: dict[int, int]):
        self.keys = keys
        self.mapping = {int(k): int(v) for k, v in mapping.items()}

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        out = dict(data)
        for k in self.keys:
            arr = out[k]
            if isinstance(arr, torch.Tensor):
                x = arr
                for src, dst in self.mapping.items():
                    x = torch.where(x == src, torch.as_tensor(dst, dtype=x.dtype, device=x.device), x)
                out[k] = x
            else:
                x = np.asarray(arr)
                for src, dst in self.mapping.items():
                    x = np.where(x == src, dst, x)
                out[k] = x
        return out


def build_train_transforms(cfg: dict[str, Any]):
    label_remap = cfg.get("label_remap") or {}
    patch_size = tuple(int(x) for x in cfg["train"]["patch_size"])
    samples_per_volume = int(cfg.get("train", {}).get("samples_per_volume", 2))

    from monai.transforms import RandCropByPosNegLabeld

    t = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ]
    if label_remap:
        t.append(LabelRemapd(keys=["label"], mapping=label_remap))

    t.extend(
        [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=max(1, samples_per_volume),
                image_key="image",
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.2),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.2),
            EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64)),
        ]
    )
    return Compose(t)


def build_val_transforms(cfg: dict[str, Any]):
    label_remap = cfg.get("label_remap") or {}
    t = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ]
    if label_remap:
        t.append(LabelRemapd(keys=["label"], mapping=label_remap))

    t.append(EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64)))
    return Compose(t)


def build_infer_transforms(cfg: dict[str, Any]):
    t = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"], dtype=torch.float32),
    ]
    return Compose(t)
