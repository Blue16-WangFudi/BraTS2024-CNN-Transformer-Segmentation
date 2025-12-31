from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy import ndimage as ndi


def postprocess_pred_label(cfg: dict[str, Any], pred_label: torch.Tensor) -> torch.Tensor:
    if not bool(cfg.get("enable_postprocess", False)):
        return pred_label
    method = str(cfg.get("postprocess", {}).get("method", "none")).lower()
    if method in {"none", ""}:
        return pred_label
    if method != "largest_cc":
        raise ValueError(f"Unknown postprocess.method={method!r}")

    if pred_label.ndim == 5:
        if pred_label.shape[1] != 1:
            raise ValueError(f"Expected pred_label as (B,1,D,H,W), got {tuple(pred_label.shape)}")
        pred_label_bdhw = pred_label[:, 0]
    elif pred_label.ndim == 4:
        pred_label_bdhw = pred_label
    else:
        raise ValueError(f"Expected pred_label as (B,1,D,H,W) or (B,D,H,W), got {tuple(pred_label.shape)}")

    num_classes = int(cfg["num_classes"])
    out = pred_label_bdhw.detach().cpu().numpy().astype(np.int32, copy=True)
    for b in range(out.shape[0]):
        vol = out[b]
        new = np.zeros_like(vol)
        for lab in range(1, num_classes):
            mask = vol == lab
            if not mask.any():
                continue
            cc, n = ndi.label(mask)
            if n <= 1:
                new[mask] = lab
                continue
            sizes = np.bincount(cc.reshape(-1))
            sizes[0] = 0
            keep = int(sizes.argmax())
            new[cc == keep] = lab
        out[b] = new
    out_t = torch.from_numpy(out).to(pred_label.device, dtype=pred_label_bdhw.dtype)
    return out_t.unsqueeze(1)
