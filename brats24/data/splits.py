from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SplitResult:
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]


def make_random_split(
    datalist: list[dict[str, Any]],
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
    max_train_cases: int | None = None,
    max_val_cases: int | None = None,
) -> SplitResult:
    if train_frac <= 0 or val_frac <= 0 or abs((train_frac + val_frac) - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac must equal 1.0 and be > 0.")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(datalist))
    rng.shuffle(idx)
    n_train = int(round(len(datalist) * train_frac))
    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train:].tolist()

    train = [datalist[i] for i in train_idx]
    val = [datalist[i] for i in val_idx]

    if max_train_cases is not None:
        train = train[: int(max_train_cases)]
    if max_val_cases is not None:
        val = val[: int(max_val_cases)]

    return SplitResult(train=train, val=val)

