from __future__ import annotations

import os
import random

import numpy as np
import torch
from monai.utils import set_determinism


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_determinism(seed=seed)

