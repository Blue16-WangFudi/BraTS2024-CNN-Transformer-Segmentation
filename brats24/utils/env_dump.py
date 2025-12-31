from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from brats24.utils.io import write_text


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e!r}"


def dump_environment(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_text(out_dir / "python_version.txt", sys.version)
    write_text(out_dir / "platform.txt", platform.platform())

    torch_info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count(),
        "device_name_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    write_text(out_dir / "torch_info.json", json.dumps(torch_info, indent=2))

    write_text(out_dir / "pip_freeze.txt", _run([sys.executable, "-m", "pip", "freeze"]))
    write_text(out_dir / "nvidia_smi.txt", _run(["nvidia-smi"]))

