from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def export_scalars_to_png(tb_dir: Path, out_dir: Path) -> list[Path]:
    tb_dir = Path(tb_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = EventAccumulator(str(tb_dir))
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    if not tags:
        return []

    out_files: list[Path] = []
    for tag in tags:
        events = acc.Scalars(tag)
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        plt.figure(figsize=(6, 4))
        plt.plot(steps, vals)
        plt.title(tag)
        plt.xlabel("step")
        plt.ylabel(tag)
        plt.tight_layout()
        out = out_dir / f"curve_{tag.replace('/', '_')}.png"
        plt.savefig(out, dpi=200)
        plt.close()
        out_files.append(out)

    return out_files

