from __future__ import annotations

import argparse
from pathlib import Path

from brats24.utils.tb_export import export_scalars_to_png


def main() -> None:
    p = argparse.ArgumentParser(description="Export TensorBoard scalar curves to PNG (paper assets).")
    p.add_argument("--tb_dir", required=True, help="TensorBoard log dir (contains event files).")
    p.add_argument("--out_dir", required=True, help="Output directory for PNGs.")
    args = p.parse_args()

    out = export_scalars_to_png(Path(args.tb_dir), Path(args.out_dir))
    print(f"Exported {len(out)} curves to {args.out_dir}")


if __name__ == "__main__":
    main()

