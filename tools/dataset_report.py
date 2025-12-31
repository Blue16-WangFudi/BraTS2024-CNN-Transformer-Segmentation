from __future__ import annotations

from pathlib import Path

from brats24.data.report import generate_dataset_report
from brats24.utils.config import load_config
from brats24.utils.io import save_json


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="artifacts/dataset_report.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    rep = generate_dataset_report(cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(out, rep)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

