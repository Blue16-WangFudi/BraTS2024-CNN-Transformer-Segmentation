from __future__ import annotations

import argparse

from brats24.engine.modeling import build_model
from brats24.utils.config import load_config
from brats24.utils.model_stats import summarize_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg)
    print(summarize_model(model))


if __name__ == "__main__":
    main()

