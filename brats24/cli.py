from __future__ import annotations

import argparse
from pathlib import Path

from brats24.utils.config import load_config


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="brats24", description="BraTS2024 3D segmentation (Stage-1)")
    sp = p.add_subparsers(dest="command", required=True)

    cfg_parent = argparse.ArgumentParser(add_help=False)
    cfg_parent.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    cfg_parent.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config with dotted keys, e.g. train.epochs=2 (repeatable).",
    )

    sp_train = sp.add_parser("train", help="Train a model.", parents=[cfg_parent])
    sp_train.add_argument("--device", type=str, default="cuda", help="cuda|cpu|cuda:0")

    sp_eval = sp.add_parser("eval", help="Evaluate a trained model.", parents=[cfg_parent])
    sp_eval.add_argument("--run_dir", type=str, required=True, help="runs/<run_name> directory.")
    sp_eval.add_argument("--ckpt", type=str, default="best.pt", help="Checkpoint name/path within run_dir.")
    sp_eval.add_argument("--device", type=str, default="cuda", help="cuda|cpu|cuda:0")

    sp_infer = sp.add_parser("infer", help="Run inference for one case.", parents=[cfg_parent])
    sp_infer.add_argument("--run_dir", type=str, required=True, help="runs/<run_name> directory.")
    sp_infer.add_argument("--ckpt", type=str, default="best.pt", help="Checkpoint name/path within run_dir.")
    sp_infer.add_argument("--case_id", type=str, default=None, help="Case folder name under data_root.")
    sp_infer.add_argument("--case_dir", type=str, default=None, help="Explicit case directory path.")
    sp_infer.add_argument("--out_dir", type=str, default=None, help="Output directory (defaults to runs/<run_name>/predictions).")
    sp_infer.add_argument("--device", type=str, default="cuda", help="cuda|cpu|cuda:0")

    sp_vis = sp.add_parser("vis_samples", help="Save dataset sample visualizations.", parents=[cfg_parent])
    sp_vis.add_argument("--out_dir", type=str, default=None, help="Output dir (defaults to runs/<run_name>/figures).")

    sp_env = sp.add_parser("env_dump", help="Dump environment information.")
    sp_env.add_argument("--output_dir", type=str, required=True, help="Directory to write env dump.")

    sp_stats = sp.add_parser("model_stats", help="Print model parameter stats.", parents=[cfg_parent])
    sp_stats.add_argument("--device", type=str, default="cpu", help="cpu|cuda|cuda:0")

    sp_report = sp.add_parser("dataset_report", help="Generate dataset report JSON.", parents=[cfg_parent])
    sp_report.add_argument("--out", type=str, default="artifacts/dataset_report.json", help="Output JSON path.")

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "env_dump":
        from brats24.utils.env_dump import dump_environment

        dump_environment(Path(args.output_dir))
        return

    cfg = load_config(args.config, overrides=getattr(args, "override", None))

    if args.command == "train":
        from brats24.engine.trainer import train

        train(cfg, device=args.device)
        return

    if args.command == "eval":
        from brats24.engine.evaluator import evaluate

        evaluate(cfg, run_dir=Path(args.run_dir), ckpt=args.ckpt, device=args.device)
        return

    if args.command == "infer":
        from brats24.engine.inference import infer

        infer(
            cfg,
            run_dir=Path(args.run_dir),
            ckpt=args.ckpt,
            case_id=args.case_id,
            case_dir=args.case_dir,
            out_dir=args.out_dir,
            device=args.device,
        )
        return

    if args.command == "vis_samples":
        from brats24.engine.visualize import vis_samples

        vis_samples(cfg, out_dir=args.out_dir)
        return

    if args.command == "model_stats":
        from brats24.engine.modeling import build_model
        from brats24.utils.model_stats import summarize_model

        model = build_model(cfg).to(args.device)
        print(summarize_model(model))
        return

    if args.command == "dataset_report":
        from brats24.data.report import generate_dataset_report
        from brats24.utils.io import save_json

        report = generate_dataset_report(cfg)
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(out_path, report)
        print(f"Wrote: {out_path}")
        return

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
