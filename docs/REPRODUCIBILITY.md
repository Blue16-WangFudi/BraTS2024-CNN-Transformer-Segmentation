# Reproducibility

This document describes the recommended way to reproduce runs and recover results.

## Environment

- Python 3.10+
- PyTorch (CPU or CUDA)
- Install repo dependencies: `python -m pip install -r requirements.txt`

PyTorch is installed separately to match your hardware (CPU vs CUDA). See https://pytorch.org/get-started/locally/

## Determinism / Seeds

- Set `seed` in your YAML config.
- Training calls `brats24.utils.seed.seed_everything`, which configures Python, NumPy, PyTorch, and MONAI determinism utilities.

Note: some GPU operations may still be non-deterministic depending on hardware/drivers and selected kernels.

## Recommended Commands

Train:

```bash
python scripts/train.py --config configs/default.yaml --data_dir brats2024-small-dataset
```

Evaluate:

```bash
python scripts/eval.py --config configs/default.yaml --run_dir runs/baseline --ckpt best.pt --data_dir brats2024-small-dataset
```

Infer:

```bash
python scripts/infer.py --config configs/default.yaml --ckpt runs/baseline/checkpoints/best.pt --input brats2024-small-dataset --output pred.nii.gz
```

## Output Recovery

Each run is written to `runs/<run_name>/`:

- `checkpoints/best.pt`, `checkpoints/last.pt`
- `metrics/metrics.csv`
- `metrics/eval_metrics.json`
- `tb/` (TensorBoard logs)
- `artifacts/` (environment dump, dataset report, split metadata)

## Troubleshooting

**CUDA out of memory**

- Reduce `train.patch_size` and/or `train.batch_size`.
- Disable AMP with `--override train.amp=false`.

**No usable cases found**

- Confirm `data_root` points to the dataset root.
- Run `python scripts/download_data.py --check` and review the reported `usable` count.
