# BraTS2024 CNN-Transformer Segmentation (Stage-1)

Stage-1 delivers a reproducible **local** 3D multi-modal MRI segmentation codebase for BraTS2024 with:

- Baselines: **3D UNet (CNN-only)**, **UNETR/SwinUNETR (Transformer-only, best-effort)**
- Our model: **CNNTransformerUNet** with explicit fusion modes:
  - `fusion_mode=serial` (CNN bottleneck → tokens → Transformer → reshape → fuse)
  - `fusion_mode=embedded` (Transformer blocks inserted at one intermediate CNN scale)
  - `fusion_mode=parallel` placeholder API only (**NotImplementedError**, Stage-2)
- YAML-driven ablations: `enable_modality_gate`, `enable_transformer`, `enable_region_aux`, `enable_postprocess`, `fusion_mode`
- Local smoke runs produce:
  - `runs/<run_name>/metrics/metrics.csv`
  - TensorBoard scalars under `runs/<run_name>/tb/`
  - PNG paper assets under `runs/<run_name>/figures/`
  - env dump under `runs/<run_name>/artifacts/env/`
  - dataset report under `artifacts/dataset_report.json` (also copied into the run)

## Environment

Use the existing conda env (do not create a new one):

```bash
conda activate pytorch
python -m pip install -r requirements.txt
```

## Dataset

Place the dataset at `brats2024-small-dataset/` (already present in this repo but **gitignored**).

Each case folder should contain 4 modalities and 1 label mask. Filenames are auto-discovered (case-insensitive):

- Modalities: `t1`, `t1c/t1ce`, `t2`, `flair` OR BraTS-style `t1n/t1c/t2w/t2f`
- Label: contains `seg`

If any required file is missing, the case is skipped and reported.

## Quickstart (Makefile)

```bash
make smoke_unet
make smoke
make smoke_ablate_no_transformer
```

Other helpers:

```bash
make vis_samples
make env_dump
```

## CLI

All commands are routed through `brats24/cli.py`.

```bash
python -m brats24.cli train --config config/smoke.yaml
python -m brats24.cli eval  --config config/smoke.yaml --run_dir runs/smoke
python -m brats24.cli infer --config config/smoke.yaml --run_dir runs/smoke --case_id <CASE_FOLDER_NAME>
python -m brats24.cli vis_samples --config config/smoke.yaml
```

Config overrides are supported via `--override key=value`, e.g.:

```bash
python -m brats24.cli train --config config/smoke.yaml --override train.epochs=3 --override fusion_mode=embedded
```

## Notes

- `fusion_mode=parallel` is a documented placeholder and raises `NotImplementedError` in Stage-1.
- No datasets, NIfTI files, LaTeX/paper sources, or `runs/` artifacts are committed.

## Model Options

- `model_name=unet3d`: CNN-only 3D UNet baseline (MONAI).
- `model_name=cnn_transformer_unet`: our CNN+Transformer fusion model (`fusion_mode=serial|embedded|parallel`).
- `model_name=unetr` / `model_name=swinunetr`: Transformer-only baselines (best-effort; may be heavier than smoke configs).
