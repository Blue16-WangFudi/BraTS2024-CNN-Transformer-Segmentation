# BraTS 2024 CNN-Transformer Segmentation

A PyTorch/MONAI codebase for 3D multi-modal brain tumor segmentation on the BraTS 2024 dataset, featuring a CNN-Transformer hybrid UNet and several baselines.

## Key Features

- Hybrid `CNNTransformerUNet` with configurable fusion (`serial` or `embedded`)
- Optional modality gating and modality dropout for robustness
- Baselines via MONAI: 3D UNet, UNETR, SwinUNETR
- YAML configs + `--override` for experiments and ablations
- Reproducible run directories with checkpoints, metrics, and TensorBoard logs

## Method Overview

`CNNTransformerUNet` is a 3D UNet-style encoder/decoder with an optional Transformer module:

- `fusion_mode=serial`: Transformer processes bottleneck tokens and fuses back into the bottleneck feature map
- `fusion_mode=embedded`: Transformer blocks are inserted at an intermediate CNN scale
- `fusion_mode=parallel`: reserved for future work and raises `NotImplementedError`

## Installation

**Requirements**

- Python 3.10+
- PyTorch (CPU or CUDA)

**Install dependencies**

```bash
python -m pip install -r requirements.txt
```

PyTorch must be installed separately (recommended) using the official instructions:
https://pytorch.org/get-started/locally/

## Data

Download BraTS 2024 from Kaggle:
https://www.kaggle.com/competitions/brats2024

Set `DATASET_DIR` to the extracted dataset root, or place the dataset at `./brats2024-small-dataset` (default).

Expected structure (one directory per case):

```text
brats2024-small-dataset/
  BraTS-GLI-02062-100/
    BraTS-GLI-02062-100-t1n.nii.gz
    BraTS-GLI-02062-100-t1c.nii.gz
    BraTS-GLI-02062-100-t2w.nii.gz
    BraTS-GLI-02062-100-t2f.nii.gz
    BraTS-GLI-02062-100-seg.nii.gz
```

Modality/label files are auto-discovered using the regex fragments in the config (`modalities.*.patterns` and `seg_pattern`).
See `docs/DATA.md` for details and validation commands.

## Quickstart (No Data Required)

Print model parameter statistics (builds the model on CPU):

```bash
python -m brats24.cli model_stats --config configs/default.yaml --device cpu
```

## Training

Run a small training job (requires data):

```bash
python scripts/train.py --config configs/smoke.yaml --data_dir brats2024-small-dataset
```

Common overrides:

```bash
python scripts/train.py --config configs/smoke.yaml --data_dir brats2024-small-dataset --override run_name=smoke --override train.overwrite=true --override model_name=cnn_transformer_unet --override fusion_mode=embedded
```

## Evaluation

Evaluate a run directory (writes `runs/<run_name>/metrics/eval_metrics.json`):

```bash
python scripts/eval.py --config configs/smoke.yaml --run_dir runs/smoke --ckpt best.pt --data_dir brats2024-small-dataset
```

## Inference

Run inference on a dataset root directory (automatically selects the first usable case):

```bash
python scripts/infer.py --config configs/smoke.yaml \
  --ckpt runs/smoke/checkpoints/best.pt \
  --input brats2024-small-dataset \
  --output pred.nii.gz
```
`scripts/infer.py` also supports a single 4D NIfTI file input (modalities stacked as 4 channels).

## Run Artifacts

Each training run is written to `runs/<run_name>/`:

- `checkpoints/best.pt`, `checkpoints/last.pt`
- `metrics/metrics.csv` (per-epoch training/validation metrics)
- `metrics/eval_metrics.json` (evaluation summary)
- `tb/` (TensorBoard event files)
- `figures/` (exported plots and sample visualizations, when enabled)
- `artifacts/` (environment dump, dataset report, split metadata)

## Reproducibility

- Random seeds are controlled by `seed` in the config and applied via MONAI determinism utilities.
- For a detailed checklist (versions, hardware, output recovery), see `docs/REPRODUCIBILITY.md`.

## Project Structure

```text
.
|-- brats24/          # library code (models, training, data, utils)
|-- configs/          # YAML experiment configs
|-- scripts/          # train/eval/infer entrypoints
|-- tools/            # optional utilities (HPO, TensorBoard export, etc.)
|-- docs/             # dataset + reproducibility notes
|-- tests/            # lightweight smoke tests
|-- Makefile
|-- requirements.txt
`-- README.md
```

## License

MIT License. See `LICENSE`.

## Citation

If you use this codebase in academic work, please use `CITATION.cff`.

## Acknowledgements

- The BraTS organizers and dataset contributors
- Kaggle for dataset distribution
- MONAI and PyTorch communities
