# Data

This project is designed for the BraTS 2024 brain tumor segmentation dataset.
The dataset is not distributed with this repository.

## Download

- Kaggle competition page: https://www.kaggle.com/competitions/brats2024

Follow the dataset license/terms on Kaggle and the BraTS website.

## Configure Dataset Location

By default, configs point to `brats2024-small-dataset/` under the repository root.

You can also set the environment variable `DATASET_DIR` and pass it via scripts:

- `python scripts/train.py --config configs/smoke.yaml --data_dir "$DATASET_DIR"`
- `python scripts/eval.py --config configs/smoke.yaml --run_dir runs/smoke --data_dir "$DATASET_DIR"`

## Expected Layout

The dataset root should contain one directory per case, each containing four modalities.
If labels are available (training/validation sets), a segmentation file is expected as well.

Example:

```text
brats2024-small-dataset/
  BraTS-GLI-02062-100/
    BraTS-GLI-02062-100-t1n.nii.gz
    BraTS-GLI-02062-100-t1c.nii.gz
    BraTS-GLI-02062-100-t2w.nii.gz
    BraTS-GLI-02062-100-t2f.nii.gz
    BraTS-GLI-02062-100-seg.nii.gz
```

## Modality Discovery

Modalities/labels are auto-discovered by filename patterns defined in the config:

- `modalities.*.patterns` (case-insensitive regex fragments)
- `seg_pattern`

Defaults support both common names (`t1`, `t1ce`, `t2`, `flair`) and BraTS-style variants (`t1n`, `t1c`, `t2w`, `t2f`).

## Validate a Dataset Directory

Run a quick scan:

```bash
python scripts/download_data.py --check
```

For training sets (expects label files):

```bash
python scripts/download_data.py --check --expect_labels
```
