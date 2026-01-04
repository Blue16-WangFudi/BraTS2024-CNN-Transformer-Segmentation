USE_CONDA ?= 1
CONDA := $(shell command -v conda 2>/dev/null)
ifeq ($(USE_CONDA),1)
  ifeq ($(CONDA),)
    PY ?= python
  else
    PY ?= conda run -n pytorch python
  endif
else
  PY ?= python
endif

CLOUD_CONFIG ?= configs/cloud.yaml
CLOUD_RUN ?= cloud_placeholder
CLOUD_RUN_NO_GATE ?= cloud_no_gate
CLOUD_RUN_NO_TRANSFORMER ?= cloud_no_transformer
CLOUD_RUN_NO_REGIONAUX ?= cloud_no_regionaux

CLOUD_OVERRIDES_BASE := --override run_name=$(CLOUD_RUN) --override train.overwrite=true
CLOUD_OVERRIDES_NO_GATE := --override run_name=$(CLOUD_RUN_NO_GATE) --override enable_modality_gate=false --override train.overwrite=true
CLOUD_OVERRIDES_NO_TRANSFORMER := --override run_name=$(CLOUD_RUN_NO_TRANSFORMER) --override enable_transformer=false --override train.overwrite=true
CLOUD_OVERRIDES_NO_REGIONAUX := --override run_name=$(CLOUD_RUN_NO_REGIONAUX) --override enable_region_aux=false --override train.overwrite=true

.PHONY: smoke smoke_unet smoke_ablate_no_gate smoke_ablate_no_transformer smoke_ablate_no_regionaux eval vis_samples env_dump \
	cloud_train cloud_ablate_no_gate cloud_ablate_no_transformer cloud_ablate_no_regionaux cloud_all cloud_assets cloud_assets_all \
	cloud_assets_no_gate cloud_assets_no_transformer cloud_assets_no_regionaux hpo

smoke:
        $(PY) -m brats24.cli train --config configs/smoke.yaml

smoke_unet:
        $(PY) -m brats24.cli train --config configs/smoke.yaml --override model_name=unet3d --override run_name=smoke_unet --override train.overwrite=true       

smoke_ablate_no_gate:
        $(PY) -m brats24.cli train --config configs/smoke_ablate_no_gate.yaml    

smoke_ablate_no_transformer:
        $(PY) -m brats24.cli train --config configs/smoke_ablate_no_transformer.yaml

smoke_ablate_no_regionaux:
        $(PY) -m brats24.cli train --config configs/smoke_ablate_no_regionaux.yaml

eval:
        $(PY) -m brats24.cli eval --config configs/local.yaml --run_dir $(RUN_DIR)

vis_samples:
        $(PY) -m brats24.cli vis_samples --config configs/local.yaml

env_dump:
	$(PY) -m brats24.cli env_dump --output_dir runs/_env_dump

cloud_train:
	$(PY) -m brats24.cli train --config $(CLOUD_CONFIG) $(CLOUD_OVERRIDES_BASE)

cloud_ablate_no_gate:
	$(PY) -m brats24.cli train --config $(CLOUD_CONFIG) $(CLOUD_OVERRIDES_NO_GATE)

cloud_ablate_no_transformer:
	$(PY) -m brats24.cli train --config $(CLOUD_CONFIG) $(CLOUD_OVERRIDES_NO_TRANSFORMER)

cloud_ablate_no_regionaux:
	$(PY) -m brats24.cli train --config $(CLOUD_CONFIG) $(CLOUD_OVERRIDES_NO_REGIONAUX)

cloud_assets:
	$(PY) -m tools.export_tb_to_png --tb_dir runs/$(CLOUD_RUN)/tb --out_dir runs/$(CLOUD_RUN)/figures
	$(PY) -m brats24.cli vis_samples --config $(CLOUD_CONFIG) $(CLOUD_OVERRIDES_BASE) --out_dir runs/$(CLOUD_RUN)/figures
	CASE_ID="$$($(PY) -c \"from pathlib import Path; import yaml; cfg=yaml.safe_load(open('$(CLOUD_CONFIG)')); root=Path(cfg['data_root']); print(next(p.name for p in root.iterdir() if p.is_dir()))\")" && \
	$(PY) -m brats24.cli infer --config $(CLOUD_CONFIG) $(CLOUD_OVERRIDES_BASE) --run_dir runs/$(CLOUD_RUN) --case_id $$CASE_ID --out_dir runs/$(CLOUD_RUN)/figures
	$(PY) -m brats24.cli dataset_report --config $(CLOUD_CONFIG) --out runs/$(CLOUD_RUN)/artifacts/dataset_report.json
	$(PY) -m brats24.cli env_dump --output_dir runs/$(CLOUD_RUN)/artifacts/env
	$(PY) -m brats24.cli eval --config $(CLOUD_CONFIG) $(CLOUD_OVERRIDES_BASE) --run_dir runs/$(CLOUD_RUN)

cloud_assets_no_gate:
	$(PY) -m tools.export_tb_to_png --tb_dir runs/$(CLOUD_RUN_NO_GATE)/tb --out_dir runs/$(CLOUD_RUN_NO_GATE)/figures
	$(PY) -m brats24.cli vis_samples --config $(CLOUD_CONFIG) $(CLOUD_OVERRIDES_NO_GATE) --out_dir runs/$(CLOUD_RUN_NO_GATE)/figures
	CASE_ID="$$($(PY) -c \"from pathlib import Path; import yaml; cfg=yaml.safe_load(open('$(CLOUD_CONFIG)')); root=Path(cfg['data_root']); print(next(p.name for p in root.iterdir() if p.is_dir()))\")" && \
	$(PY) -m brats24.cli infer --config $(CLOUD_CONFIG) $(CLOUD_OVERRIDES_NO_GATE) --run_dir runs/$(CLOUD_RUN_NO_GATE) --case_id $$CASE_ID --out_dir runs/$(CLOUD_RUN_NO_GATE)/figures
	$(PY) -m brats24.cli dataset_report --config $(CLOUD_CONFIG) --out runs/$(CLOUD_RUN_NO_GATE)/artifacts/dataset_report.json
	$(PY) -m brats24.cli env_dump --output_dir runs/$(CLOUD_RUN_NO_GATE)/artifacts/env
	$(PY) -m brats24.cli eval --config $(CLOUD_CONFIG) $(CLOUD_OVERRIDES_NO_GATE) --run_dir runs/$(CLOUD_RUN_NO_GATE)

cloud_assets_no_transformer:
	$(PY) -m tools.export_tb_to_png --tb_dir runs/$(CLOUD_RUN_NO_TRANSFORMER)/tb --out_dir runs/$(CLOUD_RUN_NO_TRANSFORMER)/figures
	$(PY) -m brats24.cli vis_samples --config $(CLOUD_CONFIG) $(CLOUD_OVERRIDES_NO_TRANSFORMER) --out_dir runs/$(CLOUD_RUN_NO_TRANSFORMER)/figures
	CASE_ID="$$($(PY) -c \"from pathlib import Path; import yaml; cfg=yaml.safe_load(open('$(CLOUD_CONFIG)')); root=Path(cfg['data_root']); print(next(p.name for p in root.iterdir() if p.is_dir()))\")" && \
	$(PY) -m brats24.cli infer --config $(CLOUD_CONFIG) $(CLOUD_OVERRIDES_NO_TRANSFORMER) --run_dir runs/$(CLOUD_RUN_NO_TRANSFORMER) --case_id $$CASE_ID --out_dir runs/$(CLOUD_RUN_NO_TRANSFORMER)/figures
	$(PY) -m brats24.cli dataset_report --config $(CLOUD_CONFIG) --out runs/$(CLOUD_RUN_NO_TRANSFORMER)/artifacts/dataset_report.json
	$(PY) -m brats24.cli env_dump --output_dir runs/$(CLOUD_RUN_NO_TRANSFORMER)/artifacts/env
	$(PY) -m brats24.cli eval --config $(CLOUD_CONFIG) $(CLOUD_OVERRIDES_NO_TRANSFORMER) --run_dir runs/$(CLOUD_RUN_NO_TRANSFORMER)

cloud_assets_no_regionaux:
	$(PY) -m tools.export_tb_to_png --tb_dir runs/$(CLOUD_RUN_NO_REGIONAUX)/tb --out_dir runs/$(CLOUD_RUN_NO_REGIONAUX)/figures
	$(PY) -m brats24.cli vis_samples --config $(CLOUD_CONFIG) $(CLOUD_OVERRIDES_NO_REGIONAUX) --out_dir runs/$(CLOUD_RUN_NO_REGIONAUX)/figures
	CASE_ID="$$($(PY) -c \"from pathlib import Path; import yaml; cfg=yaml.safe_load(open('$(CLOUD_CONFIG)')); root=Path(cfg['data_root']); print(next(p.name for p in root.iterdir() if p.is_dir()))\")" && \
	$(PY) -m brats24.cli infer --config $(CLOUD_CONFIG) $(CLOUD_OVERRIDES_NO_REGIONAUX) --run_dir runs/$(CLOUD_RUN_NO_REGIONAUX) --case_id $$CASE_ID --out_dir runs/$(CLOUD_RUN_NO_REGIONAUX)/figures
	$(PY) -m brats24.cli dataset_report --config $(CLOUD_CONFIG) --out runs/$(CLOUD_RUN_NO_REGIONAUX)/artifacts/dataset_report.json
	$(PY) -m brats24.cli env_dump --output_dir runs/$(CLOUD_RUN_NO_REGIONAUX)/artifacts/env
	$(PY) -m brats24.cli eval --config $(CLOUD_CONFIG) $(CLOUD_OVERRIDES_NO_REGIONAUX) --run_dir runs/$(CLOUD_RUN_NO_REGIONAUX)

cloud_assets_all: cloud_assets cloud_assets_no_gate cloud_assets_no_transformer cloud_assets_no_regionaux

cloud_all: cloud_train cloud_ablate_no_gate cloud_ablate_no_transformer cloud_ablate_no_regionaux cloud_assets_all

hpo:
        $(PY) -m tools.hpo --config $(CLOUD_CONFIG) --space configs/hpo.yaml --output_dir runs_hpo
