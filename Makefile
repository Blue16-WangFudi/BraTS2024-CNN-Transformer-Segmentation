PY ?= conda run -n pytorch python

.PHONY: smoke smoke_unet smoke_ablate_no_gate smoke_ablate_no_transformer smoke_ablate_no_regionaux eval vis_samples env_dump

smoke:
	$(PY) -m brats24.cli train --config config/smoke.yaml

smoke_unet:
	$(PY) -m brats24.cli train --config config/smoke.yaml --override model_name=unet3d --override run_name=smoke_unet --override train.overwrite=true

smoke_ablate_no_gate:
	$(PY) -m brats24.cli train --config config/smoke_ablate_no_gate.yaml

smoke_ablate_no_transformer:
	$(PY) -m brats24.cli train --config config/smoke_ablate_no_transformer.yaml

smoke_ablate_no_regionaux:
	$(PY) -m brats24.cli train --config config/smoke_ablate_no_regionaux.yaml

eval:
	$(PY) -m brats24.cli eval --config config/local.yaml --run_dir $(RUN_DIR)

vis_samples:
	$(PY) -m brats24.cli vis_samples --config config/local.yaml

env_dump:
	$(PY) -m brats24.cli env_dump --output_dir runs/_env_dump
