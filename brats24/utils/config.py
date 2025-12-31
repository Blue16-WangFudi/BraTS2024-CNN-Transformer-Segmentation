from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _set_by_dotted_key(d: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur: dict[str, Any] = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def load_config(path: str, *, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg_path = Path(path)
    raw = cfg_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(raw) or {}
    cfg = copy.deepcopy(cfg)
    cfg["_raw_yaml"] = raw

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Invalid override (expected k=v): {ov}")
        k, v = ov.split("=", 1)
        _set_by_dotted_key(cfg, k.strip(), yaml.safe_load(v))

    return cfg


def resolved_run_dir(cfg: dict[str, Any]) -> Path:
    output_dir = Path(cfg.get("output_dir", "runs"))
    run_name = str(cfg.get("run_name") or "run")
    run_dir = output_dir / run_name
    overwrite = bool(cfg.get("train", {}).get("overwrite", False))
    if run_dir.exists() and overwrite:
        import shutil

        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

