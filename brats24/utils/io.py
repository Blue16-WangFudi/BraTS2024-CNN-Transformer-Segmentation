from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch


def as_path(p: str | Path) -> Path:
    return Path(p).expanduser()


def save_json(path: Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")


def save_torch(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def load_torch(path: Path, *, map_location: str | None = None) -> Any:
    return torch.load(Path(path), map_location=map_location)


def save_csv_row(path: Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])

    if not header:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    new_keys = [k for k in row.keys() if k not in header]
    if new_keys:
        with path.open("r", newline="", encoding="utf-8") as f:
            dr = csv.DictReader(f)
            existing_rows = list(dr)
        new_header = header + new_keys
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=new_header)
            writer.writeheader()
            for r in existing_rows:
                writer.writerow({k: r.get(k, "") for k in new_header})
            writer.writerow({k: row.get(k, "") for k in new_header})
        tmp.replace(path)
        return

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow({k: row.get(k, "") for k in header})
