from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def _compile_patterns(modality_patterns: dict[str, list[str]]) -> dict[str, list[re.Pattern[str]]]:
    out: dict[str, list[re.Pattern[str]]] = {}
    for mod, pats in modality_patterns.items():
        out[mod] = [re.compile(p, flags=re.IGNORECASE) for p in pats]
    return out


def _match_files(files: list[Path], patterns: list[re.Pattern[str]]) -> list[Path]:
    matches = []
    for f in files:
        name = f.name
        if any(p.search(name) for p in patterns):
            matches.append(f)
    return matches


def scan_case_dir(
    *,
    case_dir: Path,
    modalities: list[str],
    pat_re: dict[str, list[re.Pattern[str]]],
    seg_re: re.Pattern[str],
    require_label: bool = True,
) -> dict[str, Any]:
    nifti_files = sorted(Path(case_dir).glob("*.nii*"))
    label_matches = [p for p in nifti_files if seg_re.search(p.name)]
    label_path = label_matches[0] if label_matches else None

    missing: dict[str, int] = {}
    multi_match: dict[str, list[str]] = {}
    img_paths: dict[str, Path] = {}

    for mod in modalities:
        matches = _match_files([p for p in nifti_files if p != label_path], pat_re[mod])
        if len(matches) == 0:
            missing[f"missing_{mod}"] = 1
        else:
            img_paths[mod] = matches[0]
            if len(matches) > 1:
                multi_match[mod] = [str(m) for m in matches]

    if label_path is None:
        missing["missing_label"] = 1

    has_modalities = all(mod in img_paths for mod in modalities)
    usable = has_modalities and (label_path is not None or not require_label)
    return {
        "case_id": Path(case_dir).name,
        "case_dir": str(case_dir),
        "usable": usable,
        "has_label": label_path is not None,
        "image": [str(img_paths[m]) for m in modalities if m in img_paths],
        "label": str(label_path) if label_path is not None else None,
        "missing": missing,
        "multi_match": multi_match,
    }


def scan_brats_dataset(
    *,
    data_root: Path,
    modalities: list[str],
    modality_patterns: dict[str, list[str]],
    seg_pattern: str = "seg",
    require_label: bool = True,
) -> dict[str, Any]:
    data_root = Path(data_root)
    nii_files = list(data_root.glob("**/*.nii*"))
    case_dirs = sorted({p.parent for p in nii_files}, key=lambda p: str(p))

    seg_re = re.compile(seg_pattern, flags=re.IGNORECASE)
    pat_re = _compile_patterns(modality_patterns)

    cases = []
    for case_dir in case_dirs:
        cases.append(
            scan_case_dir(
                case_dir=case_dir,
                modalities=modalities,
                pat_re=pat_re,
                seg_re=seg_re,
                require_label=require_label,
            )
        )

    return {"total_case_dirs": len(case_dirs), "cases": cases}


def build_datalist(
    *,
    data_root: Path,
    modalities: list[str],
    modality_patterns: dict[str, list[str]],
    seg_pattern: str,
    require_label: bool = True,
) -> list[dict[str, Any]]:
    scan = scan_brats_dataset(
        data_root=data_root,
        modalities=modalities,
        modality_patterns=modality_patterns,
        seg_pattern=seg_pattern,
        require_label=require_label,
    )
    out = []
    for c in scan["cases"]:
        if not c.get("usable", False):
            continue
        out.append({"case_id": c["case_id"], "image": c["image"], "label": c["label"]})
    return out
