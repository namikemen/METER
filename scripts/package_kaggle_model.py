#!/usr/bin/env python3
"""Package METER source files and checkpoints for Kaggle Model upload."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Iterable

SOURCE_FILES = (
    "architecture.py",
    "loss.py",
    "augmentation.py",
    "globals.py",
    "meter_data.py",
    "meter_loss.py",
    "meter_train.py",
    "notebooks/meter_nyu_kaggle_reproduction.ipynb",
)

CHECKPOINT_PATTERNS = (
    "*.pt",
    "*.pth",
    "*.ckpt",
    "best*.bin",
    "final*.bin",
)


def require_files(paths: Iterable[Path]) -> list[Path]:
    files = list(paths)
    missing = [path for path in files if not path.exists() or not path.is_file()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Required files are missing:\n{formatted}")
    return files


def collect_checkpoints(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.exists():
        return []
    checkpoints: list[Path] = []
    for pattern in CHECKPOINT_PATTERNS:
        checkpoints.extend(checkpoint_dir.glob(pattern))
    return sorted(set(path for path in checkpoints if path.is_file()))


def write_metadata(path: Path, args: argparse.Namespace, files: list[Path]) -> Path:
    metadata = {
        "model": "METER",
        "dataset": "NYU Depth v2 local-like",
        "depth_unit": "centimeters",
        "invalid_depth_rule": "depth_cm <= 1.0 is masked from loss and metrics",
        "input_size_hw": [args.input_height, args.input_width],
        "architecture": args.arch_type,
        "lambda_1": args.lambda_1,
        "lambda_2": args.lambda_2,
        "lambda_3": args.lambda_3,
        "packaged_files": [str(file) for file in files],
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def create_zip(repo_root: Path, output_zip: Path, files: list[Path]) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, arcname=file.relative_to(repo_root))


def stage_upload_dir(output_zip: Path, metadata_path: Path) -> Path:
    stage_dir = output_zip.parent / "kaggle_model_upload"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    shutil.copy2(output_zip, stage_dir / output_zip.name)
    shutil.copy2(metadata_path, stage_dir / metadata_path.name)
    return stage_dir


def maybe_upload_to_kaggle(args: argparse.Namespace, stage_dir: Path) -> None:
    if not args.upload:
        return
    if not args.kaggle_model:
        raise ValueError("--kaggle-model is required when --upload is set")

    command = [
        "kaggle",
        "models",
        "versions",
        "create",
        "--model",
        args.kaggle_model,
        "--path",
        str(stage_dir),
        "--version-notes",
        args.version_notes,
    ]
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--output-zip", type=Path, default=Path("outputs/kaggle_meter_model.zip"))
    parser.add_argument("--arch-type", default="s", choices=("s", "xs", "xxs"))
    parser.add_argument("--input-height", type=int, default=192)
    parser.add_argument("--input-width", type=int, default=256)
    parser.add_argument("--lambda-1", type=float, default=0.5)
    parser.add_argument("--lambda-2", type=float, default=100.0)
    parser.add_argument("--lambda-3", type=float, default=100.0)
    parser.add_argument("--upload", action="store_true", help="Upload output directory as a Kaggle Model version")
    parser.add_argument("--kaggle-model", default="", help="Kaggle model slug, e.g. username/meter-nyu")
    parser.add_argument("--version-notes", default="METER NYU checkpoint and source bundle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = repo_root / checkpoint_dir

    output_zip = args.output_zip
    if not output_zip.is_absolute():
        output_zip = repo_root / output_zip

    source_paths = require_files(repo_root / name for name in SOURCE_FILES)
    checkpoint_paths = collect_checkpoints(checkpoint_dir)
    output_dir = output_zip.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = write_metadata(output_dir / "meter_model_metadata.json", args, source_paths + checkpoint_paths)
    files = source_paths + checkpoint_paths + [metadata_path]

    create_zip(repo_root, output_zip, files)
    stage_dir = stage_upload_dir(output_zip, metadata_path)
    print(f"Created {output_zip}")
    print(f"Staged Kaggle upload directory at {stage_dir}")
    print("Packaged files:")
    for file in files:
        print(f"- {file.relative_to(repo_root)}")

    maybe_upload_to_kaggle(args, stage_dir)


if __name__ == "__main__":
    main()
