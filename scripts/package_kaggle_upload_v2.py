from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

try:
    import kagglehub
except ImportError:  # pragma: no cover
    kagglehub = None

MODEL_BASE_HANDLE = "ikenote/meter-nyu/pyTorch"
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_FILES = (
    Path("architecture.py"),
    Path("augmentation.py"),
    Path("globals.py"),
    Path("loss.py"),
    Path("meter_data.py"),
    Path("meter_loss.py"),
    Path("meter_train.py"),
    Path("requirements.txt"),
    Path("notebooks/meter_nyu_kaggle_reproduction.ipynb"),
    Path("scripts/package_kaggle_upload_v2.py"),
)
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}
EXCLUDED_NAMES = {".env", "kaggle.json"}
EXCLUDED_PARTS = {"__pycache__", ".git", ".ipynb_checkpoints", "wandb"}


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_variation_slug(timestamp: str) -> str:
    return f"meter-kaggle-upload-v2-{timestamp}"


def build_zip(*, repo_root: Path, checkpoint_dir: Path, output_dir: Path, timestamp: str) -> Path:
    repo_root = repo_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"meter-kaggle-upload-v2-{timestamp}.zip"
    files = sorted(_package_files(repo_root, checkpoint_dir), key=lambda path: path.as_posix())

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative_path in files:
            archive.write(repo_root / relative_path, relative_path.as_posix())

    return zip_path


def upload_zip(zip_path: Path, *, model_base_handle: str, variation_slug: str, version_notes: str, license_name: str) -> str:
    if kagglehub is None:
        raise RuntimeError("kagglehub is required for upload. Install it or run without --upload.")

    handle = f"{model_base_handle.rstrip('/')}/{variation_slug}"
    with tempfile.TemporaryDirectory(prefix=f"{variation_slug}-") as upload_dir:
        upload_path = Path(upload_dir) / zip_path.name
        shutil.copy2(zip_path, upload_path)
        kagglehub.model_upload(
            handle=handle,
            local_model_dir=str(upload_path.parent),
            version_notes=version_notes,
            license_name=license_name,
        )
    return f"https://kaggle.com/models/{handle}"


def _package_files(repo_root: Path, checkpoint_dir: Path) -> set[Path]:
    files = {path for path in PACKAGE_FILES if _should_include(path, repo_root / path)}
    checkpoints = _checkpoint_files(repo_root, checkpoint_dir)
    if not checkpoints:
        print(f"Warning: no checkpoint files found in {checkpoint_dir}")
    files.update(checkpoints)
    return files


def _checkpoint_files(repo_root: Path, checkpoint_dir: Path) -> set[Path]:
    absolute_dir = checkpoint_dir if checkpoint_dir.is_absolute() else repo_root / checkpoint_dir
    if not absolute_dir.is_dir():
        return set()

    files: set[Path] = set()
    for path in absolute_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES:
            try:
                relative_path = path.resolve().relative_to(repo_root)
            except ValueError as exc:
                raise ValueError(f"Checkpoint path must be inside repo root: {path}") from exc
            if _should_include(relative_path, path):
                files.add(relative_path)
    return files


def _should_include(relative_path: Path, absolute_path: Path) -> bool:
    if not absolute_path.is_file():
        return False
    if relative_path.name in EXCLUDED_NAMES:
        return False
    return not any(part in EXCLUDED_PARTS for part in relative_path.parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and optionally upload the METER Kaggle package ZIP v2.")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--timestamp", default=None)
    parser.add_argument("--model-base-handle", default=MODEL_BASE_HANDLE)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--version-notes", default="METER NYU checkpoint and source bundle v2")
    parser.add_argument("--license-name", default="MIT")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or make_timestamp()
    variation_slug = make_variation_slug(timestamp)
    zip_path = build_zip(
        repo_root=args.repo_root,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        timestamp=timestamp,
    )
    print(zip_path)
    if args.upload:
        print(
            upload_zip(
                zip_path,
                model_base_handle=args.model_base_handle,
                variation_slug=variation_slug,
                version_notes=args.version_notes,
                license_name=args.license_name,
            )
        )


if __name__ == "__main__":
    main()
