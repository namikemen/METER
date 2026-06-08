#!/usr/bin/env python3
"""Build a flat bundle of root-level ``*.py`` files and upload to Kaggle Models.

Only top-level Python files in the repo root are included (no ``models/``,
``notebooks/``, ``scripts/``, checkpoints, or other folders).

Example (API token username must match --owner):
  python scripts/package_kaggle_upload_v2.py --upload
  python scripts/package_kaggle_upload_v2.py --upload --variation nyu-xxs

Upload to your own Kaggle account instead of ikenote:
  python scripts/package_kaggle_upload_v2.py --upload --owner "$(python -c 'from kagglehub.config import get_kaggle_credentials as g; print(g().username)')"
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import kagglehub
    from kagglehub.config import get_kaggle_credentials
except ImportError:  # pragma: no cover
    kagglehub = None
    get_kaggle_credentials = None  # type: ignore[misc, assignment]

DEFAULT_MODEL_OWNER = "nammdt"
MODEL_SLUG = "meter"
MODEL_FRAMEWORK = "pyTorch"
DEFAULT_MODEL_BASE_HANDLE = f"{DEFAULT_MODEL_OWNER}/{MODEL_SLUG}/{MODEL_FRAMEWORK}"
DEFAULT_MODEL_PAGE_URL = f"https://www.kaggle.com/models/{DEFAULT_MODEL_OWNER}/{MODEL_SLUG}"

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VARIATION = "nyu-bundle"

EXCLUDED_PY_NAMES = {".env", "kaggle.json"}


def make_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def make_variation_slug(timestamp: str, *, explicit: str | None) -> str:
    if explicit:
        return explicit
    return f"{DEFAULT_VARIATION}-{timestamp}"


def model_page_url(owner: str) -> str:
    return f"https://www.kaggle.com/models/{owner}/{MODEL_SLUG}"


def model_base_handle(owner: str) -> str:
    return f"{owner}/{MODEL_SLUG}/{MODEL_FRAMEWORK}"


def get_authenticated_username() -> str | None:
    if get_kaggle_credentials is None:
        return None
    credentials = get_kaggle_credentials()
    if credentials is None or not credentials.username:
        return None
    return credentials.username.strip()


def preflight_upload_auth(*, requested_owner: str, skip_owner_check: bool) -> str:
    """Ensure the Kaggle API token can publish under requested_owner."""
    username = get_authenticated_username()
    if username is None:
        raise RuntimeError(
            "No Kaggle API credentials found. Create a token at "
            "https://www.kaggle.com/settings and save it to ~/.kaggle/kaggle.json "
            "or set KAGGLE_USERNAME and KAGGLE_KEY."
        )

    print(f"Kaggle API user: {username}")
    print(f"Upload target owner: {requested_owner}")

    if username.lower() == requested_owner.lower():
        return username

    if skip_owner_check:
        print(
            f"Warning: token user `{username}` != owner `{requested_owner}` (--skip-owner-check).",
            file=sys.stderr,
        )
        return username

    raise RuntimeError(
        f"Your Kaggle API token is for `{username}`, but this upload targets `{requested_owner}/{MODEL_SLUG}`.\n"
        f"Kaggle only lets you publish models under the account that owns the token.\n\n"
        f"To upload to {model_page_url(requested_owner)}:\n"
        f"  1. Log into Kaggle as `{requested_owner}` in the browser.\n"
        f"  2. Create an API token at https://www.kaggle.com/settings\n"
        f"  3. Replace ~/.kaggle/kaggle.json with that account's username/key.\n"
        f"  4. Re-run: python scripts/package_kaggle_upload_v2.py --upload\n\n"
        f"To upload under your current account instead:\n"
        f"  python scripts/package_kaggle_upload_v2.py --upload --owner {username}\n"
    )


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    return repo_root, output_dir


def build_zip(*, repo_root: Path, output_dir: Path, timestamp: str) -> Path:
    repo_root = repo_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"meter-kaggle-upload-v2-{timestamp}.zip"
    files = sorted(_package_files(repo_root), key=lambda path: path.name)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative_path in files:
            archive.write(repo_root / relative_path, relative_path.name)

    return zip_path


def stage_upload_dir(*, repo_root: Path) -> tuple[Path, set[Path]]:
    """Stage root-level .py files only (flat directory, no subfolders)."""
    files = _package_files(repo_root)
    stage_dir = Path(tempfile.mkdtemp(prefix="meter-kaggle-upload-"))
    try:
        for relative_path in sorted(files, key=lambda path: path.name):
            shutil.copy2(repo_root / relative_path, stage_dir / relative_path.name)
        return stage_dir, files
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise


def upload_with_kagglehub(
    stage_dir: Path,
    *,
    model_base_handle: str,
    variation_slug: str,
    version_notes: str,
    license_name: str,
) -> str:
    if kagglehub is None:
        raise RuntimeError(
            "kagglehub is required for upload. Install with: pip install kagglehub\n"
            "Or pass --use-kaggle-cli if the `kaggle` CLI is installed and authenticated."
        )

    handle = f"{model_base_handle.rstrip('/')}/{variation_slug}"
    kagglehub.model_upload(
        handle,
        str(stage_dir),
        version_notes=version_notes,
        license_name=license_name,
    )
    return f"https://www.kaggle.com/models/{handle}"


def upload_with_kaggle_cli(stage_dir: Path, *, model_slug: str, version_notes: str) -> str:
    subprocess.run(
        [
            "kaggle",
            "models",
            "versions",
            "create",
            "--model",
            model_slug,
            "--path",
            str(stage_dir),
            "--version-notes",
            version_notes,
        ],
        check=True,
    )
    return f"https://www.kaggle.com/models/{model_slug}"


def upload_package(
    stage_dir: Path,
    *,
    model_owner: str,
    model_base_handle: str,
    variation_slug: str,
    version_notes: str,
    license_name: str,
    use_kaggle_cli: bool,
) -> str:
    if use_kaggle_cli:
        return upload_with_kaggle_cli(
            stage_dir,
            model_slug=f"{model_owner}/{MODEL_SLUG}",
            version_notes=version_notes,
        )
    return upload_with_kagglehub(
        stage_dir,
        model_base_handle=model_base_handle,
        variation_slug=variation_slug,
        version_notes=version_notes,
        license_name=license_name,
    )


def _package_files(repo_root: Path) -> set[Path]:
    """Collect only top-level ``*.py`` files (no subdirectories)."""
    files: set[Path] = set()
    for path in repo_root.iterdir():
        if not path.is_file() or path.suffix != ".py":
            continue
        relative_path = Path(path.name)
        if path.name in EXCLUDED_PY_NAMES:
            continue
        files.add(relative_path)

    if not files:
        raise RuntimeError(f"No .py files found in repo root: {repo_root}")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Build and optionally upload the METER bundle (default target: {DEFAULT_MODEL_PAGE_URL})",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--timestamp", default=None, help="Override UTC timestamp used in artifact names")
    parser.add_argument(
        "--variation",
        default=None,
        help=(
            f"Kaggle variation slug (<owner>/{MODEL_SLUG}/{MODEL_FRAMEWORK}/<variation>). "
            f"Default: {DEFAULT_VARIATION}-<timestamp>"
        ),
    )
    parser.add_argument(
        "--owner",
        default=DEFAULT_MODEL_OWNER,
        help="Kaggle username that owns the model (must match ~/.kaggle/kaggle.json username)",
    )
    parser.add_argument(
        "--model-base-handle",
        default=None,
        help="Override full base handle (<owner>/<model>/<framework>); default is derived from --owner",
    )
    parser.add_argument(
        "--skip-owner-check",
        action="store_true",
        help="Do not verify API username matches --owner (upload will still fail if Kaggle rejects it)",
    )
    parser.add_argument("--upload", action="store_true", help="Upload staged files to Kaggle Models")
    parser.add_argument(
        "--use-kaggle-cli",
        action="store_true",
        help="Use `kaggle models versions create` instead of kagglehub.model_upload",
    )
    parser.add_argument("--version-notes", default="METER Python source files (root .py only)")
    parser.add_argument("--license-name", default="MIT")
    parser.add_argument(
        "--keep-stage-dir",
        action="store_true",
        help="Keep staged upload directory after a successful upload",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root, output_dir = resolve_paths(args)
    timestamp = args.timestamp or make_timestamp()
    variation_slug = make_variation_slug(timestamp, explicit=args.variation)
    model_owner = args.owner.strip()
    base_handle = args.model_base_handle or model_base_handle(model_owner)

    if args.upload:
        preflight_upload_auth(requested_owner=model_owner, skip_owner_check=args.skip_owner_check)

    files = _package_files(repo_root)

    zip_path = build_zip(repo_root=repo_root, output_dir=output_dir, timestamp=timestamp)
    print(f"Created zip: {zip_path}")
    print(f"Packaged {len(files)} .py files from {repo_root} (root only, flat layout):")
    for relative_path in sorted(files, key=lambda path: path.name):
        print(f"  - {relative_path.name}")

    stage_dir, _ = stage_upload_dir(repo_root=repo_root)
    print(f"Staged upload directory: {stage_dir}")

    try:
        if args.upload:
            url = upload_package(
                stage_dir,
                model_owner=model_owner,
                model_base_handle=base_handle,
                variation_slug=variation_slug,
                version_notes=args.version_notes,
                license_name=args.license_name,
                use_kaggle_cli=args.use_kaggle_cli,
            )
            print(f"Uploaded variation `{variation_slug}`")
            print(url)
        else:
            print("Skipping upload (pass --upload to publish to Kaggle).")
            print(f"Target: {base_handle}/{variation_slug}")
    finally:
        if args.upload and not args.keep_stage_dir:
            shutil.rmtree(stage_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
