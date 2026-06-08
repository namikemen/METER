from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm.std import tqdm


@dataclass
class SplitSummary:
    total: int = 0
    written: int = 0
    skipped_existing: int = 0
    failed: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ResizeSummary:
    input_root: str
    output_root: str
    height: int
    width: int
    compression: str | None
    splits: dict[str, SplitSummary]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast local resize for NYU H5 files.")
    parser.add_argument("--input-root", type=Path, default=Path("origin_data"))
    parser.add_argument("--output-root", type=Path, default=Path("nyu_depth_v2_resized_192x256"))
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--compression", choices=("none", "lzf"), default="none")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-name", default="resize_summary.json")
    return parser.parse_args()


def discover_files(input_root: Path, split: str, limit: int | None) -> list[Path]:
    split_root = input_root / split
    if split == "train":
        files = sorted(path for scene_dir in split_root.iterdir() if scene_dir.is_dir() for path in scene_dir.glob("*.h5"))
    elif split == "val":
        direct_files = sorted(split_root.glob("*.h5"))
        official_files = sorted((split_root / "official").glob("*.h5")) if (split_root / "official").exists() else []
        files = direct_files if direct_files else official_files
    else:
        raise ValueError(f"Unsupported split: {split}")
    return files[:limit] if limit is not None else files


def resize_rgb(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"Expected rgb shape (3,H,W), got {rgb.shape}")
    rgb_hwc = np.transpose(rgb, (1, 2, 0))
    resized = cv2.resize(rgb_hwc, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.transpose(resized, (2, 0, 1)).astype(rgb.dtype, copy=False)


def resize_depth(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape (H,W), got {depth.shape}")
    resized = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(depth.dtype, copy=False)


def copy_attrs(source: h5py.File, target: h5py.File) -> None:
    for key, value in source.attrs.items():
        try:
            target.attrs[key] = value
        except TypeError:
            target.attrs[key] = str(value)


def resize_one(task: tuple[str, str, str, int, int, str | None, bool]) -> dict[str, str]:
    source_text, split_root_text, target_split_root_text, height, width, compression, overwrite = task
    source_path = Path(source_text)
    split_root = Path(split_root_text)
    target_split_root = Path(target_split_root_text)
    target_path = target_split_root / source_path.relative_to(split_root)

    if target_path.exists() and not overwrite:
        return {"status": "skipped_existing", "source": source_text, "target": str(target_path)}

    try:
        with h5py.File(source_path, "r") as source:
            if "rgb" not in source or "depth" not in source:
                raise KeyError("Missing required rgb/depth datasets")

            rgb = source["rgb"][:]
            depth = source["depth"][:]
            resized_rgb = resize_rgb(rgb, height, width)
            resized_depth = resize_depth(depth, height, width)

            target_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
            if temp_path.exists():
                temp_path.unlink()

            dataset_kwargs = {} if compression is None else {"compression": compression}
            with h5py.File(temp_path, "w") as target:
                target.create_dataset("rgb", data=resized_rgb, **dataset_kwargs)
                target.create_dataset("depth", data=resized_depth, **dataset_kwargs)
                copy_attrs(source, target)
                for key in source.keys():
                    if key in {"rgb", "depth"}:
                        continue
                    try:
                        source.copy(key, target)
                    except Exception:
                        target.attrs[f"skipped_dataset_{key}"] = "copy failed during resize export"
            temp_path.replace(target_path)
    except Exception as error:
        return {"status": "failed", "source": source_text, "target": str(target_path), "error": repr(error)}

    return {"status": "written", "source": source_text, "target": str(target_path)}


def run_split(
    input_root: Path,
    output_root: Path,
    split: str,
    height: int,
    width: int,
    num_workers: int,
    limit: int | None,
    compression: str | None,
    overwrite: bool,
) -> SplitSummary:
    split_root = input_root / split
    files = discover_files(input_root, split, limit)
    summary = SplitSummary(total=len(files))
    target_split_root = output_root / split
    tasks = [
        (str(path), str(split_root), str(target_split_root), height, width, compression, overwrite)
        for path in files
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(resize_one, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=split):
            result = future.result()
            status = result["status"]
            if status == "written":
                summary.written += 1
            elif status == "skipped_existing":
                summary.skipped_existing += 1
            elif status == "failed":
                summary.failed += 1
                summary.failures.append(result)
            else:
                raise ValueError(f"Unknown status: {status}")
    return summary


def validate_roots(input_root: Path, output_root: Path, overwrite: bool) -> None:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    for split in ("train", "val"):
        if not (input_root / split).exists():
            raise FileNotFoundError(f"Missing split directory: {input_root / split}")
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    compression = None if args.compression == "none" else args.compression
    validate_roots(input_root, output_root, args.overwrite)

    summaries = {
        split: run_split(
            input_root=input_root,
            output_root=output_root,
            split=split,
            height=args.height,
            width=args.width,
            num_workers=args.num_workers,
            limit=args.limit_per_split,
            compression=compression,
            overwrite=args.overwrite,
        )
        for split in ("train", "val")
    }
    summary = ResizeSummary(
        input_root=str(input_root),
        output_root=str(output_root),
        height=args.height,
        width=args.width,
        compression=compression,
        splits=summaries,
    )
    summary_path = output_root / args.summary_name
    summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    print(json.dumps(asdict(summary), indent=2)[:5000])
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
