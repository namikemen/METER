from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn.functional as F


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
    splits: dict[str, SplitSummary]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize NYU H5 rgb/depth files to METER input size.")
    parser.add_argument("--input-root", type=Path, default=Path("origin_data"))
    parser.add_argument("--output-root", type=Path, default=Path("nyu_depth_v2_resized_192x256"))
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-name", default="resize_summary.json")
    return parser.parse_args()


def resize_rgb(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError(f"Expected rgb shape (3,H,W), got {rgb.shape}")

    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)[0]
    return resized.round().clamp(0, 255).to(torch.uint8).numpy()


def resize_depth(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape (H,W), got {depth.shape}")

    tensor = torch.from_numpy(np.ascontiguousarray(depth[None, None, :, :])).float()
    resized = F.interpolate(tensor, size=(height, width), mode="nearest")[0, 0]
    return resized.numpy().astype(depth.dtype, copy=False)


def copy_attrs(source: h5py.File, target: h5py.File) -> None:
    for key, value in source.attrs.items():
        try:
            target.attrs[key] = value
        except TypeError:
            target.attrs[key] = str(value)


def write_resized_file(source_path: Path, target_path: Path, height: int, width: int) -> None:
    with h5py.File(source_path, "r") as source:
        if "rgb" not in source or "depth" not in source:
            raise KeyError(f"Missing required rgb/depth datasets in {source_path}")

        rgb = source["rgb"][:]
        depth = source["depth"][:]
        resized_rgb = resize_rgb(rgb, height, width)
        resized_depth = resize_depth(depth, height, width)

        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        if temp_path.exists():
            temp_path.unlink()

        with h5py.File(temp_path, "w") as target:
            target.create_dataset("rgb", data=resized_rgb, compression="gzip", compression_opts=4)
            target.create_dataset("depth", data=resized_depth, compression="gzip", compression_opts=4)
            copy_attrs(source, target)
            for key in source.keys():
                if key in {"rgb", "depth"}:
                    continue
                try:
                    source.copy(key, target)
                except Exception:
                    target.attrs[f"skipped_dataset_{key}"] = "copy failed during resize export"
        temp_path.replace(target_path)


def discover_files(split_root: Path, limit: int | None) -> list[Path]:
    files = sorted(split_root.rglob("*.h5"))
    return files[:limit] if limit is not None else files


def resize_split(
    input_root: Path,
    output_root: Path,
    split: str,
    height: int,
    width: int,
    limit: int | None,
    overwrite: bool,
) -> SplitSummary:
    source_root = input_root / split
    target_root = output_root / split
    files = discover_files(source_root, limit)
    summary = SplitSummary(total=len(files))

    for index, source_path in enumerate(files, start=1):
        relative_path = source_path.relative_to(source_root)
        target_path = target_root / relative_path
        if target_path.exists() and not overwrite:
            summary.skipped_existing += 1
            continue

        try:
            write_resized_file(source_path, target_path, height, width)
            summary.written += 1
        except Exception as error:  # keep converting even when source has corrupt pseudo-H5 files
            summary.failed += 1
            summary.failures.append({"path": str(source_path), "error": repr(error)})

        if index % 1000 == 0 or index == len(files):
            print(
                f"{split}: {index}/{len(files)} "
                f"written={summary.written} failed={summary.failed} skipped={summary.skipped_existing}",
                flush=True,
            )
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
    validate_roots(input_root, output_root, args.overwrite)

    summaries = {
        split: resize_split(
            input_root=input_root,
            output_root=output_root,
            split=split,
            height=args.height,
            width=args.width,
            limit=args.limit_per_split,
            overwrite=args.overwrite,
        )
        for split in ("train", "val")
    }
    summary = ResizeSummary(
        input_root=str(input_root),
        output_root=str(output_root),
        height=args.height,
        width=args.width,
        splits=summaries,
    )
    summary_path = output_root / args.summary_name
    summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
