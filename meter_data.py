from __future__ import annotations

import random
import sys
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Sequence

CropBounds = tuple[int, int, int, int]

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DataConfig:
    input_height: int = 192
    input_width: int = 256
    max_depth_cm: float = 1000.0
    invalid_depth_threshold_cm: float = 1.0
    flip: float = 0.5
    mirror: float = 0.5
    c_swap: float = 0.5
    random_crop: float = 1.0
    shifting_strategy: float = 0.5
    color_low: float = 0.9
    color_high: float = 1.1
    depth_shift_min_cm: float = -10.0
    depth_shift_max_cm: float = 10.0


def add_code_path(code_root: str | Path) -> Path:
    path = Path(code_root).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Code path does not exist: {path}")

    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    return path


def resolve_data_root(dataset_slug: str, local_data_root: str) -> Path:
    if Path("/kaggle/input").exists():
        base = Path("/kaggle/input") / dataset_slug
        candidates = [base, base / "nyu_depth_v2", base / "nyu-depth-v2"]
    else:
        local_root = Path(local_data_root).expanduser().resolve()
        candidates = [local_root, local_root / "nyu_depth_v2", local_root / "nyu-depth-v2"]

    for candidate in candidates:
        if (candidate / "train").exists() and (candidate / "val").exists():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find NYU train/val folders. Checked: {checked}")


def discover_h5_files(data_root: Path, split: str, limit: int | None = None) -> list[Path]:
    split_root = data_root / split
    files = sorted({*split_root.glob("*.h5"), *split_root.glob("*/*.h5")})
    if not files:
        files = sorted(split_root.rglob("*.h5"))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No .h5 files found for split={split} under {split_root}")
    return files


def _crop_bounds(height: int, width: int, crop_height: int, crop_width: int, random_crop: bool) -> CropBounds:
    if height < crop_height or width < crop_width:
        raise ValueError(f"Cannot crop {(crop_height, crop_width)} from original {(height, width)}")

    if random_crop:
        top = 0 if height == crop_height else np.random.randint(0, height - crop_height + 1)
        left = 0 if width == crop_width else np.random.randint(0, width - crop_width + 1)
    else:
        top = max((height - crop_height) // 2, 0)
        left = max((width - crop_width) // 2, 0)
    return top, left, top + crop_height, left + crop_width


def _crop_pair(
    rgb: np.ndarray,
    depth_cm: np.ndarray,
    mask: np.ndarray,
    height: int,
    width: int,
    random_crop: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    top, left, bottom, right = _crop_bounds(*depth_cm.shape, height, width, random_crop)
    return rgb[:, top:bottom, left:right], depth_cm[top:bottom, left:right], mask[top:bottom, left:right]


def apply_meter_augmentations(
    rgb: np.ndarray,
    depth_cm: np.ndarray,
    mask: np.ndarray,
    config: DataConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if random.random() <= config.flip:
        rgb = rgb[:, ::-1, :].copy()
        depth_cm = depth_cm[::-1, :].copy()
        mask = mask[::-1, :].copy()

    if random.random() <= config.mirror:
        rgb = rgb[:, :, ::-1].copy()
        depth_cm = depth_cm[:, ::-1].copy()
        mask = mask[:, ::-1].copy()

    if random.random() <= config.c_swap:
        policies = list(product([0, 1, 2], repeat=3))
        rgb = rgb[list(policies[random.randint(0, len(policies) - 1)]), :, :]

    if random.random() <= config.shifting_strategy:
        gamma = random.uniform(config.color_low, config.color_high)
        brightness = random.uniform(config.color_low, config.color_high)
        colors = np.random.uniform(config.color_low, config.color_high, size=(3, 1, 1)).astype(np.float32)
        rgb = np.clip((rgb**gamma) * brightness * colors, 0.0, 1.0)

        depth_shift = random.uniform(config.depth_shift_min_cm, config.depth_shift_max_cm)
        depth_cm = depth_cm + depth_shift
        mask = mask & (depth_cm > config.invalid_depth_threshold_cm)

    if random.random() <= config.random_crop:
        for _ in range(10):
            cropped_rgb, cropped_depth, cropped_mask = _crop_pair(
                rgb,
                depth_cm,
                mask,
                config.input_height,
                config.input_width,
                random_crop=True,
            )
            if cropped_mask.any():
                return cropped_rgb.copy(), cropped_depth.copy(), cropped_mask.copy()

    rgb, depth_cm, mask = _crop_pair(
        rgb,
        depth_cm,
        mask,
        config.input_height,
        config.input_width,
        random_crop=False,
    )
    return rgb.copy(), depth_cm.copy(), mask.copy()


class NYUH5DepthDataset(Dataset):
    def __init__(self, files: Sequence[Path], config: DataConfig, train: bool) -> None:
        self.files = list(files)
        self.config = config
        self.train = train

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.files[index]
        with h5py.File(path, "r") as h5:
            depth_ds = h5["depth"]
            use_random_crop = self.train and random.random() <= self.config.random_crop
            top, left, bottom, right = _crop_bounds(
                depth_ds.shape[0],
                depth_ds.shape[1],
                self.config.input_height,
                self.config.input_width,
                random_crop=use_random_crop,
            )
            depth_cm = depth_ds[top:bottom, left:right].astype(np.float32) * 100.0
            if use_random_crop:
                for _ in range(9):
                    if (depth_cm > self.config.invalid_depth_threshold_cm).any():
                        break
                    top, left, bottom, right = _crop_bounds(
                        depth_ds.shape[0],
                        depth_ds.shape[1],
                        self.config.input_height,
                        self.config.input_width,
                        random_crop=True,
                    )
                    depth_cm = depth_ds[top:bottom, left:right].astype(np.float32) * 100.0
            rgb = h5["rgb"][:, top:bottom, left:right].astype(np.float32) / 255.0

        mask = depth_cm > self.config.invalid_depth_threshold_cm
        if self.train:
            augmentation_config = replace(self.config, random_crop=0.0)
            rgb, depth_cm, mask = apply_meter_augmentations(rgb, depth_cm, mask, augmentation_config)

        depth_cm = np.clip(depth_cm, 0.0, self.config.max_depth_cm)
        image = np.ascontiguousarray(rgb, dtype=np.float32)
        depth = np.ascontiguousarray(depth_cm[None, :, :], dtype=np.float32)
        mask_tensor = np.ascontiguousarray(mask[None, :, :], dtype=np.float32)
        return {
            "image": torch.from_numpy(image),
            "depth": torch.from_numpy(depth),
            "mask": torch.from_numpy(mask_tensor),
            "path": str(path),
        }
