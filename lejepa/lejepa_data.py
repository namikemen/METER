from __future__ import annotations

import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from lejepa_globals import IMAGENET_MEAN, IMAGENET_STD, LeJEPAConfig
from lejepa_utils import emit_log
from meter_data import DataConfig, _scaled_crop_bounds, discover_h5_files, resolve_data_root


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=image.device, dtype=image.dtype)
    std = IMAGENET_STD.to(device=image.device, dtype=image.dtype)
    return (image - mean) / std


class NYURGBDataset(Dataset):
    def __init__(self, files: list[Path], config: DataConfig, disable_meter_crop: bool) -> None:
        self.files = files
        self.config = config
        self.disable_meter_crop = disable_meter_crop

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.files[index]
        with h5py.File(path, "r") as h5:
            rgb_ds = h5["rgb"]
            if self.disable_meter_crop:
                image = torch.from_numpy(np.ascontiguousarray(rgb_ds[:], dtype=np.float32)) / 255.0
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=(self.config.input_height, self.config.input_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            else:
                height, width = rgb_ds.shape[-2:]
                top, left, bottom, right = _scaled_crop_bounds(height, width, self.config.crop_scale_min, random_crop=True)
                rgb = rgb_ds[:, top:bottom, left:right]
                image = torch.from_numpy(np.ascontiguousarray(rgb, dtype=np.float32)) / 255.0
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=(self.config.input_height, self.config.input_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
        return {"image": image, "path": str(path)}


def build_view_augment(image_size: tuple[int, int], scale: tuple[float, float]) -> v2.Compose:
    return v2.Compose(
        [
            v2.RandomResizedCrop(image_size, scale=scale, ratio=(0.75, 1.33), antialias=True),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
            v2.RandomApply([v2.RandomSolarize(threshold=0.5)], p=0.2),
            v2.RandomHorizontalFlip(),
        ]
    )


class NYUMultiViewRGBDataset(Dataset):
    def __init__(
        self,
        base_dataset: NYURGBDataset,
        global_views: int,
        local_views: int,
        image_size: tuple[int, int],
        global_crop_scale_min: float,
        local_crop_scale_min: float,
        local_crop_scale_max: float,
    ) -> None:
        self.base_dataset = base_dataset
        self.global_views = global_views
        self.local_views = local_views
        self.global_augment = build_view_augment(image_size, (global_crop_scale_min, 1.0))
        self.local_augment = build_view_augment(image_size, (local_crop_scale_min, local_crop_scale_max))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.base_dataset[index]
        image = sample["image"]
        global_views = torch.stack(
            [normalize_image(self.global_augment(image).clamp(0.0, 1.0)) for _ in range(self.global_views)]
        )
        local_views = torch.stack(
            [normalize_image(self.local_augment(image).clamp(0.0, 1.0)) for _ in range(self.local_views)]
        )
        return {"global_views": global_views, "local_views": local_views, "path": sample["path"]}


class NYURGBPreviewDataset(Dataset):
    def __init__(self, base_dataset: NYURGBDataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.base_dataset[index]
        return {"image": sample["image"], "path": sample["path"]}


def select_preview_files(files: list[Path], limit: int) -> list[Path]:
    target = max(1, min(limit, len(files)))
    selected: list[Path] = []
    seen_folders: set[Path] = set()
    for path in files:
        folder = path.parent
        if folder in seen_folders:
            continue
        selected.append(path)
        seen_folders.add(folder)
        if len(selected) >= target:
            return selected
    selected_paths = set(selected)
    for path in files:
        if path in selected_paths:
            continue
        selected.append(path)
        if len(selected) >= target:
            break
    return selected


def build_data_loaders(config: LeJEPAConfig, device: torch.device) -> tuple[DataLoader, DataLoader]:
    data_config = DataConfig(random_crop=1.0, shifting_strategy=0.0, c_swap=0.0)
    data_root = resolve_data_root(config.dataset_slug, config.local_data_root)
    train_files = discover_h5_files(data_root, "train", limit=config.train_limit)
    preview_files = select_preview_files(train_files, config.preview_limit)
    base_train = NYURGBDataset(train_files, data_config, config.disable_meter_crop)
    base_preview = NYURGBDataset(preview_files, data_config, config.disable_meter_crop)
    train_dataset = NYUMultiViewRGBDataset(
        base_train,
        config.global_views,
        config.local_views,
        (data_config.input_height, data_config.input_width),
        config.global_crop_scale_min,
        config.local_crop_scale_min,
        config.local_crop_scale_max,
    )
    preview_dataset = NYURGBPreviewDataset(base_preview)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    preview_loader = DataLoader(
        preview_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    emit_log(
        {
            "train_files": len(train_files),
            "preview_files": len(preview_files),
            "preview_folders": len({path.parent for path in preview_files}),
            "global_views": config.global_views,
            "local_views": config.local_views,
            "global_crop_scale": [config.global_crop_scale_min, 1.0],
            "local_crop_scale": [config.local_crop_scale_min, config.local_crop_scale_max],
            "local_loss_weight": config.local_loss_weight,
            "batches": len(train_loader),
        },
        config,
    )
    return train_loader, preview_loader


def build_smoke_loaders(config: LeJEPAConfig, device: torch.device) -> tuple[DataLoader, DataLoader]:
    return build_data_loaders(config, device)
