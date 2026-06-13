from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lejepa_globals import LeJEPAConfig
from lejepa_data import normalize_image
from lejepa_network import unwrap_model


def compute_pca_map(features: torch.Tensor) -> np.ndarray:
    channels, height, width = features.shape
    patches = features.reshape(channels, -1).transpose(0, 1).float().cpu()
    components = min(3, patches.shape[0], patches.shape[1])
    patches = patches - patches.mean(dim=0, keepdim=True)
    if components == 0:
        return np.zeros((height, width, 3), dtype=np.float32)
    _, _, vectors = torch.pca_lowrank(patches, q=components, center=False)
    projected = patches @ vectors[:, :components]
    pca_map = projected.reshape(height, width, components).numpy()
    for channel in range(pca_map.shape[-1]):
        component = pca_map[..., channel]
        pca_map[..., channel] = (component - component.min()) / (component.max() - component.min() + 1e-8)
    if pca_map.shape[-1] < 3:
        pad = np.zeros((height, width, 3 - pca_map.shape[-1]), dtype=np.float32)
        pca_map = np.concatenate([pca_map, pad], axis=-1)
    return pca_map.astype(np.float32)


def resize_pca_map(pca_map: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    pca_tensor = torch.from_numpy(pca_map).permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(pca_tensor, size=image_shape, mode="bilinear", align_corners=False)
    return resized.squeeze(0).permute(1, 2, 0).numpy().clip(0.0, 1.0)


def visualize_pca_embeddings(
    *,
    epoch: int,
    preview_loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    pca_dir: Path,
    config: LeJEPAConfig,
) -> Path:
    model.eval()
    rows: list[tuple[np.ndarray, np.ndarray, str]] = []
    with torch.no_grad():
        for batch in preview_loader:
            images = batch["image"].to(device)
            names = batch["path"]
            features, _ = unwrap_model(model).encoder(images)
            for image, feature, name in zip(images.cpu(), features.cpu(), names):
                pca_map = compute_pca_map(feature)
                image_np = image.permute(1, 2, 0).numpy().clip(0.0, 1.0)
                pca_map = resize_pca_map(pca_map, image_np.shape[:2])
                rows.append((image_np, pca_map, Path(name).stem))
                if len(rows) >= min(4, config.preview_limit):
                    break
            if len(rows) >= min(4, config.preview_limit):
                break
    model.train()

    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 4 * len(rows)))
    if len(rows) == 1:
        axes = np.asarray([axes])
    for row, (image_np, pca_map, scene_name) in enumerate(rows):
        axes[row, 0].imshow(image_np)
        axes[row, 0].set_title(f"Input — {scene_name}")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(pca_map)
        axes[row, 1].set_title("Encoder patch PCA")
        axes[row, 1].axis("off")
    fig.suptitle(f"LeJEPA METER {config.model_size} PCA — epoch {epoch}")
    fig.tight_layout()
    pca_path = pca_dir / f"pca_epoch_{epoch:03d}.png"
    fig.savefig(pca_path, dpi=160)
    plt.close(fig)
    return pca_path


def maybe_log_epoch_media(
    *,
    wandb_module: object | None,
    pca_path: Path,
    checkpoint_path: Path,
    wandb_logs: dict[str, object],
) -> None:
    if wandb_module is not None and pca_path.exists():
        wandb_logs["pca"] = wandb_module.Image(str(pca_path))
    wandb_logs["checkpoint_path"] = str(checkpoint_path)
