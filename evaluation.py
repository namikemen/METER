from __future__ import annotations

import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

IMAGENET_MEAN_TENSOR = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD_TENSOR = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

def compute_metrics(
    pred: torch.Tensor,
    depth: torch.Tensor,
    mask: torch.Tensor,
    max_depth_cm: float,
    invalid_depth_threshold_cm: float,
) -> dict[str, float]:
    if pred.shape[-2:] != depth.shape[-2:]:
        pred = F.interpolate(pred, size=depth.shape[-2:], mode="bilinear", align_corners=False)
    pred = pred.clamp(1.0, max_depth_cm)
    valid = (mask > 0.5) & (depth > invalid_depth_threshold_cm)
    if valid.sum() == 0:
        return {"rmse_cm": math.nan, "rmse_m": math.nan, "rel": math.nan, "delta1": math.nan}

    p = pred[valid]
    d = depth[valid]
    rmse_cm = torch.sqrt(torch.mean((p - d) ** 2))
    rel = torch.mean(torch.abs(p - d) / d.clamp_min(1.0))
    ratio = torch.maximum(p / d.clamp_min(1.0), d / p.clamp_min(1.0))
    delta1 = torch.mean((ratio < 1.25).float())
    return {
        "rmse_cm": float(rmse_cm.detach().cpu()),
        "rmse_m": float((rmse_cm / 100.0).detach().cpu()),
        "rel": float(rel.detach().cpu()),
        "delta1": float(delta1.detach().cpu()),
    }


def _denormalize_rgb(image: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN_TENSOR.to(dtype=image.dtype, device=image.device)
    std = IMAGENET_STD_TENSOR.to(dtype=image.dtype, device=image.device)
    return (image * std + mean).clamp(0.0, 1.0)


def make_visual_figure(visuals: list[dict[str, torch.Tensor]]) -> plt.Figure:
    rows = max(len(visuals), 1)
    fig, axes = plt.subplots(rows, 4, figsize=(14, 3.5 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    for row, item in enumerate(visuals):
        image = _denormalize_rgb(item["image"]).permute(1, 2, 0).numpy()
        depth = item["depth"][0].numpy()
        pred = item["pred"][0].numpy()
        mask = item["mask"][0].numpy().astype(bool)
        error = np.abs(pred - depth)
        panels = [image, np.where(mask, depth, np.nan), np.where(mask, pred, np.nan), np.where(mask, error, np.nan)]
        titles = ["RGB", "GT depth cm", "Pred depth cm", "Abs error cm"]
        cmaps = [None, "plasma_r", "plasma_r", "magma"]
        for col, (panel, title, cmap) in enumerate(zip(panels, titles, cmaps)):
            ax = axes[row, col]
            im = ax.imshow(panel, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")
            if col > 0:
                fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig

