from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
    max_depth_cm: float = 1000.0
    invalid_depth_threshold_cm: float = 1.0
    lambda_1: float = 0.5
    lambda_2: float = 100.0
    lambda_3: float = 100.0


class Sobel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        edge_kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        edge_ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer("kernel", torch.stack((edge_kx, edge_ky)).view(2, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.kernel, padding=1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(eps)


def ssim_depth(x: torch.Tensor, y: torch.Tensor, val_range: float, window_size: int = 11) -> torch.Tensor:
    c1 = (0.01 * val_range) ** 2
    c2 = (0.03 * val_range) ** 2
    pad = window_size // 2
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=pad) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=pad) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_x * mu_y
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    )


class BalancedMETERLoss(nn.Module):
    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config
        self.sobel = Sobel()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(
        self,
        pred: torch.Tensor,
        depth: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if pred.shape[-2:] != depth.shape[-2:]:
            pred = F.interpolate(pred, size=depth.shape[-2:], mode="bilinear", align_corners=False)

        pred = pred.clamp_min(0.0)
        valid = (mask > 0.5) & (depth > self.config.invalid_depth_threshold_cm)
        valid_f = valid.float()

        loss_depth = masked_mean((pred - depth).abs(), valid_f)

        depth_grad = self.sobel(depth)
        pred_grad = self.sobel(pred)
        loss_grad_raw = masked_mean((pred_grad - depth_grad).abs(), valid_f.expand_as(depth_grad))

        ones = torch.ones_like(depth)
        depth_normal = torch.cat((-depth_grad[:, 0:1], -depth_grad[:, 1:2], ones), dim=1)
        pred_normal = torch.cat((-pred_grad[:, 0:1], -pred_grad[:, 1:2], ones), dim=1)
        normal_error = (1.0 - self.cos(pred_normal, depth_normal)).abs().unsqueeze(1)
        loss_normal_raw = masked_mean(normal_error, valid_f)

        pred_for_ssim = torch.where(valid, pred, depth.detach())
        loss_ssim_raw = masked_mean(
            1.0 - ssim_depth(pred_for_ssim, depth, self.config.max_depth_cm),
            valid_f,
        )

        loss_grad = loss_grad_raw / self.config.lambda_1
        loss_normal = loss_normal_raw * self.config.lambda_2
        loss_ssim = loss_ssim_raw * self.config.lambda_3
        total = loss_depth + loss_grad + loss_normal + loss_ssim

        components = {
            "loss_depth": loss_depth.detach(),
            "loss_grad": loss_grad.detach(),
            "loss_normal": loss_normal.detach(),
            "loss_ssim": loss_ssim.detach(),
            "valid_pixels": valid_f.sum().detach(),
        }
        return total, components
