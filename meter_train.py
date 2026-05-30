from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None


@dataclass(frozen=True)
class TrainConfig:
    num_epochs: int = 60
    learning_rate: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.01
    scheduler_step_size: int = 20
    scheduler_gamma: float = 0.1
    use_amp: bool = True
    max_depth_cm: float = 1000.0
    invalid_depth_threshold_cm: float = 1.0
    output_dir: str = "outputs"
    checkpoint_name: str = "meter_nyu_best.pt"
    final_checkpoint_name: str = "meter_nyu_final.pt"
    use_wandb: bool = True
    wandb_project: str = "meter-nyu-depth"
    wandb_run_name: str = "meter-nyu"
    wandb_mode: str = "online"
    vis_every_epochs: int = 10
    vis_num_samples: int = 4


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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


def _update_component_sums(sums: dict[str, float], components: dict[str, torch.Tensor]) -> None:
    for key, value in components.items():
        if key != "valid_pixels":
            sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())


def _averages(sums: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in sums.items()}


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    config: TrainConfig,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    component_sums: dict[str, float] = {}
    batches = 0

    for batch in tqdm(loader, desc=f"Train {epoch}", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=config.use_amp and device.type == "cuda"):
            pred = model(image)
            loss, components = criterion(pred, depth, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu())
        _update_component_sums(component_sums, components)
        batches += 1
        del image, depth, mask, pred, loss, components

    logs = {"train_loss": total_loss / max(batches, 1)}
    logs.update({f"train_{key}": value for key, value in _averages(component_sums, batches).items()})
    return logs


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    config: TrainConfig,
    epoch: int,
) -> tuple[dict[str, float], list[dict[str, torch.Tensor]]]:
    model.eval()
    total_loss = 0.0
    component_sums: dict[str, float] = {}
    metric_sums = {"rmse_cm": 0.0, "rmse_m": 0.0, "rel": 0.0, "delta1": 0.0}
    metric_count = 0
    visuals: list[dict[str, torch.Tensor]] = []
    batches = 0

    for batch in tqdm(loader, desc=f"Val {epoch}", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        depth = batch["depth"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        pred = model(image)
        loss, components = criterion(pred, depth, mask)
        metrics = compute_metrics(pred, depth, mask, config.max_depth_cm, config.invalid_depth_threshold_cm)
        if not math.isnan(metrics["rmse_cm"]):
            for key, value in metrics.items():
                metric_sums[key] += value
            metric_count += 1

        total_loss += float(loss.detach().cpu())
        _update_component_sums(component_sums, components)
        batches += 1
        if len(visuals) < config.vis_num_samples:
            pred_vis = F.interpolate(pred, size=depth.shape[-2:], mode="bilinear", align_corners=False)
            visuals.append(
                {
                    "image": image[0].detach().cpu().clone(),
                    "depth": depth[0].detach().cpu().clone(),
                    "pred": pred_vis[0].detach().cpu().clone(),
                    "mask": mask[0].detach().cpu().clone(),
                }
            )
        del image, depth, mask, pred, loss, components, metrics

    logs = {"val_loss": total_loss / max(batches, 1)}
    logs.update({f"val_{key}": value for key, value in _averages(component_sums, batches).items()})
    logs.update({f"val_{key}": value / max(metric_count, 1) for key, value in metric_sums.items()})
    return logs, visuals


def make_visual_figure(visuals: list[dict[str, torch.Tensor]]) -> plt.Figure:
    rows = max(len(visuals), 1)
    fig, axes = plt.subplots(rows, 4, figsize=(14, 3.5 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    for row, item in enumerate(visuals):
        image = item["image"].permute(1, 2, 0).numpy()
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


def _init_wandb(config: TrainConfig, extra_config: dict[str, Any]):
    if not config.use_wandb or config.wandb_mode == "disabled" or wandb is None:
        return None
    merged = {**asdict(config), **extra_config}
    return wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        mode=config.wandb_mode,
        config=merged,
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    train_config: TrainConfig,
    experiment_config: dict[str, Any],
) -> list[dict[str, float]]:
    output_dir = Path(train_config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scaler = GradScaler(enabled=train_config.use_amp and device.type == "cuda")
    run = _init_wandb(train_config, experiment_config)
    history: list[dict[str, float]] = []
    best_rmse = float("inf")

    for epoch in range(1, train_config.num_epochs + 1):
        start = time.time()
        train_logs = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, train_config, epoch)
        val_logs, visuals = validate(model, val_loader, criterion, device, train_config, epoch)
        scheduler.step()
        logs = {
            **train_logs,
            **val_logs,
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "epoch_seconds": time.time() - start,
        }
        history.append(logs)
        print(json.dumps(logs, indent=2))

        if logs["val_rmse_cm"] < best_rmse:
            best_rmse = logs["val_rmse_cm"]
            save_checkpoint(
                checkpoint_dir / train_config.checkpoint_name,
                model,
                optimizer,
                scheduler,
                epoch,
                logs,
                experiment_config,
            )

        if run is not None:
            wandb_logs: dict[str, Any] = dict(logs)
            should_visualize = epoch == 1 or epoch == train_config.num_epochs or epoch % train_config.vis_every_epochs == 0
            if should_visualize:
                fig = make_visual_figure(visuals)
                wandb_logs["validation_visuals"] = wandb.Image(fig)
                plt.close(fig)
            wandb.log(wandb_logs, step=epoch)

    final_metrics = history[-1] if history else {}
    save_checkpoint(
        checkpoint_dir / train_config.final_checkpoint_name,
        model,
        optimizer,
        scheduler,
        train_config.num_epochs,
        final_metrics,
        experiment_config,
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if run is not None:
        wandb.finish()
    return history
