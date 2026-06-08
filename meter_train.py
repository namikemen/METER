from __future__ import annotations

import json
import math
import os
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
from tqdm.std import tqdm

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None


IMAGENET_MEAN_TENSOR = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD_TENSOR = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


@dataclass(frozen=True)
class TrainConfig:
    num_epochs: int = 60
    learning_rate: float = 1e-3
    encoder_learning_rate: float = 1e-4
    decoder_learning_rate: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.03
    dropout: float = 0.0
    attention_dropout: float = 0.0
    drop_path: float = 0.0
    regularization_start_epoch: int = 31
    scheduled_augmentation_start_epoch: int = 31
    freeze_encoder_epochs: int = 0
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    ema_decay: float = 0.0
    scheduler_type: str = "plateau"
    scheduler_step_size: int = 20
    scheduler_gamma: float = 0.1
    scheduler_eta_min: float = 1e-6
    scheduler_factor: float = 0.5
    scheduler_patience: int = 4
    scheduler_min_lr: float = 1e-6
    checkpoint_metric: str = "val_rmse_m"
    checkpoint_mode: str = "min"
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
    use_tqdm: bool = False
    log_every_steps: int = 25
    wandb_log_every_steps: int = 25


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


def _update_component_sums(sums: dict[str, torch.Tensor], components: dict[str, torch.Tensor]) -> None:
    for key, value in components.items():
        if key != "valid_pixels":
            detached = value.detach()
            sums[key] = detached if key not in sums else sums[key] + detached


def _averages(sums: dict[str, torch.Tensor], count: int) -> dict[str, float]:
    return {key: float((value / max(count, 1)).detach().cpu()) for key, value in sums.items()}


def _process_tree_ram_gb() -> float | None:
    if psutil is None:
        return None

    process = psutil.Process(os.getpid())
    processes = [process, *process.children(recursive=True)]
    rss_bytes = 0
    for child in processes:
        try:
            rss_bytes += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return rss_bytes / 1024**3


def _log_ram(prefix: str, step: int, every: int = 10) -> None:
    if step % every != 0:
        return
    ram_gb = _process_tree_ram_gb()
    if ram_gb is not None:
        tqdm.write(f"{prefix} step={step} cpu_ram_tree={ram_gb:.2f} GB")


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def set_encoder_trainable(model: torch.nn.Module, trainable: bool) -> None:
    module = _unwrap_model(model)
    if not hasattr(module, "encoder"):
        raise AttributeError("Model does not expose an encoder attribute to freeze")
    for param in module.encoder.parameters():
        param.requires_grad = trainable


def set_model_regularization_rates(
    model: torch.nn.Module,
    dropout: float,
    attention_dropout: float,
    drop_path: float,
) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = attention_dropout if getattr(module, "is_attention_dropout", False) else dropout
        elif hasattr(module, "drop_prob"):
            module.drop_prob = drop_path


def _set_dataset_epoch(loader: DataLoader, epoch: int) -> None:
    dataset = loader.dataset
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state_dict = model.state_dict()
        for key, value in state_dict.items():
            if key not in self.shadow or not torch.is_floating_point(value):
                continue
            self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def store(self, model: torch.nn.Module) -> None:
        self.backup = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if key in self.shadow
        }

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        state_dict = model.state_dict()
        state_dict.update({key: value.to(state_dict[key].device) for key, value in self.shadow.items()})
        model.load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        if not self.backup:
            return
        state_dict = model.state_dict()
        state_dict.update({key: value.to(state_dict[key].device) for key, value in self.backup.items()})
        model.load_state_dict(state_dict, strict=False)
        self.backup = {}


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    config: TrainConfig,
    epoch: int,
    run: Any | None = None,
    global_step: int = 0,
    ema: ExponentialMovingAverage | None = None,
) -> tuple[dict[str, float], int]:
    model.train()
    total_loss: torch.Tensor | None = None
    component_sums: dict[str, torch.Tensor] = {}
    batches = 0

    progress = tqdm(loader, desc=f"Train {epoch}", leave=False) if config.use_tqdm else loader
    for batch in progress:
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
        if ema is not None:
            ema.update(model)
        loss_detached = loss.detach()
        total_loss = loss_detached if total_loss is None else total_loss + loss_detached
        _update_component_sums(component_sums, components)
        batches += 1
        global_step += 1
        should_log = config.log_every_steps > 0 and batches % config.log_every_steps == 0
        should_log_wandb = run is not None and config.wandb_log_every_steps > 0 and global_step % config.wandb_log_every_steps == 0
        if should_log or should_log_wandb:
            loss_value = float(loss_detached.cpu())
            component_values = {key: float(value.detach().cpu()) for key, value in components.items() if key != "valid_pixels"}
            if config.use_tqdm:
                progress.set_postfix(loss=f"{loss_value:.4f}")
            elif should_log:
                print(f"Train {epoch} step={batches} global_step={global_step} loss={loss_value:.4f}")
            if should_log_wandb:
                wandb.log(
                    {
                        "train/loss_total": loss_value,
                        "train/loss_depth": component_values.get("loss_depth", math.nan),
                        "train/loss_grad": component_values.get("loss_grad", math.nan),
                        "train/loss_normal": component_values.get("loss_normal", math.nan),
                        "train/loss_ssim": component_values.get("loss_ssim", math.nan),
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )
        del batch, image, depth, mask, pred, loss, loss_detached, components

    logs = {"train_loss": float((total_loss / max(batches, 1)).detach().cpu()) if total_loss is not None else math.nan}
    logs.update({f"train_{key}": value for key, value in _averages(component_sums, batches).items()})
    return logs, global_step


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
    total_loss: torch.Tensor | None = None
    component_sums: dict[str, torch.Tensor] = {}
    metric_sums = {"rmse_cm": 0.0, "rmse_m": 0.0, "rel": 0.0, "delta1": 0.0}
    metric_count = 0
    visuals: list[dict[str, torch.Tensor]] = []
    batches = 0

    progress = tqdm(loader, desc=f"Val {epoch}", leave=False) if config.use_tqdm else loader
    for batch in progress:
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

        loss_detached = loss.detach()
        total_loss = loss_detached if total_loss is None else total_loss + loss_detached
        _update_component_sums(component_sums, components)
        batches += 1
        should_log = config.log_every_steps > 0 and batches % config.log_every_steps == 0
        if should_log:
            loss_value = float(loss_detached.cpu())
            if config.use_tqdm:
                progress.set_postfix(loss=f"{loss_value:.4f}", rmse=f"{metrics['rmse_m']:.3f}m", delta1=f"{metrics['delta1']:.3f}")
            else:
                print(f"Val {epoch} step={batches} loss={loss_value:.4f} rmse={metrics['rmse_m']:.3f}m delta1={metrics['delta1']:.3f}")
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
        del batch, image, depth, mask, pred, loss, loss_detached, components, metrics

    logs = {"val_loss": float((total_loss / max(batches, 1)).detach().cpu()) if total_loss is not None else math.nan}
    logs.update({f"val_{key}": value for key, value in _averages(component_sums, batches).items()})
    logs.update({f"val_{key}": value / max(metric_count, 1) for key, value in metric_sums.items()})
    return logs, visuals


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


def _named_trainable_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def build_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    named_parameters = _named_trainable_parameters(model)
    encoder_params = [param for name, param in named_parameters if name.startswith("encoder") or "encoder" in name]
    decoder_params = [param for name, param in named_parameters if not (name.startswith("encoder") or "encoder" in name)]
    parameter_groups: list[dict[str, Any]] = []
    if encoder_params:
        parameter_groups.append({"params": encoder_params, "lr": config.encoder_learning_rate, "name": "encoder"})
    if decoder_params:
        parameter_groups.append({"params": decoder_params, "lr": config.decoder_learning_rate, "name": "decoder"})
    if not parameter_groups:
        raise ValueError("No trainable parameters found for optimizer")

    return torch.optim.AdamW(
        parameter_groups,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig):
    if config.scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
    if config.scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs,
            eta_min=config.scheduler_eta_min,
        )
    if config.scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
        )
    raise ValueError(f"Unsupported scheduler_type: {config.scheduler_type}")


def _metric_value(logs: dict[str, float], metric_name: str) -> float:
    if metric_name not in logs:
        available = ", ".join(sorted(logs))
        raise KeyError(f"Metric {metric_name!r} was not found in logs. Available metrics: {available}")
    return logs[metric_name]


def _step_scheduler(scheduler: Any, config: TrainConfig, logs: dict[str, float]) -> None:
    if config.scheduler_type == "plateau":
        scheduler.step(_metric_value(logs, config.checkpoint_metric))
    else:
        scheduler.step()


def _optimizer_lr_logs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    logs: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{index}"))
        logs[f"lr_{name}"] = float(group["lr"])
    if optimizer.param_groups:
        logs["lr"] = float(optimizer.param_groups[0]["lr"])
    return logs


def _is_better_metric(current: float, best: float, mode: str, min_delta: float = 0.0) -> bool:
    if not math.isfinite(current):
        return False
    if mode == "min":
        return current < best - min_delta
    if mode == "max":
        return current > best + min_delta
    raise ValueError(f"Unsupported checkpoint_mode: {mode}")


def _initial_best_metric(mode: str) -> float:
    if mode == "min":
        return float("inf")
    if mode == "max":
        return -float("inf")
    raise ValueError(f"Unsupported checkpoint_mode: {mode}")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    stripped = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()}
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": stripped,
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
    scheduler: Any,
    device: torch.device,
    train_config: TrainConfig,
    experiment_config: dict[str, Any],
) -> list[dict[str, float]]:
    output_dir = Path(train_config.output_dir)
    if Path("/kaggle/working").exists() and not output_dir.is_absolute():
        output_dir = Path("/kaggle/working") / output_dir
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    scaler = GradScaler(enabled=train_config.use_amp and device.type == "cuda")
    run = _init_wandb(train_config, experiment_config)
    history: list[dict[str, float]] = []
    best_metric = _initial_best_metric(train_config.checkpoint_mode)
    global_step = 0
    epochs_without_improvement = 0
    ema: ExponentialMovingAverage | None = None

    for epoch in range(1, train_config.num_epochs + 1):
        start = time.time()
        regularization_active = epoch >= train_config.regularization_start_epoch
        _set_dataset_epoch(train_loader, epoch)
        set_model_regularization_rates(
            model,
            dropout=train_config.dropout if regularization_active else 0.0,
            attention_dropout=train_config.attention_dropout if regularization_active else 0.0,
            drop_path=train_config.drop_path if regularization_active else 0.0,
        )
        if regularization_active and ema is None and train_config.ema_decay > 0.0:
            ema = ExponentialMovingAverage(model, train_config.ema_decay)
        encoder_frozen = epoch <= train_config.freeze_encoder_epochs
        set_encoder_trainable(model, not encoder_frozen)
        train_logs, global_step = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            train_config,
            epoch,
            run,
            global_step,
            ema,
        )
        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
        try:
            val_logs, visuals = validate(model, val_loader, criterion, device, train_config, epoch)
            logs = {
                **train_logs,
                **val_logs,
                "epoch": epoch,
                "encoder_frozen": float(encoder_frozen),
                "regularization_active": float(regularization_active),
                "ema_enabled": float(ema is not None),
                "dropout_active": train_config.dropout if regularization_active else 0.0,
                "attention_dropout_active": train_config.attention_dropout if regularization_active else 0.0,
                "drop_path_active": train_config.drop_path if regularization_active else 0.0,
                "epoch_seconds": time.time() - start,
            }
            _step_scheduler(scheduler, train_config, logs)
            logs.update(_optimizer_lr_logs(optimizer))
            history.append(logs)
            print(
                f"Epoch {epoch}: "
                f"train_loss={logs['train_loss']:.4f} "
                f"val_loss={logs['val_loss']:.4f} "
                f"val_rmse={logs['val_rmse_m']:.3f}m "
                f"val_delta1={logs['val_delta1']:.3f} "
                f"encoder_frozen={bool(encoder_frozen)}"
            )

            current_metric = _metric_value(logs, train_config.checkpoint_metric)
            improved = _is_better_metric(
                current_metric,
                best_metric,
                train_config.checkpoint_mode,
                train_config.early_stopping_min_delta,
            )
            if improved:
                best_metric = current_metric
                epochs_without_improvement = 0
                checkpoint_logs = {
                    **logs,
                    "best_metric": best_metric,
                    "checkpoint_metric": train_config.checkpoint_metric,
                    "checkpoint_mode": train_config.checkpoint_mode,
                }
                save_checkpoint(
                    checkpoint_dir / train_config.checkpoint_name,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    checkpoint_logs,
                    experiment_config,
                )
                print(
                    f"Saved best checkpoint to {checkpoint_dir / train_config.checkpoint_name} "
                    f"{train_config.checkpoint_metric}={best_metric:.4f}"
                )
            elif regularization_active:
                epochs_without_improvement += 1
            else:
                epochs_without_improvement = 0

            if run is not None:
                wandb_logs: dict[str, Any] = {
                    "epoch/train_loss": logs["train_loss"],
                    "epoch/val_loss": logs["val_loss"],
                    "epoch/val_rmse_m": logs["val_rmse_m"],
                    "epoch/val_rel": logs["val_rel"],
                    "epoch/val_delta1": logs["val_delta1"],
                    "epoch/lr": logs["lr"],
                    "epoch/lr_encoder": logs.get("lr_encoder", math.nan),
                    "epoch/lr_decoder": logs.get("lr_decoder", math.nan),
                    "epoch/seconds": logs["epoch_seconds"],
                    "val/rmse_cm": logs["val_rmse_cm"],
                    "val/rmse_m": logs["val_rmse_m"],
                    "val/rel": logs["val_rel"],
                    "val/delta1": logs["val_delta1"],
                    "lr/encoder": logs.get("lr_encoder", math.nan),
                    "lr/decoder": logs.get("lr_decoder", math.nan),
                    "train/encoder_frozen": logs["encoder_frozen"],
                    "train/regularization_active": logs["regularization_active"],
                    "train/ema_enabled": logs["ema_enabled"],
                    "train/ema_decay": train_config.ema_decay,
                    "regularization/dropout": logs["dropout_active"],
                    "regularization/attention_dropout": logs["attention_dropout_active"],
                    "regularization/drop_path": logs["drop_path_active"],
                    "train/epochs_without_improvement": epochs_without_improvement,
                    "train/loss_depth_epoch": logs.get("train_loss_depth", math.nan),
                    "train/loss_grad_epoch": logs.get("train_loss_grad", math.nan),
                    "train/loss_normal_epoch": logs.get("train_loss_normal", math.nan),
                    "train/loss_ssim_epoch": logs.get("train_loss_ssim", math.nan),
                    "val/loss_depth_epoch": logs.get("val_loss_depth", math.nan),
                    "val/loss_grad_epoch": logs.get("val_loss_grad", math.nan),
                    "val/loss_normal_epoch": logs.get("val_loss_normal", math.nan),
                    "val/loss_ssim_epoch": logs.get("val_loss_ssim", math.nan),
                    "train/epoch": epoch,
                }
                should_visualize = epoch == 1 or epoch == train_config.num_epochs or epoch % train_config.vis_every_epochs == 0
                if should_visualize:
                    fig = make_visual_figure(visuals)
                    wandb_logs["validation_visuals"] = wandb.Image(fig)
                    plt.close(fig)
                wandb.log(wandb_logs, step=global_step)

            if (
                regularization_active
                and train_config.early_stopping_patience > 0
                and epochs_without_improvement >= train_config.early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch}: no {train_config.checkpoint_metric} "
                    f"improvement for {epochs_without_improvement} epochs"
                )
                break
        finally:
            if ema is not None:
                ema.restore(model)

    final_metrics = history[-1] if history else {}
    final_epoch = int(final_metrics.get("epoch", train_config.num_epochs)) if final_metrics else train_config.num_epochs
    if ema is not None:
        ema.store(model)
        ema.copy_to(model)
    try:
        save_checkpoint(
            checkpoint_dir / train_config.final_checkpoint_name,
            model,
            optimizer,
            scheduler,
            final_epoch,
            final_metrics,
            experiment_config,
        )
    finally:
        if ema is not None:
            ema.restore(model)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if run is not None:
        wandb.finish()
    return history
