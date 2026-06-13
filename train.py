from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.std import tqdm

from evaluation import compute_metrics, make_visual_figure
from utils import log_ram, ramp_value, schedule_scale, unwrap_model

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None



@dataclass(frozen=True)
class TrainConfig:
    num_epochs: int = 60
    learning_rate: float = 1e-3
    encoder_learning_rate: float = 1e-4
    decoder_learning_rate: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.03
    dropout_start: float = 0.0
    dropout: float = 0.0
    attention_dropout_start: float = 0.0
    attention_dropout: float = 0.0
    drop_path_start: float = 0.0
    drop_path: float = 0.0
    regularization_start_epoch: int = 31
    regularization_ramp_epochs: int = 10
    regularization_ramp_schedule: str = "linear"
    scheduled_augmentation_start_epoch: int = 31
    scheduled_augmentation_ramp_epochs: int = 10
    scheduled_augmentation_ramp_schedule: str = "linear"
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


def log_ram(prefix: str, step: int, every: int = 10) -> None:
    if step % every != 0:
        return
    ram_gb = _process_tree_ram_gb()
    if ram_gb is not None:
        tqdm.write(f"{prefix} step={step} cpu_ram_tree={ram_gb:.2f} GB")


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def set_encoder_trainable(model: torch.nn.Module, trainable: bool) -> None:
    module = unwrap_model(model)
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


def schedule_scale(epoch: int, start_epoch: int, ramp_epochs: int, schedule: str = "linear") -> float:
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 1:
        return 1.0
    progress = min(max((epoch - start_epoch + 1) / ramp_epochs, 0.0), 1.0)
    if schedule == "linear":
        return progress
    if schedule == "smooth":
        return progress * progress * (3.0 - 2.0 * progress)
    raise ValueError(f"Unsupported schedule: {schedule}")


def ramp_value(start_value: float, max_value: float, scale: float) -> float:
    if scale <= 0.0:
        return 0.0
    return start_value + (max_value - start_value) * scale


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
        regularization_scale = schedule_scale(
            epoch,
            train_config.regularization_start_epoch,
            train_config.regularization_ramp_epochs,
            train_config.regularization_ramp_schedule,
        )
        augmentation_scale = schedule_scale(
            epoch,
            train_config.scheduled_augmentation_start_epoch,
            train_config.scheduled_augmentation_ramp_epochs,
            train_config.scheduled_augmentation_ramp_schedule,
        )
        regularization_active = regularization_scale > 0.0
        active_dropout = ramp_value(train_config.dropout_start, train_config.dropout, regularization_scale)
        active_attention_dropout = ramp_value(
            train_config.attention_dropout_start,
            train_config.attention_dropout,
            regularization_scale,
        )
        active_drop_path = ramp_value(train_config.drop_path_start, train_config.drop_path, regularization_scale)
        _set_dataset_epoch(train_loader, epoch)
        set_model_regularization_rates(
            model,
            dropout=active_dropout,
            attention_dropout=active_attention_dropout,
            drop_path=active_drop_path,
        )
        if regularization_active and ema is None and train_config.ema_decay > 0.0:
            ema = ExponentialMovingAverage(model, train_config.ema_decay)
        encoder_frozen = epoch <= train_config.freeze_encoder_epochs
        set_encoder_trainable(model, not encoder_frozen)
        if train_config.freeze_encoder_epochs > 0 and epoch == 1:
            print(f"[epoch {epoch}] encoder frozen for warmup through epoch {train_config.freeze_encoder_epochs}")
        if train_config.freeze_encoder_epochs > 0 and epoch == train_config.freeze_encoder_epochs + 1:
            print(f"[epoch {epoch}] encoder unfrozen; encoder parameters are trainable now")
        if epoch == train_config.regularization_start_epoch:
            print(f"[epoch {epoch}] model regularization ramp started")
        if epoch == train_config.scheduled_augmentation_start_epoch:
            print(f"[epoch {epoch}] strong augmentation ramp started")
        if regularization_scale >= 1.0 and epoch == train_config.regularization_start_epoch + train_config.regularization_ramp_epochs - 1:
            print(f"[epoch {epoch}] model regularization reached full strength")
        if augmentation_scale >= 1.0 and epoch == train_config.scheduled_augmentation_start_epoch + train_config.scheduled_augmentation_ramp_epochs - 1:
            print(f"[epoch {epoch}] strong augmentation reached full strength")
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
                "regularization_scale": regularization_scale,
                "augmentation_scale": augmentation_scale,
                "ema_enabled": float(ema is not None),
                "dropout_active": active_dropout,
                "attention_dropout_active": active_attention_dropout,
                "drop_path_active": active_drop_path,
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
                f"encoder_frozen={bool(encoder_frozen)} "
                f"reg_scale={regularization_scale:.3f} "
                f"dropout={active_dropout:.3f} "
                f"drop_path={active_drop_path:.3f}"
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
                    "regularization/scale": logs["regularization_scale"],
                    "regularization/augmentation_scale": logs["augmentation_scale"],
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
