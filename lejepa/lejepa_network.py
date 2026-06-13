from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from architecture import mobilevit_s, mobilevit_xs, mobilevit_xxs
from lejepa_globals import ENCODER_CHANNELS, LeJEPAConfig
from lejepa_utils import emit_log


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17) -> None:
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        samples = proj.flatten(0, -2)
        directions = torch.randn(samples.size(-1), 256, device=samples.device, dtype=samples.dtype)
        directions = directions / directions.norm(p=2, dim=0).clamp_min(1e-6)
        x_t = (samples @ directions).unsqueeze(-1) * self.t.to(dtype=samples.dtype)
        err = (x_t.cos().mean(dim=0) - self.phi.to(dtype=samples.dtype)).square()
        err = err + x_t.sin().mean(dim=0).square()
        statistic = (err @ self.weights.to(dtype=samples.dtype)) * samples.size(0)
        return statistic.mean()


def build_mobilevit_encoder(model_size: str) -> tuple[nn.Module, str]:
    builders = {"xxs": mobilevit_xxs, "xs": mobilevit_xs, "s": mobilevit_s}
    if model_size not in builders:
        raise ValueError(f"Unsupported model size: {model_size}")
    return builders[model_size]()


class METERLeJEPAEncoder(nn.Module):
    def __init__(self, model_size: str, proj_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.model_size = model_size
        self.encoder, self.enc_type = build_mobilevit_encoder(model_size)
        input_dim = ENCODER_CHANNELS[model_size]
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, view_count = views.shape[:2]
        features, _ = self.encoder(views.flatten(0, 1))
        tokens = features.flatten(2).transpose(1, 2).contiguous()
        embeddings = tokens.mean(dim=1)
        projections = self.proj(embeddings)
        return embeddings.reshape(batch_size, view_count, -1), projections.reshape(batch_size, view_count, -1)


def unwrap_model(model: nn.Module) -> METERLeJEPAEncoder:
    return model.module if isinstance(model, nn.DataParallel) else model


def load_resume_checkpoint(
    *,
    config: LeJEPAConfig,
    device: torch.device,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
) -> tuple[int, int]:
    if config.resume_checkpoint is None:
        return 1, 0
    checkpoint_path = Path(config.resume_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    checkpoint_config = checkpoint.get("config", {})
    checkpoint_model_size = checkpoint.get("model_size") or checkpoint_config.get("model_size")
    if checkpoint_model_size is not None and checkpoint_model_size != config.model_size:
        raise ValueError(
            f"Checkpoint model_size={checkpoint_model_size!r} does not match current model_size={config.model_size!r}"
        )
    encoder_model = unwrap_model(model)
    encoder_model.encoder.load_state_dict(checkpoint["encoder"], strict=True)
    loaded = ["encoder"]
    can_load_full_state = config.resume_full_state and "model" in checkpoint
    if can_load_full_state:
        encoder_model.load_state_dict(checkpoint["model"], strict=True)
        loaded.append("model")
    if config.resume_full_state and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        loaded.append("optimizer")
    if config.resume_full_state and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
        loaded.append("scheduler")
    if config.resume_full_state and "scaler" in checkpoint and config.use_amp and device.type == "cuda":
        scaler.load_state_dict(checkpoint["scaler"])
        loaded.append("scaler")
    resume_epoch = int(checkpoint.get("epoch", 0))
    resume_step = int(checkpoint.get("global_step", 0))
    start_epoch = resume_epoch + 1 if config.resume_full_state else 1
    emit_log(
        {
            "resume_checkpoint": str(checkpoint_path),
            "loaded": loaded,
            "checkpoint_epoch": resume_epoch,
            "start_epoch": start_epoch,
            "global_step": resume_step if config.resume_full_state else 0,
            "model_size": config.model_size,
        },
        config,
    )
    return start_epoch, resume_step if config.resume_full_state else 0


def build_checkpoint_payload(
    epoch: int,
    logs: dict[str, float],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: LeJEPAConfig,
    global_step: int,
) -> dict[str, object]:
    encoder_model = unwrap_model(model)
    return {
        "arch_type": encoder_model.enc_type,
        "model_size": config.model_size,
        "enc_type": encoder_model.enc_type,
        "encoder_channels": ENCODER_CHANNELS[config.model_size],
        "encoder": encoder_model.encoder.state_dict(),
        "model": encoder_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": asdict(config),
        "logs": logs,
    }


def save_encoder_checkpoint(
    *,
    epoch: int,
    epoch_logs: dict[str, float],
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: LeJEPAConfig,
    global_step: int,
) -> Path:
    checkpoint_path = checkpoint_dir / f"encoder_epoch_{epoch:03d}.pt"
    torch.save(
        build_checkpoint_payload(epoch, epoch_logs, model, optimizer, scheduler, scaler, config, global_step),
        checkpoint_path,
    )
    return checkpoint_path
