from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from lejepa_data import build_data_loaders, build_smoke_loaders
from lejepa_evaluation import maybe_log_epoch_media, visualize_pca_embeddings
from lejepa_globals import LeJEPAConfig
from lejepa_network import (
    METERLeJEPAEncoder,
    SIGReg,
    load_resume_checkpoint,
    save_encoder_checkpoint,
)
from lejepa_utils import (
    emit_log,
    init_wandb,
    patch_conv_device,
    resolve_device,
    safe_wandb_finish,
    safe_wandb_log,
    safe_wandb_upload_checkpoint,
    seed_everything,
)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    config: LeJEPAConfig,
) -> torch.optim.lr_scheduler.LRScheduler:
    total_steps = max(1, steps_per_epoch * config.epochs)
    warmup_steps = max(1, min(total_steps // 10, steps_per_epoch))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-6, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
        min_ratio = config.min_learning_rate / config.learning_rate
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def prediction_loss(global_proj: torch.Tensor, local_proj: torch.Tensor) -> torch.Tensor:
    if global_proj.shape[1] < 2:
        global_loss = torch.zeros((), device=global_proj.device, dtype=global_proj.dtype)
    else:
        global_target = global_proj.mean(dim=1, keepdim=True).detach()
        global_loss = F.smooth_l1_loss(global_proj, global_target.expand_as(global_proj), beta=0.5)
    local_target = global_proj.mean(dim=1, keepdim=True).detach()
    local_loss = F.smooth_l1_loss(local_proj, local_target.expand_as(local_proj), beta=0.5)
    return global_loss + local_loss


def projection_diagnostics(proj: torch.Tensor) -> dict[str, torch.Tensor]:
    flat = proj.flatten(0, -2)
    centered = flat - flat.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, flat.shape[0] - 1)
    std = flat.std(dim=0).mean()
    off_diag = covariance - torch.diag(torch.diag(covariance))
    return {
        "projection_std": std,
        "projection_cov_offdiag": off_diag.abs().mean(),
    }


def lejepa_loss(
    global_proj: torch.Tensor,
    local_proj: torch.Tensor,
    sigreg: SIGReg,
    config: LeJEPAConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction = prediction_loss(global_proj, local_proj)
    all_proj = torch.cat([global_proj, local_proj], dim=1)
    regularization = sigreg(all_proj)
    loss = prediction + config.lambda_sigreg * regularization
    diagnostics = projection_diagnostics(all_proj)
    return loss, {
        "loss": loss.detach(),
        "prediction": prediction.detach(),
        "sigreg": regularization.detach(),
        **diagnostics,
    }


def tensor_logs_to_float(logs: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in logs.items()}


def view_difference(views: torch.Tensor) -> torch.Tensor:
    if views.shape[1] < 2:
        return torch.zeros((), device=views.device, dtype=views.dtype)
    return (views[:, 0] - views[:, 1]).abs().mean()


def _make_output_dirs(config: LeJEPAConfig) -> tuple[Path, Path, Path]:
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    pca_dir = output_dir / "pca"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pca_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, checkpoint_dir, pca_dir


def run_training(config: LeJEPAConfig) -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"The epoch parameter in `scheduler\.step\(\)` was not necessary.*",
        category=UserWarning,
    )
    seed_everything(config.seed)
    device = resolve_device()
    patch_conv_device(device)
    output_dir, checkpoint_dir, pca_dir = _make_output_dirs(config)
    emit_log(
        {
            "device": str(device),
            "output_dir": str(output_dir),
            "model_size": config.model_size,
            "requested_gpus": config.gpus,
            "available_cuda_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        config,
    )

    train_loader, preview_loader = (
        build_smoke_loaders(config, device) if config.smoke_test else build_data_loaders(config, device)
    )
    model: nn.Module = METERLeJEPAEncoder(config.model_size, config.proj_dim, config.hidden_dim).to(device)
    if config.gpus == 2 and torch.cuda.device_count() >= 2:
        model = nn.DataParallel(model, device_ids=[0, 1])
        emit_log({"data_parallel": True, "device_ids": [0, 1]}, config)
    elif config.gpus == 2:
        emit_log("Requested 2 GPUs but fewer than 2 CUDA devices are available; using a single device", config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = build_lr_scheduler(optimizer, len(train_loader), config)
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")
    sigreg = SIGReg(config.sigreg_knots).to(device)
    start_epoch, global_step = load_resume_checkpoint(
        config=config,
        device=device,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    wandb_module, wandb_run = init_wandb(config, log_fn=lambda message: emit_log(message, config))
    history: list[dict[str, float]] = []

    model.train()
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start = time.time()
        totals: dict[str, float] = {
            "loss": 0.0,
            "prediction": 0.0,
            "sigreg": 0.0,
            "projection_std": 0.0,
            "projection_cov_offdiag": 0.0,
            "global_view_difference": 0.0,
            "local_view_difference": 0.0,
            "lr": 0.0,
            "data_seconds": 0.0,
            "train_seconds": 0.0,
        }
        batches = 0
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False) if config.use_tqdm else train_loader
        next_data_start = time.time()
        for batch in iterator:
            data_seconds = time.time() - next_data_start
            global_step += 1
            train_start = time.time()
            global_views = batch["global_views"].to(device, non_blocking=True)
            local_views = batch["local_views"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=config.use_amp and device.type == "cuda"):
                all_views = torch.cat([global_views, local_views], dim=1)
                _, all_projections = model(all_views)
                global_projections, local_projections = all_projections.split(
                    [global_views.shape[1], local_views.shape[1]],
                    dim=1,
                )
                loss, tensor_logs = lejepa_loss(global_projections, local_projections, sigreg, config)
                tensor_logs["global_view_difference"] = view_difference(global_views).detach()
                tensor_logs["local_view_difference"] = view_difference(local_views).detach()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_seconds = time.time() - train_start
            for key, value in tensor_logs.items():
                totals[key] = totals.get(key, 0.0) + float(value.cpu())
            totals["lr"] += scheduler.get_last_lr()[0]
            totals["data_seconds"] += data_seconds
            totals["train_seconds"] += train_seconds
            batches += 1
            should_log_step = global_step == 1 or global_step % config.log_every_steps == 0
            if should_log_step:
                step_logs = tensor_logs_to_float(tensor_logs)
                step_logs.update(
                    {
                        "epoch": float(epoch),
                        "batch": float(batches),
                        "data_seconds": data_seconds,
                        "train_seconds": train_seconds,
                        "samples_per_second": global_views.shape[0] / max(data_seconds + train_seconds, 1e-6),
                        "lr": scheduler.get_last_lr()[0],
                        "embedding_dim": float(global_projections.shape[-1]),
                    }
                )
                if config.use_tqdm:
                    iterator.set_postfix(
                        loss=f"{step_logs['loss']:.4f}",
                        pred=f"{step_logs['prediction']:.4f}",
                        sigreg=f"{step_logs['sigreg']:.4f}",
                        step=global_step,
                    )
                else:
                    emit_log({**{f"step/{key}": value for key, value in step_logs.items()}, "step": global_step}, config)
                safe_wandb_log(
                    wandb_module,
                    wandb_run,
                    {f"step/{key}": value for key, value in step_logs.items()},
                    step=global_step,
                    log_fn=lambda message: emit_log(message, config),
                )
            del global_views, local_views, all_views, all_projections, global_projections, local_projections, loss, tensor_logs
            next_data_start = time.time()

        epoch_logs = {key: value / max(batches, 1) for key, value in totals.items()}
        epoch_logs = {**epoch_logs, "epoch": float(epoch), "seconds": time.time() - epoch_start}
        history.append(epoch_logs)
        emit_log(epoch_logs, config)

        should_visualize = epoch == 1 or epoch % config.pca_every_epochs == 0 or epoch == config.epochs
        checkpoint_path = save_encoder_checkpoint(
            epoch=epoch,
            epoch_logs=epoch_logs,
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            global_step=global_step,
        )
        safe_wandb_upload_checkpoint(
            wandb_module=wandb_module,
            wandb_run=wandb_run,
            config=config,
            checkpoint_path=checkpoint_path,
            epoch=epoch,
            log_fn=lambda message: emit_log(message, config),
        )
        wandb_logs: dict[str, object] = dict(epoch_logs)
        wandb_logs["checkpoint_path"] = str(checkpoint_path)
        if should_visualize:
            pca_path = visualize_pca_embeddings(
                epoch=epoch,
                preview_loader=preview_loader,
                model=model,
                device=device,
                pca_dir=pca_dir,
                config=config,
            )
            maybe_log_epoch_media(
                wandb_module=wandb_module,
                pca_path=pca_path,
                checkpoint_path=checkpoint_path,
                wandb_logs=wandb_logs,
            )
        safe_wandb_log(
            wandb_module,
            wandb_run,
            {f"epoch/{key}": value for key, value in wandb_logs.items()},
            step=global_step,
            log_fn=lambda message: emit_log(message, config),
        )

    history_path = output_dir / "history.json"
    final_checkpoint_path = checkpoint_dir / "encoder_final.pt"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    safe_wandb_finish(
        wandb_module=wandb_module,
        wandb_run=wandb_run,
        config=config,
        final_checkpoint=final_checkpoint_path,
        history_path=history_path,
        log_fn=lambda message: emit_log(message, config),
    )
    emit_log({"final_checkpoint": str(final_checkpoint_path), "history": str(history_path)}, config)
