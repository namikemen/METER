from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

import architecture as arch_module
from lejepa_globals import LeJEPAConfig


def add_repo_to_path(repo_root: Path) -> None:
    repo_root = repo_root.expanduser().resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def format_log(message: object) -> str:
    if isinstance(message, (dict, list, tuple)):
        try:
            return json.dumps(message, indent=2 if isinstance(message, dict) else None)
        except TypeError:
            return str(message)
    return str(message)


def emit_log(message: object, config: LeJEPAConfig) -> None:
    if config.use_tqdm:
        return
    print(format_log(message))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def patch_conv_device(device: torch.device) -> None:
    def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
        return nn.Sequential(
            arch_module.SeparableConv2d(
                in_channels=inp,
                out_channels=oup,
                kernel_size=kernal_size,
                stride=stride,
                bias=False,
                device=str(device),
            ),
            nn.BatchNorm2d(oup),
            nn.ReLU(),
        )

    arch_module.conv_nxn_bn = conv_nxn_bn


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def init_wandb(config: LeJEPAConfig, log_fn=print):
    if not config.use_wandb:
        return None, None
    try:
        import wandb
    except Exception:
        log_fn("wandb is not installed; continuing without W&B logging")
        return None, None
    run = wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        mode=config.wandb_mode,
        config=asdict(config),
        reinit=True,
    )
    return wandb, run


def is_wandb_run_active(run: object | None) -> bool:
    return run is not None and not bool(getattr(run, "_is_finished", False))


def safe_wandb_log(wandb_module: object | None, wandb_run: object | None, metrics: dict[str, object], step: int, log_fn=print) -> None:
    if wandb_module is None or not is_wandb_run_active(wandb_run):
        return
    try:
        wandb_module.log(metrics, step=step)
    except Exception as exc:
        log_fn(f"W&B log skipped: {exc}")


def safe_wandb_upload_checkpoint(
    *,
    wandb_module: object | None,
    wandb_run: object | None,
    config: LeJEPAConfig,
    checkpoint_path: Path,
    epoch: int | str,
    log_fn=print,
) -> None:
    if not config.wandb_upload_checkpoints or wandb_module is None or not is_wandb_run_active(wandb_run):
        return
    if epoch != "final" and int(epoch) % config.checkpoint_every_epochs != 0:
        return
    if epoch == "final" and config.epochs % config.checkpoint_every_epochs != 0:
        return
    try:
        artifact = wandb_module.Artifact(
            name=f"meter-lejepa-{config.model_size}-encoder",
            type="model",
            metadata={"model_size": config.model_size, "epoch": str(epoch)},
        )
        artifact.add_file(str(checkpoint_path))
        aliases = ["latest", "final"] if epoch == "final" else ["latest", f"epoch-{int(epoch):03d}"]
        wandb_run.log_artifact(artifact, aliases=aliases)
    except Exception as exc:
        log_fn(f"W&B checkpoint upload skipped: {exc}")


def safe_wandb_finish(
    *,
    wandb_module: object | None,
    wandb_run: object | None,
    config: LeJEPAConfig,
    final_checkpoint: Path,
    history_path: Path,
    log_fn=print,
) -> None:
    if wandb_module is None or not is_wandb_run_active(wandb_run):
        return
    try:
        wandb_run.summary.update(
            {
                "final_checkpoint": str(final_checkpoint),
                "history": str(history_path),
            }
        )
        safe_wandb_upload_checkpoint(
            wandb_module=wandb_module,
            wandb_run=wandb_run,
            config=config,
            checkpoint_path=final_checkpoint,
            epoch="final",
            log_fn=log_fn,
        )
        if config.wandb_upload_checkpoints:
            artifact = wandb_module.Artifact(
                name=f"meter-lejepa-{config.model_size}-history",
                type="dataset",
                metadata={"model_size": config.model_size},
            )
            artifact.add_file(str(history_path))
            wandb_run.log_artifact(artifact, aliases=["latest"])
    except Exception as exc:
        log_fn(f"W&B finish skipped: {exc}")
    finally:
        try:
            wandb_run.finish()
        except Exception:
            pass
