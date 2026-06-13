from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.std import tqdm

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None


def add_code_path(code_root: str | Path) -> Path:
    path = Path(code_root).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Code path does not exist: {path}")

    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    return path


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def process_tree_ram_gb() -> float | None:
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
    ram_gb = process_tree_ram_gb()
    if ram_gb is not None:
        tqdm.write(f"{prefix} step={step} cpu_ram_tree={ram_gb:.2f} GB")
