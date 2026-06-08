import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import architecture
from architecture import build_METER_model

# Local mlx PyTorch may be CPU-only; original architecture hardcodes cuda:0 in conv_nxn_bn.
def _cpu_conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return torch.nn.Sequential(
        architecture.SeparableConv2d(
            in_channels=inp,
            out_channels=oup,
            kernel_size=kernal_size,
            stride=stride,
            bias=False,
            device="cpu",
        ),
        torch.nn.BatchNorm2d(oup),
        torch.nn.ReLU(),
    )

architecture.conv_nxn_bn = _cpu_conv_nxn_bn

DEVICE = torch.device("cpu")
SIZE = (192, 256)  # H, W; paper NYU 256x192 W x H.
MAX_DEPTH_CM = 1000.0
NUM_SAMPLES = 10

WORKTREE_VAL = (
    REPO_ROOT
    / ".claude/worktrees/xenodochial-goldberg-ac26bb/nyu_depth_v2/val"
)
DEFAULT_DATA_ROOT = WORKTREE_VAL if WORKTREE_VAL.exists() else REPO_ROOT / "nyu_depth_v2" / "val"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _load_checkpoint(path: Path):
    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    state_dict = {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    return state_dict, config


def _load_model(path: Path, fallback_arch: str):
    state_dict, config = _load_checkpoint(path)
    arch_type = config.get("arch_type", fallback_arch)
    model = build_METER_model(device=str(DEVICE), arch_type=arch_type).to(DEVICE)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Loaded {path}")
    print(f"  arch_type={arch_type} missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  first missing:", missing[:8])
    if unexpected:
        print("  first unexpected:", unexpected[:8])
    return model, arch_type


def _discover_h5_files(root: Path, limit: int):
    files = sorted(root.glob("*.h5"))
    if not files:
        files = sorted(root.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 validation files found under {root}")
    return files[:limit]


def _load_h5_sample(path: Path):
    with h5py.File(path, "r") as h5:
        rgb = h5["rgb"][:].astype(np.float32) / 255.0
        depth_cm = h5["depth"][:].astype(np.float32) * 100.0

    rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb)).unsqueeze(0)
    depth_tensor = torch.from_numpy(np.ascontiguousarray(depth_cm[None, :, :])).unsqueeze(0)

    rgb_resized = F.interpolate(rgb_tensor, size=SIZE, mode="bilinear", align_corners=False)[0]
    depth_resized = F.interpolate(depth_tensor, size=SIZE, mode="nearest")[0]
    depth_resized = depth_resized.clamp(1.0, MAX_DEPTH_CM)
    mask = depth_resized > 1.0

    rgb_normalized = (rgb_resized - IMAGENET_MEAN) / IMAGENET_STD
    rgb_display = rgb_resized.permute(1, 2, 0).numpy().clip(0.0, 1.0)
    return rgb_normalized, rgb_display, depth_resized, mask


@torch.no_grad()
def _predict(model, image, target_shape):
    pred = model(image.unsqueeze(0).to(DEVICE))
    if pred.shape[-2:] != target_shape:
        pred = F.interpolate(pred, size=target_shape, mode="bilinear", align_corners=False)
    return pred[0].detach().cpu().clamp(1.0, MAX_DEPTH_CM)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two METER checkpoints on NYU val")
    parser.add_argument("--left", type=Path, default=REPO_ROOT / "outputs/meter_checkpoints_60/meter_60_40.pt")
    parser.add_argument("--right", type=Path, default=REPO_ROOT / "outputs/meter_checkpoints_60/meter-lejepa-60.pt")
    parser.add_argument("--left-label", type=str, default=None)
    parser.add_argument("--right-label", type=str, default=None)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/meter_checkpoints_60/meter_60_40_vs_meter_lejepa_60_val_10.png")
    parser.add_argument("--limit", type=int, default=NUM_SAMPLES)
    return parser.parse_args()


def main():
    args = parse_args()
    left_path = args.left.expanduser().resolve()
    right_path = args.right.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    out_path = args.out.expanduser().resolve()
    left_label = args.left_label or left_path.stem
    right_label = args.right_label or right_path.stem

    official_model, official_arch = _load_model(left_path, "xxs")
    trained_model, trained_arch = _load_model(right_path, "xxs")
    files = _discover_h5_files(data_root, args.limit)
    print(f"Validation files: {len(files)} from {data_root}")

    rows = len(files)
    fig, axes = plt.subplots(rows, 5, figsize=(20, 3.5 * rows), squeeze=False)
    for row, path in enumerate(files):
        image, rgb_display, gt_depth, mask = _load_h5_sample(path)
        pred_official = _predict(official_model, image, gt_depth.shape[-2:])
        pred_trained = _predict(trained_model, image, gt_depth.shape[-2:])
        err_trained = (pred_trained - gt_depth).abs()

        mask_np = mask[0].numpy().astype(bool)
        panels = [
            rgb_display,
            np.where(mask_np, gt_depth[0].numpy(), np.nan),
            np.where(mask_np, pred_official[0].numpy(), np.nan),
            np.where(mask_np, pred_trained[0].numpy(), np.nan),
            np.where(mask_np, err_trained[0].numpy(), np.nan),
        ]
        titles = [
            f"RGB\n{path.name}",
            "GT depth cm",
            f"{left_label} {official_arch}",
            f"{right_label} {trained_arch}",
            f"{right_label} abs error cm",
        ]
        cmaps = [None, "plasma_r", "plasma_r", "plasma_r", "magma"]
        for col, (panel, title, cmap) in enumerate(zip(panels, titles, cmaps)):
            ax = axes[row, col]
            im = ax.imshow(panel, cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            if col > 0:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
