import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import architecture
from architecture import build_METER_model
from meter_data import DataConfig, NYUH5DepthDataset, discover_h5_files


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


def normalize_rgb_for_display(rgb: np.ndarray) -> np.ndarray:
    out = np.empty_like(rgb, dtype=np.float32)
    for channel in range(rgb.shape[-1]):
        plane = rgb[..., channel]
        low = float(plane.min())
        high = float(plane.max())
        out[..., channel] = 0.0 if high <= low else (plane - low) / (high - low)
    return np.clip(out, 0.0, 1.0)


def load_checkpoint_state_dict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    return {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in checkpoint.items()
    }


def load_model(path: Path, device: torch.device, arch_type: str) -> torch.nn.Module:
    model = build_METER_model(device="cpu", arch_type=arch_type)
    state_dict = load_checkpoint_state_dict(path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"{path.name}: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"{path.name} first missing: {missing[:5]}")
    if unexpected:
        print(f"{path.name} first unexpected: {unexpected[:5]}")
    model = model.to(device)
    model.eval()
    return model


def predict_depth(model: torch.nn.Module, image: torch.Tensor, depth_shape: tuple[int, int], device: torch.device) -> np.ndarray:
    pred = model(image.unsqueeze(0).to(device))
    if pred.shape[-2:] != depth_shape:
        pred = F.interpolate(pred, size=depth_shape, mode="bilinear", align_corners=False)
    pred = pred.clamp(1.0, 1000.0)
    return pred[0, 0].detach().cpu().numpy()


def parse_args():
    worktree_data = REPO_ROOT / ".claude/worktrees/xenodochial-goldberg-ac26bb/nyu_depth_v2"
    parser = argparse.ArgumentParser(description="Compare two METER S checkpoints on resized NYU val samples")
    parser.add_argument("--checkpoint-a", type=Path, default=REPO_ROOT / "outputs/meter_checkpoints_60/S-100.pt")
    parser.add_argument("--checkpoint-b", type=Path, default=REPO_ROOT / "models/build_model_best_nyu_s")
    parser.add_argument("--label-a", default="S-100")
    parser.add_argument("--label-b", default="best_nyu_s")
    parser.add_argument("--arch-type", default="s", choices=("xxs", "xs", "s"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=worktree_data if worktree_data.exists() else REPO_ROOT / "nyu_depth_v2",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/visualizations/S-100_vs_best_nyu_s.png")
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_a = args.checkpoint_a.expanduser().resolve()
    checkpoint_b = args.checkpoint_b.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    out_path = args.out.expanduser().resolve()

    print("checkpoint_a:", checkpoint_a)
    print("checkpoint_b:", checkpoint_b)
    print("arch_type:", args.arch_type)
    print("device:", device)
    print("data_root:", data_root)

    config = DataConfig(input_height=192, input_width=256, max_depth_cm=1000.0, invalid_depth_threshold_cm=1.0)
    files = discover_h5_files(data_root, "val", limit=args.limit)
    dataset = NYUH5DepthDataset(files, config, train=False)

    model_a = load_model(checkpoint_a, device, args.arch_type)
    model_b = load_model(checkpoint_b, device, args.arch_type)

    rows = len(dataset)
    fig, axes = plt.subplots(rows, 6, figsize=(24, 3.5 * rows), squeeze=False)

    with torch.no_grad():
        for idx in range(rows):
            item = dataset[idx]
            image = item["image"]
            depth_np = item["depth"][0].numpy()
            mask_np = item["mask"][0].numpy().astype(bool)
            pred_a = predict_depth(model_a, image, depth_np.shape, device)
            pred_b = predict_depth(model_b, image, depth_np.shape, device)
            err_a = np.abs(pred_a - depth_np)
            err_b = np.abs(pred_b - depth_np)

            image_np = normalize_rgb_for_display(image.permute(1, 2, 0).numpy())
            panels = [
                image_np,
                np.where(mask_np, depth_np, np.nan),
                np.where(mask_np, pred_a, np.nan),
                np.where(mask_np, err_a, np.nan),
                np.where(mask_np, pred_b, np.nan),
                np.where(mask_np, err_b, np.nan),
            ]
            titles = [
                f"RGB\n{Path(item['path']).name}",
                "GT depth cm",
                f"{args.label_a} pred cm",
                f"{args.label_a} abs err cm",
                f"{args.label_b} pred cm",
                f"{args.label_b} abs err cm",
            ]
            cmaps = [None, "plasma_r", "plasma_r", "magma", "plasma_r", "magma"]
            for col, (panel, title, cmap) in enumerate(zip(panels, titles, cmaps)):
                ax = axes[idx, col]
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
