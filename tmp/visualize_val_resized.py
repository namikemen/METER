import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import architecture
from architecture import build_METER_model
from meter_data import DataConfig, NYUH5DepthDataset, discover_h5_files

# Local mlx PyTorch is CPU-only; the original architecture hardcodes cuda:0 in conv_nxn_bn.
def _cpu_conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return torch.nn.Sequential(
        architecture.SeparableConv2d(
            in_channels=inp,
            out_channels=oup,
            kernel_size=kernal_size,
            stride=stride,
            bias=False,
            device='cpu',
        ),
        torch.nn.BatchNorm2d(oup),
        torch.nn.ReLU(),
    )

architecture.conv_nxn_bn = _cpu_conv_nxn_bn


def normalize_rgb_for_display(rgb: np.ndarray) -> np.ndarray:
    """Per-image min-max over HW per channel for matplotlib display."""
    out = np.empty_like(rgb, dtype=np.float32)
    for channel in range(rgb.shape[-1]):
        plane = rgb[..., channel]
        low = float(plane.min())
        high = float(plane.max())
        if high <= low:
            out[..., channel] = 0.0
        else:
            out[..., channel] = (plane - low) / (high - low)
    return np.clip(out, 0.0, 1.0)


WORKTREE_DATA = (
    REPO_ROOT
    / '.claude/worktrees/xenodochial-goldberg-ac26bb/nyu_depth_v2'
)


def parse_args():
    parser = argparse.ArgumentParser(description='NYU val depth inference visualization')
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=REPO_ROOT / 'models/build_model_best_nyu_xxs',
        help='Model checkpoint path',
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        default=WORKTREE_DATA if WORKTREE_DATA.exists() else REPO_ROOT / 'nyu_depth_v2',
        help='NYU root with train/ and val/ folders',
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=None,
        help='Output PNG path (default: outputs/<checkpoint_stem>_val_resized_10.png)',
    )
    parser.add_argument('--limit', type=int, default=10)
    return parser.parse_args()


args = parse_args()
checkpoint_path = args.checkpoint.expanduser().resolve()
data_root = args.data_root.expanduser().resolve()
out_path = (
    args.out.expanduser().resolve()
    if args.out is not None
    else (REPO_ROOT / 'outputs' / f'{checkpoint_path.stem}_val_resized_{args.limit}.png').resolve()
)

config = DataConfig(
    input_height=192,
    input_width=256,
    max_depth_cm=1000.0,
    invalid_depth_threshold_cm=1.0,
    random_crop=0.5,
)

files = discover_h5_files(data_root, 'val', limit=args.limit)
dataset = NYUH5DepthDataset(files, config, train=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
state_dict = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in state_dict.items()}
arch_type = (
    checkpoint.get('config', {}).get('arch_type', 'xxs')
    if isinstance(checkpoint, dict) and 'config' in checkpoint
    else 'xxs'
)
print('checkpoint:', checkpoint_path)
print('arch_type:', arch_type)
print('device:', device)
print('val files:', len(files))

model = build_METER_model(device=str(device), arch_type=arch_type).to(device)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print('missing:', len(missing), 'unexpected:', len(unexpected))
if missing:
    print('first missing:', missing[:5])
if unexpected:
    print('first unexpected:', unexpected[:5])
model.eval()

rows = len(dataset)
fig, axes = plt.subplots(rows, 4, figsize=(16, 3.5 * rows), squeeze=False)

with torch.no_grad():
    for idx in range(rows):
        item = dataset[idx]
        image = item['image'].unsqueeze(0).to(device)
        depth = item['depth'].unsqueeze(0).to(device)
        pred = model(image)
        if pred.shape[-2:] != depth.shape[-2:]:
            pred = F.interpolate(pred, size=depth.shape[-2:], mode='bilinear', align_corners=False)
        pred = pred.clamp(1.0, config.max_depth_cm)

        image_np = normalize_rgb_for_display(item['image'].permute(1, 2, 0).numpy())
        depth_np = item['depth'][0].numpy()
        mask_np = item['mask'][0].numpy().astype(bool)
        pred_np = pred[0, 0].detach().cpu().numpy()
        err_np = np.abs(pred_np - depth_np)

        panels = [
            image_np,
            np.where(mask_np, depth_np, np.nan),
            np.where(mask_np, pred_np, np.nan),
            np.where(mask_np, err_np, np.nan),
        ]
        titles = [
            f'RGB\n{Path(item["path"]).name}',
            'GT depth cm',
            'Pred depth cm',
            'Abs error cm',
        ]
        cmaps = [None, 'plasma_r', 'plasma_r', 'magma']
        for col, (panel, title, cmap) in enumerate(zip(panels, titles, cmaps)):
            ax = axes[idx, col]
            im = ax.imshow(panel, cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.axis('off')
            if col > 0:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

fig.tight_layout()
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=160, bbox_inches='tight')
plt.close(fig)
print('saved:', out_path)
