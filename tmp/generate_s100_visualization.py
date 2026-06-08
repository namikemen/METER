from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import architecture
from meter_data import DataConfig, NYUH5DepthDataset, discover_h5_files, resolve_data_root
from scripts.visualization import _visualize_depth_prediction

CHECKPOINT = ROOT / "outputs/meter_checkpoints_60/S-100.pt"
OUT_DIR = ROOT / "outputs/visualizations"
OUT_PATH = OUT_DIR / "S-100_depth_prediction.png"


def conv_nxn_bn_local(inp, oup, kernal_size=3, stride=1):
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


def extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def resolve_local_data_root():
    candidates = [
        "../nyu_depth_v2",
        ".claude/worktrees/xenodochial-goldberg-ac26bb/nyu_depth_v2",
        "nyu_depth_v2",
    ]
    for candidate in candidates:
        try:
            return resolve_data_root("nyu-depth-v2", str(ROOT / candidate))
        except FileNotFoundError:
            continue
    raise FileNotFoundError("Could not find NYU validation data in known local locations")


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    architecture.conv_nxn_bn = conv_nxn_bn_local
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = architecture.build_METER_model(
        device="cpu",
        arch_type="s",
        dropout=0.0,
        attention_dropout=0.0,
        drop_path=0.0,
    )

    try:
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("missing sample:", missing[:10])
    if unexpected:
        print("unexpected sample:", unexpected[:10])

    model = model.to(device)
    data_root = resolve_local_data_root()
    files = discover_h5_files(data_root, "val", limit=1)
    dataset = NYUH5DepthDataset(
        files,
        DataConfig(input_height=192, input_width=256, resize_inputs=True, normalize_rgb=True),
        train=False,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))

    source_path = batch["path"][0]
    fig = _visualize_depth_prediction(
        model,
        batch["image"][0],
        batch["depth"][0],
        device,
        title=f"METER S-100: {Path(source_path).name}",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"saved={OUT_PATH}")
    print(f"source={source_path}")
    print(f"device={device}")


if __name__ == "__main__":
    main()
