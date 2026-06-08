# METER Server Setup Guide

This guide describes how to upload NYU Depth v2 data and run METER on a GPU server. The current repo defaults are NYU Depth v2 with depth values in centimeters.

## 1. Server requirements

Install these on the server:

- NVIDIA GPU driver
- Docker
- NVIDIA Container Toolkit
- `rsync`

Check GPU access:

```bash
nvidia-smi
```

Check Docker GPU support:

```bash
docker run --rm --gpus all nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04 nvidia-smi
```

## 2. Clone the repo on the server

```bash
mkdir -p ~/METER
cd ~/METER
git clone <repo-url> code
cd code
```

Expected server layout:

```text
~/METER/
├── code/              # this git repo
├── data/
│   └── nyu_depth_v2/  # uploaded dataset
├── outputs/           # logs and metrics
└── checkpoints/       # model weights
```

## 3. Upload NYU Depth v2 data

From your local machine, run:

```bash
scripts/upload_nyu_data.sh /local/path/to/nyu_depth_v2 user@gpu-server ~/METER
```

This uploads to:

```text
~/METER/data/nyu_depth_v2/
```

The script uses `rsync --partial --progress`, so interrupted uploads can resume.

Verify on the server:

```bash
ssh user@gpu-server
ls ~/METER/data/nyu_depth_v2
du -sh ~/METER/data/nyu_depth_v2
find ~/METER/data/nyu_depth_v2 -type f | wc -l
```

## 4. Confirm NYU/cm defaults

The repo defaults are in `globals.py`:

```python
RGB_img_res = (3, 192, 256)
dts_type = "nyu"
depth_unit = "cm"
max_depth = 1000.0
```

Augmentation probabilities are all `0.5`.

For NYU, depth shift augmentation is `[-10, +10] cm`.

## 5. Run inside Docker

From the server repo directory:

```bash
cd ~/METER/code
scripts/run_server_docker.sh -- python meter_train.py \
    --data-dir /workspace/data/nyu_depth_v2 \
    --output-dir /workspace/outputs/run_001 \
    --checkpoint-dir /workspace/checkpoints
```

The helper script mounts:

| Host path | Container path |
|---|---|
| repo | `/workspace/METER` |
| `~/METER/data` | `/workspace/data` |
| `~/METER/outputs` | `/workspace/outputs` |
| `~/METER/checkpoints` | `/workspace/checkpoints` |

Override paths if needed:

```bash
SERVER_ROOT=/scratch/$USER/METER \
IMAGE_NAME=meter:latest \
scripts/run_server_docker.sh -- python meter_train.py \
    --data-dir /workspace/data/nyu_depth_v2 \
    --output-dir /workspace/outputs/run_001 \
    --checkpoint-dir /workspace/checkpoints
```

## 6. Keep long runs alive

Use `tmux`:

```bash
tmux new -s meter
cd ~/METER/code
scripts/run_server_docker.sh -- python meter_train.py \
    --data-dir /workspace/data/nyu_depth_v2 \
    --output-dir /workspace/outputs/run_001 \
    --checkpoint-dir /workspace/checkpoints
```

Detach with:

```text
Ctrl-b d
```

Reattach later:

```bash
tmux attach -t meter
```

## 7. Save logs

If your training command does not write logs itself, pipe output with `tee`:

```bash
mkdir -p ~/METER/outputs/run_001
scripts/run_server_docker.sh -- bash -lc 'python meter_train.py \
    --data-dir /workspace/data/nyu_depth_v2 \
    --output-dir /workspace/outputs/run_001 \
    --checkpoint-dir /workspace/checkpoints 2>&1 | tee /workspace/outputs/run_001/train.log'
```

## 8. Before launching a full run

Run a short smoke test first:

- Load one NYU sample.
- Check RGB shape is compatible with `(3, 192, 256)`.
- Check depth is in centimeters.
- Run one forward pass.
- Save one checkpoint.

## 9. Do not commit data or outputs

Keep these out of git:

```text
data/
outputs/
checkpoints/
*.pt
*.pth
*.h5
```

Dataset files and model checkpoints should stay on the server storage, not in the repository.
