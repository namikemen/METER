# METER + LeJEPA: Lightweight Monocular Depth Estimation with Self-Supervised Representation Learning

This project improves [METER](https://github.com/lorenzopapa5/METER), a lightweight MobileViT-based monocular depth estimation model, by incorporating **LeJEPA-style self-supervised representation learning**. The goal is to improve depth prediction quality while keeping the model compact, fast, and suitable for robotics applications.

## Project Structure

```
meter/
├── globals.py           # Shared constants & config defaults
├── utils.py             # Support utilities (seeding, RAM logging, schedule helpers)
├── data.py              # Dataset discovery, H5 loading, augmentation pipeline
├── network.py           # METER MobileViT encoder-decoder architecture
├── train.py             # Training loop, loss, optimizer, scheduler, EMA, checkpointing, W&B
├── evaluation.py        # Metrics (RMSE, AbsRel, δ1) and visualization helpers
├── notebooks/
│   └── meter_nyu_kaggle_reproduction.ipynb   # Kaggle launcher for METER training
│
├── lejepa/
│   ├── lejepa_globals.py       # LeJEPA config dataclass & parser
│   ├── lejepa_utils.py         # Seeding, device, W&B helpers, architecture patching
│   ├── lejepa_data.py          # Multi-crop view dataset & augmentation
│   ├── lejepa_network.py       # LeJEPA encoder + projector + SIGReg
│   ├── lejepa_train.py         # LeJEPA pretraining loop (prediction loss + SIGReg)
│   ├── lejepa_evaluation.py    # PCA probing, embedding visualization
│   └── lejepa_pretrain_meter_nyu.ipynb    # Kaggle launcher for LeJEPA pretraining
│
├── architecture.py      # Original METER architecture (imported by lejepa/ modules)
├── augmentation.py      # Original augmentation helpers
├── loss.py              # Original METER loss
├── models/              # Pretrained checkpoints
└── outputs/             # Training outputs, visualizations
```

## Pipeline Overview

### 1. METER Depth Estimation (6 files)

| Module | File | Purpose |
|--------|------|---------|
| Config | `globals.py` | Shared depth/image constants (`MAX_DEPTH_CM`, `INVALID_DEPTH_THRESHOLD_CM`, `RGB_img_res`) |
| Utilities | `utils.py` | `seed_everything`, `schedule_scale`, `ramp_value`, `unwrap_model`, RAM logging |
| Data | `data.py` | `DataConfig`, `NYUH5DepthDataset`, `discover_h5_files`, augmentation pipeline with progressive ramping |
| Model | `network.py` | `build_METER_model(device, arch_type, dropout, attention_dropout, drop_path)` — MobileViT encoder + conv decoder |
| Training | `train.py` | `TrainConfig`, `LossConfig`, `BalancedMETERLoss`, optimizer/scheduler, EMA, checkpointing, `fit()` |
| Evaluation | `evaluation.py` | `compute_metrics` (RMSE, AbsRel, δ1), `make_visual_figure`, image denormalization |

**Training losses**: L1 depth reconstruction + Sobel gradient matching + surface-normal cosine loss + SSIM loss.

### 2. LeJEPA Pretraining (6 files)

| Module | File | Purpose |
|--------|------|---------|
| Config | `lejepa_globals.py` | `LeJEPAConfig` dataclass with CLI arg parser |
| Utilities | `lejepa_utils.py` | Device resolution, W&B init/logging, architecture patching for CPU/MPS |
| Data | `lejepa_data.py` | Multi-crop view generation (2 global × 256×192 + 4 local views), augmentations |
| Model | `lejepa_network.py` | Shared MobileViT encoder + projection MLP + `SIGReg` module |
| Training | `lejepa_train.py` | Pretraining loop: prediction loss + SIGReg regularization, checkpointing, PCA probing |
| Evaluation | `lejepa_evaluation.py` | PCA-based zero-shot geometric probing of pretrained encoder features |

**Pretraining objective**: `L = λ · SIGReg(embeddings) + (1 − λ) · L_pred`

### 3. Workflow

1. **Pretrain** the encoder on unlabeled NYU images using LeJEPA (`lejepa/` pipeline).
2. **Transfer** the pretrained encoder weights into the METER depth estimation model.
3. **Fine-tune** the full model on RGB-depth pairs using the METER training pipeline.
4. **Evaluate** using RMSE, AbsRel, and δ1 metrics.

## Datasets

- **NYU Depth v2**: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html


## Key Configurations

### METER Training

| Parameter | Value |
|-----------|-------|
| Input resolution | 256 × 192 |
| Optimizer | AdamW (lr=1e-3, weight_decay=0.03) |
| LR schedule | Cosine annealing (2e-3 → 1e-4) |
| Epochs | 60 |
| Batch size | 128 |
| Max depth | 1000 cm |

### LeJEPA Pretraining

| Parameter | Value |
|-----------|-------|
| Views | 2 global (256×192) + 4 local |
| Prediction loss | L2 between global mean and each view embedding |
| SIGReg λ | 0.02 |
| Optimizer | AdamW (lr=2e-3, weight_decay=0.05) |
| Projector dim | 128 |

## Hardware

All training and evaluation are performed on Kaggle Tesla T4 GPUs.

## Pretrained Models

Available under `models/`:

- `build_model_best_nyu_{xxs,xs,s}` — Baseline METER on NYU
- `build_model_best_kitti_{xxs,xs,s}` — Baseline METER on KITTI
- `lemeter-kaggle-h5-full-none-e2e_best.pt` — METER + LeJEPA fine-tuned
- `xxs_encoder_timm_init.pt` — Pretrained encoder initialization

## Results

LeJEPA-pretrained METER achieves smoother depth predictions and lower RMSE compared to training from scratch, with the trade-off of approximately 5 hours of additional pretraining time.

## References

- METER: https://github.com/lorenzopapa5/METER
- MobileViT: https://arxiv.org/abs/2110.02178
- LeJEPA: Joint Embedding Predictive Architecture for self-supervised learning
- NYU Depth V2: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
