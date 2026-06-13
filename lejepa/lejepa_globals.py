from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import torch

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
ENCODER_CHANNELS = {"xxs": 160, "xs": 192, "s": 320}
LOCAL_CHANNELS = {"xxs": 64, "xs": 80, "s": 128}


@dataclass(frozen=True)
class LeJEPAConfig:
    dataset_slug: str = "/kaggle/input/datasets/artemmmtry/nyu-depth-v2"
    local_data_root: str = "data/nyu_depth_v2"
    output_dir: str = "/kaggle/working/lejepa"
    model_size: str = "xxs"
    train_limit: int | None = None
    preview_limit: int = 128
    epochs: int = 100
    batch_size: int = 64
    num_workers: int = 4
    global_views: int = 2
    local_views: int = 2
    global_crop_scale_min: float = 0.5
    local_crop_scale_min: float = 0.25
    local_crop_scale_max: float = 0.6
    local_loss_weight: float = 0.5
    learning_rate: float = 2e-3
    min_learning_rate: float = 1e-3
    weight_decay: float = 0.05
    lambda_sigreg: float = 0.02
    proj_dim: int = 128
    hidden_dim: int = 256
    crop_scale_min: float = 0.7
    disable_meter_crop: bool = True
    sigreg_knots: int = 17
    pca_every_epochs: int = 5
    checkpoint_every_epochs: int = 5
    log_every_steps: int = 5
    resume_checkpoint: str | None = None
    resume_full_state: bool = False
    seed: int = 42
    use_amp: bool = True
    use_tqdm: bool = True
    gpus: int = 1
    use_wandb: bool = True
    wandb_upload_checkpoints: bool = True
    wandb_project: str = "meter-lejepa"
    wandb_run_name: str = "lejepa-meter-xxs-nyu"
    wandb_mode: str = "online"
    smoke_test: bool = False


def parse_config() -> LeJEPAConfig:
    parser = argparse.ArgumentParser(add_help=__name__ == "__main__")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dataset-slug", default=LeJEPAConfig.dataset_slug)
    parser.add_argument("--local-data-root", default=LeJEPAConfig.local_data_root)
    parser.add_argument("--output-dir", default=LeJEPAConfig.output_dir)
    parser.add_argument("--model-size", choices=("xxs", "xs", "s"), default=LeJEPAConfig.model_size)
    parser.add_argument("--epochs", type=int, default=LeJEPAConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=LeJEPAConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=LeJEPAConfig.num_workers)
    parser.add_argument("--global-views", type=int, default=LeJEPAConfig.global_views)
    parser.add_argument("--local-views", type=int, default=LeJEPAConfig.local_views)
    parser.add_argument("--global-crop-scale-min", type=float, default=LeJEPAConfig.global_crop_scale_min)
    parser.add_argument("--local-crop-scale-min", type=float, default=LeJEPAConfig.local_crop_scale_min)
    parser.add_argument("--local-crop-scale-max", type=float, default=LeJEPAConfig.local_crop_scale_max)
    parser.add_argument("--local-loss-weight", type=float, default=LeJEPAConfig.local_loss_weight)
    parser.add_argument("--learning-rate", type=float, default=LeJEPAConfig.learning_rate)
    parser.add_argument("--min-learning-rate", type=float, default=LeJEPAConfig.min_learning_rate)
    parser.add_argument("--weight-decay", type=float, default=LeJEPAConfig.weight_decay)
    parser.add_argument("--crop-scale-min", type=float, default=LeJEPAConfig.crop_scale_min)
    parser.add_argument("--disable-meter-crop", dest="disable_meter_crop", action="store_true", default=None)
    parser.add_argument("--no-meter-crop", dest="disable_meter_crop", action="store_true")
    parser.add_argument("--enable-meter-crop", dest="disable_meter_crop", action="store_false")
    parser.add_argument("--train-limit", type=int, default=LeJEPAConfig.train_limit)
    parser.add_argument("--preview-limit", type=int, default=LeJEPAConfig.preview_limit)
    parser.add_argument("--gpus", type=int, choices=(1, 2), default=LeJEPAConfig.gpus)
    parser.add_argument("--log-every-steps", type=int, default=LeJEPAConfig.log_every_steps)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=LeJEPAConfig.checkpoint_every_epochs)
    parser.add_argument("--resume-checkpoint", default=LeJEPAConfig.resume_checkpoint)
    parser.add_argument("--encoder-only-resume", action="store_true")
    parser.add_argument("--use-tqdm", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-upload-checkpoints", action="store_true")
    parser.add_argument("--wandb-project", default=LeJEPAConfig.wandb_project)
    parser.add_argument("--wandb-run-name", default=LeJEPAConfig.wandb_run_name)
    parser.add_argument("--wandb-mode", default=LeJEPAConfig.wandb_mode)
    args, _ = parser.parse_known_args()
    config = LeJEPAConfig(
        dataset_slug=args.dataset_slug,
        local_data_root=args.local_data_root,
        output_dir=args.output_dir,
        model_size=args.model_size,
        train_limit=args.train_limit,
        preview_limit=args.preview_limit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        global_views=args.global_views,
        local_views=args.local_views,
        global_crop_scale_min=args.global_crop_scale_min,
        local_crop_scale_min=args.local_crop_scale_min,
        local_crop_scale_max=args.local_crop_scale_max,
        local_loss_weight=args.local_loss_weight,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        crop_scale_min=args.crop_scale_min,
        disable_meter_crop=LeJEPAConfig.disable_meter_crop if args.disable_meter_crop is None else args.disable_meter_crop,
        gpus=args.gpus,
        log_every_steps=args.log_every_steps,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        resume_checkpoint=args.resume_checkpoint or None,
        resume_full_state=not args.encoder_only_resume,
        use_tqdm=args.use_tqdm,
        use_wandb=not args.no_wandb,
        wandb_upload_checkpoints=args.wandb_upload_checkpoints,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        smoke_test=args.smoke_test,
    )
    if args.smoke_test:
        smoke_epochs = max(1, args.epochs if args.resume_checkpoint else 1)
        return replace(
            config,
            epochs=smoke_epochs,
            batch_size=1,
            num_workers=0,
            train_limit=2,
            preview_limit=2,
            global_views=1,
            local_views=1,
            use_amp=False,
            use_wandb=False,
        )
    return config
