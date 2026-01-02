from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from bloodcancer.data.datasets import DataSpec, build_dataloaders
from bloodcancer.models.modeling import create_classifier
from bloodcancer.train.trainer import TrainConfig, train as train_loop


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a blood cancer classifier (ImageFolder format).")
    ap.add_argument("--data_dir", type=str, required=True, help="Processed dataset root containing train/val/test folders.")
    ap.add_argument("--model_name", type=str, default="tf_efficientnetv2_s")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision (CUDA only).")
    ap.add_argument("--no_pretrained", action="store_true", help="Disable pretrained weights.")
    args = ap.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    spec = DataSpec(
        data_dir=data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_dl, val_dl, _test_dl, classes = build_dataloaders(spec)
    if len(classes) < 2:
        raise ValueError(f"Expected >=2 classes under {data_dir/'train'}; got: {classes}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_classifier(
        args.model_name,
        num_classes=len(classes),
        pretrained=(not args.no_pretrained),
    )

    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amp=bool(args.amp and device.startswith("cuda")),
    )

    train_loop(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        device=device,
        out_dir=out_dir,
        cfg=cfg,
        class_names=classes,
    )


if __name__ == "__main__":
    main()
