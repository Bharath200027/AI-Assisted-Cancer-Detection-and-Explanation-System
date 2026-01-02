from __future__ import annotations
import argparse
from pathlib import Path

from bloodcancer.data.datasets import DataSpec, build_dataloaders
from bloodcancer.models.modeling import create_classifier
from bloodcancer.train.trainer import TrainConfig, train

import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="data/processed/cnmc")
    ap.add_argument("--model_name", type=str, default="tf_efficientnetv2_s")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    train_dl, val_dl, test_dl, classes = build_dataloaders(DataSpec(data_dir=data_dir, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers))
    print(f"Classes: {classes}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_classifier(args.model_name, num_classes=len(classes), pretrained=True)

    cfg = TrainConfig(epochs=args.epochs, lr=args.lr)
    train(model, train_dl, val_dl, device, out_dir, cfg, class_names=classes)

if __name__ == "__main__":
    main()
