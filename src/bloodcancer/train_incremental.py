from __future__ import annotations
import argparse
from pathlib import Path
import json
import os

import torch
from torch.utils.data import ConcatDataset, DataLoader
from bloodcancer.data.datasets import ImageFolderDataset
from bloodcancer.models.modeling import create_classifier
from bloodcancer.train.trainer import TrainConfig, train as train_loop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_data_dir", type=str, required=True, help="data/processed/cnmc")
    ap.add_argument("--feedback_dir", type=str, default="data/feedback", help="data/feedback/<class>/imgs")
    ap.add_argument("--model_name", type=str, default="tf_efficientnetv2_s")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    base_dir = Path(args.base_data_dir)
    fb_dir = Path(args.feedback_dir)

    train_dir = base_dir / "train"
    val_dir = base_dir / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Missing base train dir: {train_dir}")

    # Build base datasets
    base_train = ImageFolderDataset(str(train_dir), transform=build_transforms(args.img_size, train=True))
    classes = base_train.classes

    # Feedback dataset must match class folder names
    fb_train = None
    if fb_dir.exists():
        # Ensure same class folders exist
        if any((fb_dir / c).exists() for c in classes):
            fb_train = ImageFolderDataset(str(fb_dir), transform=build_transforms(args.img_size, train=True))
            # ImageFolder sorts classes; if mismatch, warn
            if fb_train.classes != classes:
                print("WARNING: feedback classes differ from base classes.")
                print("Base:", classes)
                print("Feedback:", fb_train.classes)

    merged_train = base_train if fb_train is None else ConcatDataset([base_train, fb_train])

    val_ds = ImageFolderDataset(str(val_dir), transform=build_transforms(args.img_size, train=False)) if val_dir.exists() else None

    train_dl = DataLoader(merged_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) if val_ds else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_classifier(args.model_name, num_classes=len(classes), pretrained=True)

    cfg = TrainConfig(epochs=args.epochs, lr=args.lr)
    hist = train_loop(model, train_dl, val_dl, device, Path(args.out_dir), cfg)

    # Write last run meta
    meta = {
        "base_data_dir": str(base_dir),
        "feedback_dir": str(fb_dir),
        "classes": classes,
        "epochs": args.epochs,
        "model_name": args.model_name,
        "device": device,
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "last_incremental_run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved:", Path(args.out_dir) / "last_incremental_run.json")

if __name__ == "__main__":
    main()
