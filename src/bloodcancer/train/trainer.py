from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
import json

import torch
from torch import nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from bloodcancer.common.metrics import compute_metrics

@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    amp: bool = True

def _run_epoch(model: nn.Module, dl, device: str, optimizer=None, scaler=None, class_names=None):
    is_train = optimizer is not None
    model.train(is_train)
    loss_fn = nn.CrossEntropyLoss()

    all_y, all_prob = [], []
    total_loss = 0.0
    n = 0

    pbar = tqdm(dl, leave=False)
    for xb, yb in pbar:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(scaler is not None)):
            logits = model(xb)
            loss = loss_fn(logits, yb)

        if is_train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * xb.size(0)
        n += xb.size(0)

        prob = torch.softmax(logits.detach(), dim=1).cpu().numpy()
        all_prob.append(prob)
        all_y.append(yb.detach().cpu().numpy())

    all_prob = __import__("numpy").concatenate(all_prob, axis=0)
    all_y = __import__("numpy").concatenate(all_y, axis=0)
    metrics = compute_metrics(all_y, all_prob, class_names=class_names)
    metrics["loss"] = total_loss / max(1, n)
    return metrics

def train(model: nn.Module, train_dl, val_dl, device: str, out_dir: Path, cfg: TrainConfig, class_names=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = GradScaler(enabled=(cfg.amp and device.startswith("cuda")))

    best_metric = -1.0
    history = []

    model.to(device)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_metrics = _run_epoch(model, train_dl, device, optimizer=optimizer, scaler=scaler, class_names=class_names)
        val_metrics = _run_epoch(model, val_dl, device, class_names=class_names) if val_dl else {}

        # choose a stable criterion: accuracy
        key = "accuracy"
        score = float(val_metrics.get(key, train_metrics.get(key, 0.0)))

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

        (models_dir / "last.pt").write_bytes(torch.save(ckpt, models_dir / "last.pt") or b"")
        # torch.save returns None; above trick doesn't work reliably on all FS; do proper save:
        torch.save(ckpt, models_dir / "last.pt")

        if score > best_metric:
            best_metric = score
            torch.save(ckpt, models_dir / "best.pt")

        rec = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "seconds": round(time.time() - t0, 2),
            "best_accuracy": best_metric,
        }
        history.append(rec)
        print(json.dumps(rec, indent=2))

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history
