# Deprecated: node logic moved to src/bloodcancer/agents/agents.py and src/bloodcancer/agents/nodes.py
from __future__ import annotations
import os
from pathlib import Path

import torch
from PIL import Image
from bloodcancer.vision.preprocess import preprocess_pil, batchify
from bloodcancer.models.modeling import create_classifier, load_checkpoint

def _preprocess(img: Image.Image, img_size: int):
    return preprocess_pil(img, img_size=img_size, train=False)

def inference_node(state: dict, cfg) -> dict:
    if state.get("errors"):
        return {}
    ckpt = os.getenv("MODEL_CHECKPOINT", "artifacts/models/best.pt")
    model_name = os.getenv("MODEL_NAME", "tf_efficientnetv2_s")
    class_names = cfg.class_names
    device = cfg.device
    img_size = cfg.img_size

    if not Path(ckpt).exists():
        return {"errors": state.get("errors", []) + [f"Checkpoint not found: {ckpt}. Train first or set MODEL_CHECKPOINT."]}

    model = create_classifier(model_name, num_classes=len(class_names), pretrained=False)
    load_checkpoint(model, ckpt, map_location=device)
    model.to(device).eval()

    img = Image.open(state["image_path"]).convert("RGB")
    xb = _preprocess(img, img_size).to(device)

    with torch.no_grad():
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()

    pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    pred_label = class_names[pred_idx]
    conf = float(probs[pred_idx])

    return {
        "probs": {class_names[i]: float(p) for i,p in enumerate(probs)},
        "predicted_label": pred_label,
        "confidence": conf,
    }
