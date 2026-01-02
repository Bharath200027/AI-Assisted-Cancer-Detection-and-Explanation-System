# Deprecated: node logic moved to src/bloodcancer/agents/agents.py and src/bloodcancer/agents/nodes.py
from __future__ import annotations
from pathlib import Path
import os
from PIL import Image

from bloodcancer.explain.gradcam import gradcam_explain
from bloodcancer.models.modeling import create_classifier, load_checkpoint

import torch

def explain_node(state: dict, cfg) -> dict:
    if state.get("errors"):
        return {}
    out_dir = Path(os.getenv("OUT_DIR", "artifacts"))
    out_path = out_dir / "explanations" / f"gradcam_{Path(state['image_path']).stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = os.getenv("MODEL_CHECKPOINT", "artifacts/models/best.pt")
    model_name = os.getenv("MODEL_NAME", "tf_efficientnetv2_s")
    device = cfg.device

    model = create_classifier(model_name, num_classes=len(cfg.class_names), pretrained=False)
    load_checkpoint(model, ckpt, map_location=device)
    model.to(device).eval()

    img = Image.open(state["image_path"]).convert("RGB")
    class_idx = cfg.class_names.index(state["predicted_label"])
    summary = gradcam_explain(model, img, class_idx, str(out_path), device=device, img_size=cfg.img_size)

    return {"heatmap_path": str(out_path), "explain_summary": summary}
