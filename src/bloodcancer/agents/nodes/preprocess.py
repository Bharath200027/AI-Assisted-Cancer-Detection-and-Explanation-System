# Deprecated: node logic moved to src/bloodcancer/agents/agents.py and src/bloodcancer/agents/nodes.py
from __future__ import annotations
from pathlib import Path
from PIL import Image
import numpy as np

def preprocess_node(state: dict, cfg) -> dict:
    # Basic QC checks: loadable, not tiny, not blank-ish
    image_path = state["image_path"]
    try:
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)
        h,w = arr.shape[:2]
        if h < 64 or w < 64:
            return {"errors": state.get("errors", []) + [f"Image too small: {w}x{h}"]}
        if arr.std() < 2.0:
            return {"errors": state.get("errors", []) + ["Image appears nearly uniform (low contrast)."]}
        return {"errors": state.get("errors", [])}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Failed to load image: {e}"]}
