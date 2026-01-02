# Deprecated: node logic moved to src/bloodcancer/agents/agents.py and src/bloodcancer/agents/nodes.py
from __future__ import annotations
from pathlib import Path

def triage_node(state: dict, cfg) -> dict:
    image_path = state.get("image_path")
    if not image_path:
        return {"errors": ["image_path missing"]}
    p = Path(image_path)
    return {
        "modality": cfg.raw["app"].get("modality", "blood_smear"),
        "image_filename": p.name,
        "errors": state.get("errors", []),
    }
