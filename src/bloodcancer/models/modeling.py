from __future__ import annotations
try:
    import timm
except ImportError:  # pragma: no cover
    timm = None
import torch
from torch import nn

def create_classifier(model_name: str, num_classes: int, pretrained: bool=True) -> nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model

def load_checkpoint(model: nn.Module, ckpt_path: str, map_location: str="cpu") -> dict:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=True)
        return ckpt
    # fallback: raw state dict
    model.load_state_dict(ckpt, strict=True)
    return {"model_state": ckpt}
