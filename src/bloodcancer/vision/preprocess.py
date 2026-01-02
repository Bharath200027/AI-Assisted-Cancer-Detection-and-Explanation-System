from __future__ import annotations
from typing import Tuple
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_pil(img: Image.Image, img_size: int = 224, train: bool = False) -> torch.Tensor:
    """Convert PIL image to normalized CHW float tensor (no torchvision).

    - Resize to (img_size, img_size)
    - Optional lightweight augmentation when train=True
    - Normalize with ImageNet mean/std
    """
    img = img.convert("RGB")
    img = img.resize((img_size, img_size))

    if train:
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # small rotation
        if random.random() < 0.3:
            angle = random.uniform(-10, 10)
            img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0,0,0))
        # brightness/contrast jitter
        if random.random() < 0.3:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.3:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t

def preprocess_path(path: str, img_size: int = 224, train: bool = False) -> torch.Tensor:
    img = Image.open(path)
    return preprocess_pil(img, img_size=img_size, train=train)

def batchify(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 3:
        return t.unsqueeze(0)
    return t
