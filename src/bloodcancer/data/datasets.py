from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List
import random

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from bloodcancer.vision.preprocess import preprocess_pil

IMG_EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp"}

@dataclass
class DataSpec:
    data_dir: Path
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4

class ImageFolderDataset(Dataset):
    """Lightweight ImageFolder replacement (no torchvision)."""
    def __init__(self, root: Path, img_size: int, train: bool):
        self.root = Path(root)
        self.img_size = img_size
        self.train = train

        # Discover classes by subfolder names
        classes = [p.name for p in self.root.iterdir() if p.is_dir()]
        classes.sort()
        self.classes = classes
        self.class_to_idx = {c:i for i,c in enumerate(classes)}

        self.items: List[tuple[str,int]] = []
        for c in classes:
            for fp in (self.root/c).rglob("*"):
                if fp.is_file() and fp.suffix.lower() in IMG_EXTS:
                    self.items.append((str(fp), self.class_to_idx[c]))

        random.shuffle(self.items)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        img = Image.open(path)
        x = preprocess_pil(img, img_size=self.img_size, train=self.train)
        return x, int(y)

def build_dataloaders(spec: DataSpec) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Expected directory layout:
    data_dir/
      train/<class>/*.png
      val/<class>/*.png
      test/<class>/*.png
    """
    tr_root = spec.data_dir / "train"
    va_root = spec.data_dir / "val"
    te_root = spec.data_dir / "test"

    train_ds = ImageFolderDataset(tr_root, img_size=spec.img_size, train=True)
    classes = train_ds.classes

    val_ds = ImageFolderDataset(va_root, img_size=spec.img_size, train=False) if va_root.exists() else None
    test_ds = ImageFolderDataset(te_root, img_size=spec.img_size, train=False) if te_root.exists() else None

    train_dl = DataLoader(train_ds, batch_size=spec.batch_size, shuffle=True, num_workers=spec.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=spec.batch_size, shuffle=False, num_workers=spec.num_workers, pin_memory=True) if val_ds else None
    test_dl = DataLoader(test_ds, batch_size=spec.batch_size, shuffle=False, num_workers=spec.num_workers, pin_memory=True) if test_ds else None
    return train_dl, val_dl, test_dl, classes
