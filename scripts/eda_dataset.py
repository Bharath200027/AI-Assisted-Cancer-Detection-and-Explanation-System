from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import random

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

IMAGE_EXTS = {".png",".jpg",".jpeg",".webp"}

def iter_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]

def image_stats(paths: List[Path], max_n: int = 2000) -> Dict[str, float]:
    sel = paths[:max_n]
    sizes = []
    means = []
    stds = []
    sharp = []
    for p in sel:
        try:
            img = Image.open(p).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            h,w,_ = arr.shape
            sizes.append((h,w))
            means.append(arr.mean())
            stds.append(arr.std())
            # simple sharpness proxy: variance of Laplacian
            gray = arr.mean(axis=2)
            lap = np.abs(np.gradient(gray)[0]) + np.abs(np.gradient(gray)[1])
            sharp.append(float(lap.var()))
        except Exception:
            continue
    if not sizes:
        return {}
    hs = np.array([s[0] for s in sizes], dtype=np.float32)
    ws = np.array([s[1] for s in sizes], dtype=np.float32)
    return {
        "n": float(len(sizes)),
        "h_mean": float(hs.mean()),
        "w_mean": float(ws.mean()),
        "h_p10": float(np.percentile(hs, 10)),
        "h_p90": float(np.percentile(hs, 90)),
        "w_p10": float(np.percentile(ws, 10)),
        "w_p90": float(np.percentile(ws, 90)),
        "brightness_mean": float(np.mean(means) if means else 0.0),
        "contrast_mean": float(np.mean(stds) if stds else 0.0),
        "sharpness_mean": float(np.mean(sharp) if sharp else 0.0),
    }

def class_counts(split_dir: Path) -> Dict[str, int]:
    counts = {}
    for cls in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        counts[cls.name] = len([x for x in cls.rglob("*") if x.suffix.lower() in IMAGE_EXTS])
    return counts

def save_bar(counts: Dict[str,int], out: Path, title: str):
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(counts.keys())
    vals = [counts[k] for k in keys]
    plt.figure()
    plt.bar(keys, vals)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def save_montage(paths: List[Path], out: Path, n: int = 16, title: str = "Samples"):
    out.parent.mkdir(parents=True, exist_ok=True)
    if not paths:
        return
    sel = random.sample(paths, min(n, len(paths)))
    cols = 4
    rows = int(np.ceil(len(sel)/cols))
    plt.figure(figsize=(10, 2.5*rows))
    for i,p in enumerate(sel):
        plt.subplot(rows, cols, i+1)
        try:
            plt.imshow(Image.open(p).convert("RGB"))
        except Exception:
            pass
        plt.axis("off")
        plt.title(p.parent.name)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Processed dataset root (expects train/val/test)")
    ap.add_argument("--out_dir", type=str, default="artifacts/eda")
    ap.add_argument("--max_images", type=int, default=2000)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md = []
    md.append(f"# Dataset EDA Report\n\n**Dataset:** `{data_dir}`\n")

    for split in ["train","val","test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        md.append(f"## Split: {split}\n")
        counts = class_counts(split_dir)
        total = sum(counts.values())
        md.append(f"- Total images: **{total}**\n")
        for k,v in counts.items():
            md.append(f"  - {k}: {v}\n")

        save_bar(counts, out_dir / f"{split}_class_counts.png", f"{split} class counts")

        paths = iter_images(split_dir)
        st = image_stats(paths, max_n=args.max_images)
        if st:
            md.append("\n**Image statistics (sampled):**\n")
            for k,v in st.items():
                md.append(f"- {k}: {v:.4f}" if isinstance(v,float) else f"- {k}: {v}")
            md.append("\n")
        save_montage(paths, out_dir / f"{split}_montage.png", n=16, title=f"{split} samples")

    # Save markdown
    report = out_dir / "report.md"
    report.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved EDA to {out_dir}")
    print(f"- {report}")

if __name__ == "__main__":
    main()
