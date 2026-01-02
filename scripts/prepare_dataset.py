from __future__ import annotations
import argparse
from pathlib import Path
import random
import shutil

def _collect_images(in_dir: Path):
    exts = {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    return [p for p in in_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True, help="Input directory containing labeled folders or exported splits")
    ap.add_argument("--out_dir", type=str, required=True, help="Output ImageFolder: {train,val,test}/{class}/")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--class_map", type=str, default="", help="Optional mapping like '0:normal,1:leukemia'")
    args = ap.parse_args()

    random.seed(args.seed)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_map = {}
    if args.class_map.strip():
        for item in args.class_map.split(","):
            k,v = item.split(":")
            class_map[k.strip()] = v.strip()

    # Support two input styles:
    # 1) ImageFolder: in_dir/{class}/img.png
    # 2) Split folders: in_dir/{train,val,test}/{class}/img.png (HF export)
    has_splits = any((in_dir / s).exists() for s in ["train","validation","val","test"])

    def norm_label(lbl: str) -> str:
        return class_map.get(lbl, lbl)

    if has_splits:
        split_alias = {"validation":"val"}
        for s in ["train","val","validation","test"]:
            sdir = in_dir / s
            if not sdir.exists():
                continue
            out_split = split_alias.get(s, s)
            for class_dir in [p for p in sdir.iterdir() if p.is_dir()]:
                lbl = norm_label(class_dir.name)
                dest = out_dir / out_split / lbl
                dest.mkdir(parents=True, exist_ok=True)
                for img in _collect_images(class_dir):
                    shutil.copy2(img, dest / img.name)
        print(f"Prepared dataset with existing splits at: {out_dir}")
        return

    # If no explicit splits: do random split per class
    for class_dir in [p for p in in_dir.iterdir() if p.is_dir()]:
        imgs = _collect_images(class_dir)
        if not imgs:
            continue
        random.shuffle(imgs)
        n = len(imgs)
        n_test = int(n * args.test_frac)
        n_val = int(n * args.val_frac)
        test_imgs = imgs[:n_test]
        val_imgs = imgs[n_test:n_test+n_val]
        train_imgs = imgs[n_test+n_val:]

        lbl = norm_label(class_dir.name)
        for split, split_imgs in [("train",train_imgs),("val",val_imgs),("test",test_imgs)]:
            dest = out_dir / split / lbl
            dest.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                shutil.copy2(img, dest / img.name)

    print(f"Prepared dataset at: {out_dir}")

if __name__ == "__main__":
    main()
