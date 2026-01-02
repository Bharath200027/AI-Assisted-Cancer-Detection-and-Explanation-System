from __future__ import annotations

import argparse
from pathlib import Path
import shutil

IMAGE_EXTS = {".png",".jpg",".jpeg",".webp"}

def copy_split(src_root: Path, dst_root: Path, classes: list[str]):
    for split in ["train","val","test"]:
        src_split = src_root / split
        if not src_split.exists():
            continue
        for cls in classes:
            sdir = src_split / cls
            if not sdir.exists():
                continue
            ddir = dst_root / split / cls
            ddir.mkdir(parents=True, exist_ok=True)
            for p in sdir.rglob("*"):
                if p.suffix.lower() in IMAGE_EXTS:
                    shutil.copy2(p, ddir / p.name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True, help="Source dataset dir (ImageFolder: train/val/test/<class>)")
    ap.add_argument("--dst", type=str, required=True, help="Destination root to create family datasets")
    ap.add_argument("--lymphoid", type=str, default="all,cll", help="Comma-separated lymphoid classes")
    ap.add_argument("--myeloid", type=str, default="aml,cml", help="Comma-separated myeloid classes")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    lymph = [x.strip() for x in args.lymphoid.split(",") if x.strip()]
    myel = [x.strip() for x in args.myeloid.split(",") if x.strip()]

    lymph_dst = dst / "subtype_lymphoid"
    myel_dst = dst / "subtype_myeloid"
    lymph_dst.mkdir(parents=True, exist_ok=True)
    myel_dst.mkdir(parents=True, exist_ok=True)

    copy_split(src, lymph_dst, lymph)
    copy_split(src, myel_dst, myel)

    print(f"Created family datasets:")
    print(f"- {lymph_dst} (classes={lymph})")
    print(f"- {myel_dst} (classes={myel})")

if __name__ == "__main__":
    main()
