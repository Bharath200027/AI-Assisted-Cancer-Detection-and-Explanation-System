from __future__ import annotations
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Output directory, e.g., data/raw/cnmc_hf")
    ap.add_argument("--dataset", type=str, default="dwb2023/cnmc-leukemia-2019")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from PIL import Image

    ds = load_dataset(args.dataset)

    export_dir = out_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Attempt common fields: image + label
    # We'll export into ImageFolder-like structure: export/{split}/{label}/img.png
    label_names = None
    if "train" in ds and hasattr(ds["train"].features.get("label", None), "names"):
        label_names = ds["train"].features["label"].names

    for split_name, split in ds.items():
        split_dir = export_dir / split_name
        for i, row in enumerate(split):
            img = row.get("image")
            label = row.get("label")
            if img is None:
                continue
            if label_names and isinstance(label, int) and 0 <= label < len(label_names):
                label_str = label_names[label]
            else:
                label_str = str(label)
            (split_dir / label_str).mkdir(parents=True, exist_ok=True)

            # datasets may return PIL.Image.Image already
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img.save(split_dir / label_str / f"{i:07d}.png")

    print(f"Exported HF dataset to: {export_dir}")
    print("Next: python scripts/prepare_dataset.py --in_dir <export_dir> --out_dir data/processed/cnmc")

if __name__ == "__main__":
    main()
