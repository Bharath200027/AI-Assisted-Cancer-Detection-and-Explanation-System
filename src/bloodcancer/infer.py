from __future__ import annotations
import argparse
from pathlib import Path
import json

import torch
from PIL import Image

from bloodcancer.vision.preprocess import preprocess_pil, batchify
from bloodcancer.models.modeling import create_classifier, load_checkpoint
from bloodcancer.explain.gradcam import gradcam_explain

def preprocess(img: Image.Image, img_size: int) -> torch.Tensor:
    return batchify(preprocess_pil(img, img_size=img_size, train=False))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--image_path", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="tf_efficientnetv2_s")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--class_names", type=str, default="normal,leukemia")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = [x.strip() for x in args.class_names.split(",") if x.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_classifier(args.model_name, num_classes=len(class_names), pretrained=False)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.to(device).eval()

    img = Image.open(args.image_path).convert("RGB")
    xb = preprocess(img, args.img_size).to(device)

    with torch.no_grad():
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()

    pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    pred_label = class_names[pred_idx]
    conf = float(probs[pred_idx])

    # Explain (Grad-CAM)
    heatmap_path = str(out_dir / "explanations" / "gradcam_overlay.png")
    Path(heatmap_path).parent.mkdir(parents=True, exist_ok=True)
    explain_summary = gradcam_explain(model, img, pred_idx, heatmap_path, device=device, img_size=args.img_size)

    result = {
        "image": args.image_path,
        "predicted_label": pred_label,
        "confidence": conf,
        "probs": {class_names[i]: float(p) for i, p in enumerate(probs)},
        "heatmap_path": heatmap_path,
        "explain_summary": explain_summary,
    }
    (out_dir / "prediction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
