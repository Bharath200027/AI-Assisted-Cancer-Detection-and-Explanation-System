from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except Exception as e:
    GradCAM = None
    show_cam_on_image = None

def _auto_target_layer(model: torch.nn.Module):
    # Heuristic: last module with weight shape resembling conv features
    # For timm models, often 'conv_head' or 'features' exist.
    candidates = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            candidates.append((name, m))
    return candidates[-1][1] if candidates else None

def _overlay_cam_on_image(img_np: np.ndarray, cam: np.ndarray) -> Image.Image:
    """Overlay a [H,W] CAM on an RGB image array in [0,1]."""
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam_uint8 = (cam * 255).astype(np.uint8)

    # simple heatmap using OpenCV if available; otherwise grayscale overlay
    try:
        import cv2
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = (0.55 * img_np + 0.45 * heatmap)
        overlay = np.clip(overlay, 0, 1)
        return Image.fromarray((overlay * 255).astype(np.uint8))
    except Exception:
        # fallback: blend with grayscale cam
        heat = np.stack([cam, cam, cam], axis=-1)
        overlay = (0.7 * img_np + 0.3 * heat)
        overlay = np.clip(overlay, 0, 1)
        return Image.fromarray((overlay * 255).astype(np.uint8))

def _simple_gradcam(model: torch.nn.Module, x: torch.Tensor, class_idx: int, target_layer: torch.nn.Module) -> np.ndarray:
    """Minimal Grad-CAM implementation for CNN-like models (torchvision-free, grad-cam-free)."""
    activations = None
    gradients = None

    def fwd_hook(_, __, out):
        nonlocal activations
        activations = out

    def bwd_hook(_, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad(set_to_none=True)
    logits = model(x)
    score = logits[:, class_idx].sum()
    score.backward(retain_graph=False)

    h1.remove()
    h2.remove()

    if activations is None or gradients is None:
        raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

    # weights: global average pooling of gradients
    w = gradients.mean(dim=(2,3), keepdim=True)
    cam = torch.relu((w * activations).sum(dim=1, keepdim=False))
    cam = cam.detach().cpu().numpy()[0]
    return cam

def gradcam_explain(
    model: torch.nn.Module,
    pil_img: Image.Image,
    class_idx: int,
    out_path: str,
    device: str = "cpu",
    img_size: int = 224,
    target_layer=None
) -> str:
    """Generate a Grad-CAM-like overlay.

    Priority:
    1) Use external package `grad-cam` (import name `pytorch_grad_cam`) if available.
    2) Otherwise, use a minimal internal Grad-CAM for CNN conv layers.
    3) If no conv layer exists (e.g., pure ViT), save the resized input image and return a note.

    Always writes an *image file* to out_path so the UI never breaks.
    """
    model.eval()

    if target_layer is None:
        target_layer = _auto_target_layer(model)

    img = pil_img.convert("RGB").resize((img_size, img_size))
    img_np = np.array(img).astype(np.float32) / 255.0

    # preprocess: ImageNet normalize
    x = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0)
    x = (x - torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)) / torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    x = x.to(device)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # 1) External library path
    if GradCAM is not None and show_cam_on_image is not None and target_layer is not None:
        try:
            cam = GradCAM(model=model, target_layers=[target_layer])
            grayscale_cam = cam(input_tensor=x, targets=None)[0, :]
            out_img = _overlay_cam_on_image(img_np, grayscale_cam)
            out_img.save(out_path)
            q = float(np.quantile(grayscale_cam, 0.90))
            focus = float((grayscale_cam > q).mean())
            return f"Grad-CAM generated (grad-cam). Top-10% activation area fraction: {focus:.3f}."
        except Exception:
            pass

    # 2) Internal fallback for CNNs
    if target_layer is not None:
        try:
            cam = _simple_gradcam(model, x, class_idx, target_layer)
            out_img = _overlay_cam_on_image(img_np, cam)
            out_img.save(out_path)
            q = float(np.quantile(cam, 0.90))
            focus = float((cam > q).mean())
            return f"Grad-CAM generated (internal fallback). Top-10% activation area fraction: {focus:.3f}."
        except Exception:
            pass

    # 3) No conv layer (e.g., ViT): save image only
    Image.fromarray((img_np * 255).astype(np.uint8)).save(out_path)
    return "Grad-CAM unavailable for this architecture (no conv layer). Saved resized image instead."
