#!/usr/bin/env python3
"""
figures/gradcam_panel.py

Grad-CAM panel for the ResNet-50 branch (last conv stage, layer4), overlaid
on sample test images.

This is the one figure script that can't be driven by aggregate JSON alone —
Grad-CAM needs the actual model weights and pixels. It still only *consumes*
existing artifacts (read-only), same as analysis/common.py's
load_model_from_seed_dir(); it does not train or re-evaluate anything.

Three modes, tried in order:

  1. --explainability-json PATH (default: analysis/results/explainability/
     explainability_summary.json, if it exists). analysis/explainability.py
     already runs the real thing — Grad-CAM for ResNet-50, Attention Rollout
     for ViT — on a curated pool of confident-correct / misclassified /
     high-entropy examples per class, and saves per-example heatmap PNGs.
     This mode composes those pre-rendered PNGs into one manuscript-grid
     figure with true/pred/entropy/category captions, rather than
     recomputing anything. Preferred whenever it's available since the
     example selection is principled, not random, and it also covers ViT
     (attention rollout) alongside ResNet-50 (Grad-CAM).

  2. --checkpoint PATH --images-dir DIR   Self-contained real Grad-CAM (this
     script's own hook-based implementation) on a trained ResNet-50
     checkpoint and a directory of sample images, picked at random. Useful
     when explainability_summary.json doesn't exist yet, or to sanity-check
     a specific checkpoint directly.

  3. (neither available)   Synthetic placeholder panel — random blob
     heatmaps clearly watermarked MOCK.

For mode 2, the ResNet-50 architecture is reconstructed locally (matches
src/code.py::BrainTumorResNet50 — selective layer4+head fine-tuning,
IMAGENET1K_V2 backbone) so this script has no import-time dependency on
src/code.py; only the state_dict is consumed from the .pth file.

Usage:
    # Preferred — compose from analysis/explainability.py's real output:
    python figures/gradcam_panel.py --out-dir figures/output

    # Explicit real Grad-CAM from a specific checkpoint, bypassing mode 1:
    python figures/gradcam_panel.py --no-explainability-json \\
        --checkpoint "results/20260620_165427/resnet50/seed_42/best_resnet50_seed42.pth" \\
        --images-dir data/brain_tumor --n-samples 8 --out-dir figures/output

    # Mock panel, zero dependencies:
    python figures/gradcam_panel.py --no-explainability-json --out-dir figures/output
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CLASS_NAMES, MissingDataError, ensure_output_dir, set_style,
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _build_resnet50_head(num_classes: int = 4):
    """Reconstructs src/code.py::BrainTumorResNet50's architecture (backbone +
    LayerNorm/Dropout/Linear/GELU/Dropout/Linear head) so a saved state_dict
    loads cleanly, without importing src/code.py itself.
    """
    import torch.nn as nn
    from torchvision import models

    backbone = models.resnet50(weights=None)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return backbone


class _GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx: Optional[int] = None):
        import torch
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)[0, 0].cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, class_idx, torch.softmax(logits, dim=-1)[0, class_idx].item()


def _load_real_model_and_gradcam(checkpoint_path: str):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_resnet50_head(num_classes=len(CLASS_NAMES)).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    # src/code.py wraps the resnet in `self.backbone`; the state dict is prefixed accordingly.
    state_dict = {k.replace("backbone.", "", 1) if k.startswith("backbone.") else k: v
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    cam_engine = _GradCAM(model, model.layer4[-1])
    return model, cam_engine, device


def _sample_images(images_dir: str, n_samples: int) -> List[Tuple[Path, Optional[str]]]:
    root = Path(images_dir)
    samples: List[Tuple[Path, Optional[str]]] = []
    class_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()] if root.is_dir() else []
    if class_dirs:
        per_class = max(1, n_samples // max(1, len(class_dirs)))
        rng = random.Random(0)
        for d in class_dirs:
            files = [f for f in d.iterdir() if f.suffix.lower() in _IMG_EXTS]
            for f in rng.sample(files, min(per_class, len(files))):
                samples.append((f, d.name))
    else:
        files = [f for f in root.rglob("*") if f.suffix.lower() in _IMG_EXTS]
        samples = [(f, None) for f in files[:n_samples]]
    return samples[:n_samples]


def _preprocess(path: Path):
    import torch
    from PIL import Image
    from torchvision import transforms

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])
    img = Image.open(path).convert("RGB")
    x = tfm(img).unsqueeze(0)
    disp = np.array(img.resize((224, 224))) / 255.0
    return x, disp


def _overlay(disp: np.ndarray, cam: np.ndarray) -> np.ndarray:
    import cv2  # noqa: WPS433 -- optional dep, degrade gracefully below
    cam_resized = cv2.resize(cam, (disp.shape[1], disp.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = heatmap[:, :, ::-1] / 255.0
    return 0.55 * disp + 0.45 * heatmap


def _overlay_no_cv2(disp: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Fallback overlay (no OpenCV dependency): nearest-neighbor resize of the
    CAM via numpy repeat + matplotlib's 'jet' colormap.
    """
    h, w = disp.shape[:2]
    ch, cw = cam.shape
    ys = (np.arange(h) * ch / h).astype(int).clip(0, ch - 1)
    xs = (np.arange(w) * cw / w).astype(int).clip(0, cw - 1)
    cam_resized = cam[ys][:, xs]
    heatmap = plt.get_cmap("jet")(cam_resized)[..., :3]
    return 0.55 * disp + 0.45 * heatmap


def build_real_panel(checkpoint: str, images_dir: str, n_samples: int, out_dir: Path) -> Path:
    import torch

    model, cam_engine, device = _load_real_model_and_gradcam(checkpoint)
    samples = _sample_images(images_dir, n_samples)
    if not samples:
        raise FileNotFoundError(f"No images found under {images_dir}")

    set_style()
    n = len(samples)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, (path, true_class) in zip(axes, samples):
        x, disp = _preprocess(path)
        x = x.to(device)
        cam, pred_idx, pred_prob = cam_engine(x)
        try:
            overlay = _overlay(disp, cam)
        except ImportError:
            overlay = _overlay_no_cv2(disp, cam)

        ax.imshow(np.clip(overlay, 0, 1))
        ax.axis("off")
        pred_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)
        title = f"pred={pred_name} ({pred_prob:.2f})"
        if true_class:
            title = f"true={true_class}\n{title}"
        ax.set_title(title, fontsize=8)

    for ax in axes[len(samples):]:
        ax.axis("off")

    fig.suptitle(f"Grad-CAM — ResNet-50 (layer4), checkpoint: {Path(checkpoint).name}", y=1.02)
    fig.text(0.5, -0.01, "Checkpoint provenance not verified as a final multi-seed run — "
                          "treat as a pipeline check unless confirmed otherwise.",
              ha="center", fontsize=8, color="darkred")
    fig.tight_layout()

    out_path = out_dir / "gradcam_panel.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_mock_panel(out_dir: Path, n_samples: int = 8) -> Path:
    set_style()
    rng = np.random.default_rng(0)
    ncols = min(4, n_samples)
    nrows = (n_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, ax in enumerate(axes[:n_samples]):
        base = rng.uniform(0.05, 0.15, size=(224, 224))
        cx, cy = rng.integers(60, 164, size=2)
        yy, xx = np.mgrid[0:224, 0:224]
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 35 ** 2)))
        img = np.clip(base[..., None].repeat(3, axis=2), 0, 1)
        heat = plt.get_cmap("jet")(blob)[..., :3]
        overlay = 0.6 * img + 0.4 * heat
        ax.imshow(overlay)
        ax.axis("off")
        cls = CLASS_NAMES[i % len(CLASS_NAMES)]
        ax.set_title(f"pred={cls} (mock)", fontsize=8)

    for ax in axes[n_samples:]:
        ax.axis("off")

    fig.suptitle("Grad-CAM — ResNet-50 (MOCK synthetic panel, no checkpoint given)", y=1.02)
    stamp_provenance_fig(fig)
    fig.tight_layout()

    out_path = out_dir / "gradcam_panel.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def stamp_provenance_fig(fig) -> None:
    fig.text(0.5, 0.5, "MOCK FIXTURE DATA", transform=fig.transFigure,
              fontsize=28, color="red", alpha=0.15, ha="center", va="center",
              rotation=30, zorder=100)


def build_panel_from_explainability_json(
    summary_path: Path, models: List[str], out_dir: Path
) -> Path:
    """Compose a manuscript grid from analysis/explainability.py's real output:
    curated per-class (confident-correct / misclassified / high-entropy)
    examples, with their already-rendered Grad-CAM (ResNet-50) / Attention
    Rollout (ViT) heatmap PNGs. See analysis/results/explainability/
    explainability_summary.{json,md} for the source.
    """
    import json
    import matplotlib.image as mpimg

    with open(summary_path) as f:
        summary = json.load(f)

    heatmap_root = summary_path.parent
    is_smoke = False
    rows = []  # (model, class, example_dict, heatmap_path)
    for model in models:
        model_block = summary.get("models", {}).get(model)
        if not model_block or "error" in model_block:
            continue
        is_smoke = is_smoke or model_block.get("provenance", {}).get("is_smoke_test_run", False)
        for cls_block in model_block.get("selections", []):
            cls = cls_block["class"]
            for ex in cls_block.get("examples", []):
                stem = Path(ex["filepath"]).stem
                heatmap_path = heatmap_root / model / f"{model}_{cls}_{stem}.png"
                if heatmap_path.exists():
                    rows.append((model, cls, ex, heatmap_path))

    if not rows:
        raise MissingDataError(
            f"[PLACEHOLDER: awaiting analysis/results/explainability/*] — "
            f"{summary_path} loaded but no matching heatmap PNGs found under {heatmap_root}."
        )

    set_style()
    ncols = min(4, len(rows))
    nrows = (len(rows) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    method_label = {"resnet50": "Grad-CAM", "vit": "Attn. Rollout"}
    for ax, (model, cls, ex, heatmap_path) in zip(axes, rows):
        img = mpimg.imread(heatmap_path)
        ax.imshow(img)
        ax.axis("off")
        cats = "/".join(ex.get("categories", []))
        title = (f"{method_label.get(model, model)} — true={ex['true_class']}, "
                  f"pred={ex['pred_class']}\nH={ex['entropy']:.2f}  [{cats}]")
        ax.set_title(title, fontsize=7)

    for ax in axes[len(rows):]:
        ax.axis("off")

    fig.suptitle(f"Explainability panel — curated examples ({', '.join(method_label.get(m, m) for m in models)})", y=1.02)
    if is_smoke:
        fig.text(0.5, -0.01,
                  "SMOKE-TEST RUN — checkpoint is a 1-epoch pipeline check, not a final model. Not real explanations.",
                  ha="center", fontsize=8, color="darkred")
    fig.tight_layout()

    out_path = out_dir / "gradcam_panel.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explainability-json", type=str, default=None,
                         help="Path to analysis/results/explainability/explainability_summary.json "
                              "(default: that exact path, if it exists).")
    parser.add_argument("--no-explainability-json", action="store_true",
                         help="Skip mode 1 even if the default file exists.")
    parser.add_argument("--explainability-models", type=str, nargs="+",
                         default=["resnet50", "vit"], choices=["resnet50", "vit"])
    parser.add_argument("--checkpoint", type=str, default=None,
                         help="Mode 2: path to a best_resnet50_seed*.pth checkpoint.")
    parser.add_argument("--images-dir", type=str, default=None,
                         help="Mode 2: directory of class-subfoldered images, e.g. data/brain_tumor.")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = ensure_output_dir(args.out_dir)

    if not args.no_explainability_json:
        default_path = Path(__file__).resolve().parent.parent / "analysis" / "results" / "explainability" / "explainability_summary.json"
        summary_path = Path(args.explainability_json) if args.explainability_json else default_path
        if summary_path.exists():
            try:
                out_path = build_panel_from_explainability_json(summary_path, args.explainability_models, out_dir)
                print(f"Wrote {out_path} (composed from {summary_path})")
                return
            except Exception as exc:
                print(f"[figures] Composing from {summary_path} failed ({exc!r}) - trying mode 2.")
        else:
            print(f"[figures] {summary_path} not found - trying mode 2.")

    if args.checkpoint and args.images_dir:
        try:
            out_path = build_real_panel(args.checkpoint, args.images_dir, args.n_samples, out_dir)
            print(f"Wrote {out_path} (real Grad-CAM from {args.checkpoint})")
            return
        except Exception as exc:
            print(f"[figures] Real Grad-CAM failed ({exc!r}) - falling back to mock panel.")

    out_path = build_mock_panel(out_dir, args.n_samples)
    print(f"Wrote {out_path} (MOCK - pass --checkpoint and --images-dir for a real panel, "
          f"or provide analysis/results/explainability/explainability_summary.json)")


if __name__ == "__main__":
    main()
