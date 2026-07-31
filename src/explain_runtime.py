#!/usr/bin/env python3
"""
src/explain_runtime.py

Runtime heatmap generation for the clinic application. Contract 4.

Two public functions, and nothing else the app should depend on:

    generate_heatmap(model, image_tensor, model_kind, target_class=None) -> np.ndarray
    overlay_heatmap(original_rgb, heatmap, alpha=0.4) -> np.ndarray

Design rules this file holds to:

1. **Numerically consistent with `analysis/explainability.py`.** The heatmap the
   clinician sees must be the same heatmap that was validated. The Grad-CAM math,
   the normalisation, and the up-sampling interpolation are all reproduced
   exactly. `tests/test_explain_runtime.py` proves it against the research code.

2. **CPU only, and fast.** The ResNet-50 Grad-CAM path runs the trunk under
   `torch.no_grad()` and only builds an autograd graph for the pooling + head,
   which is all the gradient w.r.t. the layer4 feature map actually needs. Same
   numbers, a fraction of the cost of a full backward pass.

3. **No heavy imports.** numpy, torch, PIL, cv2. No matplotlib, no sklearn, no
   pandas. The jet colour table is baked in as a 256x3 byte array that is checked
   against matplotlib's in the test suite.

4. **The model is not mutated.** Dropout / train-eval state is saved and restored
   around every call.

Read `analysis/results/explainability_clinical/EXPLAINABILITY_RESULTS.md` before
putting these heatmaps in front of a clinician. It measures how often they land
on the tumour, and the answer is not the same for both backbones.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

__all__ = [
    "generate_heatmap",
    "overlay_heatmap",
    "MODEL_KINDS",
    "VIT_HEAD_FUSION",
    "VIT_DISCARD_RATIO",
    "HEATMAP_CAVEAT",
]

MODEL_KINDS: Tuple[str, ...] = ("resnet50", "vit")

# Frozen to match analysis/explainability.py's call into
# xai_and_stats.generate_attention_heatmap. Changing either of these breaks
# consistency with the validated heatmaps.
VIT_HEAD_FUSION: str = "mean"
VIT_DISCARD_RATIO: float = 0.9

# One honest line for the app to show under the overlay. Session C: use this
# verbatim. See the clinical explainability report for the numbers behind it.
HEATMAP_CAVEAT: str = (
    "Shows where the model looked, not where the tumour is. "
    "Use it to catch obviously wrong calls, not to confirm right ones."
)


# =============================================================================
# Colour table (matplotlib "jet", baked to stay matplotlib-free at runtime)
# =============================================================================

_JET_LUT_B64 = (
    "AAB/AACEAACIAACNAACRAACWAACaAACfAACjAACoAACsAACxAAC2AAC6AAC/AADDAADIAADMAADRAADVAADaAADe"
    "AADjAADoAADsAADxAAD1AAD6AAD+AAD/AAD/AAD/AAD/AAT/AAj/AAz/ABD/ABT/ABj/ABz/ACD/ACT/ACj/ACz/"
    "ADD/ADT/ADj/ADz/AED/AET/AEj/AEz/AFD/AFT/AFj/AFz/AGD/AGT/AGj/AGz/AHD/AHT/AHj/AHz/AID/AIT/"
    "AIj/AIz/AJD/AJT/AJj/AJz/AKD/AKT/AKj/AKz/ALD/ALT/ALj/ALz/AMD/AMT/AMj/AMz/AND/ANT/ANj/ANz+"
    "AOD6AOT3Auj0BezxCPDtDPTqD/jnEvzkFf/hGP/dHP/aH//XIv/UJf/QKf/NLP/KL//HMv/DNv/AOf+9PP+6P/+3"
    "Qv+zRv+wSf+tTP+qT/+mU/+jVv+gWf+dXP+aX/+WY/+TZv+Qaf+NbP+JcP+Gc/+Ddv+Aef99fP95gP92g/9zhv9w"
    "if9sjf9pkP9mk/9jlv9fmv9cnf9ZoP9Wo/9Tpv9Pqv9Mrf9JsP9Gs/9Ct/8/uv88vf85wP82w/8yx/8vyv8szf8p"
    "0P8l1P8i1/8f2v8c3f8Y4P8V5P8S5/8P6v8M7f8I8fwF9PgC9/QA+vAA/u0A/+kA/+UA/+IA/94A/9oA/9cA/9MA"
    "/88A/8sA/8gA/8QA/8AA/70A/7kA/7UA/7EA/64A/6oA/6YA/6MA/58A/5sA/5gA/5QA/5AA/4wA/4kA/4UA/4EA"
    "/34A/3oA/3YA/3MA/28A/2sA/2cA/2QA/2AA/1wA/1kA/1UA/1EA/00A/0oA/0YA/0IA/z8A/zsA/zcA/zQA/zAA"
    "/ywA/ygA/yUA/yEA/x0A/xoA/xYA/hIA+g8A9QsA8QcA7AMA6AAA4wAA3gAA2gAA1QAA0QAAzAAAyAAAwwAAvwAA"
    "ugAAtgAAsQAArAAAqAAAowAAnwAAmgAAlgAAkQAAjQAAiAAAhAAAfwAA"
)

_JET_LUT: np.ndarray = np.frombuffer(base64.b64decode(_JET_LUT_B64), dtype=np.uint8).reshape(256, 3)


def _apply_jet(heatmap: np.ndarray) -> np.ndarray:
    """[H,W] float in [0,1] -> [H,W,3] uint8, identical to
    ``(matplotlib.cm.jet(heatmap)[:, :, :3] * 255).astype(np.uint8)``."""
    x = np.clip(np.asarray(heatmap, dtype=np.float64), 0.0, 1.0) * 256.0
    idx = np.clip(x.astype(np.int64), 0, 255)
    return _JET_LUT[idx]


# =============================================================================
# Small utilities
# =============================================================================

@contextmanager
def _frozen_eval(model: nn.Module) -> Iterator[None]:
    """Force every submodule to eval() for the duration, then restore.

    This matters. `predict_with_uncertainty()` in src/code.py leaves the dropout
    layers in train() mode when it returns, and it never puts them back. A
    Grad-CAM computed in that state is stochastic: run it twice, get two
    different heatmaps. The app must never do that, so we force eval and restore
    whatever the caller had.
    """
    was_training = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, flag in was_training:
            module.training = flag


def _as_batch(image_tensor: torch.Tensor) -> torch.Tensor:
    """Accept [3,H,W] or [1,3,H,W]; return [1,3,H,W]. Reject real batches."""
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError(f"image_tensor must be a torch.Tensor, got {type(image_tensor).__name__}")
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise ValueError(
            f"Expected a single image, shape [3,H,W] or [1,3,H,W]; got {tuple(image_tensor.shape)}"
        )
    return image_tensor


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


def _resize_pil_bicubic(small: np.ndarray, width: int, height: int) -> np.ndarray:
    """Exactly the up-sampling analysis/explainability.py's GradCAM uses:
    quantise to uint8, PIL BICUBIC, back to float. The uint8 round-trip is part
    of the reference behaviour, so it is reproduced rather than improved."""
    quantised = (small * 255).astype(np.uint8)
    resized = np.array(Image.fromarray(quantised).resize((width, height), Image.BICUBIC))
    return resized.astype(np.float32) / 255.0


# =============================================================================
# Grad-CAM (ResNet-50)
# =============================================================================

_RESNET_TRUNK = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")


def _resnet_trunk(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """conv1 .. layer4, i.e. torchvision ResNet forward minus avgpool/flatten/fc."""
    for name in _RESNET_TRUNK:
        x = getattr(backbone, name)(x)
    return x


def _gradcam_resnet50(
    model: nn.Module,
    x: torch.Tensor,
    target_class: Optional[int],
) -> np.ndarray:
    """Grad-CAM on layer4. Same numbers as analysis/explainability.py's GradCAM.

    Speed note: the gradient of a logit w.r.t. the layer4 feature map only
    travels back through avgpool + the classifier head. So the trunk runs under
    no_grad (no graph, no stored intermediates) and autograd only sees the tail.
    That is roughly a forward pass in cost instead of forward + full backward.
    """
    backbone = getattr(model, "backbone", model)

    with torch.no_grad():
        acts = _resnet_trunk(backbone, x)  # [1, C, h, w]

    acts = acts.detach().requires_grad_(True)
    with torch.enable_grad():
        pooled = backbone.avgpool(acts)
        logits = backbone.fc(torch.flatten(pooled, 1))  # [1, num_classes]
        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())
        (grads,) = torch.autograd.grad(logits[0, target_class], acts)

    alpha = grads[0].mean(dim=(1, 2))                                    # [C]
    cam = torch.relu((alpha[:, None, None] * acts.detach()[0]).sum(0))   # [h, w]
    cam = _minmax(cam.cpu().numpy())

    return _resize_pil_bicubic(cam, x.shape[-1], x.shape[-2])


# =============================================================================
# Attention rollout (ViT-B/16)
# =============================================================================

@contextmanager
def _capture_attention(vit: nn.Module) -> Iterator[List[torch.Tensor]]:
    """Patch each encoder block's MultiheadAttention to emit per-head weights.

    Reproduces src/xai_and_stats.py::_capture_attention. torchvision's ViT calls
    self_attention(..., need_weights=False); we flip that on for the duration and
    restore the original forward afterwards.
    """
    captured: List[torch.Tensor] = []
    blocks = list(vit.encoder.layers)
    originals = []

    for block in blocks:
        attn = block.self_attention
        original = attn.forward
        originals.append(original)

        def _make_patched(orig):
            def patched(query, key, value, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                out, weights = orig(query, key, value, **kwargs)
                if weights is not None:
                    captured.append(weights.detach().cpu())
                return out, None
            return patched

        attn.forward = _make_patched(original)

    try:
        yield captured
    finally:
        for block, original in zip(blocks, originals):
            block.self_attention.forward = original


def _attention_rollout(
    attention_maps: List[torch.Tensor],
    head_fusion: str = VIT_HEAD_FUSION,
    discard_ratio: float = VIT_DISCARD_RATIO,
) -> torch.Tensor:
    """Abnar & Zuidema 2020. Reproduces src/xai_and_stats.py::attention_rollout."""
    if not attention_maps:
        raise RuntimeError(
            "No attention weights captured. Is this a torchvision ViT-B/16 with "
            "the standard encoder.layers structure?"
        )
    fuse = {
        "mean": lambda t: t.mean(dim=1),
        "max": lambda t: t.max(dim=1).values,
        "min": lambda t: t.min(dim=1).values,
    }
    if head_fusion not in fuse:
        raise ValueError(f"head_fusion must be one of {list(fuse)}")

    result: Optional[torch.Tensor] = None
    for attn in attention_maps:                 # [B, H, N, N]
        fused = fuse[head_fusion](attn)         # [B, N, N]

        if discard_ratio > 0.0:
            flat = fused.flatten(start_dim=1)
            threshold = torch.quantile(flat, discard_ratio, dim=1)
            fused = fused * (fused >= threshold[:, None, None])

        n = fused.shape[-1]
        fused = fused + torch.eye(n, device=fused.device).unsqueeze(0)
        fused = fused / fused.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        result = fused if result is None else torch.bmm(fused, result)

    return result


def _attention_rollout_heatmap(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Full-resolution CLS-attention heatmap. Same numbers as
    xai_and_stats.generate_attention_heatmap(head_fusion='mean', discard_ratio=0.9).

    Note for the caller: this is **class-agnostic**. It shows what the ViT's CLS
    token pooled over, not what evidence supported a particular class. Passing
    target_class does nothing here, and generate_heatmap() will tell you so.
    """
    import cv2  # local import: keeps cv2 off the ResNet path's cost

    vit = getattr(model, "backbone", model)

    with torch.no_grad(), _capture_attention(vit) as captured:
        _ = vit(x)

    rollout = _attention_rollout(captured)
    cls_attn = rollout[0, 0, 1:].numpy()                       # drop CLS->CLS

    side = int(round(cls_attn.shape[0] ** 0.5))
    small = _minmax(cls_attn.reshape(side, side))

    height, width = x.shape[-2], x.shape[-1]
    heatmap = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.clip(heatmap, 0.0, 1.0).astype(np.float32)


# =============================================================================
# Contract 4 public surface
# =============================================================================

def generate_heatmap(
    model: nn.Module,
    image_tensor: torch.Tensor,
    model_kind: str,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """Where the model looked, as an [H,W] float array in [0,1].

    Args:
        model:        BrainTumorResNet50 or BrainTumorViT from src/code.py (a bare
                      torchvision backbone also works).
        image_tensor: Normalised input, [3,H,W] or [1,3,H,W]. H and W must be
                      multiples of 16 for the ViT path.
        model_kind:   "resnet50" -> Grad-CAM on layer4.
                      "vit"      -> attention rollout over all 12 encoder blocks.
        target_class: Class to explain. None means the model's own argmax.
                      **Ignored on the ViT path** — attention rollout is
                      class-agnostic and cannot answer "why this class".

    Returns:
        [H,W] float32 in [0,1], same spatial size as the input, 1.0 at the
        most-attended pixel.

    The model is left in whatever train/eval state it arrived in.
    """
    if model_kind not in MODEL_KINDS:
        raise ValueError(f"model_kind must be one of {MODEL_KINDS}, got {model_kind!r}")

    x = _as_batch(image_tensor)

    with _frozen_eval(model):
        if model_kind == "resnet50":
            heatmap = _gradcam_resnet50(model, x, target_class)
        else:
            heatmap = _attention_rollout_heatmap(model, x)

    return heatmap.astype(np.float32)


def overlay_heatmap(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a heatmap over the scan. Returns [H,W,3] uint8, ready to display.

    Args:
        original_rgb: [H,W,3] uint8 RGB, or [H,W] greyscale (it gets stacked).
        heatmap:      [H,W] float in [0,1]. If its size differs from
                      original_rgb it is resized (bicubic) to match, so you can
                      overlay a 224x224 heatmap on the full-resolution scan.
        alpha:        Heatmap weight. 0.0 is the bare scan, 1.0 hides it.

    Colour map is jet, matching analysis/explainability.py so the app shows the
    same picture that was validated. Jet is not perceptually uniform; do not read
    fine gradations off it.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    original = np.asarray(original_rgb)
    if original.ndim == 2:
        original = np.stack([original] * 3, axis=-1)
    if original.ndim != 3 or original.shape[2] != 3:
        raise ValueError(f"original_rgb must be [H,W,3] or [H,W]; got {original.shape}")
    if original.dtype != np.uint8:
        original = np.clip(original, 0, 255).astype(np.uint8)

    heat = np.asarray(heatmap, dtype=np.float32)
    if heat.ndim != 2:
        raise ValueError(f"heatmap must be [H,W]; got {heat.shape}")

    if heat.shape != original.shape[:2]:
        import cv2
        heat = cv2.resize(
            heat, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_CUBIC
        )
        heat = np.clip(heat, 0.0, 1.0)

    colored = _apply_jet(heat).astype(np.float32)
    blended = original.astype(np.float32) * (1.0 - alpha) + colored * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)
