#!/usr/bin/env python3
"""
src/xai_and_stats.py

Model Interpretability (XAI) and Statistical Rigor for ViT-B/16 Brain Tumor Classification.
Designed to meet Nature Scientific Reports submission standards.

Two self-contained modules
--------------------------
1. Attention Maps     – Attention Rollout across all 12 encoder blocks of a
                        torchvision ViT-B/16.  No model surgery required; uses
                        a temporary forward-hook + monkey-patch so that
                        nn.MultiheadAttention returns per-head weights even when
                        the model was compiled with need_weights=False.

2. Bootstrapped 95% CIs – Empirical resampling (n=1000) for Accuracy,
                           Macro AUC, and multiclass Brier Score.

Importable surface
------------------
    from xai_and_stats import (
        generate_attention_heatmap,   # → np.ndarray [H, W] in [0, 1]
        visualize_attention,          # → plt.Figure (side-by-side)
        calculate_bootstrapped_metrics,  # → pd.DataFrame
    )
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Dict, Generator, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torchvision import models

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (ViT-B/16 defaults)
# ─────────────────────────────────────────────────────────────────────────────

PATCH_SIZE: int = 16
IMAGE_SIZE: int = 224
GRID_SIZE: int = IMAGE_SIZE // PATCH_SIZE   # 14 patches per side
NUM_PATCHES: int = GRID_SIZE ** 2           # 196 spatial tokens


# =============================================================================
# SECTION 1 – Explainable AI: Attention Rollout
# =============================================================================

@contextmanager
def _capture_attention(
    model: nn.Module,
) -> Generator[List[torch.Tensor], None, None]:
    """Temporarily patch every EncoderBlock's self_attention to emit per-head
    weights.

    torchvision ViT calls ``self_attention(x, x, x, need_weights=False)``.
    This manager intercepts that call at the ``nn.MultiheadAttention`` level,
    overrides ``need_weights=True`` and ``average_attn_weights=False``, then
    restores the original forward when the context exits.

    Setting ``need_weights=True`` automatically disables PyTorch 2.x's
    ``scaled_dot_product_attention`` fast-path inside MHA, so no extra
    ``torch.backends`` flags are needed.

    Yields:
        captured: Mutable list that is populated with one ``[B, H, N, N]``
                  tensor per encoder block during the forward pass (layer order).
    """
    captured: List[torch.Tensor] = []
    originals: List = []

    blocks = list(model.encoder.layers)   # nn.Sequential of EncoderBlock

    for block in blocks:
        attn = block.self_attention        # nn.MultiheadAttention
        orig = attn.forward
        originals.append(orig)

        # Build patched forward with orig captured in closure (avoids late-binding)
        def _make_patched(original):
            def patched(query, key, value, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                out, weights = original(query, key, value, **kwargs)
                if weights is not None:
                    captured.append(weights.detach().cpu())
                return out, None   # EncoderBlock discards the 2nd return anyway
            return patched

        attn.forward = _make_patched(orig)

    try:
        yield captured
    finally:
        for block, orig in zip(blocks, originals):
            block.self_attention.forward = orig


def attention_rollout(
    attention_maps: List[torch.Tensor],
    head_fusion: str = "mean",
    discard_ratio: float = 0.9,
) -> torch.Tensor:
    """Implements Attention Rollout (Abnar & Zuidema, 2020, arXiv:2005.00928).

    For each layer the per-head attention matrices are fused, augmented with an
    identity (modelling the residual connection), row-normalised, and chained via
    matrix multiplication.  The resulting matrix propagates the CLS token's
    effective attention across all layers.

    Args:
        attention_maps: List of ``[B, H, N, N]`` tensors in layer order.
                        Captured by :func:`_capture_attention`.
        head_fusion:    Strategy for collapsing the head axis.
                        ``"mean"`` | ``"max"`` | ``"min"``.
        discard_ratio:  Fraction of lowest-weight entries to zero out before
                        chain-multiplication (noise suppression).  ``0.0``
                        disables.  ``0.9`` is the published default.

    Returns:
        Rollout tensor ``[B, N, N]``.  Index ``[:, 0, 1:]`` to obtain the CLS
        token's aggregated attention over the ``N-1`` patch tokens.

    Raises:
        ValueError: On an empty list or unsupported ``head_fusion`` value.
    """
    if not attention_maps:
        raise ValueError("`attention_maps` is empty — no attention was captured.")

    _fuse_fn = {
        "mean": lambda t: t.mean(dim=1),
        "max":  lambda t: t.max(dim=1).values,
        "min":  lambda t: t.min(dim=1).values,
    }
    if head_fusion not in _fuse_fn:
        raise ValueError(f"`head_fusion` must be one of {list(_fuse_fn)}.")

    result: Optional[torch.Tensor] = None

    for attn in attention_maps:   # attn: [B, H, N, N]
        B, H, N, _ = attn.shape

        attn_fused = _fuse_fn[head_fusion](attn)   # [B, N, N]

        # Optional noise suppression: zero entries below the discard_ratio quantile
        if discard_ratio > 0.0:
            flat = attn_fused.flatten(start_dim=1)   # [B, N*N]
            threshold = torch.quantile(flat, discard_ratio, dim=1)  # [B]
            mask = attn_fused >= threshold[:, None, None]
            attn_fused = attn_fused * mask

        # Add identity matrix (residual) and row-normalise
        I = torch.eye(N, device=attn_fused.device).unsqueeze(0)   # [1, N, N]
        attn_fused = attn_fused + I
        attn_fused = attn_fused / attn_fused.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        result = attn_fused if result is None else torch.bmm(attn_fused, result)

    return result   # [B, N, N]


def generate_attention_heatmap(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    head_fusion: str = "mean",
    discard_ratio: float = 0.9,
) -> np.ndarray:
    """Run one forward pass, apply Attention Rollout, return a full-resolution
    heatmap.

    This is the primary entry point for XAI.  The model is **not** modified
    permanently.

    Args:
        model:          torchvision ViT-B/16 (any head; must be in eval mode).
        image_tensor:   Pre-processed float tensor ``[1, 3, H, W]``.  ``H`` and
                        ``W`` must be divisible by ``PATCH_SIZE`` (16).
        device:         Torch device for inference.
        head_fusion:    Passed to :func:`attention_rollout`.
        discard_ratio:  Passed to :func:`attention_rollout`.

    Returns:
        ``heatmap``: float32 ndarray ``[H, W]`` with values in ``[0, 1]``.
                     Ready to pass to :func:`visualize_attention`.

    Raises:
        RuntimeError: If no attention weights were captured (wrong model type).
        ValueError:   If ``image_tensor`` does not have the expected 4-D shape.
    """
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise ValueError(
            f"Expected image_tensor shape [1, C, H, W]; got {tuple(image_tensor.shape)}"
        )

    model = model.to(device).eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad(), _capture_attention(model) as captured:
        _ = model(image_tensor)

    if not captured:
        raise RuntimeError(
            "No attention weights were captured.  Verify that `model` is a "
            "torchvision ViT-B/16 with the standard `encoder.layers` structure."
        )

    rollout = attention_rollout(
        captured, head_fusion=head_fusion, discard_ratio=discard_ratio
    )   # [1, N+1, N+1]  (N+1 because of CLS token)

    # CLS token row → patch tokens; strip the CLS→CLS diagonal entry
    cls_attn = rollout[0, 0, 1:].numpy()   # [N_patches]

    grid_side = int(round(cls_attn.shape[0] ** 0.5))
    heatmap_small = cls_attn.reshape(grid_side, grid_side)

    # Min-max normalise to [0, 1]
    lo, hi = heatmap_small.min(), heatmap_small.max()
    heatmap_small = (heatmap_small - lo) / (hi - lo + 1e-8)

    H, W = image_tensor.shape[-2], image_tensor.shape[-1]
    heatmap = cv2.resize(heatmap_small, (W, H), interpolation=cv2.INTER_CUBIC)

    # Bicubic can overshoot; re-clip to [0, 1]
    heatmap = np.clip(heatmap, 0.0, 1.0)

    return heatmap.astype(np.float32)


def visualize_attention(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
    title: str = "Attention Rollout",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Produce a side-by-side figure: [Original MRI] | [Attention Overlay].

    Args:
        original_image: uint8 RGB ndarray ``[H, W, 3]`` or greyscale ``[H, W]``.
        heatmap:        float32 ``[H, W]`` in ``[0, 1]`` from
                        :func:`generate_attention_heatmap`.
        alpha:          Heatmap blending weight.  ``0.0`` = image only,
                        ``1.0`` = heatmap only.  ``0.5`` recommended.
        colormap:       OpenCV colormap constant.  Default: ``cv2.COLORMAP_JET``.
        title:          Figure super-title string.
        save_path:      Optional file path to save the figure (PNG/PDF).

    Returns:
        ``fig`` (plt.Figure) — caller is responsible for ``plt.close(fig)``.
    """
    # ── Normalise original image to uint8 RGB ──────────────────────────────
    img = np.array(original_image)
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    # ── Build heatmap overlay ──────────────────────────────────────────────
    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_u8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 1.0 - alpha, heatmap_rgb, alpha, 0)

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].imshow(img)
    axes[0].set_title("Original MRI", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Attention Heatmap Overlay", fontsize=12)
    axes[1].axis("off")

    sm = plt.cm.ScalarMappable(
        cmap="jet", norm=plt.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    fig.colorbar(
        sm, ax=axes[1], fraction=0.046, pad=0.04, label="Attention Intensity"
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualize_attention] Figure saved -> {save_path}")

    return fig


# =============================================================================
# SECTION 2 – Statistical Rigor: Bootstrapped 95% Confidence Intervals
# =============================================================================

def calculate_bootstrapped_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Empirical bootstrap estimation of 95% CIs for three primary metrics.

    Metrics
    -------
    * **Accuracy**   – ``argmax(y_probs)`` vs ``y_true``.
    * **Macro AUC**  – One-vs-rest macro-averaged AUROC.  Bootstrap samples
                       where a class is absent are skipped for this metric.
    * **Brier Score** – Multiclass generalisation:
                        ``mean_n( sum_k (p_nk − I[y_n=k])^2 )``.
                        Lower is better (0 = perfect; 2·(1-1/C) = worst).

    Args:
        y_true:        Integer class labels, shape ``[N]``.
        y_probs:       Predicted probabilities, shape ``[N, C]``.  Binary case:
                       pass ``[N, 2]`` (column 1 = P(positive)).
        n_iterations:  Number of bootstrap resamples.  ≥1000 required for
                       publication-grade intervals.
        confidence:    Confidence level; default ``0.95`` gives 2.5/97.5 pct.
        random_state:  Seed for reproducibility.

    Returns:
        ``pd.DataFrame`` indexed by Metric with columns
        ``[Mean, CI Lower, CI Upper, CI Width]``, all rounded to 4 d.p.

    Raises:
        ValueError: On shape mismatch or ``n_iterations < 100``.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.asarray(y_probs, dtype=np.float64)

    if y_probs.ndim == 1:
        # Treat as P(class=1) for binary classification
        y_probs = np.column_stack([1.0 - y_probs, y_probs])

    N, C = y_probs.shape

    if y_true.shape[0] != N:
        raise ValueError(
            f"Shape mismatch: y_true has {y_true.shape[0]} samples, "
            f"y_probs has {N} rows."
        )
    if n_iterations < 100:
        raise ValueError("`n_iterations` must be ≥ 100 for reliable CI estimates.")

    alpha = 1.0 - confidence
    lo_pct = (alpha / 2.0) * 100.0
    hi_pct = (1.0 - alpha / 2.0) * 100.0

    rng = np.random.default_rng(random_state)

    acc_boot:   List[float] = []
    auc_boot:   List[float] = []
    brier_boot: List[float] = []

    for _ in range(n_iterations):
        idx = rng.integers(0, N, size=N)
        yt = y_true[idx]
        yp = y_probs[idx]

        # ── Accuracy ──────────────────────────────────────────────────────
        acc_boot.append(float(accuracy_score(yt, yp.argmax(axis=1))))

        # ── Macro AUC ─────────────────────────────────────────────────────
        # Skip samples where the bootstrap doesn't contain every class
        if np.unique(yt).shape[0] == C:
            try:
                if C == 2:
                    auc = roc_auc_score(yt, yp[:, 1])
                else:
                    auc = roc_auc_score(yt, yp, multi_class="ovr", average="macro")
                auc_boot.append(float(auc))
            except ValueError:
                pass  # Degenerate sample (all one class after sklearn checks)

        # ── Brier Score (multiclass) ───────────────────────────────────────
        y_onehot = np.eye(C)[yt]                          # [N, C]
        brier = float(np.mean(np.sum((yp - y_onehot) ** 2, axis=1)))
        brier_boot.append(brier)

    # ── Summarise ─────────────────────────────────────────────────────────

    def _ci(scores: List[float], name: str) -> Dict:
        arr = np.array(scores)
        mean = arr.mean()
        lo   = float(np.percentile(arr, lo_pct))
        hi   = float(np.percentile(arr, hi_pct))
        return {
            "Metric":   name,
            "Mean":     round(mean, 4),
            "CI Lower": round(lo, 4),
            "CI Upper": round(hi, 4),
            "CI Width": round(hi - lo, 4),
        }

    rows = [
        _ci(acc_boot,   "Accuracy"),
        _ci(auc_boot,   "Macro AUC"),
        _ci(brier_boot, "Brier Score"),
    ]

    return pd.DataFrame(rows).set_index("Metric")


# =============================================================================
# Demo  (python src/xai_and_stats.py)
# =============================================================================

if __name__ == "__main__":
    import torchvision.transforms as T
    from pathlib import Path as _Path

    ASSETS_DIR = _Path(__file__).resolve().parent.parent / "assets"
    ASSETS_DIR.mkdir(exist_ok=True)

    CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    NUM_CLASSES  = len(CLASS_NAMES)
    N_SAMPLES    = 300

    print("=" * 60)
    print(" ViT-B/16 XAI + Stats Demo")
    print("=" * 60)

    # ── Build a ViT-B/16 with a 4-class head (no pretrained weights needed) ──
    print("\n[1/4] Building ViT-B/16 model...")
    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, NUM_CLASSES)
    model.eval()

    # ── Simulate a preprocessed MRI tensor (normally from your DataLoader) ──
    print("[2/4] Generating dummy MRI tensor and running Attention Rollout...")
    preprocess = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    rng_np = np.random.default_rng(0)
    raw_np   = rng_np.uniform(0.0, 1.0, (IMAGE_SIZE, IMAGE_SIZE, 3)).astype(np.float32)
    raw_uint8 = (raw_np * 255).astype(np.uint8)   # for visualisation

    img_tensor = torch.from_numpy(raw_np).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    img_tensor = preprocess(img_tensor)

    heatmap = generate_attention_heatmap(
        model, img_tensor,
        device=torch.device("cpu"),
        head_fusion="mean",
        discard_ratio=0.9,
    )
    print(f"    Heatmap shape: {heatmap.shape},  range [{heatmap.min():.3f}, {heatmap.max():.3f}]")

    fig_attn = visualize_attention(
        original_image=raw_uint8,
        heatmap=heatmap,
        alpha=0.5,
        title="Brain Tumor Classification — Attention Rollout (Demo)",
        save_path=str(ASSETS_DIR / "attention_demo.png"),
    )
    plt.close(fig_attn)

    # ── Generate synthetic prediction data ─────────────────────────────────
    print("[3/4] Computing bootstrapped 95% CIs (n=1000 iterations)...")
    y_true_demo = rng_np.integers(0, NUM_CLASSES, size=N_SAMPLES)

    # Add a small signal toward the true class so AUC is meaningfully above 0.5
    logits = rng_np.normal(size=(N_SAMPLES, NUM_CLASSES))
    logits[np.arange(N_SAMPLES), y_true_demo] += 1.5
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    y_probs_demo = (exp_logits / exp_logits.sum(axis=1, keepdims=True)).astype(np.float64)

    metrics_df = calculate_bootstrapped_metrics(
        y_true=y_true_demo,
        y_probs=y_probs_demo,
        n_iterations=1000,
        confidence=0.95,
        random_state=42,
    )

    print("\n-- Bootstrapped Performance Metrics (95% CI) ------------------")
    print(metrics_df.to_string())
    print()

    # ── CI bar chart ────────────────────────────────────────────────────────
    print("[4/4] Saving metrics CI chart...")
    df = metrics_df.reset_index()

    fig_ci, ax = plt.subplots(figsize=(8, 5))
    x      = np.arange(len(df))
    colors = ["#4878CF", "#2CA02C", "#D62728"]
    errs   = np.array([
        (df["Mean"] - df["CI Lower"]).values,
        (df["CI Upper"] - df["Mean"]).values,
    ])

    bars = ax.bar(
        x, df["Mean"], yerr=errs, capsize=7, width=0.5,
        color=colors, alpha=0.80, ecolor="black", linewidth=0,
    )

    for bar, row in zip(bars, df.itertuples()):
        # row._3 = CI Lower, row._4 = CI Upper (spaces → invalid identifiers → _N)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row._4 + 0.015,                           # place label above upper CI cap
            f"{row.Mean:.3f}\n[{row._3:.3f}, {row._4:.3f}]",
            ha="center", va="bottom", fontsize=8.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["Metric"], fontsize=12)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.20)
    ax.set_title(
        "Bootstrapped Performance Metrics with 95% Confidence Intervals\n"
        "(Empirical Bootstrap, n = 1 000 resamples)",
        fontsize=12, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig_ci.tight_layout()
    fig_ci.savefig(str(ASSETS_DIR / "metrics_ci_demo.png"), dpi=150, bbox_inches="tight")
    print(f"    Saved -> {ASSETS_DIR / 'metrics_ci_demo.png'}")
    plt.close(fig_ci)

    print(f"\nDemo complete.  Outputs: {ASSETS_DIR / 'attention_demo.png'}, {ASSETS_DIR / 'metrics_ci_demo.png'}")
