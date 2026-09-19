#!/usr/bin/env python3
"""
src/input_validation.py

Input validation for the brain-MRI triage tool. Owned by session B.
Contract 3. Session C imports this. The signature of
``validate_image`` is frozen once published.

WHAT THIS IS
    Input validation answers one question: *should the model judge this image
    at all?* It is a gate that runs before the classifier. A knee MRI, a chest
    X-ray, a blank file, a photo of a document, a corrupted scan: none of these
    are brain MRI, and the classifier will still sort every one of them into one
    of its four classes with no warning.

WHAT THIS IS NOT
    This is not deferral. Deferral (session A) answers a different question:
    *this is a valid brain MRI, but should a human double-check the answer?*
    Both are needed. They are not the same signal and must never be reported as
    one number.

TWO STAGES
    Stage 1, precheck. Cheap image statistics. No model, no GPU, milliseconds.
    Catches operator error: wrong file, blank scan, photograph, colour image,
    wrong body part with obviously wrong geometry.

    Stage 2, score-based rejection. Needs the trained model. Uses an
    uncertainty score (mutual information, predictive entropy, max softmax) or a
    feature-space distance to catch images that look plausible to the precheck
    but sit outside anything the model was trained on.

    Stage 1 alone is always available. Stage 2 activates only when a model and
    fitted statistics are supplied.

USAGE
    from input_validation import validate_image
    r = validate_image("scan.jpg")
    if not r.is_in_scope:
        show_to_user(r.reason)

    # With the model, for the full two-stage gate:
    r = validate_image("scan.jpg", model=model, device=device)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "analysis" / "results" / "ood" / "rejector_config.json"
_DEFAULT_STATS_PATH = _REPO_ROOT / "analysis" / "results" / "ood" / "rejector_stats.npz"

# Working resolution for precheck geometry. Independent of the model's 224.
_WORK_SIZE = 256
_BORDER_PX = 8

CLASS_NAMES: Tuple[str, ...] = ("glioma", "meningioma", "pituitary", "notumor")


# =============================================================================
# Result type
# =============================================================================

@dataclass
class InputValidationResult:
    """Verdict on whether an image is inside the model's scope.

    Contract 3 fields, frozen:
        is_in_scope : True if the model may judge this image.
        reason      : Plain-words explanation, safe to show a clinic user.
        score       : The score the decision was made on. Higher means more
                      out-of-scope. For precheck-only decisions this is the
                      number of failed rules.
        method      : Which method produced the decision.

    Extra fields below are additive and may be ignored by callers.
    """

    is_in_scope: bool
    reason: str
    score: float
    method: str

    stage: str = "precheck"                       # "precheck" | "score" | "accept"
    failed_rules: List[str] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)
    threshold: Optional[float] = None
    score_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_in_scope": bool(self.is_in_scope),
            "reason": self.reason,
            "score": float(self.score),
            "method": self.method,
            "stage": self.stage,
            "failed_rules": list(self.failed_rules),
            "threshold": self.threshold,
            "score_name": self.score_name,
        }


# =============================================================================
# Feature extraction
# =============================================================================

def _to_rgb_array(image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    """Load anything into an HxWx3 uint8 array. Raises on unreadable input."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            im.load()
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        return arr
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            a = arr.astype(np.float64)
            lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
            if hi <= 1.0 + 1e-6 and lo >= -1e-6:
                a = a * 255.0
            arr = np.clip(a, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"cannot interpret array of shape {image.shape} as an image")
        return arr
    raise TypeError(f"unsupported image input type: {type(image)!r}")


def _largest_component_stats(gray: np.ndarray) -> Dict[str, float]:
    """Otsu-threshold the image, measure the largest bright blob.

    Brain MRI has one big bright object roughly in the middle of a dark field.
    A photograph, a document scan, or a corner crop does not.
    """
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    h, w = gray.shape
    total = float(h * w)
    out = {
        "fg_area_frac": 0.0,
        "fg_centroid_offset": 1.0,
        "fg_border_touch": 1.0,
        "fg_solidity": 0.0,
        "fg_extent": 0.0,
        "fg_n_components": 0.0,
    }

    finite = gray[np.isfinite(gray)]
    if finite.size == 0 or float(finite.max() - finite.min()) < 1e-6:
        return out                                  # constant image: no blob at all

    try:
        thr = float(threshold_otsu(gray))
    except Exception:
        thr = float(np.mean(gray))
    mask = gray > thr
    if not mask.any():
        return out

    labels, n = ndimage.label(mask)
    if n == 0:
        return out
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    out["fg_n_components"] = float(int((sizes > 0.01 * total).sum()))

    big = int(np.argmax(sizes)) + 1
    comp = labels == big
    area = float(sizes[big - 1])
    out["fg_area_frac"] = area / total

    ys, xs = np.nonzero(comp)
    cy, cx = float(ys.mean()), float(xs.mean())
    half_diag = 0.5 * math.hypot(h, w)
    out["fg_centroid_offset"] = math.hypot(cy - h / 2.0, cx - w / 2.0) / half_diag

    border = np.zeros_like(comp)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    out["fg_border_touch"] = float((comp & border).sum()) / max(1.0, float(border.sum()))

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    out["fg_extent"] = area / float(max(1, (y1 - y0) * (x1 - x0)))

    try:
        from skimage.morphology import convex_hull_image
        hull = convex_hull_image(comp)
        out["fg_solidity"] = area / float(max(1, hull.sum()))
    except Exception:
        out["fg_solidity"] = out["fg_extent"]

    return out


def extract_precheck_features(
    image: Union[str, Path, np.ndarray, Image.Image],
) -> Dict[str, float]:
    """Cheap statistics describing whether an image looks like a brain MRI slice.

    All geometric features are measured on a 256x256 resize so that the numbers
    do not depend on the source resolution. Dimension and colour features are
    measured on the original.
    """
    rgb = _to_rgb_array(image)
    h0, w0 = rgb.shape[:2]

    rgb_f = rgb.astype(np.float32)
    color_spread = float(np.mean(rgb_f.max(axis=2) - rgb_f.min(axis=2)) / 255.0)

    small = np.asarray(
        Image.fromarray(rgb).convert("L").resize((_WORK_SIZE, _WORK_SIZE), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    p01, p50, p99 = (float(v) for v in np.percentile(small, [1, 50, 99]))

    border = np.ones((_WORK_SIZE, _WORK_SIZE), dtype=bool)
    border[_BORDER_PX:-_BORDER_PX, _BORDER_PX:-_BORDER_PX] = False

    hist, _ = np.histogram(small, bins=64, range=(0.0, 1.0))
    p = hist.astype(np.float64) / max(1.0, hist.sum())
    nz = p[p > 0]
    hist_entropy = float(-(nz * np.log2(nz)).sum())

    # Blur and noise: Laplacian variance is the standard sharpness proxy.
    lap = (
        -4.0 * small
        + np.roll(small, 1, 0) + np.roll(small, -1, 0)
        + np.roll(small, 1, 1) + np.roll(small, -1, 1)
    )[1:-1, 1:-1]
    laplacian_var = float(np.var(lap))

    feats: Dict[str, float] = {
        "width": float(w0),
        "height": float(h0),
        "min_side": float(min(h0, w0)),
        "max_side": float(max(h0, w0)),
        "aspect_ratio": float(max(h0, w0) / max(1.0, min(h0, w0))),
        "color_spread": color_spread,
        "intensity_mean": float(small.mean()),
        "intensity_std": float(small.std()),
        "p01": p01,
        "p50": p50,
        "p99": p99,
        "dynamic_range": p99 - p01,
        "frac_dark": float((small < 0.10).mean()),
        "frac_bright": float((small > 0.90).mean()),
        "border_mean": float(small[border].mean()),
        "hist_entropy": hist_entropy,
        "laplacian_var": laplacian_var,
    }
    feats.update(_largest_component_stats(small))
    return feats


# =============================================================================
# Precheck rules
# =============================================================================
# Each rule is (feature, low, high, plain-words reason). A rule fails when the
# feature falls outside [low, high]. None means unbounded on that side.
# These defaults are replaced by the fitted values in rejector_config.json when
# that file exists. They are deliberately loose so the module is usable, and
# honest about it, before fitting.

_Rule = Tuple[str, Optional[float], Optional[float], str]

DEFAULT_PRECHECK_RULES: Dict[str, _Rule] = {
    "min_side": ("min_side", 64.0, None, "the image is too small to be a diagnostic scan"),
    "aspect_ratio": ("aspect_ratio", None, 2.0, "the image is far wider than it is tall, or the reverse, which a brain slice is not"),
    "color_spread": ("color_spread", None, 0.02, "the image is in colour, and MRI is not"),
    "intensity_std": ("intensity_std", 0.05, None, "the image is nearly blank"),
    "dynamic_range": ("dynamic_range", 0.30, None, "the image has almost no contrast"),
    "frac_dark": ("frac_dark", 0.15, 0.95, "the image does not have the dark background a brain scan has"),
    "border_mean": ("border_mean", None, 0.20, "the edges of the image are bright, and a brain scan has air at its edges"),
    "hist_entropy": ("hist_entropy", 1.5, None, "the image carries almost no detail"),
    "fg_area_frac": ("fg_area_frac", 0.05, 0.75, "no single object of brain-like size was found"),
    "fg_centroid_offset": ("fg_centroid_offset", None, 0.30, "the main object is not near the centre, and a brain slice is centred"),
    "fg_border_touch": ("fg_border_touch", None, 0.25, "the main object runs off the edge of the image, and a brain does not"),
    "fg_solidity": ("fg_solidity", 0.60, None, "the main object is not a compact rounded shape"),
}

# Rules that fire on tampered-but-still-brain inputs and rules that fire on
# wrong-modality inputs are kept apart so the report can say which is which.
RULE_GROUPS: Dict[str, str] = {
    "min_side": "format",
    "aspect_ratio": "format",
    "color_spread": "modality",
    "intensity_std": "exposure",
    "dynamic_range": "exposure",
    "frac_dark": "exposure",
    "border_mean": "geometry",
    "hist_entropy": "exposure",
    "fg_area_frac": "geometry",
    "fg_centroid_offset": "geometry",
    "fg_border_touch": "geometry",
    "fg_solidity": "geometry",
}


def apply_precheck_rules(
    feats: Dict[str, float],
    rules: Optional[Dict[str, _Rule]] = None,
) -> Tuple[List[str], List[str]]:
    """Return (failed rule names, plain-words reasons)."""
    rules = DEFAULT_PRECHECK_RULES if rules is None else rules
    failed: List[str] = []
    reasons: List[str] = []
    for name, (feat, lo, hi, reason) in rules.items():
        v = feats.get(feat)
        if v is None or not np.isfinite(v):
            failed.append(name)
            reasons.append(f"{reason} (could not be measured)")
            continue
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            failed.append(name)
            reasons.append(reason)
    return failed, reasons


def precheck(
    image: Union[str, Path, np.ndarray, Image.Image],
    rules: Optional[Dict[str, _Rule]] = None,
) -> InputValidationResult:
    """Stage 1 only. No model needed. Milliseconds."""
    try:
        feats = extract_precheck_features(image)
    except Exception as exc:
        return InputValidationResult(
            is_in_scope=False,
            reason=f"the file could not be read as an image ({exc})",
            score=float("inf"),
            method="precheck",
            stage="precheck",
            failed_rules=["unreadable"],
        )

    failed, reasons = apply_precheck_rules(feats, rules)
    if failed:
        return InputValidationResult(
            is_in_scope=False,
            reason="This does not look like a brain MRI slice: " + "; ".join(reasons[:3]) + ".",
            score=float(len(failed)),
            method="precheck",
            stage="precheck",
            failed_rules=failed,
            features=feats,
        )
    return InputValidationResult(
        is_in_scope=True,
        reason="Passed the image checks.",
        score=0.0,
        method="precheck",
        stage="precheck",
        failed_rules=[],
        features=feats,
    )


# =============================================================================
# Configuration
# =============================================================================

class RejectorConfig:
    """Loaded rejector_config.json plus, if needed, the fitted statistics."""

    def __init__(self, cfg: Dict[str, Any], stats: Optional[Dict[str, np.ndarray]] = None) -> None:
        for key in ("schema_version", "method", "threshold", "precheck_enabled"):
            if key not in cfg:
                raise ValueError(f"rejector_config.json is missing required key {key!r}")
        self.raw = cfg
        self.schema_version: str = cfg["schema_version"]
        self.method: str = cfg["method"]
        self.threshold: float = float(cfg["threshold"])
        self.precheck_enabled: bool = bool(cfg["precheck_enabled"])
        self.stats = stats or {}

        raw_rules = cfg.get("precheck_rules") or {}
        self.rules: Dict[str, _Rule] = {}
        for name, spec in raw_rules.items():
            self.rules[name] = (
                spec["feature"],
                None if spec.get("low") is None else float(spec["low"]),
                None if spec.get("high") is None else float(spec["high"]),
                spec.get("reason", "the image failed an image-quality check"),
            )
        if not self.rules:
            self.rules = dict(DEFAULT_PRECHECK_RULES)

    @property
    def is_fitted(self) -> bool:
        return self.schema_version != "0.0-unfitted"


_UNFITTED = {
    "schema_version": "0.0-unfitted",
    "method": "precheck_only",
    "threshold": float("inf"),
    "precheck_enabled": True,
    "precheck_rules": {},
}

_config_cache: Dict[str, RejectorConfig] = {}


def load_config(
    config_path: Union[str, Path, None] = None,
    stats_path: Union[str, Path, None] = None,
) -> RejectorConfig:
    """Load the published rejector config. Falls back to loose built-in defaults.

    The fallback is deliberate: session C must be able to import and run this
    module before session B publishes. A fallback config reports
    ``schema_version == "0.0-unfitted"`` so callers can tell.
    """
    cp = Path(config_path) if config_path is not None else _DEFAULT_CONFIG_PATH
    key = str(cp)
    if key in _config_cache:
        return _config_cache[key]

    if not cp.exists():
        cfg = RejectorConfig(dict(_UNFITTED))
        _config_cache[key] = cfg
        return cfg

    with open(cp, "r", encoding="utf-8") as f:
        raw = json.load(f)

    stats: Dict[str, np.ndarray] = {}
    sp = Path(stats_path) if stats_path is not None else None
    if sp is None:
        rel = raw.get("fitted_statistics_path")
        if rel:
            cand = Path(rel)
            sp = cand if cand.is_absolute() else _REPO_ROOT / cand
    if sp is not None and Path(sp).exists():
        with np.load(sp, allow_pickle=False) as z:
            stats = {k: z[k] for k in z.files}

    cfg = RejectorConfig(raw, stats)
    _config_cache[key] = cfg
    return cfg


def clear_config_cache() -> None:
    _config_cache.clear()


# =============================================================================
# Stage 2: model-based scores
# =============================================================================

def _mc_scores(model: Any, x: Any, T: int = 20) -> Dict[str, float]:
    """Predictive entropy, mutual information and max softmax for one image.

    Entropy is in bits, matching ``src/code.py``'s ``predict_with_uncertainty``.

    Mutual information is entropy of the mean minus the mean of the per-pass
    entropies. It is the part of the uncertainty that comes from the model not
    knowing, rather than from the image being genuinely ambiguous. That is the
    part that matters for "I have never seen anything like this".
    """
    import torch

    eps = 1e-10
    with torch.no_grad():
        out = model.predict_with_uncertainty(x, T=T)
    all_probs = out["all_probs"]                        # (T, B, C)
    mean_probs = out["mean_probs"]                      # (B, C)
    ent_mean = -(mean_probs * torch.log2(mean_probs + eps)).sum(-1)          # (B,)
    mean_ent = -(all_probs * torch.log2(all_probs + eps)).sum(-1).mean(0)    # (B,)
    mi = ent_mean - mean_ent
    return {
        "entropy": float(ent_mean[0].item()),
        "mutual_information": float(mi[0].item()),
        "max_softmax": float(mean_probs[0].max().item()),
        "mean_probs": mean_probs[0].detach().cpu().numpy().tolist(),
    }


def _mahalanobis(feat: np.ndarray, stats: Dict[str, np.ndarray]) -> float:
    """Smallest Mahalanobis distance to any class centroid in feature space.

    Centroids and the shared inverse covariance are fitted on the internal
    training split only. Never on BRISC.
    """
    mu = stats["mahalanobis_means"]                     # (C, D)
    prec = stats["mahalanobis_precision"]               # (D, D)
    d = feat[None, :] - mu                              # (C, D)
    m = np.einsum("cd,de,ce->c", d, prec, d)
    return float(np.sqrt(np.maximum(m, 0.0)).min())


_SCORE_DIRECTION = {                    # +1: higher means more out-of-scope
    "entropy": +1.0,
    "mutual_information": +1.0,
    "max_softmax": -1.0,
    "mahalanobis": +1.0,
}

_SCORE_REASON = {
    "entropy": "the model is very unsure about this image",
    "mutual_information": "the model has not seen anything like this image before",
    "max_softmax": "the model's best guess is weak",
    "mahalanobis": "this image sits far away from everything the model was trained on",
}


# =============================================================================
# The published entry point
# =============================================================================

def validate_image(
    image_path_or_array: Union[str, Path, np.ndarray, Image.Image],
    model: Any = None,
    device: Any = None,
    config: Optional[RejectorConfig] = None,
    config_path: Union[str, Path, None] = None,
    mc_T: Optional[int] = None,
) -> InputValidationResult:
    """Decide whether the model is allowed to judge this image. Contract 3.

    Args:
        image_path_or_array: A path, a PIL image, or an HxW / HxWx3 array.
        model:  Optional. A ``BrainTumorViT`` or ``BrainTumorResNet50``. When
                given, stage 2 runs and the model-based score is used. When
                omitted, only the precheck runs.
        device: Optional torch device for the model.
        config: Optional pre-loaded config. Defaults to the published one.
        config_path: Optional path override, mostly for tests.
        mc_T:   Optional override for the number of MC Dropout passes.

    Returns:
        InputValidationResult with ``.is_in_scope``, ``.reason``, ``.score``,
        ``.method``.

    A rejection here means the image goes to a human unread. That is the safe
    outcome. Accepting an image the model cannot read is the dangerous one.
    """
    cfg = config if config is not None else load_config(config_path)

    if cfg.precheck_enabled:
        pre = precheck(image_path_or_array, cfg.rules)
        if not pre.is_in_scope:
            return pre
    else:
        pre = precheck(image_path_or_array, {})       # features only, no rules

    if model is None or cfg.method == "precheck_only":
        note = "" if cfg.is_fitted else " (input validation is running on unfitted defaults)"
        return InputValidationResult(
            is_in_scope=True,
            reason="Passed the image checks." + note,
            score=0.0,
            method="precheck_only",
            stage="accept",
            features=pre.features,
        )

    import torch
    from torchvision import transforms as T_

    tf = T_.Compose([
        T_.Resize((224, 224)),
        T_.ToTensor(),
        T_.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    rgb = _to_rgb_array(image_path_or_array)
    x = tf(Image.fromarray(rgb)).unsqueeze(0)
    if device is not None:
        x = x.to(device)

    method = cfg.method
    if method == "mahalanobis":
        with torch.no_grad():
            feat = _penultimate_features(model, x).cpu().numpy()[0]
        raw = _mahalanobis(feat, cfg.stats)
    else:
        T_passes = mc_T if mc_T is not None else int(cfg.raw.get("mc_T", 20))
        scores = _mc_scores(model, x, T=T_passes)
        raw = scores[method]

    direction = _SCORE_DIRECTION.get(method, +1.0)
    reject = (raw > cfg.threshold) if direction > 0 else (raw < cfg.threshold)

    if reject:
        return InputValidationResult(
            is_in_scope=False,
            reason=(
                "This image is outside what the model can judge: "
                f"{_SCORE_REASON.get(method, 'it does not match the training data')}. "
                "Send it to a human."
            ),
            score=float(raw),
            method=method,
            stage="score",
            threshold=cfg.threshold,
            score_name=method,
            features=pre.features,
        )
    return InputValidationResult(
        is_in_scope=True,
        reason="Passed the image checks and looks like data the model knows.",
        score=float(raw),
        method=method,
        stage="accept",
        threshold=cfg.threshold,
        score_name=method,
        features=pre.features,
    )


def _penultimate_features(model: Any, x: Any) -> Any:
    """Backbone features just before the classification head, shape (B, D).

    ResNet-50: the 2048-d pooled vector. ViT-B/16: the 768-d class token after
    the encoder LayerNorm. Both are deterministic. All of the MC Dropout
    stochasticity in these models lives in the head, so this is a single pass.
    """
    import torch
    import torch.nn as nn

    model.eval()
    bb = model.backbone
    with torch.no_grad():
        if hasattr(bb, "fc"):                          # ResNet-50
            head, bb.fc = bb.fc, nn.Identity()
            try:
                return bb(x)
            finally:
                bb.fc = head
        if hasattr(bb, "heads"):                       # ViT-B/16
            head, bb.heads = bb.heads, nn.Identity()
            try:
                return bb(x)
            finally:
                bb.heads = head
    raise TypeError(f"do not know how to take features from {type(model)!r}")


__all__ = [
    "InputValidationResult",
    "validate_image",
    "precheck",
    "extract_precheck_features",
    "apply_precheck_rules",
    "load_config",
    "clear_config_cache",
    "RejectorConfig",
    "DEFAULT_PRECHECK_RULES",
    "RULE_GROUPS",
]
