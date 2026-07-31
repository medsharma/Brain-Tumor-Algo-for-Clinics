"""Adapter for session D's runtime explainer (Contract 4).

Session D owns ``src/explain_runtime.py`` with ``generate_heatmap()`` and
``overlay_heatmap()``. Until that lands, this module supplies a stub.

The stub heatmap is deliberately, unmistakably synthetic: hard diagonal
stripes that no saliency method would ever produce. A plausible-looking fake
heatmap is worse than none at all, because the heatmap is the layer a doctor
uses to sanity check the call. If it looks real and means nothing, it destroys
exactly the trust it exists to build. So the stub looks like a test card, and
the app refuses to start on it outside development.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

import numpy as np

STUB_LABEL = "STUB — NOT A REAL HEATMAP"

#: Shown under the overlay when session D's module is not available. D
#: publishes its own wording as ``HEATMAP_CAVEAT`` and asks that it be used
#: verbatim, so the real one always wins over this fallback.
FALLBACK_CAVEAT = (
    "Shows where the model looked, not where the tumour is. "
    "Use it to catch obviously wrong calls, not to confirm right ones."
)

STUB_CAVEAT = (
    "This is a test pattern, not where the model looked. The real heatmap is "
    "not installed in this build."
)


class Explainer:
    """Wraps session D's explainer, or stands in for it.

    Attributes:
        is_stub: True when D's module was not importable.
        detail: Why, in words.
        caveat: The one honest line shown under the overlay. Taken verbatim
            from session D's ``HEATMAP_CAVEAT`` when available, because D owns
            the wording and measured what backs it.
    """

    def __init__(self, force_stub: bool = False) -> None:
        self._generate: Callable[..., Any] | None = None
        self._overlay: Callable[..., Any] | None = None
        self.is_stub = True
        self.detail = ""
        self.caveat = STUB_CAVEAT

        if force_stub:
            self.detail = "stub forced by caller"
            return

        try:
            module = importlib.import_module("src.explain_runtime")
        except ImportError as exc:
            self.detail = f"src/explain_runtime.py not importable ({exc})"
            return

        generate = getattr(module, "generate_heatmap", None)
        overlay = getattr(module, "overlay_heatmap", None)
        if not callable(generate) or not callable(overlay):
            self.detail = (
                "src/explain_runtime.py is missing generate_heatmap() or "
                "overlay_heatmap()"
            )
            return

        self._generate = generate
        self._overlay = overlay
        self.is_stub = False
        self.detail = "session D's explainer is loaded"

        published = getattr(module, "HEATMAP_CAVEAT", None)
        self.caveat = published if isinstance(published, str) and published else FALLBACK_CAVEAT

    # -- Contract 4 surface -------------------------------------------------

    def generate_heatmap(
        self,
        model: Any,
        image_tensor: Any,
        model_kind: str,
        target_class: int | None = None,
    ) -> np.ndarray:
        """HxW float array in [0, 1], same spatial size as the input tensor."""
        if self._generate is None:
            return self._stub_heatmap(image_tensor)

        heatmap = np.asarray(
            self._generate(model, image_tensor, model_kind, target_class),
            dtype=np.float32,
        )
        return self._check_heatmap(heatmap, image_tensor)

    def overlay_heatmap(
        self,
        original_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """HxWx3 uint8 image ready to display."""
        if self._overlay is None:
            return self._stub_overlay(original_rgb, heatmap, alpha)

        result = np.asarray(self._overlay(original_rgb, heatmap, alpha))
        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    # -- Validation of D's output ------------------------------------------

    @staticmethod
    def _check_heatmap(heatmap: np.ndarray, image_tensor: Any) -> np.ndarray:
        """Enforce Contract 4's shape and range promises.

        Loud rather than lenient. A heatmap silently transposed or out of range
        would draw attention to the wrong part of the brain, which is worse
        than showing nothing.
        """
        if heatmap.ndim != 2:
            raise ValueError(
                f"Contract 4 says generate_heatmap returns an HxW array. "
                f"Session D returned shape {heatmap.shape}."
            )

        expected = tuple(int(v) for v in np.shape(image_tensor)[-2:])
        if heatmap.shape != expected:
            raise ValueError(
                f"Heatmap shape {heatmap.shape} does not match the input's "
                f"spatial size {expected}. Contract 4 requires them equal."
            )

        finite = np.isfinite(heatmap)
        if not finite.all():
            raise ValueError("Heatmap contains NaN or infinity.")

        low, high = float(heatmap.min()), float(heatmap.max())
        if low < -1e-6 or high > 1.0 + 1e-6:
            raise ValueError(
                f"Heatmap values run {low:.4f} to {high:.4f}. Contract 4 "
                f"requires the range [0, 1]."
            )
        return np.clip(heatmap, 0.0, 1.0)

    # -- Stubs --------------------------------------------------------------

    @staticmethod
    def _stub_heatmap(image_tensor: Any) -> np.ndarray:
        """Diagonal stripes. Obviously not saliency, at a glance."""
        height, width = (int(v) for v in np.shape(image_tensor)[-2:])
        rows = np.arange(height).reshape(-1, 1)
        cols = np.arange(width).reshape(1, -1)
        stripes = ((rows + cols) // 16) % 2
        return stripes.astype(np.float32)

    @staticmethod
    def _stub_overlay(
        original_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Flat magenta stripes over the scan. Impossible to mistake for real."""
        base = np.asarray(original_rgb)
        if base.dtype != np.uint8:
            base = np.clip(base, 0, 255).astype(np.uint8)

        colour = np.zeros_like(base)
        mask = heatmap[..., None]
        colour[..., 0] = 255  # red
        colour[..., 2] = 255  # blue, so red + blue reads as magenta

        blended = base * (1.0 - alpha * mask) + colour * (alpha * mask)
        return np.clip(blended, 0, 255).astype(np.uint8)


_explainer: Explainer | None = None


def get_explainer(force_stub: bool = False) -> Explainer:
    global _explainer
    if _explainer is None or force_stub:
        _explainer = Explainer(force_stub=force_stub)
    return _explainer


def reset_explainer() -> None:
    """Test hook, and the way the app picks up D's module after a rebase."""
    global _explainer
    _explainer = None
