#!/usr/bin/env python3
"""
tests/test_explain_runtime.py

Proof that src/explain_runtime.py produces the same heatmaps as the already
validated research code in analysis/explainability.py. Session C ships whatever
this file says is correct, so this is the test that matters most in session D.

What is compared, on real checkpoints and real images:

  ResNet-50 : runtime Grad-CAM      vs analysis/explainability.py::GradCAM
  ViT-B/16  : runtime rollout       vs src/xai_and_stats.py::generate_attention_heatmap
  overlay   : runtime jet blend     vs analysis/explainability.py::overlay_heatmap
  jet LUT   : baked table           vs matplotlib.colormaps['jet']

Everything runs on CPU.

    python tests/test_explain_runtime.py             # real checkpoints, 8 images
    python tests/test_explain_runtime.py --n-images 24
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_REPO = Path(r"C:\Users\medha\OneDrive\Documents\MRI ALGO")
CHECKPOINTS = MAIN_REPO / "results" / "20260703_155524"

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "analysis"))

import explain_runtime  # noqa: E402
from explain_runtime import generate_heatmap, overlay_heatmap  # noqa: E402

# The research reference implementations.
from explainability import GradCAM, denormalize_to_uint8  # noqa: E402
from explainability import overlay_heatmap as reference_overlay  # noqa: E402
from xai_and_stats import generate_attention_heatmap  # noqa: E402
from common import get_manifest, get_transforms  # noqa: E402
from code import BrainTumorResNet50, BrainTumorViT, NUM_CLASSES  # noqa: E402

DEVICE = torch.device("cpu")

# Tolerances. Both paths run the identical arithmetic in the identical order, so
# the only expected difference is float non-associativity in the reduction.
ATOL_HEATMAP = 1e-5
ATOL_OVERLAY = 0  # uint8, must match exactly

_FAILURES: List[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        _FAILURES.append(f"{name}: {detail}")


def load_model(kind: str, seed: int = 42):
    cls = {"resnet50": BrainTumorResNet50, "vit": BrainTumorViT}[kind]
    ckpt_path = CHECKPOINTS / kind / f"seed_{seed}" / f"best_{kind}_seed{seed}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")
    model = cls(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)["model_state_dict"])
    model.eval()
    return model


def fixed_image_batch(n: int) -> List[torch.Tensor]:
    """A fixed, seed-stable set of internal test-split images. Internal data, not
    BRISC, so nothing here can leak the external test set into a design choice."""
    manifest = get_manifest()
    test_df = manifest[manifest["split"] == "test"].sort_values("filepath").reset_index(drop=True)
    picks = test_df.sample(n=n, random_state=20260730).reset_index(drop=True)
    transform = get_transforms("test")
    return [
        transform(Image.open(row["filepath"]).convert("RGB")).unsqueeze(0).to(DEVICE)
        for _, row in picks.iterrows()
    ]


# =============================================================================

def test_jet_lut() -> None:
    print("\n[1] Baked jet colour table vs matplotlib")
    try:
        import matplotlib
    except ImportError:
        check("jet LUT", False, "matplotlib not installed, cannot verify")
        return
    cmap = matplotlib.colormaps["jet"]
    rng = np.random.default_rng(0)
    probe = np.concatenate([rng.random(4096), [0.0, 1.0, 0.5, 1e-12, 1 - 1e-12]])
    probe = probe.reshape(1, -1)
    expected = (cmap(probe)[:, :, :3] * 255).astype(np.uint8)
    got = explain_runtime._apply_jet(probe)
    check("_apply_jet matches matplotlib jet exactly", np.array_equal(expected, got),
          f"max abs diff {int(np.abs(expected.astype(int) - got.astype(int)).max())}")


def test_resnet_gradcam(images: List[torch.Tensor]) -> None:
    print("\n[2] ResNet-50 Grad-CAM: runtime vs analysis/explainability.py")
    model = load_model("resnet50")

    reference = GradCAM(model, model.backbone.layer4)
    worst = 0.0
    worst_argmax_gap = 0
    for x in images:
        model.eval()  # reference GradCAM is stochastic if dropout is left active
        ref, used_class = reference(x, target_class=None)
        got = generate_heatmap(model, x, "resnet50", target_class=None)
        worst = max(worst, float(np.abs(ref - got).max()))
        ref_peak = np.unravel_index(np.argmax(ref), ref.shape)
        got_peak = np.unravel_index(np.argmax(got), got.shape)
        worst_argmax_gap = max(
            worst_argmax_gap,
            abs(ref_peak[0] - got_peak[0]) + abs(ref_peak[1] - got_peak[1]),
        )
    reference.remove()

    check(f"Grad-CAM matches on {len(images)} images", worst <= ATOL_HEATMAP,
          f"max abs diff {worst:.3e} (tol {ATOL_HEATMAP:.0e})")
    check("peak pixel identical", worst_argmax_gap == 0,
          f"max peak displacement {worst_argmax_gap} px")

    # An explicit target_class must be honoured, and must change the map.
    x = images[0]
    maps = [generate_heatmap(model, x, "resnet50", target_class=c) for c in range(NUM_CLASSES)]
    distinct = any(np.abs(maps[0] - m).max() > 1e-4 for m in maps[1:])
    check("target_class changes the Grad-CAM", distinct,
          "class-conditional as expected" if distinct else "all four classes gave the same map")

    # Determinism: the app must not show a different picture on a re-run.
    a = generate_heatmap(model, x, "resnet50")
    b = generate_heatmap(model, x, "resnet50")
    check("deterministic across repeat calls", np.array_equal(a, b))

    # And it must stay deterministic after MC-Dropout has left dropout active.
    model.predict_with_uncertainty(x, T=2)
    c_ = generate_heatmap(model, x, "resnet50")
    check("deterministic after predict_with_uncertainty()", np.array_equal(a, c_),
          "" if np.array_equal(a, c_) else "dropout leaked into the heatmap")
    model.eval()

    # Range and shape contract.
    check("output shape == input spatial shape", a.shape == (x.shape[-2], x.shape[-1]), str(a.shape))
    check("output in [0,1]", float(a.min()) >= 0.0 and float(a.max()) <= 1.0,
          f"[{a.min():.4f}, {a.max():.4f}]")
    check("output dtype float32", a.dtype == np.float32, str(a.dtype))


def test_vit_rollout(images: List[torch.Tensor]) -> None:
    print("\n[3] ViT-B/16 attention rollout: runtime vs src/xai_and_stats.py")
    model = load_model("vit")

    worst = 0.0
    for x in images:
        ref = generate_attention_heatmap(
            model.backbone, x, device=DEVICE, head_fusion="mean", discard_ratio=0.9
        )
        got = generate_heatmap(model, x, "vit")
        worst = max(worst, float(np.abs(ref - got).max()))

    check(f"attention rollout matches on {len(images)} images", worst <= ATOL_HEATMAP,
          f"max abs diff {worst:.3e} (tol {ATOL_HEATMAP:.0e})")

    x = images[0]
    a = generate_heatmap(model, x, "vit")
    b = generate_heatmap(model, x, "vit", target_class=0)
    c_ = generate_heatmap(model, x, "vit", target_class=3)
    check("ViT rollout is class-agnostic (documented, not a bug)",
          np.array_equal(a, b) and np.array_equal(b, c_),
          "target_class has no effect, as the docstring states")
    check("deterministic across repeat calls", np.array_equal(a, generate_heatmap(model, x, "vit")))
    check("output in [0,1]", float(a.min()) >= 0.0 and float(a.max()) <= 1.0,
          f"[{a.min():.4f}, {a.max():.4f}]")

    # Forward hooks must be fully unpatched afterwards, or the app's normal
    # inference silently pays the slow-attention cost forever.
    patched = [
        blk for blk in model.backbone.encoder.layers
        if blk.self_attention.forward.__func__ is not type(blk.self_attention).forward
        if hasattr(blk.self_attention.forward, "__func__")
    ]
    unpatched = all(
        getattr(blk.self_attention.forward, "__qualname__", "").startswith("MultiheadAttention")
        for blk in model.backbone.encoder.layers
    )
    check("attention hooks removed after the call", unpatched,
          "" if unpatched else "model left monkey-patched")


def test_overlay(images: List[torch.Tensor]) -> None:
    print("\n[4] Overlay: runtime vs analysis/explainability.py::overlay_heatmap")
    model = load_model("resnet50")
    x = images[0]
    heat = generate_heatmap(model, x, "resnet50")
    rgb = denormalize_to_uint8(x[0])

    for alpha in (0.0, 0.25, 0.4, 0.45, 1.0):
        ref = reference_overlay(rgb, heat, alpha=alpha)
        got = overlay_heatmap(rgb, heat, alpha=alpha)
        check(f"overlay exact at alpha={alpha}", np.array_equal(ref, got),
              f"max abs diff {int(np.abs(ref.astype(int) - got.astype(int)).max())}")

    got = overlay_heatmap(rgb, heat)
    check("overlay shape/dtype", got.shape == rgb.shape and got.dtype == np.uint8,
          f"{got.shape} {got.dtype}")

    # Overlay a 224x224 heatmap onto a full-resolution scan.
    big = np.array(Image.fromarray(rgb).resize((512, 512), Image.BICUBIC))
    up = overlay_heatmap(big, heat)
    check("resizes heatmap to a larger scan", up.shape == (512, 512, 3), str(up.shape))

    grey = overlay_heatmap(rgb[:, :, 0], heat)
    check("accepts greyscale input", grey.shape == rgb.shape, str(grey.shape))


def test_speed(images: List[torch.Tensor], repeats: int = 10) -> None:
    print(f"\n[5] CPU speed ({torch.get_num_threads()} threads)")
    x = images[0]
    for kind in ("resnet50", "vit"):
        model = load_model(kind)
        for _ in range(2):  # warm up
            generate_heatmap(model, x, kind)
        timings = []
        for i in range(repeats):
            img = images[i % len(images)]
            t0 = time.perf_counter()
            generate_heatmap(model, img, kind)
            timings.append((time.perf_counter() - t0) * 1000.0)
        rgb = denormalize_to_uint8(x[0])
        heat = generate_heatmap(model, x, kind)
        t0 = time.perf_counter()
        for _ in range(repeats):
            overlay_heatmap(rgb, heat)
        overlay_ms = (time.perf_counter() - t0) * 1000.0 / repeats

        arr = np.array(timings)
        print(f"  {kind:9s} heatmap  median {np.median(arr):7.1f} ms   "
              f"mean {arr.mean():7.1f} ms   p90 {np.percentile(arr, 90):7.1f} ms")
        print(f"  {kind:9s} overlay  median {overlay_ms:7.1f} ms")
        del model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=8)
    parser.add_argument("--skip-speed", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    print(f"Consistency test — explain_runtime vs analysis/explainability.py")
    print(f"device=cpu  checkpoints={CHECKPOINTS}")

    images = fixed_image_batch(args.n_images)
    print(f"fixed image set: {len(images)} internal test-split images")

    test_jet_lut()
    test_resnet_gradcam(images)
    test_vit_rollout(images)
    test_overlay(images)
    if not args.skip_speed:
        test_speed(images)

    print("\n" + "=" * 70)
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILURE(S):")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("All consistency checks passed.")


if __name__ == "__main__":
    main()
