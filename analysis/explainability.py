#!/usr/bin/env python3
"""
analysis/explainability.py

Visual explainability for both architectures, as required for a clinical
imaging submission:
  - ResNet-50 : Grad-CAM (Selvaraju et al. 2017) on the layer4 feature map.
  - ViT-B/16  : Attention Rollout (Abnar & Zuidema 2020), reusing the
                existing implementation in src/xai_and_stats.py rather than
                duplicating it.

For each of the 4 classes we save heatmap overlays for:
  - up to 2 confidently-correct examples (lowest MC-Dropout entropy, correct)
  - up to 2 misclassified examples found in the sampled candidate pool
  - up to 2 high-entropy examples (highest MC-Dropout entropy, regardless of
    correctness)
(sets may overlap and are de-duplicated; an example is only rendered once
but its metadata records every category it qualified for)

Candidate pool: per class, a fixed-seed random sample of the test split
(default 80/class) is scored with MC-Dropout (T=20) to find these examples,
rather than scoring the full test set, to keep runtime bounded on CPU.

Usage:
    python analysis/explainability.py [--per-class-pool 80] [--mc-T 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import (  # noqa: E402
    CLASS_NAMES,
    DEVICE,
    RESULTS_OUT_DIR,
    REPO_ROOT,
    all_seed_dirs,
    find_latest_run,
    get_manifest,
    get_transforms,
    load_model_from_seed_dir,
    run_provenance,
)
from xai_and_stats import generate_attention_heatmap, visualize_attention  # noqa: E402

OUT_DIR = RESULTS_OUT_DIR / "explainability"
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize_to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """[3,H,W] normalized tensor -> [H,W,3] uint8 RGB."""
    arr = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * _IMAGENET_STD + _IMAGENET_MEAN
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


# =============================================================================
# Grad-CAM (ResNet-50)
# =============================================================================

class GradCAM:
    """Grad-CAM against a single target module (ResNet-50 layer4).

    Standard implementation: forward hook captures activations A [B,C,h,w];
    backward hook captures d(logit_target)/dA. CAM = ReLU(sum_c alpha_c * A_c)
    where alpha_c = global-average-pooled gradient for channel c.
    """

    def __init__(self, model: nn.Module, target_module: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._fwd_handle = target_module.register_forward_hook(self._save_activation)
        self._bwd_handle = target_module.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, image_tensor: torch.Tensor, target_class: int | None = None) -> Tuple[np.ndarray, int]:
        """image_tensor: [1,3,H,W]. Returns (cam [H,W] in [0,1], target_class used)."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)  # [1, C]
        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())
        score = logits[0, target_class]
        score.backward()

        acts = self.activations[0]      # [C, h, w]
        grads = self.gradients[0]       # [C, h, w]
        alpha = grads.mean(dim=(1, 2))  # [C]
        cam = torch.relu((alpha[:, None, None] * acts).sum(dim=0))  # [h, w]
        cam = cam.cpu().numpy()

        lo, hi = cam.min(), cam.max()
        cam = (cam - lo) / (hi - lo + 1e-8)

        H, W = image_tensor.shape[-2], image_tensor.shape[-1]
        cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC))
        cam_resized = cam_resized.astype(np.float32) / 255.0
        return cam_resized, target_class


def overlay_heatmap(img_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    colored = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    overlay = (img_uint8.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_panel(img_uint8: np.ndarray, heatmap: np.ndarray, meta: Dict[str, Any], save_path: Path) -> None:
    overlay = overlay_heatmap(img_uint8, heatmap)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.suptitle(
        f"{meta['method']} | true={meta['true_class']} pred={meta['pred_class']} "
        f"{'CORRECT' if meta['correct'] else 'MISCLASSIFIED'} | entropy={meta['entropy']:.3f} bits",
        fontsize=10, fontweight="bold",
    )
    axes[0].imshow(img_uint8)
    axes[0].set_title("Input MRI", fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"{meta['method']} overlay", fontsize=10)
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Candidate selection via MC-Dropout on a bounded per-class pool
# =============================================================================

def select_examples_for_model(
    model: nn.Module,
    manifest_df,
    per_class_pool: int,
    mc_T: int,
    device: torch.device,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    test_df = manifest_df[manifest_df["split"] == "test"].reset_index(drop=True)
    transform = get_transforms("test")

    selected: List[Dict[str, Any]] = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_rows = test_df[test_df["label"] == class_idx]
        n_pool = min(per_class_pool, len(class_rows))
        pool = class_rows.sample(n=n_pool, random_state=int(rng.integers(0, 1_000_000)))

        images, filepaths = [], []
        for _, row in pool.iterrows():
            img = Image.open(row["filepath"]).convert("RGB")
            images.append(transform(img))
            filepaths.append(row["filepath"])
        batch = torch.stack(images).to(device)

        entropies: List[float] = []
        preds: List[int] = []
        with torch.no_grad():
            bs = 16
            for i in range(0, len(batch), bs):
                chunk = batch[i:i + bs]
                out = model.predict_with_uncertainty(chunk, T=mc_T)
                entropies.extend(out["entropy"].cpu().tolist())
                preds.extend(out["predictions"].cpu().tolist())

        candidates = []
        for i, fp in enumerate(filepaths):
            candidates.append({
                "filepath": fp,
                "true_class_idx": class_idx,
                "true_class": class_name,
                "pred_class_idx": preds[i],
                "pred_class": CLASS_NAMES[preds[i]],
                "correct": preds[i] == class_idx,
                "entropy": float(entropies[i]),
            })

        correct_sorted = sorted([c for c in candidates if c["correct"]], key=lambda c: c["entropy"])
        misclassified = [c for c in candidates if not c["correct"]]
        misclassified_sorted = sorted(misclassified, key=lambda c: -c["entropy"])
        high_entropy_sorted = sorted(candidates, key=lambda c: -c["entropy"])

        picks: Dict[str, Dict[str, Any]] = {}
        for c in correct_sorted[:2]:
            picks.setdefault(c["filepath"], {**c, "categories": []})
            picks[c["filepath"]]["categories"].append("confident_correct")
        for c in misclassified_sorted[:2]:
            picks.setdefault(c["filepath"], {**c, "categories": []})
            picks[c["filepath"]]["categories"].append("misclassified")
        for c in high_entropy_sorted[:2]:
            picks.setdefault(c["filepath"], {**c, "categories": []})
            picks[c["filepath"]]["categories"].append("high_entropy")

        selected.append({
            "class": class_name,
            "pool_size": n_pool,
            "n_misclassified_in_pool": len(misclassified),
            "examples": list(picks.values()),
        })

    return selected


# =============================================================================
# Per-model runners
# =============================================================================

def run_resnet_gradcam(model: nn.Module, selections: List[Dict[str, Any]], out_dir: Path) -> List[Dict[str, Any]]:
    target_module = model.backbone.layer4
    cam = GradCAM(model, target_module)
    transform = get_transforms("test")
    saved: List[Dict[str, Any]] = []

    for class_block in selections:
        for ex in class_block["examples"]:
            img = Image.open(ex["filepath"]).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            heatmap, used_class = cam(tensor, target_class=ex["pred_class_idx"])
            img_uint8 = denormalize_to_uint8(tensor[0])

            fname = f"resnet50_{class_block['class']}_{Path(ex['filepath']).stem}.png"
            save_path = out_dir / fname
            meta = {
                "method": "Grad-CAM",
                "true_class": ex["true_class"], "pred_class": ex["pred_class"],
                "correct": ex["correct"], "entropy": ex["entropy"],
            }
            save_panel(img_uint8, heatmap, meta, save_path)

            saved.append({**ex, "heatmap_file": str(save_path.relative_to(REPO_ROOT)), "method": "Grad-CAM"})

    cam.remove()
    return saved


def run_vit_attention_rollout(model: nn.Module, selections: List[Dict[str, Any]], out_dir: Path) -> List[Dict[str, Any]]:
    vit_backbone = model.backbone  # raw torchvision ViT — xai_and_stats expects .encoder.layers
    transform = get_transforms("test")
    saved: List[Dict[str, Any]] = []

    for class_block in selections:
        for ex in class_block["examples"]:
            img = Image.open(ex["filepath"]).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            heatmap = generate_attention_heatmap(vit_backbone, tensor, device=DEVICE, head_fusion="mean", discard_ratio=0.9)
            img_uint8 = denormalize_to_uint8(tensor[0])

            fname = f"vit_{class_block['class']}_{Path(ex['filepath']).stem}.png"
            save_path = out_dir / fname
            meta = {
                "method": "Attention Rollout",
                "true_class": ex["true_class"], "pred_class": ex["pred_class"],
                "correct": ex["correct"], "entropy": ex["entropy"],
            }
            save_panel(img_uint8, heatmap, meta, save_path)

            saved.append({**ex, "heatmap_file": str(save_path.relative_to(REPO_ROOT)), "method": "Attention Rollout"})

    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class-pool", type=int, default=80)
    parser.add_argument("--mc-T", type=int, default=20)
    args = parser.parse_args()

    run_dir = find_latest_run()
    if run_dir is None:
        print("ERROR: no completed run found under results/.", file=sys.stderr)
        sys.exit(1)
    print(f"Using run: {run_dir}")

    manifest_df = get_manifest()

    resnet_dir = OUT_DIR / "resnet50"
    vit_dir = OUT_DIR / "vit"
    resnet_dir.mkdir(parents=True, exist_ok=True)
    vit_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {"run_dir": run_dir.name, "models": {}}

    for model_name in ("resnet50", "vit"):
        seed_dirs = all_seed_dirs(run_dir, model_name)
        if not seed_dirs:
            summary["models"][model_name] = {"error": "no seed dirs"}
            continue
        seed_dir = seed_dirs[0]  # explainability examples from one representative seed
        model, ckpt_path = load_model_from_seed_dir(model_name, seed_dir, device=DEVICE)
        provenance = run_provenance(run_dir, seed_dir)

        print(f"\n=== {model_name} [{seed_dir.name}] — selecting candidate examples "
              f"(pool={args.per_class_pool}/class, MC-T={args.mc_T}) ===")
        selections = select_examples_for_model(model, manifest_df, args.per_class_pool, args.mc_T, DEVICE)
        for block in selections:
            print(f"  {block['class']}: pool={block['pool_size']}  "
                  f"misclassified_in_pool={block['n_misclassified_in_pool']}  "
                  f"examples_saved={len(block['examples'])}")

        if model_name == "resnet50":
            saved = run_resnet_gradcam(model, selections, resnet_dir)
        else:
            saved = run_vit_attention_rollout(model, selections, vit_dir)

        summary["models"][model_name] = {
            "provenance": provenance,
            "checkpoint": str(ckpt_path.relative_to(REPO_ROOT)),
            "per_class_pool": args.per_class_pool,
            "mc_T": args.mc_T,
            "selections": selections,
            "n_heatmaps_saved": len(saved),
        }

        del model

    out_json = OUT_DIR / "explainability_summary.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nJSON -> {out_json}")

    md_lines = [
        "# Explainability — Grad-CAM (ResNet-50) & Attention Rollout (ViT-B/16)",
        "",
        f"Run: `{run_dir.name}`",
        "",
    ]
    any_smoke = any(
        summary["models"].get(m, {}).get("provenance", {}).get("is_smoke_test_run")
        for m in ("resnet50", "vit")
    )
    if any_smoke:
        md_lines += [
            "> **WARNING: at least one model ran from a --smoke test checkpoint "
            "(1 epoch, 8 samples/class). The resulting model is near-chance and "
            "its heatmaps/predictions below are pipeline-validation only, not "
            "real clinical explanations. Re-run after Session A's full "
            "multi-seed training completes.**",
            "",
        ]
    for model_name in ("resnet50", "vit"):
        block = summary["models"].get(model_name, {})
        if "error" in block:
            md_lines.append(f"## {model_name}: ERROR — {block['error']}")
            continue
        method = "Grad-CAM" if model_name == "resnet50" else "Attention Rollout"
        md_lines.append(f"## {model_name} ({method})")
        md_lines.append("")
        for cls_block in block["selections"]:
            md_lines.append(
                f"- **{cls_block['class']}**: pool={cls_block['pool_size']}, "
                f"misclassified_in_pool={cls_block['n_misclassified_in_pool']}, "
                f"{len(cls_block['examples'])} example(s) saved"
            )
        md_lines.append("")
    md_lines.append(f"Heatmap PNGs saved under `analysis/results/explainability/{{resnet50,vit}}/`.")

    out_md = OUT_DIR / "explainability_summary.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown -> {out_md}")


if __name__ == "__main__":
    main()
