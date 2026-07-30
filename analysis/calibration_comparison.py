#!/usr/bin/env python3
"""
analysis/calibration_comparison.py

Post-hoc temperature scaling (Guo et al. 2017) as a second calibration method
alongside the MC-Dropout ECE/Brier already computed by src/code.py's main
pipeline. Fits a single scalar T per model on held-out VALIDATION logits by
minimizing NLL, then reports ECE / Brier / NLL / reliability diagrams on the
TEST split BEFORE and AFTER scaling.

Scope note (read before citing these numbers): temperature scaling here is
applied to the DETERMINISTIC single-forward-pass logits (dropout OFF), which
is the standard setting in the calibration literature and is what a single
non-MC-Dropout inference call from this model would return. It is a
complementary calibration check to the MC-Dropout mean-probability ECE/Brier
that src/code.py already reports in summary.json — it does not replace or
recompute that number, and the two are not directly comparable because one
is calibrating a point-estimate softmax and the other an averaged predictive
distribution over T stochastic passes.

Usage:
    python analysis/calibration_comparison.py
"""

from __future__ import annotations

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CLASS_NAMES,
    DEVICE,
    RESULTS_OUT_DIR,
    all_seed_dirs,
    build_dataloaders_from_manifest,
    compute_calibration_metrics,
    find_latest_run,
    get_manifest,
    load_model_from_seed_dir,
    run_provenance,
)

OUT_DIR = RESULTS_OUT_DIR / "calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def collect_deterministic_logits(
    model: nn.Module, loader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single forward pass per batch, dropout OFF (model.eval(), no MC)."""
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.clone())
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> Dict[str, float]:
    """Fit a single scalar T minimizing NLL on (logits, labels) via LBFGS.

    Returns dict with fitted T and pre/post NLL on the fitting set itself
    (val NLL before/after — the val-set analogue of the test-set numbers
    reported by evaluate_before_after below).
    """
    logits = logits.detach()
    labels = labels.detach()
    nll_criterion = nn.CrossEntropyLoss()

    nll_before = float(nll_criterion(logits, labels).item())

    T = nn.Parameter(torch.ones(1) * 1.5)
    # line_search_fn="strong_wolfe" is required here, not optional: plain LBFGS
    # takes an unguarded quasi-Newton step scaled only by lr, which for this
    # 1-D problem reliably diverges to the T.clamp(min=1e-3) floor instead of
    # converging (verified: without it, every seed/model fit landed on
    # T=0.001 and *increased* val NLL by 250-400x, the opposite of what
    # temperature scaling should do). This is the same fix used in the
    # reference temperature-scaling implementation (Guo et al. 2017).
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=max_iter, line_search_fn="strong_wolfe")

    def _closure():
        optimizer.zero_grad()
        loss = nll_criterion(logits / T.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    optimizer.step(_closure)
    T_fitted = float(T.detach().clamp(min=1e-3).item())

    nll_after = float(nll_criterion(logits / T_fitted, labels).item())

    return {"temperature": T_fitted, "val_nll_before": nll_before, "val_nll_after": nll_after}


def evaluate_before_after(
    logits: torch.Tensor, labels: torch.Tensor, T: float
) -> Dict[str, Any]:
    labels_np = labels.numpy()

    probs_before = F.softmax(logits, dim=-1).numpy()
    probs_after = F.softmax(logits / T, dim=-1).numpy()

    nll_criterion = nn.CrossEntropyLoss()
    nll_before = float(nll_criterion(logits, labels).item())
    nll_after = float(nll_criterion(logits / T, labels).item())

    cal_before = compute_calibration_metrics(probs_before, labels_np, n_bins=15)
    cal_after = compute_calibration_metrics(probs_after, labels_np, n_bins=15)

    acc = float((probs_before.argmax(axis=1) == labels_np).mean())  # T scaling doesn't change argmax/accuracy

    return {
        "test_accuracy": acc,
        "before": {"ece": cal_before["ece"], "brier": cal_before["brier"], "nll": nll_before},
        "after": {"ece": cal_after["ece"], "brier": cal_after["brier"], "nll": nll_after},
        "probs_before": probs_before,
        "probs_after": probs_after,
        "labels": labels_np,
    }


def reliability_diagram(
    probs: np.ndarray, labels: np.ndarray, title: str, save_path: Path, n_bins: int = 15
) -> None:
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    acc_bins, conf_bins, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        n = int(in_bin.sum())
        counts.append(n)
        acc_bins.append(correct[in_bin].mean() if n > 0 else 0.0)
        conf_bins.append(confidence[in_bin].mean() if n > 0 else (lo + hi) / 2)

    ece = compute_calibration_metrics(probs, labels, n_bins=n_bins)["ece"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.bar(bins[:-1], acc_bins, width=1 / n_bins, align="edge", alpha=0.7,
           edgecolor="black", linewidth=0.3, label="Accuracy per bin")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def run_for_model(model_name: str, run_dir: Path, manifest_df) -> Dict[str, Any]:
    seed_dirs = all_seed_dirs(run_dir, model_name)
    if not seed_dirs:
        return {"error": f"no seed dirs for {model_name}"}

    loaders = build_dataloaders_from_manifest(manifest_df, batch_size=32, num_workers=0)
    val_loader, test_loader = loaders.get("val"), loaders.get("test")
    if val_loader is None or test_loader is None:
        return {"error": "manifest missing val/test split"}

    per_seed_results: List[Dict[str, Any]] = []

    for seed_dir in seed_dirs:
        model, ckpt_path = load_model_from_seed_dir(model_name, seed_dir, device=DEVICE)
        provenance = run_provenance(run_dir, seed_dir)

        val_logits, val_labels = collect_deterministic_logits(model, val_loader, DEVICE)
        test_logits, test_labels = collect_deterministic_logits(model, test_loader, DEVICE)

        fit = fit_temperature(val_logits, val_labels)
        T = fit["temperature"]

        result = evaluate_before_after(test_logits, test_labels, T)

        seed_tag = seed_dir.name
        reliability_diagram(
            result["probs_before"], result["labels"],
            f"{model_name.upper()} [{seed_tag}] — Before Temperature Scaling",
            OUT_DIR / f"{model_name}_{seed_tag}_reliability_before.png",
        )
        reliability_diagram(
            result["probs_after"], result["labels"],
            f"{model_name.upper()} [{seed_tag}] — After Temperature Scaling (T={T:.3f})",
            OUT_DIR / f"{model_name}_{seed_tag}_reliability_after.png",
        )

        per_seed_results.append({
            "seed_dir": seed_tag,
            "checkpoint": str(ckpt_path.relative_to(Path(__file__).resolve().parent.parent)),
            "provenance": provenance,
            "n_val": int(len(val_labels)),
            "n_test": int(len(test_labels)),
            "temperature": T,
            "val_nll_before": fit["val_nll_before"],
            "val_nll_after": fit["val_nll_after"],
            "test_accuracy": result["test_accuracy"],
            "test_ece_before": result["before"]["ece"],
            "test_ece_after": result["after"]["ece"],
            "test_brier_before": result["before"]["brier"],
            "test_brier_after": result["after"]["brier"],
            "test_nll_before": result["before"]["nll"],
            "test_nll_after": result["after"]["nll"],
            "ece_improved": result["after"]["ece"] < result["before"]["ece"],
            "brier_improved": result["after"]["brier"] < result["before"]["brier"],
        })

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {"model": model_name, "per_seed": per_seed_results}


def main() -> None:
    run_dir = find_latest_run()
    if run_dir is None:
        print("ERROR: no completed run (results/<ts>/comparison_summary.json) found. "
              "Run src/code.py first (Session A's job).", file=sys.stderr)
        sys.exit(1)

    print(f"Using run: {run_dir}")
    manifest_df = get_manifest()

    all_results: Dict[str, Any] = {"run_dir": str(run_dir.name), "models": {}}

    for model_name in ("vit", "resnet50"):
        print(f"\n=== {model_name} ===")
        res = run_for_model(model_name, run_dir, manifest_df)
        all_results["models"][model_name] = res
        for row in res.get("per_seed", []):
            print(
                f"  [{row['seed_dir']}] T={row['temperature']:.3f}  "
                f"ECE {row['test_ece_before']:.4f} -> {row['test_ece_after']:.4f}  "
                f"Brier {row['test_brier_before']:.4f} -> {row['test_brier_after']:.4f}  "
                f"NLL {row['test_nll_before']:.4f} -> {row['test_nll_after']:.4f}"
            )

    out_json = OUT_DIR / "calibration_comparison.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON -> {out_json}")

    # Markdown summary table
    md_lines = [
        "# Temperature Scaling — Calibration Comparison",
        "",
        f"Run: `{run_dir.name}`",
        "",
    ]
    any_smoke = False
    for model_name, res in all_results["models"].items():
        if "error" in res:
            md_lines.append(f"## {model_name}: ERROR — {res['error']}")
            continue
        md_lines.append(f"## {model_name}")
        md_lines.append("")
        md_lines.append("| seed | n_test | T | ECE before | ECE after | Brier before | Brier after | NLL before | NLL after |")
        md_lines.append("|---|---|---|---|---|---|---|---|---|")
        for row in res["per_seed"]:
            if row["provenance"]["is_smoke_test_run"]:
                any_smoke = True
            md_lines.append(
                f"| {row['seed_dir']} | {row['n_test']} | {row['temperature']:.3f} | "
                f"{row['test_ece_before']:.4f} | {row['test_ece_after']:.4f} | "
                f"{row['test_brier_before']:.4f} | {row['test_brier_after']:.4f} | "
                f"{row['test_nll_before']:.4f} | {row['test_nll_after']:.4f} |"
            )
        md_lines.append("")

    if any_smoke:
        md_lines.insert(2, "> **WARNING: at least one model above ran from a --smoke test checkpoint "
                            "(1 epoch, 8 samples/class). These numbers validate the pipeline only and "
                            "must NOT be cited as real calibration results. Re-run after Session A's "
                            "full multi-seed training completes.**")
        md_lines.insert(3, "")

    md_lines += [
        "## Method",
        "",
        "Temperature T is fit by minimizing NLL on VALIDATION-split deterministic "
        "(dropout-off, single forward pass) logits via LBFGS, following Guo et al. "
        "(2017), *On Calibration of Modern Neural Networks*. T is then applied to "
        "TEST-split logits and ECE (15-bin)/Brier/NLL are recomputed. Temperature "
        "scaling is a monotonic rescaling of logits and therefore never changes "
        "argmax predictions — test accuracy is identical before/after.",
        "",
        "This calibrates the deterministic (non-MC-Dropout) softmax output. It is "
        "complementary to, not a replacement for, the MC-Dropout mean-probability "
        "ECE/Brier already reported in `results/*/*/summary.json` by the main "
        "training pipeline — those numbers reflect a different (averaged, "
        "stochastic) predictive distribution and are not directly comparable to "
        "the before/after pair above.",
    ]

    out_md = OUT_DIR / "calibration_comparison.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown -> {out_md}")


if __name__ == "__main__":
    main()
