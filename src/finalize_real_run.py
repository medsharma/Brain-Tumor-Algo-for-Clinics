#!/usr/bin/env python3
"""
src/finalize_real_run.py

One-time closer for the real (non-smoke) 5-seed x 2-backbone run at
results/20260703_155524, whose training completed (all 10 checkpoints +
summary_{vit,resnet50}.csv exist per resume_run.log) but whose final
aggregation step (per-seed McNemar + comparison_summary.json) was cut off
mid-way (log stops after seed=7's McNemar, before seed=2024/31).

This does NOT retrain anything. It only re-loads the already-trained
checkpoints and runs evaluate_with_uncertainty (forward-pass only) to get
fresh predictions for the McNemar test, and to emit the per-seed
predictions.json files figures/README.md documents as the input every
manuscript figure script needs. Per-seed scalar metrics (test_acc, macro_f1,
ece, etc.) are taken directly from the existing summary_{vit,resnet50}.csv
(already computed during the real training run) rather than recomputed, so
they match exactly what's on disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from code import (  # noqa: E402
    CLASS_NAMES,
    DEVICE,
    NUM_CLASSES,
    BrainTumorResNet50,
    BrainTumorViT,
    build_dataloaders_from_manifest,
    build_split_manifest,
    evaluate_with_uncertainty,
    mcnemar_test,
)

RUN_DIR = REPO_ROOT / "results" / "20260703_155524"
ANALYSIS_RESULTS_DIR = REPO_ROOT / "analysis" / "results"
MODEL_CLASSES = {"vit": BrainTumorViT, "resnet50": BrainTumorResNet50}


def load_checkpoint(model_name: str, seed: int) -> torch.nn.Module:
    seed_dir = RUN_DIR / model_name / f"seed_{seed}"
    ckpt_files = list(seed_dir.glob(f"best_{model_name}_seed*.pth"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint in {seed_dir}")
    model = MODEL_CLASSES[model_name](num_classes=NUM_CLASSES).to(DEVICE)
    ckpt = torch.load(ckpt_files[0], map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def provenance_for(seed: int, model_name: str) -> dict:
    cfg_path = RUN_DIR / model_name / f"seed_{seed}" / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    smoke = cfg.get("num_epochs") == 1 and cfg.get("mc_T") == 2
    return {
        "run_dir": str(RUN_DIR.relative_to(REPO_ROOT)),
        "seed_dir": str((RUN_DIR / model_name / f"seed_{seed}").relative_to(REPO_ROOT)),
        "is_smoke_test_run": smoke,
        "caveat": "Real training run (not a smoke test).",
        "config": cfg,
    }


def main() -> None:
    print(f"Run dir: {RUN_DIR}")
    assert RUN_DIR.exists(), f"missing {RUN_DIR}"

    manifest_path = REPO_ROOT / "data" / "split_manifest.csv"
    manifest_df = build_split_manifest(
        raw_root=str(REPO_ROOT / "data" / "brain_tumor"),
        manifest_path=str(manifest_path),
        force_rebuild=False,
    )
    test_loader = build_dataloaders_from_manifest(manifest_df, batch_size=32, num_workers=0).get("test")
    assert test_loader is not None, "manifest has no test split"

    seeds = [42, 123, 7, 2024, 31]
    ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_preds: dict[tuple[str, int], dict] = {}

    for model_name in ("vit", "resnet50"):
        for seed in seeds:
            print(f"\n=== evaluating {model_name} seed={seed} (T=20 MC Dropout) ===")
            model = load_checkpoint(model_name, seed)
            r = evaluate_with_uncertainty(model, test_loader, DEVICE, T=20)
            raw_preds[(model_name, seed)] = r

            prov = provenance_for(seed, model_name)
            out = {
                "model": model_name,
                "seed": seed,
                "class_names": CLASS_NAMES,
                "mc_T": 20,
                "provenance": prov,
                "y_true": r["labels"].tolist(),
                "y_pred": r["predictions"].tolist(),
                "mean_probs": r["mean_probs"].tolist(),
                "entropy": r["entropy"].tolist(),
            }
            out_path = ANALYSIS_RESULTS_DIR / f"{model_name}_seed{seed}_predictions.json"
            with open(out_path, "w") as f:
                json.dump(out, f)
            acc = float((r["predictions"] == r["labels"]).float().mean())
            print(f"  acc={acc:.4f}  -> {out_path}")

            del model

    # Per-seed McNemar (ViT vs ResNet-50, same seed, same test set)
    per_seed_mcnemar = []
    for seed in seeds:
        vit_r = raw_preds[("vit", seed)]
        rn_r = raw_preds[("resnet50", seed)]
        y_true = vit_r["labels"].numpy()
        preds_vit = vit_r["predictions"].numpy()
        preds_rn = rn_r["predictions"].numpy()

        stat, pval = mcnemar_test(y_true, preds_vit, preds_rn)
        vit_acc = float((preds_vit == y_true).mean())
        rn_acc = float((preds_rn == y_true).mean())
        direction = "vit_better" if vit_acc > rn_acc else "resnet50_better"

        per_seed_mcnemar.append({
            "seed": seed,
            "chi2": stat,
            "p_value": pval,
            "significant": bool(pval < 0.05),
            "direction": direction,
            "vit_acc": vit_acc,
            "rn_acc": rn_acc,
        })
        print(f"McNemar [seed={seed}]  chi2={stat:.4f}  p={pval:.4f}  dir={direction}  (vit={vit_acc:.4f}  rn={rn_acc:.4f})")

    n_sig = sum(r["significant"] for r in per_seed_mcnemar)
    sig_dirs = [r["direction"] for r in per_seed_mcnemar if r["significant"]]
    if sig_dirs:
        dominant = max(set(sig_dirs), key=sig_dirs.count)
        consistent = len(set(sig_dirs)) == 1
        print(f"McNemar summary: {n_sig}/{len(per_seed_mcnemar)} seeds significant | dominant: {dominant} | consistent: {consistent}")
    else:
        print(f"McNemar summary: 0/{len(per_seed_mcnemar)} seeds significant (p<0.05).")

    # Per-seed scalar metrics: reuse exactly what's already on disk from the real run
    vit_summary_df = pd.read_csv(RUN_DIR / "summary_vit.csv")
    rn_summary_df = pd.read_csv(RUN_DIR / "summary_resnet50.csv")

    comparison = {
        "vit": vit_summary_df.to_dict(orient="list"),
        "resnet50": rn_summary_df.to_dict(orient="list"),
        "mcnemar_per_seed": per_seed_mcnemar,
    }
    out_path = RUN_DIR / "comparison_summary.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
