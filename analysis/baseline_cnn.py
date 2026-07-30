#!/usr/bin/env python3
"""
analysis/baseline_cnn.py

A small CNN trained FROM SCRATCH (no ImageNet pretraining) as a third,
weaker-prior baseline so the ViT-B/16 vs ResNet-50 story isn't the only
data point in the manuscript. This directly reuses src/code.py's training loop,
evaluation, calibration, risk-coverage, bootstrap-CI, and results-logging
machinery (run_multi_seed / run_single_seed / ResultsLogger) — nothing here
duplicates that logic. Only the architecture is new.

Output layout mirrors src/code.py's own results/<ts>/<model>/seed_<n>/ schema
exactly (config.json, epoch_metrics.csv, confusion_matrix.png,
calibration.png, risk_coverage.png, summary.json), just rooted under
analysis/results/baseline/ instead of results/, so a manuscript session can
fold it into the same table trivially.

Usage:
    python analysis/baseline_cnn.py [--epochs 25] [--seeds 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import (  # noqa: E402
    DEVICE,
    NUM_CLASSES,
    RESULTS_OUT_DIR,
    REPO_ROOT,
    all_seed_dirs,
    find_latest_run,
    get_manifest,
    is_smoke_run,
)
from code import mcnemar_test, run_multi_seed  # noqa: E402


class SimpleCNN(nn.Module):
    """4-block conv/BN/ReLU/pool CNN, trained from scratch (no pretrained
    weights). ~420K parameters — roughly 60x smaller than ResNet-50's
    trainable head+layer4 and 34x smaller than the ViT's trainable blocks,
    with no ImageNet prior at all. Serves as a "how much is transfer
    learning buying us" reference point.

    Mirrors BrainTumorViT / BrainTumorResNet50's public interface
    (forward, predict_with_uncertainty via MC-Dropout) so it drops directly
    into src/code.py's run_single_seed / run_multi_seed / evaluate_with_uncertainty.
    """

    def __init__(self, num_classes: int = 4, dropout_p: float = 0.3) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_classes),
        )

        n_params = sum(p.numel() for p in self.parameters())
        import logging
        logging.getLogger(__name__).info("SimpleCNN (from scratch) | params: %s", f"{n_params:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

    @staticmethod
    def _activate_dropout(model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, T: int = 20, epsilon: float = 1e-10) -> Dict[str, torch.Tensor]:
        self.eval()
        self._activate_dropout(self)
        sample_probs: List[torch.Tensor] = []
        for _ in range(T):
            logits = self(x)
            sample_probs.append(torch.softmax(logits, dim=-1))
        stacked = torch.stack(sample_probs, dim=0)
        mean_probs = stacked.mean(dim=0)
        entropy = -(mean_probs * torch.log2(mean_probs + epsilon)).sum(dim=-1)
        std_probs = stacked.std(dim=0)
        return {
            "predictions": mean_probs.argmax(dim=-1),
            "mean_probs": mean_probs,
            "entropy": entropy,
            "std_probs": std_probs,
            "all_probs": stacked,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--mc-T", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="Default: reuse whichever seeds exist for vit/resnet50 in the latest run.")
    args = parser.parse_args()

    manifest_df = get_manifest()

    seeds = args.seeds
    reference_run = find_latest_run()
    reference_smoke = None
    if seeds is None:
        if reference_run is not None:
            vit_seed_dirs = all_seed_dirs(reference_run, "vit")
            seeds = [int(d.name.split("_")[1]) for d in vit_seed_dirs]
            reference_smoke = any(is_smoke_run(d) for d in vit_seed_dirs)
        if not seeds:
            seeds = [42]
    print(f"Training SimpleCNN-from-scratch baseline for seeds={seeds}")
    if reference_smoke:
        print("WARNING: reference ViT/ResNet50 run is a --smoke test run. This baseline "
              "will be trained for real (full epochs), so any head-to-head comparison "
              "against the current results/ checkpoints is NOT apples-to-apples until "
              "Session A's full run completes.")

    results_dir = RESULTS_OUT_DIR / "baseline"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs: Dict[str, Any] = dict(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        num_workers=args.workers,
        mc_T=args.mc_T,
        early_stop_patience=args.early_stop_patience,
    )

    summary_df, models = run_multi_seed(
        manifest_df, SimpleCNN, "simplecnn", results_dir, seeds=seeds, **train_kwargs,
    )

    # Head-to-head McNemar vs. whatever ViT/ResNet50 checkpoints exist for the
    # same seeds, so the comparison is ready to go the moment it's apples-to-apples.
    comparison_rows: List[Dict[str, Any]] = []
    if reference_run is not None:
        from common import load_model_from_seed_dir, build_dataloaders_from_manifest
        from code import evaluate_with_uncertainty

        test_loader = build_dataloaders_from_manifest(manifest_df, batch_size=32, num_workers=0).get("test")
        if test_loader is not None:
            for seed in seeds:
                if seed not in models:
                    continue
                cnn_mc = evaluate_with_uncertainty(models[seed], test_loader, DEVICE, T=args.mc_T)
                y_true = cnn_mc["labels"].numpy()
                cnn_preds = cnn_mc["predictions"].numpy()
                cnn_acc = float((cnn_preds == y_true).mean())

                for other_model in ("vit", "resnet50"):
                    try:
                        seed_dir = reference_run / other_model / f"seed_{seed}"
                        other_model_obj, _ = load_model_from_seed_dir(other_model, seed_dir, device=DEVICE)
                    except FileNotFoundError:
                        continue
                    other_mc = evaluate_with_uncertainty(other_model_obj, test_loader, DEVICE, T=args.mc_T)
                    other_preds = other_mc["predictions"].numpy()
                    other_acc = float((other_preds == y_true).mean())
                    stat, pval = mcnemar_test(y_true, cnn_preds, other_preds)
                    comparison_rows.append({
                        "seed": seed,
                        "simplecnn_acc": cnn_acc,
                        "other_model": other_model,
                        "other_model_acc": other_acc,
                        "mcnemar_chi2": stat,
                        "mcnemar_p": pval,
                        "significant": bool(pval < 0.05),
                        "reference_run_is_smoke": bool(reference_smoke),
                    })
                    del other_model_obj

    comparison = {
        "baseline_summary": summary_df.to_dict(orient="list"),
        "reference_run": reference_run.name if reference_run else None,
        "reference_run_is_smoke": bool(reference_smoke) if reference_smoke is not None else None,
        "mcnemar_vs_reference": comparison_rows,
    }
    out_json = results_dir / "baseline_comparison.json"
    with open(out_json, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nComparison JSON -> {out_json}")

    md_lines = ["# SimpleCNN-from-scratch Baseline", ""]
    if reference_smoke:
        md_lines += [
            "> **WARNING: the ViT/ResNet50 checkpoints used for the head-to-head "
            "McNemar comparison below are from a --smoke test run (1 epoch, "
            "8 samples/class) and are near-chance. The comparison rows are a "
            "pipeline check only. Re-run this script once Session A's full "
            "multi-seed training completes for a real comparison.**",
            "",
        ]
    md_lines.append("## Baseline (trained from scratch, full run — not a smoke test)")
    md_lines.append("")
    for _, row in summary_df.iterrows():
        md_lines.append(
            f"- seed {int(row['seed'])}: test_acc={row.get('test_acc'):.4f}  "
            f"mc_acc={row.get('mc_acc'):.4f}  macro_f1={row.get('macro_f1'):.4f}  "
            f"macro_auc={row.get('macro_auc'):.4f}  ece={row.get('ece'):.4f}  "
            f"brier={row.get('brier'):.4f}"
        )
    md_lines.append("")
    md_lines.append("## McNemar vs. ViT / ResNet-50 (same seed, same test set)")
    md_lines.append("")
    if comparison_rows:
        md_lines.append("| seed | SimpleCNN acc | vs | other acc | chi2 | p | significant |")
        md_lines.append("|---|---|---|---|---|---|---|")
        for row in comparison_rows:
            md_lines.append(
                f"| {row['seed']} | {row['simplecnn_acc']:.4f} | {row['other_model']} | "
                f"{row['other_model_acc']:.4f} | {row['mcnemar_chi2']:.4f} | "
                f"{row['mcnemar_p']:.4g} | {row['significant']} |"
            )
    else:
        md_lines.append("(no reference checkpoints available for comparison)")

    out_md = results_dir / "baseline_comparison.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Comparison Markdown -> {out_md}")


if __name__ == "__main__":
    main()
