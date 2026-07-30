#!/usr/bin/env python3
"""
figures/roc_curves.py

Per-class one-vs-rest ROC curves + macro-average, one panel per model, pooled
across all seeds found by --analysis-glob.

Usage:
    python figures/roc_curves.py \
        --analysis-glob "analysis/results/*_predictions.json" --out-dir figures/output

    # Test against mock fixtures:
    python figures/roc_curves.py \
        --analysis-glob "figures/fixtures/analysis_results/*_predictions.json" --out-dir figures/output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CLASS_NAMES, MODEL_LABELS, MODEL_NAMES, MissingDataError,
    ensure_output_dir, load_predictions, pooled_arrays, set_style,
    stamp_provenance,
)

CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-glob", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = ensure_output_dir(args.out_dir)

    try:
        records = load_predictions(args.analysis_glob)
    except MissingDataError as exc:
        print(str(exc))
        sys.exit(1)

    set_style()
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(5.5 * len(MODEL_NAMES), 5), sharey=True)
    if len(MODEL_NAMES) == 1:
        axes = [axes]

    for ax, model in zip(axes, MODEL_NAMES):
        y_true, y_pred, mean_probs, entropy = pooled_arrays(records, model)
        n_classes = mean_probs.shape[1]

        all_fpr = np.linspace(0, 1, 200)
        mean_tpr = np.zeros_like(all_fpr)
        for c in range(n_classes):
            y_true_c = (y_true == c).astype(int)
            if y_true_c.sum() == 0 or y_true_c.sum() == len(y_true_c):
                continue
            fpr, tpr, _ = roc_curve(y_true_c, mean_probs[:, c])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=CLASS_COLORS[c % len(CLASS_COLORS)],
                     label=f"{CLASS_NAMES[c]}  (AUC={roc_auc:.3f})", lw=1.5)
            mean_tpr += np.interp(all_fpr, fpr, tpr)

        mean_tpr /= n_classes
        macro_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, color="black", lw=2, linestyle="--",
                 label=f"Macro-average  (AUC={macro_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=0.8, linestyle=":")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False positive rate")
        ax.set_title(f"{MODEL_LABELS[model]}  (N={len(y_true)})")
        ax.legend(fontsize=7, loc="lower right")

    axes[0].set_ylabel("True positive rate")
    fig.suptitle("Per-class ROC curves (one-vs-rest) — MC-Dropout mean probabilities", y=1.03)
    stamp_provenance(fig, records)
    fig.tight_layout()

    out_path = out_dir / "roc_curves.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
