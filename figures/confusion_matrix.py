#!/usr/bin/env python3
"""
figures/confusion_matrix.py

Row-normalized confusion matrix per model, pooled across all seeds found by
--analysis-glob.

Usage:
    python figures/confusion_matrix.py \
        --analysis-glob "analysis/results/*_predictions.json" --out-dir figures/output

    # Test against mock fixtures:
    python figures/confusion_matrix.py \
        --analysis-glob "figures/fixtures/analysis_results/*_predictions.json" --out-dir figures/output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CLASS_NAMES, MODEL_LABELS, MODEL_NAMES, MissingDataError,
    ensure_output_dir, load_predictions, pooled_arrays, set_style,
    stamp_provenance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-glob", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--normalize", action="store_true", default=True)
    args = parser.parse_args()

    out_dir = ensure_output_dir(args.out_dir)

    try:
        records = load_predictions(args.analysis_glob)
    except MissingDataError as exc:
        print(str(exc))
        sys.exit(1)

    set_style()
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(5.5 * len(MODEL_NAMES), 5))
    if len(MODEL_NAMES) == 1:
        axes = [axes]

    for ax, model in zip(axes, MODEL_NAMES):
        y_true, y_pred, mean_probs, entropy = pooled_arrays(records, model)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
        if args.normalize:
            with np.errstate(invalid="ignore", divide="ignore"):
                cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm_display = np.nan_to_num(cm_display)
            fmt, vmax = ".2f", 1.0
        else:
            cm_display, fmt, vmax = cm, "d", cm.max()

        im = ax.imshow(cm_display, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{MODEL_LABELS[model]}  (N={len(y_true)})")

        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                val = cm_display[i, j]
                text = f"{val:{fmt}}"
                color = "white" if val > vmax * 0.6 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Confusion matrices — pooled across seeds" + (" (row-normalized)" if args.normalize else ""), y=1.03)
    stamp_provenance(fig, records)
    fig.tight_layout()

    out_path = out_dir / "confusion_matrix.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
