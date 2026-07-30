#!/usr/bin/env python3
"""
figures/calibration_reliability.py

Reliability diagram (binned confidence vs. accuracy) + ECE per model, from
MC-Dropout mean-probability predictions. If analysis/results/calibration/
calibration_comparison.json also exists (analysis/calibration_comparison.py's
deterministic-logit temperature-scaling export), adds a bonus ECE
before/after-scaling comparison panel — otherwise that panel is skipped.

Usage:
    python figures/calibration_reliability.py \
        --analysis-glob "analysis/results/*_predictions.json" \
        --out-dir figures/output

    # Test against mock fixtures:
    python figures/calibration_reliability.py \
        --analysis-glob "figures/fixtures/analysis_results/*_predictions.json" \
        --out-dir figures/output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CLASS_NAMES, MODEL_LABELS, MODEL_NAMES, MissingDataError,
    ensure_output_dir, load_calibration_comparison, load_predictions,
    pooled_arrays, reliability_bins, set_style, stamp_provenance,
)


def plot_reliability(records, out_dir: Path) -> Path:
    set_style()
    fig, axes = plt.subplots(1, len(MODEL_NAMES), figsize=(5 * len(MODEL_NAMES), 5), sharey=True)
    if len(MODEL_NAMES) == 1:
        axes = [axes]

    for ax, model in zip(axes, MODEL_NAMES):
        y_true, y_pred, mean_probs, entropy = pooled_arrays(records, model)
        edges, bin_acc, bin_conf, bin_count, ece = reliability_bins(mean_probs, y_true)
        n_bins = len(bin_acc)

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.bar(
            edges[:-1], np.nan_to_num(bin_acc), width=1 / n_bins, align="edge",
            alpha=0.75, edgecolor="black", linewidth=0.4, color="#4C72B0",
            label="Accuracy per bin",
        )
        gap_x = np.nan_to_num(bin_conf)
        gap_y = np.nan_to_num(bin_acc)
        valid = bin_count > 0
        ax.scatter(gap_x[valid], gap_y[valid], s=14, color="#C44E52", zorder=5, label="Bin mean confidence")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_title(f"{MODEL_LABELS[model]}\nECE = {ece:.4f}  (N={len(y_true)}, {len(set(r['seed'] for r in records if r['model'] == model))} seeds pooled)")
        ax.legend(fontsize=7, loc="upper left")

    axes[0].set_ylabel("Accuracy")
    fig.suptitle("Calibration reliability — MC-Dropout mean probabilities", y=1.02)
    stamp_provenance(fig, records)
    fig.tight_layout()

    out_path = out_dir / "calibration_reliability.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_temperature_scaling_bonus(calib_json: dict, out_dir: Path) -> Path:
    """Bonus panel: MC-Dropout ECE has no 'before/after' — this instead shows
    the deterministic-softmax temperature-scaling before/after ECE that
    analysis/calibration_comparison.py already computes, per model per seed.
    """
    set_style()
    models = list(calib_json.get("models", {}).keys())
    fig, ax = plt.subplots(figsize=(6, 4.5))

    width = 0.35
    x = np.arange(len(models))
    before_means, after_means = [], []
    any_smoke = False
    for m in models:
        rows = calib_json["models"][m].get("per_seed", [])
        if not rows:
            before_means.append(np.nan)
            after_means.append(np.nan)
            continue
        before_means.append(np.mean([r["test_ece_before"] for r in rows]))
        after_means.append(np.mean([r["test_ece_after"] for r in rows]))
        any_smoke = any_smoke or any(r["provenance"]["is_smoke_test_run"] for r in rows)

    ax.bar(x - width / 2, before_means, width, label="Before temp. scaling", color="#DD8452")
    ax.bar(x + width / 2, after_means, width, label="After temp. scaling", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.set_ylabel("ECE (deterministic softmax, 15-bin)")
    ax.set_title("Temperature scaling — mean ECE across seeds\n(complementary to MC-Dropout ECE above; not directly comparable)")
    ax.legend(fontsize=8)
    if any_smoke:
        fig.text(0.5, -0.02, "Contains smoke-test-run data — not real calibration numbers.",
                  ha="center", fontsize=8, color="darkred", transform=ax.transAxes)
    fig.tight_layout()

    out_path = out_dir / "calibration_temperature_scaling_bonus.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-glob", type=str, default=None)
    parser.add_argument("--calibration-comparison-json", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = ensure_output_dir(args.out_dir)

    try:
        records = load_predictions(args.analysis_glob)
    except MissingDataError as exc:
        print(str(exc))
        sys.exit(1)

    out_path = plot_reliability(records, out_dir)
    print(f"Wrote {out_path}")

    calib_json = load_calibration_comparison(args.calibration_comparison_json)
    if calib_json is not None:
        bonus_path = plot_temperature_scaling_bonus(calib_json, out_dir)
        print(f"Wrote {bonus_path}")
    else:
        print("[figures] analysis/results/calibration/calibration_comparison.json not found - "
              "skipping temperature-scaling bonus panel.")


if __name__ == "__main__":
    main()
