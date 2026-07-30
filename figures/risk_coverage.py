#!/usr/bin/env python3
"""
figures/risk_coverage.py

Selective-prediction risk-coverage curve per model (entropy-ordered rejection,
identical construction to src/code.py::compute_risk_coverage_curve), with AURC and
acc@{80,90,95}% coverage annotated. Cross-checks the recomputed AURC against
the aggregate value in master_summary.json as a sanity check.

Usage:
    python figures/risk_coverage.py \
        --analysis-glob "analysis/results/*_predictions.json" \
        --master-summary results/master_summary.json --out-dir figures/output

    # Test against mock fixtures:
    python figures/risk_coverage.py \
        --analysis-glob "figures/fixtures/analysis_results/*_predictions.json" \
        --master-summary figures/fixtures/master_summary.json --out-dir figures/output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    MODEL_LABELS, MODEL_NAMES, MissingDataError, ensure_output_dir,
    load_master_summary, load_predictions, pooled_arrays,
    risk_coverage_curve, set_style, stamp_provenance,
)

COLORS = {"vit": "#4C72B0", "resnet50": "#DD8452"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-glob", type=str, default=None)
    parser.add_argument("--master-summary", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = ensure_output_dir(args.out_dir)

    try:
        records = load_predictions(args.analysis_glob)
    except MissingDataError as exc:
        print(str(exc))
        sys.exit(1)

    try:
        master = load_master_summary(args.master_summary)
    except MissingDataError as exc:
        print(f"[figures] {exc}\n[figures] Continuing without master_summary cross-check.")
        master = None

    set_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for model in MODEL_NAMES:
        y_true, y_pred, mean_probs, entropy = pooled_arrays(records, model)
        curve = risk_coverage_curve(mean_probs, y_true)
        ax.plot(curve["coverage"], curve["risk"], color=COLORS.get(model, "gray"),
                label=f"{MODEL_LABELS[model]}  (AURC={curve['aurc']:.4f})", lw=1.8)

        if master is not None and model in master:
            agg_aurc_list = master[model].get("aurc", [])
            if agg_aurc_list:
                agg_aurc_mean = sum(agg_aurc_list) / len(agg_aurc_list)
                delta = abs(agg_aurc_mean - curve["aurc"])
                flag = "  <-- differs from master_summary mean by > 0.05" if delta > 0.05 else ""
                print(f"[figures] {model}: recomputed AURC (pooled seeds) = {curve['aurc']:.4f}  "
                      f"vs. master_summary per-seed mean = {agg_aurc_mean:.4f}{flag}")

    ax.set_xlabel("Coverage (fraction of test set retained)")
    ax.set_ylabel("Risk (1 − accuracy on retained samples)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_title("Risk-coverage curve — MC-Dropout entropy-based rejection")
    ax.legend(fontsize=8)
    stamp_provenance(fig, records)
    fig.tight_layout()

    out_path = out_dir / "risk_coverage.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
