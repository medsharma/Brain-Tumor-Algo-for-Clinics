#!/usr/bin/env python3
"""
figures/common.py

Shared I/O, math, and styling helpers for the manuscript figure scripts in
this directory. Every script here is a pure consumer: it reads JSON written
by the training pipeline (src/code.py) and the analysis suite (analysis/*.py)
and renders a publication panel. Nothing in figures/ trains a model, runs
inference, or touches results/ or analysis/ — see figures/README.md for the
exact input schema each script expects.

Two input families:

  1. Aggregate summary  — results/master_summary.json
     Falls back to the latest results/<timestamp>/comparison_summary.json
     (the file src/code.py's run_comparison() already writes today) if
     master_summary.json doesn't exist yet. Same shape either way:
     per-model lists of scalar metrics across seeds, plus mcnemar_per_seed.

  2. Per-sample predictions — analysis/results/*_predictions.json
     One file per (model, seed): y_true / y_pred / mean_probs / entropy
     from an MC-Dropout evaluation pass. This is a thin JSON dump of exactly
     what src/code.py's evaluate_with_uncertainty() already returns in memory —
     see the docstring in load_predictions() for the one-line export Session
     A/B can add. Doesn't exist in the repo yet, hence the mock fixtures.

Until real files land, point scripts at figures/fixtures/ with --use-fixtures.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = FIGURES_DIR / "fixtures"
DEFAULT_OUTPUT_DIR = FIGURES_DIR / "output"

CLASS_NAMES: List[str] = ["glioma", "meningioma", "pituitary", "notumor"]
MODEL_NAMES: List[str] = ["vit", "resnet50"]
MODEL_LABELS: Dict[str, str] = {"vit": "ViT-B/16", "resnet50": "ResNet-50"}
_trapz = getattr(np, "trapezoid", np.trapz)


class MissingDataError(RuntimeError):
    """Raised when a required results/analysis file isn't available yet.

    Every message is written to read directly as the placeholder marker the
    manuscript uses, so a script that fails this way tells you exactly what
    it's still waiting on.
    """


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": False,
    })


def stamp_provenance(fig: plt.Figure, records: List[Dict[str, Any]]) -> None:
    """Watermark a figure if any contributing record is mock or smoke-test data.

    records: list of per-(model,seed) dicts that may carry "_mock": true
    and/or a "provenance": {"is_smoke_test_run": bool, ...} block, matching
    analysis/common.py's run_provenance() convention.
    """
    is_mock = any(r.get("_mock") for r in records)
    is_smoke = any(r.get("provenance", {}).get("is_smoke_test_run") for r in records)
    if not (is_mock or is_smoke):
        return
    label = "MOCK FIXTURE DATA" if is_mock else "SMOKE-TEST RUN — NOT REAL RESULTS"
    fig.text(
        0.5, 0.5, label, transform=fig.transFigure,
        fontsize=28, color="red", alpha=0.15, ha="center", va="center",
        rotation=30, zorder=100,
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _find_latest_comparison_summary(results_root: Path) -> Optional[Path]:
    if not results_root.exists():
        return None
    candidates = sorted(
        p / "comparison_summary.json"
        for p in results_root.iterdir()
        if p.is_dir() and (p / "comparison_summary.json").exists()
    )
    return candidates[-1] if candidates else None


def load_master_summary(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the aggregate multi-seed metrics summary.

    Resolution order when `path` is None:
      1. results/master_summary.json
      2. latest results/<timestamp>/comparison_summary.json
      3. raise MissingDataError
    """
    if path is not None:
        candidate = Path(path)
        if not candidate.exists():
            raise MissingDataError(
                f"[PLACEHOLDER: awaiting results/master_summary.json] - "
                f"explicit --master-summary path does not exist: {candidate}"
            )
    else:
        candidate = REPO_ROOT / "results" / "master_summary.json"
        if not candidate.exists():
            fallback = _find_latest_comparison_summary(REPO_ROOT / "results")
            if fallback is None:
                raise MissingDataError(
                    "[PLACEHOLDER: awaiting results/master_summary.json] - "
                    "not found, and no results/<timestamp>/comparison_summary.json "
                    "exists yet either. Test this script with "
                    "--master-summary figures/fixtures/master_summary.json"
                )
            candidate = fallback

    with open(candidate) as f:
        data = json.load(f)
    data["_source_path"] = str(candidate)
    if data.get("_mock"):
        print(f"[figures] WARNING: {candidate} is MOCK fixture data, not real results.")
    return data


def load_predictions(pattern: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load per-(model,seed) MC-Dropout prediction dumps.

    Expected file: analysis/results/<model>_seed<seed>_predictions.json
        {
          "model": "vit" | "resnet50", "seed": int,
          "class_names": [...4 names...],
          "provenance": {"is_smoke_test_run": bool, "caveat": str, ...},
          "y_true": [N ints], "y_pred": [N ints],
          "mean_probs": [[N x 4 floats]], "entropy": [N floats]
        }

    This is a one-line dump of evaluate_with_uncertainty()'s return value:
        r = evaluate_with_uncertainty(model, test_loader, DEVICE, T=mc_T)
        json.dump({
            "model": model_name, "seed": seed, "class_names": CLASS_NAMES,
            "provenance": run_provenance(run_dir, seed_dir),
            "y_true": r["labels"].tolist(), "y_pred": r["predictions"].tolist(),
            "mean_probs": r["mean_probs"].tolist(), "entropy": r["entropy"].tolist(),
        }, f)
    Doesn't exist in the repo yet — no analysis/*.py script currently exports
    raw predictions (analysis/calibration_comparison.py only writes aggregate
    scalars + PNGs). Test with --analysis-glob "figures/fixtures/analysis_results/*_predictions.json".
    """
    glob_pattern = pattern or str(REPO_ROOT / "analysis" / "results" / "*_predictions.json")
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        raise MissingDataError(
            f"[PLACEHOLDER: awaiting analysis/results/*_predictions.json] - "
            f"no files matched {glob_pattern}. Test this script with "
            f"--analysis-glob \"figures/fixtures/analysis_results/*_predictions.json\""
        )
    out: List[Dict[str, Any]] = []
    any_mock = False
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        d["_source_path"] = p
        if d.get("_mock"):
            any_mock = True
        out.append(d)
    if any_mock:
        print(f"[figures] WARNING: {sum(1 for d in out if d.get('_mock'))}/{len(out)} "
              f"prediction file(s) matched by {glob_pattern} are MOCK fixture data.")
    return out


def load_calibration_comparison(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Optional bonus loader for analysis/results/calibration/calibration_comparison.json,
    the temperature-scaling before/after export analysis/calibration_comparison.py
    already produces today. Returns None (not an error) if absent — this input
    is a bonus panel, not a hard requirement for the calibration figure.
    """
    candidate = Path(path) if path else (
        REPO_ROOT / "analysis" / "results" / "calibration" / "calibration_comparison.json"
    )
    if not candidate.exists():
        return None
    with open(candidate) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metric math (mirrors src/code.py's compute_calibration_metrics /
# compute_risk_coverage_curve / mcnemar_test so figures reproduce the exact
# manuscript-reported numbers when given the same per-sample arrays).
# ---------------------------------------------------------------------------

def pooled_arrays(
    records: List[Dict[str, Any]], model: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate y_true / y_pred / mean_probs / entropy across all seeds for one model."""
    recs = [r for r in records if r.get("model") == model]
    if not recs:
        raise MissingDataError(f"No prediction records found for model={model!r}")
    y_true = np.concatenate([np.asarray(r["y_true"]) for r in recs])
    y_pred = np.concatenate([np.asarray(r["y_pred"]) for r in recs])
    mean_probs = np.concatenate([np.asarray(r["mean_probs"]) for r in recs], axis=0)
    entropy = np.concatenate([np.asarray(r["entropy"]) for r in recs])
    return y_true, y_pred, mean_probs, entropy


def reliability_bins(
    mean_probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """15-bin reliability diagram data + ECE, identical binning to
    src/code.py::compute_calibration_metrics.
    """
    confidence = mean_probs.max(axis=1)
    preds = mean_probs.argmax(axis=1)
    correct = (preds == y_true).astype(float)
    N = len(y_true)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc = np.full(n_bins, np.nan)
    bin_conf = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=int)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (confidence > lo) & (confidence <= hi)
        n = int(in_bin.sum())
        bin_count[i] = n
        if n == 0:
            continue
        bin_acc[i] = correct[in_bin].mean()
        bin_conf[i] = confidence[in_bin].mean()
        ece += abs(bin_acc[i] - bin_conf[i]) * n / N

    return edges, bin_acc, bin_conf, bin_count, float(ece)


def risk_coverage_curve(
    mean_probs: np.ndarray,
    y_true: np.ndarray,
    coverage_levels: Tuple[float, ...] = (0.80, 0.90, 0.95),
) -> Dict[str, Any]:
    """Identical logic to src/code.py::compute_risk_coverage_curve."""
    entropy = -(mean_probs * np.log2(mean_probs + 1e-10)).sum(axis=1)
    correct = (mean_probs.argmax(axis=1) == y_true).astype(float)
    order = np.argsort(entropy)
    correct_s = correct[order]

    N = len(y_true)
    coverage = np.arange(1, N + 1) / N
    risk = 1.0 - np.cumsum(correct_s) / np.arange(1, N + 1)
    aurc = float(_trapz(risk, coverage))

    acc_at = {}
    for level in coverage_levels:
        n_covered = max(1, int(N * level))
        acc_at[level] = float(correct_s[:n_covered].mean())

    return {"coverage": coverage, "risk": risk, "aurc": aurc, "acc_at_coverage": acc_at}


def ensure_output_dir(path: Optional[str] = None) -> Path:
    out = Path(path) if path else DEFAULT_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out
