#!/usr/bin/env python3
"""
analysis/safety_metrics.py  -  Session A.

The metric library for external validation. Every number Session A reports is
computed here, so there is one definition of each and no chance of two scripts
disagreeing about what "miss rate" means.

Safety definitions, stated once and used everywhere
---------------------------------------------------
**Tumour miss rate** is the headline number of this project:

    of all images whose true class is glioma, meningioma or pituitary,
    the fraction the model predicted as `notumor`

"Predicted as notumor" means argmax over the four MC-Dropout mean probabilities.
That is the class the tool would display. This is `tumor_miss_rate`.

A second, threshold-based version exists because a deployed triage tool calls
"tumour" on `p_tumor >= threshold`, not on argmax. That is
`1 - binary_sensitivity` and it is reported alongside. The two differ: argmax can
pick `notumor` at p_notumor = 0.4, while p_tumor = 0.6 >= 0.5 would refer. Both
are reported so nobody can quietly use whichever looks better.

**Binary confusion**, positive class = tumour:
    TP tumour called tumour, FN tumour called no-tumour (the killer),
    TN no-tumour called no-tumour, FP no-tumour called tumour (a false referral).

Nothing here fits anything. Every function takes predictions that already exist
and scores them. Thresholds arrive as arguments; they are chosen in
`analysis/operating_point.py` on the internal validation split.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASS_NAMES, REPO_ROOT  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))
from code import (  # noqa: E402
    bootstrap_ci,
    compute_calibration_metrics,
    compute_risk_coverage_curve,
    mcnemar_test,
)

PROB_COLS = ["p_glioma", "p_meningioma", "p_pituitary", "p_notumor"]
LABEL_INDEX = {"glioma": 0, "meningioma": 1, "pituitary": 2, "notumor": 3}
NOTUMOR_IDX = LABEL_INDEX["notumor"]
TUMOR_CLASSES = ["glioma", "meningioma", "pituitary"]

N_BOOT = 1000


# ===========================================================================
# Bootstrap
# ===========================================================================

def bootstrap_stat_ci(
    fn: Callable[[np.ndarray], float],
    n: int,
    n_resamples: int = N_BOOT,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for any statistic of a row index set.

    `src/code.py`'s `bootstrap_ci` only knows accuracy and macro-AUC. Miss rate,
    sensitivity, specificity, PPV and NPV are none of those, so they resample
    through here. Same percentile method, same default of 1000 resamples.

    `fn` receives an array of row indices and returns a scalar. Resamples where
    the statistic is undefined (empty denominator) return NaN and are dropped,
    so a CI over a small subgroup stays honest rather than silently narrow.
    """
    rng = np.random.default_rng(seed)
    point = fn(np.arange(n))
    est = np.empty(n_resamples)
    for i in range(n_resamples):
        est[i] = fn(rng.integers(0, n, size=n))
    valid = est[~np.isnan(est)]
    if valid.size == 0 or np.isnan(point):
        return float("nan"), float("nan"), float("nan")
    return (
        float(point),
        float(np.percentile(valid, 100 * alpha / 2)),
        float(np.percentile(valid, 100 * (1 - alpha / 2))),
    )


def _rate(numer: np.ndarray, denom: np.ndarray) -> float:
    d = int(denom.sum())
    if d == 0:
        return float("nan")
    return float(numer[denom].sum() / d)


def _ci_dict(point: float, lo: float, hi: float, n: int) -> Dict[str, Any]:
    return {"value": point, "ci_lo": lo, "ci_hi": hi, "n": int(n)}


# ===========================================================================
# Safety metrics
# ===========================================================================

def tumor_miss_rate(
    df: pd.DataFrame, seed: int = 0, n_resamples: int = N_BOOT
) -> Dict[str, Any]:
    """Fraction of real tumours the model called `notumor` (argmax)."""
    y = df["true_label"].to_numpy()
    p = df["pred_label"].to_numpy()
    is_tumor = y != NOTUMOR_IDX
    called_none = p == NOTUMOR_IDX

    def stat(idx: np.ndarray) -> float:
        return _rate(called_none[idx], is_tumor[idx])

    pt, lo, hi = bootstrap_stat_ci(stat, len(df), n_resamples, seed=seed)
    return _ci_dict(pt, lo, hi, int(is_tumor.sum()))


def per_class_miss_rate(
    df: pd.DataFrame, seed: int = 0, n_resamples: int = N_BOOT
) -> Dict[str, Dict[str, Any]]:
    """Miss rate for glioma, meningioma and pituitary separately.

    An aggregate can hide one class failing badly. It usually does.
    """
    out: Dict[str, Dict[str, Any]] = {}
    y = df["true_label"].to_numpy()
    p = df["pred_label"].to_numpy()
    called_none = p == NOTUMOR_IDX
    for cls in TUMOR_CLASSES:
        k = LABEL_INDEX[cls]
        mask = y == k

        def stat(idx: np.ndarray, mask=mask) -> float:
            return _rate(called_none[idx], mask[idx])

        pt, lo, hi = bootstrap_stat_ci(stat, len(df), n_resamples, seed=seed)
        out[cls] = _ci_dict(pt, lo, hi, int(mask.sum()))
    return out


def binary_metrics(
    df: pd.DataFrame,
    threshold: float = 0.5,
    seed: int = 0,
    n_resamples: int = N_BOOT,
) -> Dict[str, Any]:
    """Tumour vs no-tumour at a `p_tumor` threshold. Positive class = tumour."""
    y_pos = (df["true_label"].to_numpy() != NOTUMOR_IDX)
    pred_pos = df["p_tumor"].to_numpy() >= threshold

    tp = y_pos & pred_pos
    fn_ = y_pos & ~pred_pos
    tn = ~y_pos & ~pred_pos
    fp = ~y_pos & pred_pos
    correct = tp | tn

    defs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "accuracy": (correct, np.ones(len(df), dtype=bool)),
        "sensitivity": (tp, y_pos),
        "specificity": (tn, ~y_pos),
        "ppv": (tp, pred_pos),
        "npv": (tn, ~pred_pos),
        "miss_rate_at_threshold": (fn_, y_pos),
        "false_referral_rate": (fp, ~y_pos),
        "referral_rate": (pred_pos, np.ones(len(df), dtype=bool)),
    }

    res: Dict[str, Any] = {
        "threshold": float(threshold),
        "counts": {"tp": int(tp.sum()), "fn": int(fn_.sum()),
                   "tn": int(tn.sum()), "fp": int(fp.sum())},
    }
    for name, (num, den) in defs.items():
        def stat(idx: np.ndarray, num=num, den=den) -> float:
            return _rate(num[idx], den[idx])

        pt, lo, hi = bootstrap_stat_ci(stat, len(df), n_resamples, seed=seed)
        res[name] = _ci_dict(pt, lo, hi, int(den.sum()))
    return res


# ===========================================================================
# Standard classification metrics
# ===========================================================================

def standard_metrics(
    df: pd.DataFrame, seed: int = 0, n_resamples: int = N_BOOT
) -> Dict[str, Any]:
    """Four-way accuracy, macro F1, macro AUC, ECE, Brier, risk-coverage."""
    y = df["true_label"].to_numpy()
    p = df["pred_label"].to_numpy()
    probs = df[PROB_COLS].to_numpy()

    acc, acc_lo, acc_hi = bootstrap_ci("accuracy", y, p, n_resamples=n_resamples, seed=seed)

    def f1_stat(idx: np.ndarray) -> float:
        yi = y[idx]
        if len(np.unique(yi)) < 2:
            return float("nan")
        return float(f1_score(yi, p[idx], average="macro", zero_division=0))

    f1_pt, f1_lo, f1_hi = bootstrap_stat_ci(f1_stat, len(df), n_resamples, seed=seed)

    try:
        auc, auc_lo, auc_hi = bootstrap_ci(
            "auc", y, p, probs, n_resamples=n_resamples, seed=seed
        )
    except ValueError:
        auc = auc_lo = auc_hi = float("nan")

    cal = compute_calibration_metrics(probs, y, n_bins=15)
    rc = compute_risk_coverage_curve(probs, y, coverage_levels=(0.80, 0.90, 0.95))

    return {
        "n": int(len(df)),
        "four_way_accuracy": _ci_dict(acc, acc_lo, acc_hi, len(df)),
        "macro_f1": _ci_dict(f1_pt, f1_lo, f1_hi, len(df)),
        "macro_auc": _ci_dict(auc, auc_lo, auc_hi, len(df)),
        "ece_15bin": float(cal["ece"]),
        "brier": float(cal["brier"]),
        "aurc": float(rc["aurc"]),
        "acc_at_coverage": {f"{int(k*100)}": float(v) for k, v in rc["acc_at_coverage"].items()},
        "mean_entropy_nats": float(df["entropy"].mean()),
        "mean_entropy_bits": float(df["entropy"].mean() / np.log(2.0)),
        "per_class_recall": {
            CLASS_NAMES[k]: (float((p[y == k] == k).mean()) if (y == k).any() else float("nan"))
            for k in range(4)
        },
        "confusion_matrix": _confusion(y, p),
    }


def _confusion(y: np.ndarray, p: np.ndarray) -> List[List[int]]:
    cm = np.zeros((4, 4), dtype=int)
    for t, q in zip(y, p):
        cm[t, q] += 1
    return cm.tolist()


# ===========================================================================
# Uncertainty behaviour: does deferral actually work?
# ===========================================================================

def entropy_behaviour(
    df: pd.DataFrame, seed: int = 0, n_resamples: int = N_BOOT
) -> Dict[str, Any]:
    """Entropy on correct vs wrong, entropy on missed tumours, error AUROC.

    If entropy on wrong answers looks like entropy on right answers, deferral
    cannot work: the tool has no signal telling it which cases to escalate.
    If missed tumours specifically carry low entropy, the tool is confidently
    wrong about the one error that kills people, and deferral will not catch it.
    """
    y = df["true_label"].to_numpy()
    p = df["pred_label"].to_numpy()
    ent = df["entropy"].to_numpy()
    wrong = p != y
    missed = (y != NOTUMOR_IDX) & (p == NOTUMOR_IDX)

    def summarise(mask: np.ndarray) -> Dict[str, Any]:
        v = ent[mask]
        if v.size == 0:
            return {"n": 0}
        return {
            "n": int(v.size),
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "p10": float(np.percentile(v, 10)),
            "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75)),
            "p90": float(np.percentile(v, 90)),
        }

    correct_s = summarise(~wrong)
    wrong_s = summarise(wrong)
    missed_s = summarise(missed)

    # AUROC of entropy as a detector of "this prediction is wrong".
    if wrong.any() and (~wrong).any():
        err_auroc = float(roc_auc_score(wrong.astype(int), ent))
        def auroc_stat(idx: np.ndarray) -> float:
            w = wrong[idx]
            if len(np.unique(w)) < 2:
                return float("nan")
            return float(roc_auc_score(w.astype(int), ent[idx]))
        _, ea_lo, ea_hi = bootstrap_stat_ci(auroc_stat, len(df), n_resamples, seed=seed)
    else:
        err_auroc = ea_lo = ea_hi = float("nan")

    # Same question restricted to the safety-critical error.
    if missed.any():
        keep = (~wrong) | missed  # correct predictions vs missed tumours only
        lab = missed[keep].astype(int)
        miss_auroc = (float(roc_auc_score(lab, ent[keep]))
                      if len(np.unique(lab)) > 1 else float("nan"))
    else:
        miss_auroc = float("nan")

    # Overlap: what fraction of wrong answers sit below the median entropy of
    # correct answers? A high number means deferral will not separate them.
    med_correct = correct_s.get("median", float("nan"))
    frac_wrong_below = (float((ent[wrong] < med_correct).mean())
                        if wrong.any() and not np.isnan(med_correct) else float("nan"))
    frac_missed_below = (float((ent[missed] < med_correct).mean())
                         if missed.any() and not np.isnan(med_correct) else float("nan"))

    return {
        "entropy_correct": correct_s,
        "entropy_wrong": wrong_s,
        "entropy_missed_tumors": missed_s,
        "error_detection_auroc": {"value": err_auroc, "ci_lo": ea_lo, "ci_hi": ea_hi},
        "missed_tumor_detection_auroc": miss_auroc,
        "frac_wrong_below_median_correct_entropy": frac_wrong_below,
        "frac_missed_below_median_correct_entropy": frac_missed_below,
    }


def deferral_curve(
    df: pd.DataFrame,
    defer_thresholds: Dict[float, float],
    seed: int = 0,
    n_resamples: int = N_BOOT,
) -> List[Dict[str, Any]]:
    """Miss rate among cases the model did NOT defer, at fixed entropy cutoffs.

    `defer_thresholds` maps a target deferral budget (0.05, 0.10, ...) to an
    entropy cutoff **fitted on the internal validation split**. Cases with
    entropy above the cutoff are sent to a human and are not scored. The number
    that matters is the miss rate on what is left, because those are the
    patients the tool sends home without a second look.
    """
    rows: List[Dict[str, Any]] = []
    for budget, cut in sorted(defer_thresholds.items()):
        keep = df["entropy"].to_numpy() < cut
        kept = df.loc[keep]
        actual_defer = float(1.0 - keep.mean())
        row: Dict[str, Any] = {
            "target_defer_rate": float(budget),
            "entropy_threshold_nats": float(cut),
            "actual_defer_rate": actual_defer,
            "n_kept": int(keep.sum()),
            "n_deferred": int((~keep).sum()),
        }
        if len(kept) == 0:
            row.update({"miss_rate_kept": float("nan"), "accuracy_kept": float("nan")})
            rows.append(row)
            continue
        mr = tumor_miss_rate(kept, seed=seed, n_resamples=n_resamples)
        row["miss_rate_kept"] = mr["value"]
        row["miss_rate_kept_ci"] = [mr["ci_lo"], mr["ci_hi"]]
        row["n_tumors_kept"] = mr["n"]
        row["accuracy_kept"] = float(
            (kept["pred_label"].to_numpy() == kept["true_label"].to_numpy()).mean()
        )
        # How many of the missed tumours did deferral actually catch?
        y, p = df["true_label"].to_numpy(), df["pred_label"].to_numpy()
        missed_all = (y != NOTUMOR_IDX) & (p == NOTUMOR_IDX)
        row["n_missed_total"] = int(missed_all.sum())
        row["n_missed_caught_by_defer"] = int((missed_all & ~keep).sum())
        row["n_missed_still_missed"] = int((missed_all & keep).sum())
        rows.append(row)
    return rows


# ===========================================================================
# Subgroups
# ===========================================================================

def subgroup_report(
    df: pd.DataFrame,
    by: str,
    threshold: float = 0.5,
    seed: int = 0,
    n_resamples: int = N_BOOT,
    min_n: int = 30,
) -> Dict[str, Any]:
    """Run the safety and standard metrics separately within each subgroup."""
    out: Dict[str, Any] = {}
    for value, sub in df.groupby(by, sort=True):
        entry: Dict[str, Any] = {"n": int(len(sub))}
        if len(sub) < min_n:
            entry["note"] = f"only {len(sub)} images, metrics omitted as unreliable"
            out[str(value)] = entry
            continue
        entry["tumor_miss_rate"] = tumor_miss_rate(sub, seed=seed, n_resamples=n_resamples)
        entry["per_class_miss_rate"] = per_class_miss_rate(sub, seed=seed, n_resamples=n_resamples)
        entry["binary"] = binary_metrics(sub, threshold, seed=seed, n_resamples=n_resamples)
        entry["standard"] = standard_metrics(sub, seed=seed, n_resamples=n_resamples)
        entry["entropy"] = entropy_behaviour(sub, seed=seed, n_resamples=n_resamples)
        out[str(value)] = entry
    return out


# ===========================================================================
# Full report for one prediction frame
# ===========================================================================

def full_report(
    df: pd.DataFrame,
    threshold: float = 0.5,
    seed: int = 0,
    n_resamples: int = N_BOOT,
    subgroups: Sequence[str] = (),
) -> Dict[str, Any]:
    rep: Dict[str, Any] = {
        "n": int(len(df)),
        "n_tumor": int((df["true_label"] != NOTUMOR_IDX).sum()),
        "n_notumor": int((df["true_label"] == NOTUMOR_IDX).sum()),
        "tumor_miss_rate": tumor_miss_rate(df, seed=seed, n_resamples=n_resamples),
        "per_class_miss_rate": per_class_miss_rate(df, seed=seed, n_resamples=n_resamples),
        "binary": binary_metrics(df, threshold, seed=seed, n_resamples=n_resamples),
        "standard": standard_metrics(df, seed=seed, n_resamples=n_resamples),
        "entropy": entropy_behaviour(df, seed=seed, n_resamples=n_resamples),
    }
    for col in subgroups:
        if col in df.columns:
            rep[f"by_{col}"] = subgroup_report(
                df, col, threshold=threshold, seed=seed, n_resamples=n_resamples
            )
    return rep


def confident_misses(df: pd.DataFrame, entropy_cut: float) -> pd.DataFrame:
    """True tumour, predicted `notumor`, entropy below `entropy_cut`.

    The cases that kill people. Deferral does not catch them because the model
    is not uncertain, and the clinician is never prompted to look twice.
    """
    y = df["true_label"].to_numpy()
    p = df["pred_label"].to_numpy()
    missed = (y != NOTUMOR_IDX) & (p == NOTUMOR_IDX)
    return df.loc[missed & (df["entropy"].to_numpy() < entropy_cut)].copy()


def drop_with_ci(
    internal: Dict[str, Any],
    external: Dict[str, Any],
    key_path: Sequence[str],
) -> Dict[str, Any]:
    """internal minus external for a metric that carries a CI on both sides.

    The CI on the difference is the naive independent-samples combination
    (the two datasets are different images, so independence holds), computed
    from the half-widths. Stated as an approximation because it is one.
    """
    def dig(d: Dict[str, Any]) -> Dict[str, Any]:
        for k in key_path:
            d = d[k]
        return d

    a, b = dig(internal), dig(external)
    diff = a["value"] - b["value"]
    ha = (a["ci_hi"] - a["ci_lo"]) / 2.0
    hb = (b["ci_hi"] - b["ci_lo"]) / 2.0
    h = float(np.sqrt(ha ** 2 + hb ** 2))
    return {
        "internal": a["value"], "external": b["value"],
        "drop": float(diff), "drop_ci_approx": [float(diff - h), float(diff + h)],
        "note": "CI on the difference combines the two percentile-bootstrap "
                "half-widths in quadrature; approximate.",
    }
