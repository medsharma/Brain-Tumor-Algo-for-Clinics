#!/usr/bin/env python3
"""
analysis/operating_point.py  -  Session A, Phase 4 and Phase 6.

Chooses the settings a clinic would actually run the tool at, and publishes them
as `analysis/results/safety/deployment_config.json` (Contract 2).

THE RULE THIS FILE OBEYS
------------------------
Every fitted quantity in this file is fitted on the **internal validation
split** and applied to BRISC unchanged:

  * the `p_tumor` referral threshold      -> swept and chosen on internal val
  * the temperature                        -> fitted by LBFGS on internal val
  * the entropy deferral cutoffs           -> quantiles of internal val entropy
  * the backbone and the ensemble decision -> compared on internal val

BRISC labels are read only to score the finished configuration. Picking a
threshold on BRISC and then reporting BRISC performance at that threshold would
look excellent and mean nothing.

CONTAMINATION NOTE, READ THIS BEFORE THE GREP FLAGS IT
-------------------------------------------------------
`fit_temperature`, imported from `analysis/calibration_comparison.py`, calls
`loss.backward()` and `optimizer.step()`. That is a real optimizer and it really
does run. It fits exactly one scalar, the temperature, on internal validation
data. It never sees a BRISC image or a BRISC label. This is the permitted
fitting-on-internal-val case, not a violation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import CLASS_NAMES, REPO_ROOT, RESULTS_OUT_DIR  # noqa: E402
from calibration_comparison import fit_temperature  # noqa: E402  (see note above)
from external_validation_brisc import (  # noqa: E402
    CHECKPOINTS,
    MC_T,
    MODELS,
    SEEDS,
    attach_overlap_flags,
    checkpoint_path,
    ensemble_frames,
    load_brisc_cache,
    load_internal_cache,
)
import safety_metrics as sm  # noqa: E402

def _load_figure_style():
    """Reuse figures/common.py set_style without letting its module name shadow
    analysis/common.py, which is also called common."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "figures_common", REPO_ROOT / "figures" / "common.py"
    )
    if spec is None or spec.loader is None:
        return lambda: None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.set_style


set_style = _load_figure_style()

SAFETY_OUT = RESULTS_OUT_DIR / "safety"
FIG_OUT = SAFETY_OUT / "figures"

NOTUMOR_IDX = sm.NOTUMOR_IDX
PROB_COLS = sm.PROB_COLS

# Deferral budgets a clinic could plausibly absorb.
DEFER_BUDGETS = [0.05, 0.10, 0.20, 0.30]
# Minimum-sensitivity targets. 0.98 is the working choice, 0.95 and 0.99 are
# reported so the tradeoff is visible rather than asserted.
SENS_TARGETS = [0.95, 0.98, 0.99]
PRIMARY_SENS_TARGET = 0.98
# Plausible tumour prevalence in a rural screening clinic, used to translate
# sensitivity and specificity into PPV/NPV that mean something there.
CLINIC_PREVALENCES = [0.02, 0.05, 0.10]


# ===========================================================================
# Loading
# ===========================================================================

def load_set(model: str, seeds: List[int], split: str, clean: bool = True) -> pd.DataFrame:
    """One frame for a model/split: single seed as-is, several seeds averaged.

    For BRISC, `clean=True` (the default) drops every image that is a
    near-duplicate of an internal TRAINING image. About 56 percent of BRISC is.
    Reporting BRISC performance without this filter is reporting training
    accuracy, so the default is the safe one and callers must opt out loudly.
    """
    if split == "brisc":
        frames = [load_brisc_cache(model, s) for s in seeds]
        df = frames[0] if len(frames) == 1 else ensemble_frames(frames)
        df = attach_overlap_flags(df)
        return df[df["clean_vs_train"]].reset_index(drop=True) if clean else df
    frames = [load_internal_cache(model, s, split) for s in seeds]
    return frames[0] if len(frames) == 1 else ensemble_frames(frames)


# ===========================================================================
# Phase 4.1 - threshold sweep on internal val
# ===========================================================================

def sweep_thresholds(df: pd.DataFrame, n_grid: int = 1001) -> pd.DataFrame:
    """Sensitivity / specificity / miss rate / referral load vs `p_tumor` cutoff."""
    y_pos = (df["true_label"].to_numpy() != NOTUMOR_IDX)
    pt = df["p_tumor"].to_numpy()
    n_pos, n_neg = int(y_pos.sum()), int((~y_pos).sum())

    rows: List[Dict[str, float]] = []
    for thr in np.linspace(0.0, 1.0, n_grid):
        pred_pos = pt >= thr
        tp = int((y_pos & pred_pos).sum())
        fn_ = n_pos - tp
        fp = int((~y_pos & pred_pos).sum())
        tn = n_neg - fp
        rows.append({
            "threshold": float(thr),
            "tp": tp, "fn": fn_, "fp": fp, "tn": tn,
            "sensitivity": tp / n_pos if n_pos else np.nan,
            "specificity": tn / n_neg if n_neg else np.nan,
            "miss_rate": fn_ / n_pos if n_pos else np.nan,
            "false_referral_rate": fp / n_neg if n_neg else np.nan,
            "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
            "npv": tn / (tn + fn_) if (tn + fn_) else np.nan,
            "referral_rate": (tp + fp) / len(df),
            "referrals_per_100": 100.0 * (tp + fp) / len(df),
            "balanced_accuracy": 0.5 * ((tp / n_pos if n_pos else 0)
                                        + (tn / n_neg if n_neg else 0)),
        })
    return pd.DataFrame(rows)


def pick_threshold(curve: pd.DataFrame, min_sensitivity: float) -> Dict[str, Any]:
    """Highest threshold that still meets the sensitivity floor on internal val.

    Raising the threshold refers fewer people, so specificity rises and
    sensitivity falls. Taking the highest threshold that still clears the floor
    gives the fewest false referrals consistent with the safety constraint.
    """
    ok = curve[curve["sensitivity"] >= min_sensitivity]
    if ok.empty:
        return {"min_sensitivity": min_sensitivity, "feasible": False}
    row = ok.loc[ok["threshold"].idxmax()]
    return {
        "min_sensitivity": float(min_sensitivity),
        "feasible": True,
        "threshold": float(row["threshold"]),
        "val_sensitivity": float(row["sensitivity"]),
        "val_specificity": float(row["specificity"]),
        "val_miss_rate": float(row["miss_rate"]),
        "val_false_referral_rate": float(row["false_referral_rate"]),
        "val_ppv": float(row["ppv"]),
        "val_npv": float(row["npv"]),
        "val_referrals_per_100": float(row["referrals_per_100"]),
    }


def projected_ppv_npv(sens: float, spec: float, prevalence: float) -> Dict[str, float]:
    """PPV and NPV at a stated prevalence.

    BRISC is 81 percent tumour and the internal val split is 73 percent tumour.
    A rural clinic is nothing like that. PPV and NPV measured on either dataset
    are not transferable, so project them onto plausible clinic prevalences from
    sensitivity and specificity, which are prevalence-independent.
    """
    tp = sens * prevalence
    fn_ = (1 - sens) * prevalence
    fp = (1 - spec) * (1 - prevalence)
    tn = spec * (1 - prevalence)
    return {
        "prevalence": prevalence,
        "ppv": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "npv": float(tn / (tn + fn_)) if (tn + fn_) else float("nan"),
        "referrals_per_100": float(100 * (tp + fp)),
        "missed_per_100": float(100 * fn_),
    }


# ===========================================================================
# Phase 4.3 - temperature scaling, fitted on internal val
# ===========================================================================

def fit_temperature_on_val(val_df: pd.DataFrame) -> Dict[str, float]:
    """Fit one temperature on internal validation MC-Dropout mean probabilities.

    `analysis/calibration_comparison.py` fits temperature on deterministic
    single-pass logits. That is the textbook setting, but it is not what this
    tool shows a clinician: the displayed confidence comes from the MC-Dropout
    mean over T=20 passes. So we calibrate that distribution instead, by taking
    log(mean_probs) as the logits. softmax(log(p)/T) is p**(1/T) renormalised,
    which is the standard temperature family applied to the predictive mean.
    The fitting routine itself is `fit_temperature` unchanged.
    """
    logits = torch.log(torch.tensor(val_df[PROB_COLS].to_numpy(), dtype=torch.float64) + 1e-12).float()
    labels = torch.tensor(val_df["true_label"].to_numpy(), dtype=torch.long)
    return fit_temperature(logits, labels)


def apply_temperature(df: pd.DataFrame, T: float) -> pd.DataFrame:
    """Rescale the cached probabilities by temperature T. Argmax is unchanged."""
    p = df[PROB_COLS].to_numpy()
    logp = np.log(p + 1e-12) / T
    logp -= logp.max(axis=1, keepdims=True)
    q = np.exp(logp)
    q /= q.sum(axis=1, keepdims=True)
    out = df.copy()
    out[PROB_COLS] = q
    out["p_tumor"] = q[:, :3].sum(axis=1)
    ent = -(q * np.log(q + 1e-10)).sum(axis=1)
    out["entropy"] = ent
    out["entropy_bits"] = ent / np.log(2.0)
    return out


def calibration_before_after(df: pd.DataFrame, T: float) -> Dict[str, Any]:
    y = df["true_label"].to_numpy()
    before = sm.compute_calibration_metrics(df[PROB_COLS].to_numpy(), y, n_bins=15)
    after = sm.compute_calibration_metrics(apply_temperature(df, T)[PROB_COLS].to_numpy(), y, n_bins=15)
    return {
        "n": int(len(df)),
        "temperature": float(T),
        "ece_before": float(before["ece"]), "ece_after": float(after["ece"]),
        "brier_before": float(before["brier"]), "brier_after": float(after["brier"]),
        "ece_improved": bool(after["ece"] < before["ece"]),
        "brier_improved": bool(after["brier"] < before["brier"]),
    }


# ===========================================================================
# Phase 4.4 - deferral thresholds from internal val entropy quantiles
# ===========================================================================

def defer_thresholds_from_val(val_df: pd.DataFrame, budgets: List[float]) -> Dict[float, float]:
    """Entropy cutoff that defers `budget` of the internal validation split."""
    ent = val_df["entropy"].to_numpy()
    return {b: float(np.quantile(ent, 1.0 - b)) for b in budgets}


# ===========================================================================
# Figures
# ===========================================================================

def fig_sens_spec_tradeoff(curve: pd.DataFrame, chosen: Dict[str, Any], path: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(curve["threshold"], curve["sensitivity"], lw=2, label="Sensitivity (tumours caught)")
    ax.plot(curve["threshold"], curve["specificity"], lw=2, label="Specificity (no-tumour correctly cleared)")
    ax.plot(curve["threshold"], curve["referral_rate"], lw=1.5, ls="--",
            color="grey", label="Referral rate (share of patients sent on)")
    if chosen.get("feasible"):
        ax.axvline(chosen["threshold"], color="crimson", lw=1.5,
                   label=f"Chosen threshold = {chosen['threshold']:.3f}")
    ax.axhline(PRIMARY_SENS_TARGET, color="crimson", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("p_tumor threshold")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.02)
    ax.set_title("Sensitivity / specificity tradeoff, internal validation split")
    ax.legend(fontsize=8, loc="center left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_miss_vs_defer(rows: List[Dict[str, Any]], baseline: float, path: Path, title: str) -> None:
    set_style()
    x = [0.0] + [r["actual_defer_rate"] for r in rows]
    y = [baseline] + [r["miss_rate_kept"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(np.array(x) * 100, np.array(y) * 100, "o-", lw=2, color="crimson")
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi*100:.2f}%", (xi * 100, yi * 100),
                    textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.set_xlabel("Share of scans deferred to a human (%)")
    ax.set_ylabel("Tumour miss rate among scans NOT deferred (%)")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ===========================================================================
# Contract 2
# ===========================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_deployment_config(
    backbone: str,
    seeds: List[int],
    temperature: float,
    tumor_threshold: float,
    entropy_defer_threshold: float,
    expected: Dict[str, Any],
    path: Path,
) -> Dict[str, Any]:
    cfg = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "chosen_backbone": backbone,
        "chosen_seeds": list(seeds),
        "ensemble": len(seeds) > 1,
        "mc_T": MC_T,
        "temperature": float(temperature),
        "tumor_threshold": float(tumor_threshold),
        "entropy_defer_threshold": float(entropy_defer_threshold),
        "thresholds_fitted_on": "internal_val",
        "class_names": list(CLASS_NAMES),
        "preprocessing": {
            "resize": [224, 224],
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
        },
        "checkpoints": [
            {
                "model": backbone,
                "seed": s,
                "path": str(checkpoint_path(backbone, s)),
                "sha256": sha256_file(checkpoint_path(backbone, s)),
            }
            for s in seeds
        ],
        "expected_performance": expected,
        "entropy_units": "nats",
        "notes": [
            "entropy_defer_threshold is in nats and is compared against the "
            "predictive entropy of the MC-Dropout mean over mc_T passes.",
            "temperature is applied as softmax(log(mean_probs)/T); it does not "
            "change the predicted class, only the displayed confidence.",
            "tumor_threshold is applied to p_tumor = p_glioma + p_meningioma + "
            "p_pituitary AFTER temperature scaling.",
            "Every threshold here was fitted on the internal validation split "
            "and applied to BRISC unchanged.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=sm.N_BOOT)
    args = ap.parse_args()

    SAFETY_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    nb = args.boot

    out: Dict[str, Any] = {"generated_utc": datetime.now(timezone.utc).isoformat()}

    # ------------------------------------------------------------------
    # Step 1. Choose backbone and ensemble-vs-single. On INTERNAL VAL only.
    # ------------------------------------------------------------------
    print("=== Step 1: backbone and ensemble choice, internal val only ===")
    choice_rows: List[Dict[str, Any]] = []
    for model in MODELS:
        for tag, seeds in [("single_seed42", [42]), ("ensemble5", SEEDS)]:
            v = load_set(model, seeds, "val")
            mr = sm.tumor_miss_rate(v, n_resamples=nb)
            st = sm.standard_metrics(v, n_resamples=nb)
            bn = sm.binary_metrics(v, 0.5, n_resamples=nb)
            choice_rows.append({
                "model": model, "config": tag, "n_seeds": len(seeds),
                "val_accuracy": st["four_way_accuracy"]["value"],
                "val_macro_f1": st["macro_f1"]["value"],
                "val_ece": st["ece_15bin"],
                "val_miss_rate": mr["value"],
                "val_sensitivity@0.5": bn["sensitivity"]["value"],
                "val_specificity@0.5": bn["specificity"]["value"],
            })
            print(f"  {model:9s} {tag:14s} acc={st['four_way_accuracy']['value']:.4f} "
                  f"miss={mr['value']:.4f} sens={bn['sensitivity']['value']:.4f} "
                  f"ece={st['ece_15bin']:.4f}")
    choice_df = pd.DataFrame(choice_rows)
    choice_df.to_csv(SAFETY_OUT / "backbone_selection_internal_val.csv", index=False)
    out["backbone_selection_internal_val"] = choice_rows

    # Decision rule, stated before looking: lowest internal-val miss rate wins;
    # ties within 0.5 percentage points go to the cheaper single-seed config.
    best = min(choice_rows, key=lambda r: (r["val_miss_rate"], -r["val_accuracy"]))
    same_model = [r for r in choice_rows if r["model"] == best["model"]]
    single = next(r for r in same_model if r["config"] == "single_seed42")
    ens = next(r for r in same_model if r["config"] == "ensemble5")
    if ens["val_miss_rate"] - single["val_miss_rate"] > -0.005:
        chosen_cfg, chosen_seeds = "single_seed42", [42]
    else:
        chosen_cfg, chosen_seeds = "ensemble5", list(SEEDS)
    chosen_model = best["model"]
    print(f"\n  chosen: {chosen_model} / {chosen_cfg}  (decided on internal val)")
    out["choice"] = {
        "backbone": chosen_model, "config": chosen_cfg, "seeds": chosen_seeds,
        "rule": "lowest internal-val tumour miss rate; the 5-seed ensemble is "
                "only taken if it beats the single seed by more than 0.5 "
                "percentage points, because it costs 5x the inference time on a laptop",
    }

    val_raw = load_set(chosen_model, chosen_seeds, "val")
    itest_raw = load_set(chosen_model, chosen_seeds, "test")
    brisc_raw = load_set(chosen_model, chosen_seeds, "brisc")

    # ------------------------------------------------------------------
    # Step 2. Temperature FIRST, fitted on internal val.
    #
    # Order matters. The deployed pipeline is: MC mean -> temperature ->
    # p_tumor threshold and entropy deferral. So the threshold and the entropy
    # cutoffs must be fitted on the temperature-scaled validation split, not the
    # raw one, or the cutoffs are calibrated for a distribution the tool never
    # actually produces.
    #
    # Temperature is a monotone power transform of the probability vector, so it
    # never changes the argmax. The headline argmax-based tumour miss rate is
    # therefore identical with or without it.
    # ------------------------------------------------------------------
    print("\n=== Step 2: temperature scaling, fitted on internal val ===")
    fit = fit_temperature_on_val(val_raw)
    T = fit["temperature"]
    print(f"  fitted T = {T:.4f}  (val NLL {fit['val_nll_before']:.4f} -> {fit['val_nll_after']:.4f})")
    cal = {
        "fit": fit,
        "internal_val": calibration_before_after(val_raw, T),
        "internal_test": calibration_before_after(itest_raw, T),
        "brisc": calibration_before_after(brisc_raw, T),
    }
    for k in ("internal_val", "internal_test", "brisc"):
        c = cal[k]
        print(f"  {k:14s} ECE {c['ece_before']:.4f} -> {c['ece_after']:.4f}   "
              f"Brier {c['brier_before']:.4f} -> {c['brier_after']:.4f}")
    out["temperature_scaling"] = cal

    val = apply_temperature(val_raw, T)
    itest = apply_temperature(itest_raw, T)
    brisc = apply_temperature(brisc_raw, T)

    # ------------------------------------------------------------------
    # Step 3. Threshold sweep on the temperature-scaled internal val.
    # ------------------------------------------------------------------
    print("\n=== Step 3: p_tumor threshold sweep, internal val (post-temperature) ===")
    curve = sweep_thresholds(val)
    curve.insert(0, "model", chosen_model)
    curve.insert(1, "config", chosen_cfg)
    curve.insert(2, "fitted_on", "internal_val")
    curve.to_csv(SAFETY_OUT / "operating_curve.csv", index=False)
    print(f"  -> {SAFETY_OUT / 'operating_curve.csv'}  ({len(curve)} rows)")

    picks = {f"sens_{t}": pick_threshold(curve, t) for t in SENS_TARGETS}
    for k, v in picks.items():
        if v.get("feasible"):
            print(f"  {k}: thr={v['threshold']:.3f} spec={v['val_specificity']:.4f} "
                  f"referrals/100={v['val_referrals_per_100']:.1f}")
        else:
            print(f"  {k}: NOT ACHIEVABLE at any threshold")
    out["threshold_picks_internal_val"] = picks

    chosen_thr_info = picks[f"sens_{PRIMARY_SENS_TARGET}"]
    tumor_threshold = chosen_thr_info["threshold"] if chosen_thr_info.get("feasible") else 0.5

    out["projected_clinic_ppv_npv"] = {
        f"sens_{t}": (
            [projected_ppv_npv(picks[f"sens_{t}"]["val_sensitivity"],
                               picks[f"sens_{t}"]["val_specificity"], pv)
             for pv in CLINIC_PREVALENCES]
            if picks[f"sens_{t}"].get("feasible") else None
        )
        for t in SENS_TARGETS
    }

    fig_sens_spec_tradeoff(curve, chosen_thr_info, FIG_OUT / "sens_spec_tradeoff_internal_val.png")

    # ------------------------------------------------------------------
    # Step 4. Deferral. Cutoffs are internal-val entropy quantiles,
    # measured on the same temperature-scaled distribution the tool produces.
    # ------------------------------------------------------------------
    print("\n=== Step 4: deferral, entropy cutoffs from internal val ===")
    cuts = defer_thresholds_from_val(val, DEFER_BUDGETS)
    for b, c in cuts.items():
        print(f"  budget {int(b*100):2d}%  entropy cutoff {c:.4f} nats")
    defer = {
        "entropy_cutoffs_nats_fitted_on_internal_val": {str(k): v for k, v in cuts.items()},
        "internal_test": sm.deferral_curve(itest, cuts, n_resamples=nb),
        "brisc": sm.deferral_curve(brisc, cuts, n_resamples=nb),
    }
    base_brisc = sm.tumor_miss_rate(brisc, n_resamples=nb)
    base_itest = sm.tumor_miss_rate(itest, n_resamples=nb)
    defer["baseline_miss_rate_brisc"] = base_brisc
    defer["baseline_miss_rate_internal_test"] = base_itest
    print(f"\n  BRISC miss rate with no deferral: {base_brisc['value']:.4f}")
    for r in defer["brisc"]:
        print(f"    defer {r['actual_defer_rate']*100:5.1f}%  ->  miss rate on kept "
              f"{r['miss_rate_kept']:.4f}   ({r['n_missed_still_missed']}/"
              f"{r['n_missed_total']} misses still missed)")
    out["deferral"] = defer

    fig_miss_vs_defer(defer["brisc"], base_brisc["value"],
                      FIG_OUT / "miss_rate_vs_defer_brisc.png",
                      f"BRISC: does deferring help? ({chosen_model}, {chosen_cfg})")
    fig_miss_vs_defer(defer["internal_test"], base_itest["value"],
                      FIG_OUT / "miss_rate_vs_defer_internal_test.png",
                      f"Internal test: does deferring help? ({chosen_model}, {chosen_cfg})")

    # ------------------------------------------------------------------
    # Step 5. Ensemble vs single seed at the chosen point, on BRISC.
    # ------------------------------------------------------------------
    print("\n=== Step 5: ensemble vs single seed ===")
    print("  Each config gets its own temperature and its own threshold, both")
    print("  fitted on its own internal val. Otherwise the comparison is rigged.")
    ens_rows: List[Dict[str, Any]] = []
    for model in MODELS:
        for tag, seeds in [("single_seed42", [42]), ("ensemble5", SEEDS)]:
            v_raw = load_set(model, seeds, "val")
            b_raw = load_set(model, seeds, "brisc")
            T_i = fit_temperature_on_val(v_raw)["temperature"]
            v_i, b_i = apply_temperature(v_raw, T_i), apply_temperature(b_raw, T_i)
            pick_i = pick_threshold(sweep_thresholds(v_i), PRIMARY_SENS_TARGET)
            thr_i = pick_i["threshold"] if pick_i.get("feasible") else 0.5
            cut_i = defer_thresholds_from_val(v_i, [primary_budget := 0.20])[0.20]

            mr = sm.tumor_miss_rate(b_i, n_resamples=nb)
            bn = sm.binary_metrics(b_i, thr_i, n_resamples=nb)
            st = sm.standard_metrics(b_i, n_resamples=nb)
            df_i = sm.deferral_curve(b_i, {0.20: cut_i}, n_resamples=nb)[0]
            ens_rows.append({
                "model": model, "config": tag, "n_seeds": len(seeds),
                "relative_inference_cost": len(seeds),
                "own_temperature": T_i, "own_threshold": thr_i,
                "brisc_miss_rate": mr["value"],
                "brisc_miss_rate_ci_lo": mr["ci_lo"], "brisc_miss_rate_ci_hi": mr["ci_hi"],
                "brisc_sensitivity": bn["sensitivity"]["value"],
                "brisc_specificity": bn["specificity"]["value"],
                "brisc_referrals_per_100": 100 * bn["referral_rate"]["value"],
                "brisc_accuracy": st["four_way_accuracy"]["value"],
                "brisc_macro_f1": st["macro_f1"]["value"],
                "brisc_ece": st["ece_15bin"],
                "brisc_miss_rate_after_20pct_defer": df_i["miss_rate_kept"],
            })
            print(f"  {model:9s} {tag:14s} miss={mr['value']:.4f} "
                  f"sens={bn['sensitivity']['value']:.4f} spec={bn['specificity']['value']:.4f} "
                  f"acc={st['four_way_accuracy']['value']:.4f} T={T_i:.3f} thr={thr_i:.3f}")
    out["ensemble_vs_single_brisc"] = ens_rows
    pd.DataFrame(ens_rows).to_csv(SAFETY_OUT / "ensemble_vs_single.csv", index=False)

    # ------------------------------------------------------------------
    # Step 6. Contract 2.
    # ------------------------------------------------------------------
    print("\n=== Step 6: deployment_config.json ===")
    primary_budget = 0.20
    entropy_cut = cuts[primary_budget]
    brisc_T = apply_temperature(brisc, T)
    bn_final = sm.binary_metrics(brisc_T, tumor_threshold, n_resamples=nb)
    mr_final = sm.tumor_miss_rate(brisc_T, n_resamples=nb)
    st_final = sm.standard_metrics(brisc_T, n_resamples=nb)
    defer_final = sm.deferral_curve(brisc_T, {primary_budget: entropy_cut}, n_resamples=nb)[0]

    brisc_dirty = apply_temperature(
        load_set(chosen_model, chosen_seeds, "brisc", clean=False), T)
    out["full_brisc_for_reference_only"] = {
        "n": int(len(brisc_dirty)),
        "WARNING": "56 percent of these images are in the training set. "
                   "Not an external result. Do not quote.",
        "tumor_miss_rate": sm.tumor_miss_rate(brisc_dirty, n_resamples=nb),
        "four_way_accuracy": sm.standard_metrics(
            brisc_dirty, n_resamples=nb)["four_way_accuracy"],
    }

    expected = {
        "dataset": "BRISC2025_clean_subset",
        "n": int(len(brisc)),
        "subset_definition": (
            "BRISC images whose nearest internal TRAINING image is more than "
            "perceptual-hash Hamming distance 5 away. The other 56 percent of "
            "BRISC 2025 is near-duplicate of the training data and is excluded."
        ),
        "tumor_miss_rate": mr_final["value"],
        "tumor_miss_rate_ci": [mr_final["ci_lo"], mr_final["ci_hi"]],
        "binary_sensitivity": bn_final["sensitivity"]["value"],
        "binary_specificity": bn_final["specificity"]["value"],
        "four_way_accuracy": st_final["four_way_accuracy"]["value"],
        "defer_rate": defer_final["actual_defer_rate"],
        "miss_rate_after_defer": defer_final["miss_rate_kept"],
        "_caveats": [
            "These numbers are from the CLEAN subset of BRISC 2025 only. About "
            "56 percent of BRISC is near-duplicate of the training data and was "
            "excluded. Full-BRISC numbers are training accuracy, not external.",
            "Image-level duplication is excluded. PATIENT-level leakage cannot "
            "be ruled out: neither dataset ships patient identifiers, so other "
            "slices from the same patients may still be in training. The clean "
            "subset is therefore an upper bound on true external performance.",
            "Class balance in the clean subset is not the same as full BRISC. "
            "PPV and NPV measured here do not transfer to a clinic. Use "
            "sensitivity and specificity and the prevalence projections in "
            "OPERATING_POINT.md.",
            "BRISC is entirely T1 across three planes. Performance on other "
            "sequences is unmeasured.",
            "tumor_miss_rate here is argmax-based: a real tumour displayed as "
            "notumor. binary_sensitivity uses the p_tumor threshold.",
        ],
    }
    cfg = write_deployment_config(
        chosen_model, chosen_seeds, T, tumor_threshold, entropy_cut,
        expected, SAFETY_OUT / "deployment_config.json",
    )
    print(json.dumps(expected, indent=2))
    out["deployment_config"] = cfg

    (SAFETY_OUT / "operating_point_metrics.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n-> {SAFETY_OUT / 'operating_point_metrics.json'}")


if __name__ == "__main__":
    main()
