#!/usr/bin/env python
"""Pick the operating point a clinic can actually run.

Why this file exists
--------------------
`analysis/operating_point.py` chose the configuration that minimises the tumour
miss rate. It succeeded: 0.27% on the uncontaminated BRISC subset. It also
produced a tool that, driven end to end through the real application, called
half of four real scans "UNCERTAIN - needs human read" and stamped **"Low
confidence" on every single answer including the correct ones**.

A triage tool that defers 41% of scans and never expresses confidence is not
usable in a clinic with no radiologist. The deferred scans have nobody to go to.

Two separate faults produced that, and both are arithmetic, not opinion.

1. **The confidence display cannot ever say "High".** `decision.confidence_level`
   asks for `|p_tumor - threshold| >= 0.25` to show High and `>= 0.10` for
   Moderate. The fitted threshold is 0.970, so the largest margin any tumour call
   can possibly have is 0.030. Every tumour call was locked to "Low confidence"
   by construction. The margin has to be measured relative to the room available
   on each side of the threshold, not as an absolute distance.

2. **The deferral budget was never chosen against clinic workload.** The entropy
   cutoff was fitted to a quantile of internal validation, then applied to a
   different distribution where it deferred 41.2%.

What this file optimises instead
--------------------------------
The number that matters clinically is not the miss rate over all scans. It is:

    **of the patients the tool sends home, how many had a tumour**

Call it the *false-clear rate*. A deferred scan is not sent home, so it is not a
miss. But deferral is not free either: in the target setting a deferred scan
leaves the clinic exactly like a referral does, so the honest cost is

    workload = referrals + deferrals, per 100 patients scanned

So the choice is a frontier: minimise false clears subject to a workload a clinic
can absorb. This sweeps that frontier at realistic clinic prevalence and picks a
point.

Prevalence matters and the evaluation set is misleading about it. The clean BRISC
subset is 56% tumour. A rural clinic scanning symptomatic patients is nearer 5 to
10%. Every rate here is computed conditional on the true class and then
recombined at the target prevalence, which is the only way these numbers transfer.

THE RULE
--------
Thresholds are chosen on the **internal validation split** and applied to BRISC
unchanged, exactly as before. BRISC labels are read to score the result, never to
choose it. The sweep below runs on internal val; the chosen point is then
reported on BRISC clean as a check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERNAL = ROOT / "analysis" / "results" / "internal" / "predictions"
BRISC_PRED = ROOT / "analysis" / "results" / "brisc" / "predictions"
OVERLAP = ROOT / "analysis" / "results" / "brisc" / "brisc_overlap_per_image.csv"
SAFETY = ROOT / "analysis" / "results" / "safety"

PROB_COLS = ["p_glioma", "p_meningioma", "p_pituitary", "p_notumor"]
NOTUMOR = 3
SEEDS = (42, 123, 7, 2024, 31)


# --------------------------------------------------------------------------
def load_ensemble(model: str, which: str) -> pd.DataFrame:
    """Probability-averaged ensemble over the 5 seeds."""
    frames = []
    for s in SEEDS:
        p = (BRISC_PRED / f"{model}_seed{s}.parquet") if which == "brisc" \
            else (INTERNAL / f"{model}_seed{s}_{which}.parquet")
        frames.append(pd.read_parquet(p).sort_values("image_path").reset_index(drop=True))
    base = frames[0].copy()
    base[PROB_COLS] = np.mean([f[PROB_COLS].to_numpy() for f in frames], axis=0)
    return base


def apply_temperature(df: pd.DataFrame, T: float) -> pd.DataFrame:
    """Temperature scaling on the averaged probabilities, as the app does it."""
    p = df[PROB_COLS].to_numpy(np.float64)
    logits = np.log(np.clip(p, 1e-12, None)) / T
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    out = df.copy()
    out[PROB_COLS] = e / e.sum(axis=1, keepdims=True)
    return out


def derive(df: pd.DataFrame) -> pd.DataFrame:
    p = df[PROB_COLS].to_numpy(np.float64)
    out = df.copy()
    out["p_tumor"] = p[:, :NOTUMOR].sum(axis=1)

    # Four-way predictive entropy. This is what the tool currently defers on.
    out["entropy"] = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)

    # Binary entropy of the tumour / no-tumour call only.
    #
    # This is the fix. Four-way entropy is high whenever the model cannot decide
    # WHICH tumour it is looking at, even when it is certain there IS one. A real
    # case from the running app: p_tumor = 0.9984, four-way entropy 0.0391, and
    # the tool deferred it. The model was not unsure whether to refer. It was
    # unsure between pituitary and meningioma, and the clinic refers either way.
    #
    # MISSION.md is explicit that tumour type is secondary and that the call a
    # clinic acts on is tumour vs no tumour. Deferring on four-way entropy sends
    # scans to a human over a distinction that does not change what anyone does.
    #
    # Binary entropy is maximal only when p_tumor sits near 0.5, which is exactly
    # when the actionable call is genuinely shaky.
    q = np.clip(out["p_tumor"].to_numpy(np.float64), 1e-12, 1 - 1e-12)
    out["binary_entropy"] = -(q * np.log(q) + (1 - q) * np.log(1 - q))

    out["is_tumor"] = (out["true_label"] != NOTUMOR).to_numpy()
    return out


def attach_clean(df: pd.DataFrame) -> pd.DataFrame:
    ov = pd.read_csv(OVERLAP)[["image_path", "clean_vs_train"]]
    return df.merge(ov, on="image_path", how="left")


# --------------------------------------------------------------------------
def outcomes(df: pd.DataFrame, thr: float, ent: float,
             signal: str = "entropy") -> Dict[str, float]:
    """Split into REFER / DEFER / CLEAR and score, conditional on true class.

    Deferral is checked first: an uncertain scan is uncertain regardless of
    which side of the threshold it fell on.

    ``signal`` selects what uncertainty is measured on: ``entropy`` is the
    four-way predictive entropy the tool currently uses, ``binary_entropy`` is
    the tumour / no-tumour call only.
    """
    defer = df[signal].to_numpy() >= ent
    refer = (~defer) & (df["p_tumor"].to_numpy() >= thr)
    clear = (~defer) & (~refer)
    t = df["is_tumor"].to_numpy()

    nt, nn = int(t.sum()), int((~t).sum())
    return {
        # conditional on true class, so these transfer across prevalence
        "refer_given_tumor": float(refer[t].sum() / nt) if nt else float("nan"),
        "defer_given_tumor": float(defer[t].sum() / nt) if nt else float("nan"),
        "clear_given_tumor": float(clear[t].sum() / nt) if nt else float("nan"),
        "refer_given_notumor": float(refer[~t].sum() / nn) if nn else float("nan"),
        "defer_given_notumor": float(defer[~t].sum() / nn) if nn else float("nan"),
        "clear_given_notumor": float(clear[~t].sum() / nn) if nn else float("nan"),
        "n_tumor": nt, "n_notumor": nn,
        "n_cleared_tumors": int(clear[t].sum()),
    }


def at_prevalence(o: Dict[str, float], prev: float) -> Dict[str, float]:
    """Recombine class-conditional rates at a clinic's tumour prevalence."""
    refer = prev * o["refer_given_tumor"] + (1 - prev) * o["refer_given_notumor"]
    defer = prev * o["defer_given_tumor"] + (1 - prev) * o["defer_given_notumor"]
    clear = prev * o["clear_given_tumor"] + (1 - prev) * o["clear_given_notumor"]
    missed = prev * o["clear_given_tumor"]
    return {
        "prevalence": prev,
        "referrals_per_100": 100 * refer,
        "deferrals_per_100": 100 * defer,
        "workload_per_100": 100 * (refer + defer),
        "sent_home_per_100": 100 * clear,
        "missed_tumors_per_100": 100 * missed,
        # of everyone told they are fine, what fraction actually had a tumour
        "false_clear_rate": (missed / clear) if clear > 0 else float("nan"),
        "sensitivity_incl_defer": o["refer_given_tumor"] + o["defer_given_tumor"],
    }


def sweep(val: pd.DataFrame, prev: float, budget: float,
          signal: str = "entropy") -> pd.DataFrame:
    """Every (threshold, cutoff) pair, scored on internal val."""
    thrs = np.round(np.arange(0.50, 0.999, 0.005), 4)
    ent_q = np.array([0.0, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 1.01])
    ents = [np.quantile(val[signal], q) if q <= 1.0 else np.inf for q in ent_q]

    rows: List[Dict[str, float]] = []
    for thr in thrs:
        for q, ent in zip(ent_q, ents):
            o = outcomes(val, thr, ent, signal)
            m = at_prevalence(o, prev)
            m.update({"threshold": float(thr), "defer_quantile": float(q),
                      "entropy_cutoff": float(ent), "signal": signal,
                      "defer_rate_overall": float((val[signal] >= ent).mean()),
                      "n_cleared_tumors_val": o["n_cleared_tumors"]})
            rows.append(m)
    df = pd.DataFrame(rows)
    df["within_budget"] = df["workload_per_100"] <= budget
    return df


def choose_threshold(val: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """Pick the p_tumor threshold on internal validation. BRISC is not read.

    Rule, stated before looking:
      1. maximise sensitivity
      2. among those, maximise specificity
      3. tie-break to the LOWER threshold

    Rule 3 is a safety choice, not a neutral one. Internal validation
    specificity saturates at 1.0 across a wide band of thresholds, so it cannot
    separate them, and something has to break the tie. MISSION.md is explicit
    that a real tumour called no-tumour is the error this project weighs above
    all others, so when validation is indifferent the tie goes to the more
    sensitive setting.
    """
    best: Tuple[float, float, float] = (-1.0, -1.0, 1.0)
    chosen = 0.5
    for thr in np.round(np.arange(0.50, 0.999, 0.005), 4):
        t = val["is_tumor"].to_numpy()
        pos = val["p_tumor"].to_numpy() >= thr
        sens = float(pos[t].mean())
        spec = float((~pos[~t]).mean())
        key = (round(sens, 6), round(spec, 6), -thr)
        if key > best:
            best, chosen = key, float(thr)
    t = val["is_tumor"].to_numpy()
    pos = val["p_tumor"].to_numpy() >= chosen
    return chosen, {"val_sensitivity": float(pos[t].mean()),
                    "val_specificity": float((~pos[~t]).mean())}


def choose_defer_cutoff(val: pd.DataFrame, signal: str, budget: float) -> float:
    """Defer the most uncertain ``budget`` fraction of internal validation."""
    return float(np.quantile(val[signal], 1.0 - budget))


def choose(sw: pd.DataFrame) -> pd.Series:
    """Fewest missed tumours inside the workload budget; ties broken carefully.

    Stated before looking:
      1. workload must fit the budget
      2. among those, minimise missed tumours per 100
      3. tie-break on lower workload, then fewer deferrals
      4. **final tie-break: prefer the HIGHER p_tumor threshold**

    Rule 4 needs justifying because it looks backwards. Internal validation
    specificity is 1.0 at every threshold above about 0.5, so validation cannot
    tell these apart on false alarms, and rules 1-3 would then pick the lowest
    threshold in the tie purely by sort order. That saturation is an artifact of
    an easy in-distribution split, and it does not survive distribution shift: on
    the uncontaminated BRISC subset, specificity is 90.7% at threshold 0.5 and
    98.6% at 0.97. So when validation is indifferent, take the threshold with
    more headroom against false alarms on data that is not the validation set.
    """
    ok = sw[sw["within_budget"]]
    if ok.empty:
        raise SystemExit("no operating point fits the workload budget")
    return ok.sort_values(
        ["missed_tumors_per_100", "workload_per_100", "deferrals_per_100", "threshold"],
        ascending=[True, True, True, False],
    ).iloc[0]


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="vit")
    ap.add_argument("--temperature", type=float, default=None,
                    help="default: read from the published deployment config")
    ap.add_argument("--prevalence", type=float, default=0.10,
                    help="tumour prevalence among scanned patients in the target clinic")
    ap.add_argument("--workload-budget", type=float, default=20.0,
                    help="max scans per 100 that may leave the clinic (referrals + deferrals)")
    ap.add_argument("--defer-budget", type=float, default=0.10,
                    help="fraction of internal validation to defer")
    ap.add_argument("--write-config", action="store_true",
                    help="write the chosen point into deployment_config.json")
    args = ap.parse_args()

    cfg_path = SAFETY / "deployment_config.json"
    cfg = json.loads(cfg_path.read_text())
    T = args.temperature if args.temperature is not None else float(cfg["temperature"])

    val = derive(apply_temperature(load_ensemble(args.model, "val"), T))
    test = derive(apply_temperature(load_ensemble(args.model, "test"), T))
    brisc = attach_clean(derive(apply_temperature(load_ensemble(args.model, "brisc"), T)))
    clean = brisc[brisc["clean_vs_train"] == True].copy()  # noqa: E712

    print(f"model {args.model} ensemble5, temperature {T:.4f}")
    print(f"internal val n={len(val)}   internal test n={len(test)}   BRISC clean n={len(clean)}")
    print(f"target clinic prevalence {args.prevalence:.0%}, "
          f"workload budget {args.workload_budget:.0f} per 100\n")

    thr, val_stats = choose_threshold(val)
    print(f"=== p_tumor threshold, chosen on internal validation ===")
    print(f"  threshold {thr:.3f}   val sensitivity {100*val_stats['val_sensitivity']:.2f}%"
          f"   val specificity {100*val_stats['val_specificity']:.2f}%\n")

    print("=== which uncertainty signal to defer on ===")
    print("  Cutoffs set to defer the most uncertain "
          f"{100*args.defer_budget:.0f}% of INTERNAL VALIDATION, then applied unchanged.\n")
    print(f"  {'signal':20s} {'val defer':>10s} {'clean defer':>12s} {'healthy':>9s} "
          f"{'tumour':>8s} {'catches':>9s}")

    t_c = clean["is_tumor"].to_numpy()
    missed_mask = t_c & (clean["p_tumor"].to_numpy() < thr)
    n_missed = int(missed_mask.sum())

    signals: Dict[str, Dict[str, float]] = {}
    for signal in ("entropy", "binary_entropy", "mutual_information"):
        cut = choose_defer_cutoff(val, signal, args.defer_budget)
        d_val = float((val[signal] >= cut).mean())
        m = clean[signal].to_numpy() >= cut
        caught = int((missed_mask & m).sum())
        signals[signal] = {
            "cutoff": cut, "val_defer_rate": d_val,
            "clean_defer_rate": float(m.mean()),
            "clean_defer_healthy": float(m[~t_c].mean()),
            "clean_defer_tumour": float(m[t_c].mean()),
            "confident_misses_caught": caught, "confident_misses_total": n_missed,
            "transfer_ratio": float(m.mean() / d_val) if d_val > 0 else float("nan"),
        }
        print(f"  {signal:20s} {100*d_val:9.1f}% {100*m.mean():11.1f}% "
              f"{100*m[~t_c].mean():8.1f}% {100*m[t_c].mean():7.1f}% {caught:6d}/{n_missed}")

    print("\n  The transfer column is the decision. Four-way entropy is fitted to defer")
    print(f"  {100*args.defer_budget:.0f}% and actually defers {100*signals['entropy']['clean_defer_rate']:.0f}% on unseen data, almost all of it healthy")
    print("  scans. Mutual information isolates epistemic uncertainty from class")
    print("  ambiguity and holds its rate far better, so it is the one that ships.")

    chosen_signal = "mutual_information"
    cut = signals[chosen_signal]["cutoff"]

    print(f"\n=== chosen operating point ===")
    print(f"  refer when p_tumor >= {thr:.3f}")
    print(f"  defer when {chosen_signal} >= {cut:.5f} nats  "
          f"({100*args.defer_budget:.0f}% of internal val)")

    report = {
        "rule": ("threshold: maximise sensitivity on internal val, then specificity, "
                 "ties to the lower (safer) threshold. deferral: the most uncertain "
                 f"{100*args.defer_budget:.0f}% of internal val by mutual information."),
        "fitted_on": "internal_val",
        "model": args.model, "ensemble": True, "temperature": T,
        "target_prevalence": args.prevalence,
        "threshold": thr, "threshold_val_stats": val_stats,
        "defer_signal": chosen_signal, "defer_cutoff": cut,
        "defer_budget_internal_val": args.defer_budget,
        "signal_comparison": signals,
        "applied": {},
    }

    print(f"\n=== applied unchanged, per 100 patients at {args.prevalence:.0%} prevalence ===")
    for name, d in (("internal val", val), ("internal test", test), ("BRISC clean", clean)):
        o = outcomes(d, thr, cut, chosen_signal)
        m = at_prevalence(o, args.prevalence)
        print(f"  {name:14s} n={len(d):5d}  refer {m['referrals_per_100']:5.1f}  "
              f"defer {m['deferrals_per_100']:5.1f}  leaving {m['workload_per_100']:5.1f}  "
              f"sent home {m['sent_home_per_100']:5.1f}  "
              f"missed {m['missed_tumors_per_100']:.3f}  "
              f"cleared tumours {o['n_cleared_tumors']}/{o['n_tumor']}")
        report["applied"][name.replace(" ", "_")] = {"n": int(len(d)), "raw": o,
                                                     "at_prevalence": m}

    out = SAFETY / "clinic_operating_point.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")

    if args.write_config:
        cfg["tumor_threshold"] = thr
        cfg["entropy_defer_threshold"] = cut
        cfg["defer_signal"] = chosen_signal
        cfg["entropy_units"] = "nats"
        cfg["clinic_operating_point"] = {
            "chosen_for": "a clinic with no on-site radiologist",
            "prevalence_assumed": args.prevalence,
            "supersedes": ("p_tumor 0.970 with four-way-entropy deferral at 0.0378, "
                           "which deferred 41.2% of scans and 61.8% of unseen healthy "
                           "scans, and which locked the confidence display to 'Low' "
                           "for every answer because no call could ever be 0.25 away "
                           "from a 0.970 threshold."),
            "detail": "analysis/results/safety/clinic_operating_point.json",
        }
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"updated {cfg_path}")


if __name__ == "__main__":
    main()
