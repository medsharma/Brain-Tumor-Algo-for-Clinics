"""Internal held-out test-split safety metrics, computed for the model card.

Why this exists
---------------
The model card leads with the tumor miss rate: how often a real tumor is called
no-tumor. Session A owns the canonical safety analysis. This script computes the
same quantity independently from the prediction exports that already existed in
`analysis/results/*_predictions.json` (produced from the real 5-seed run in
`results/20260703_155524`), so the model card has a documented internal number
and so A's figure has something to be checked against.

If this disagrees with A's number, A's is canonical and this script is the bug.

Scope
-----
Internal held-out test split only. n = 1,112. Same merged Kaggle pool the model
trained on, held out at the phash near-duplicate-cluster level. **This is not an
external result and nothing here should ever be presented as one.**

No GPU. No model loaded. Reads cached predictions only.

Usage
-----
    python docs/internal_safety_metrics.py

Writes docs/results/internal_safety_metrics.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PRED_DIR = Path("analysis/results")
OUT = Path("docs/results/internal_safety_metrics.json")

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]
NOTUMOR = 3
TUMOR = (0, 1, 2)
SEEDS = [42, 123, 7, 2024, 31]
MODELS = ["resnet50", "vit"]
N_BOOT = 2000
BOOT_SEED = 12345


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval. Behaves sanely when k is 0 or n, which a normal
    approximation does not, and the miss rate is expected to be near 0."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


def boot_ci(values: np.ndarray, stat, rng, n_boot: int = N_BOOT):
    """Percentile bootstrap over image indices."""
    n = len(values)
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = np.array([stat(values[i]) for i in idx])
    stats = stats[~np.isnan(stats)]
    if stats.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def load(model: str, seed: int):
    p = PRED_DIR / f"{model}_seed{seed}_predictions.json"
    d = json.loads(p.read_text())
    assert d["class_names"] == CLASS_NAMES, d["class_names"]
    assert not d["provenance"]["is_smoke_test_run"], f"{p} is a smoke run"
    return (
        np.array(d["y_true"]),
        np.array(d["y_pred"]),
        np.array(d["mean_probs"]),
        np.array(d["entropy"]),
        d["mc_T"],
    )


def metrics_for(y_true, y_pred, probs, entropy, rng):
    is_tumor = np.isin(y_true, TUMOR)
    pred_tumor = np.isin(y_pred, TUMOR)

    n_tumor = int(is_tumor.sum())
    n_notumor = int((~is_tumor).sum())

    # THE number: a real tumor called no-tumor.
    missed = int((is_tumor & ~pred_tumor).sum())
    miss_rate = missed / n_tumor

    # A no-tumor scan called a tumor. Costs a referral, not a life.
    fp = int((~is_tumor & pred_tumor).sum())
    false_alarm = fp / n_notumor

    sens = (n_tumor - missed) / n_tumor
    spec = (n_notumor - fp) / n_notumor
    acc4 = float((y_true == y_pred).mean())
    acc2 = float((is_tumor == pred_tumor).mean())

    # Per-class miss: of true class c, how often predicted notumor.
    per_class_miss = {}
    for c in TUMOR:
        m = y_true == c
        k = int((m & ~pred_tumor).sum())
        per_class_miss[CLASS_NAMES[c]] = {
            "n": int(m.sum()),
            "missed": k,
            "rate": k / int(m.sum()),
            "ci95_wilson": list(wilson(k, int(m.sum()))),
        }

    # Deferral: send the most-uncertain fraction to a human, by entropy.
    defer = {}
    for frac in (0.05, 0.10, 0.20):
        thr = float(np.quantile(entropy, 1 - frac))
        keep = entropy < thr
        kt = is_tumor & keep
        km = int((kt & ~pred_tumor).sum())
        defer[f"defer_{int(frac*100)}pct"] = {
            "entropy_threshold": thr,
            "coverage": float(keep.mean()),
            "accuracy_on_kept": float((y_true[keep] == y_pred[keep]).mean()),
            "n_tumor_kept": int(kt.sum()),
            "tumor_missed_among_kept": km,
            "miss_rate_among_kept": km / int(kt.sum()) if kt.sum() else float("nan"),
        }

    # Do the misses look uncertain? If a miss is confident, deferral cannot save it.
    miss_idx = np.where(is_tumor & ~pred_tumor)[0]
    missed_entropy = entropy[miss_idx].tolist()
    missed_conf = probs[miss_idx].max(axis=1).tolist() if miss_idx.size else []

    return {
        "n": int(len(y_true)),
        "n_tumor": n_tumor,
        "n_notumor": n_notumor,
        "four_way_accuracy": acc4,
        "binary_accuracy": acc2,
        "tumor_miss_count": missed,
        "tumor_miss_rate": miss_rate,
        "tumor_miss_rate_ci95_wilson": list(wilson(missed, n_tumor)),
        "binary_sensitivity": sens,
        "binary_sensitivity_ci95_wilson": list(wilson(n_tumor - missed, n_tumor)),
        "binary_specificity": spec,
        "binary_specificity_ci95_wilson": list(wilson(n_notumor - fp, n_notumor)),
        "false_alarm_count": fp,
        "false_alarm_rate": false_alarm,
        "per_class_miss": per_class_miss,
        "deferral": defer,
        "missed_case_entropies": missed_entropy,
        "missed_case_max_prob": missed_conf,
        "mean_entropy": float(entropy.mean()),
    }


def main() -> None:
    rng = np.random.default_rng(BOOT_SEED)
    out = {
        "_what_this_is": (
            "Internal held-out test split of the same merged Kaggle pool the "
            "model trained on, grouped at the phash near-duplicate-cluster "
            "level. n=1112. NOT an external result. Do not present as one."
        ),
        "_source": "analysis/results/{model}_seed{seed}_predictions.json, run results/20260703_155524",
        "_computed_by": "session E, docs/internal_safety_metrics.py",
        "_canonical_owner": "session A owns the authoritative safety analysis; if A disagrees, A wins",
        "dataset": "internal_test",
        "n": 1112,
        "mc_T": 20,
        "per_model": {},
    }

    for model in MODELS:
        per_seed = {}
        pooled_missed = 0
        pooled_tumor = 0
        for seed in SEEDS:
            y_true, y_pred, probs, entropy, mc_T = load(model, seed)
            assert mc_T == 20
            m = metrics_for(y_true, y_pred, probs, entropy, rng)
            per_seed[str(seed)] = m
            pooled_missed += m["tumor_miss_count"]
            pooled_tumor += m["n_tumor"]

        rates = np.array([per_seed[str(s)]["tumor_miss_rate"] for s in SEEDS])
        sens = np.array([per_seed[str(s)]["binary_sensitivity"] for s in SEEDS])
        spec = np.array([per_seed[str(s)]["binary_specificity"] for s in SEEDS])
        acc4 = np.array([per_seed[str(s)]["four_way_accuracy"] for s in SEEDS])

        out["per_model"][model] = {
            "per_seed": per_seed,
            "across_seeds": {
                "tumor_miss_rate_mean": float(rates.mean()),
                "tumor_miss_rate_min": float(rates.min()),
                "tumor_miss_rate_max": float(rates.max()),
                "binary_sensitivity_mean": float(sens.mean()),
                "binary_sensitivity_min": float(sens.min()),
                "binary_sensitivity_max": float(sens.max()),
                "binary_specificity_mean": float(spec.mean()),
                "binary_specificity_min": float(spec.min()),
                "binary_specificity_max": float(spec.max()),
                "four_way_accuracy_mean": float(acc4.mean()),
            },
            "pooled_over_seeds": {
                "_note": (
                    "Seeds score the same 1,112 images, so these are 5 correlated "
                    "reads of one test set, not 5,560 independent images. The CI "
                    "below is therefore optimistic. Treat the per-seed range as "
                    "the honest spread."
                ),
                "tumor_missed": pooled_missed,
                "tumor_evaluations": pooled_tumor,
                "tumor_miss_rate": pooled_missed / pooled_tumor,
                "ci95_wilson": list(wilson(pooled_missed, pooled_tumor)),
            },
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))

    # human-readable
    print(f"internal test split, n=1112, {out['per_model']['resnet50']['per_seed']['42']['n_tumor']} tumor images\n")
    for model in MODELS:
        a = out["per_model"][model]["across_seeds"]
        p = out["per_model"][model]["pooled_over_seeds"]
        print(f"{model}:")
        print(f"  tumor miss rate   mean {a['tumor_miss_rate_mean']:.4f}  "
              f"range {a['tumor_miss_rate_min']:.4f}-{a['tumor_miss_rate_max']:.4f}")
        print(f"    pooled {p['tumor_missed']}/{p['tumor_evaluations']} = {p['tumor_miss_rate']:.4f} "
              f"95% CI [{p['ci95_wilson'][0]:.4f}, {p['ci95_wilson'][1]:.4f}]")
        print(f"  sensitivity       mean {a['binary_sensitivity_mean']:.4f}  "
              f"range {a['binary_sensitivity_min']:.4f}-{a['binary_sensitivity_max']:.4f}")
        print(f"  specificity       mean {a['binary_specificity_mean']:.4f}  "
              f"range {a['binary_specificity_min']:.4f}-{a['binary_specificity_max']:.4f}")
        print(f"  four-way accuracy mean {a['four_way_accuracy_mean']:.4f}")
        for seed in SEEDS:
            s = out["per_model"][model]["per_seed"][str(seed)]
            d5 = s["deferral"]["defer_5pct"]
            print(f"   seed {seed:>4}: missed {s['tumor_miss_count']}/{s['n_tumor']}"
                  f"  false alarms {s['false_alarm_count']}/{s['n_notumor']}"
                  f"  | defer 5%: {d5['tumor_missed_among_kept']} misses remain")
        print()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
