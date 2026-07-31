"""List the tumors the model missed while sounding confident, with filenames.

Why this exists
---------------
The model card's "Known failure modes" section needs concrete cases, not
adjectives. A missed tumor that the model flagged as uncertain is a case the
deferral machinery catches. A missed tumor the model was confident about is one
nothing catches, and it is the failure that reaches a clinician looking clean.

This script names those images so a human can open them and look.

Scope
-----
Internal held-out test split only, n = 1,112. Same merged Kaggle pool the model
trained on. **Not an external result.**

Alignment
---------
`analysis/export_predictions.py` builds the test loader via
`build_dataloaders_from_manifest`, which sets `shuffle=(split == "train")`, so
the test loader is unshuffled and its order is the manifest's test rows in
manifest order. This script asserts that alignment against the labels before
using it, and refuses to run if it does not hold.

Usage
-----
    python docs/confident_misses.py

Writes docs/results/confident_misses.json and copies the example images to
docs/results/confident_miss_examples/.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PRED_DIR = Path("analysis/results")
OUT_JSON = Path("docs/results/confident_misses.json")
OUT_IMG = Path("docs/results/confident_miss_examples")

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]
NOTUMOR = 3
SEEDS = [42, 123, 7, 2024, 31]
MODELS = ["resnet50", "vit"]

#: A miss is "confident" when the model put this much probability on no-tumor.
CONFIDENT = 0.80


def load(model: str, seed: int):
    d = json.loads((PRED_DIR / f"{model}_seed{seed}_predictions.json").read_text())
    assert d["class_names"] == CLASS_NAMES
    assert not d["provenance"]["is_smoke_test_run"]
    return (
        np.array(d["y_true"]),
        np.array(d["y_pred"]),
        np.array(d["mean_probs"]),
        np.array(d["entropy"]),
    )


def main() -> None:
    man = pd.read_csv("data/split_manifest.csv")
    test = man[man["split"] == "test"].reset_index(drop=True)
    print(f"manifest test rows: {len(test)}")

    y_true_ref, _, _, _ = load("resnet50", 42)
    assert len(y_true_ref) == len(test), (len(y_true_ref), len(test))
    # The alignment check. If the loader order ever changes, this fails loudly
    # instead of silently naming the wrong files.
    assert (test["label"].to_numpy() == y_true_ref).all(), (
        "Prediction order does not match manifest test order. "
        "Do not trust any filename this script would produce."
    )
    print("alignment check passed: prediction order == manifest test order")

    # How many of the 5 seeds missed each image, and how confident they were.
    missed_by = defaultdict(list)
    for model in MODELS:
        for seed in SEEDS:
            y_true, y_pred, probs, entropy = load(model, seed)
            miss = (y_true != NOTUMOR) & (y_pred == NOTUMOR)
            for i in np.where(miss)[0]:
                missed_by[(model, int(i))].append(
                    {
                        "seed": seed,
                        "p_notumor": float(probs[i][NOTUMOR]),
                        "entropy_bits": float(entropy[i]),
                    }
                )

    records = []
    for (model, i), events in missed_by.items():
        row = test.iloc[i]
        p_max = max(e["p_notumor"] for e in events)
        records.append(
            {
                "model": model,
                "test_index": i,
                "filepath": row["filepath"],
                "filename": Path(row["filepath"]).name,
                "true_class": row["class_name"],
                "phash_cluster": int(row["patient_id"]),
                "n_seeds_missed": len(events),
                "max_p_notumor": p_max,
                "min_entropy_bits": min(e["entropy_bits"] for e in events),
                "confident": p_max >= CONFIDENT,
                "events": sorted(events, key=lambda e: -e["p_notumor"]),
            }
        )

    records.sort(key=lambda r: (-r["n_seeds_missed"], -r["max_p_notumor"]))

    confident = [r for r in records if r["confident"]]
    unanimous = [r for r in records if r["n_seeds_missed"] == len(SEEDS)]

    # Distinct images, since the same image can be missed by both backbones.
    distinct_files = {r["filename"] for r in records}
    distinct_confident = {r["filename"] for r in confident}

    summary = {
        "_what_this_is": (
            "Tumors called no-tumor on the internal held-out test split "
            "(n=1112). Same pool the model trained on, held out at "
            "near-duplicate-cluster level. NOT an external result."
        ),
        "_confident_threshold_p_notumor": CONFIDENT,
        "_entropy_units": "bits (src/code.py uses torch.log2)",
        "n_distinct_images_missed_by_any_model_or_seed": len(distinct_files),
        "n_distinct_images_missed_confidently": len(distinct_confident),
        "n_missed_by_all_5_seeds_of_a_backbone": len(unanimous),
        "by_true_class": {
            k: int(v)
            for k, v in pd.Series([r["true_class"] for r in records])
            .value_counts()
            .items()
        },
        "confident_by_true_class": {
            k: int(v)
            for k, v in pd.Series([r["true_class"] for r in confident])
            .value_counts()
            .items()
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"summary": summary, "cases": records}, indent=2))

    # Copy the worst cases so a human can actually look at them.
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    copied = []
    for r in records:
        if not (r["confident"] and r["n_seeds_missed"] >= 3):
            continue
        src = Path(r["filepath"])
        if not src.exists():
            continue
        dst = OUT_IMG / (
            f"{r['true_class']}_{Path(r['filename']).stem}"
            f"_{r['model']}_{r['n_seeds_missed']}of5"
            f"_p{int(round(r['max_p_notumor'] * 100))}.jpg"
        )
        if not dst.exists():
            shutil.copy2(src, dst)
        copied.append(dst.name)

    print(json.dumps(summary, indent=2, default=str))
    print(f"\ncopied {len(copied)} example images to {OUT_IMG}")
    print("\nworst cases, missed by the most seeds at the highest confidence:")
    for r in records[:12]:
        print(
            f"  {r['true_class']:11} {r['filename']:22} {r['model']:9} "
            f"missed by {r['n_seeds_missed']}/5 seeds  "
            f"max p(no tumor)={r['max_p_notumor']:.3f}  "
            f"min entropy={r['min_entropy_bits']:.2f} bits"
        )


if __name__ == "__main__":
    main()
