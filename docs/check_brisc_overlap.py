"""Measure image overlap between the internal training pool and BRISC 2025.

Why this exists
---------------
The BRISC 2025 paper (Fateh et al., Scientific Data, 2026) states that BRISC
images were collated from the Cheng/Figshare, SARTAJ and Br35H collections,
aggregated via the Kaggle "Brain Tumor MRI Dataset" (Nickparvar). That is the
same pool this project trained on. So BRISC may not be an independent external
cohort at all. This script measures how much of BRISC is a near-duplicate of an
image the model was trained on.

Method
------
Perceptual hash (phash, 64-bit) of every image in both sets, then a Hamming
distance match. Threshold 5 is the same one src/code.py::_phash_cluster_ids uses
to define a near-duplicate cluster for the leakage-safe split, so a match here
means "close enough that the internal split logic would have refused to put
these two images on opposite sides of a train/test boundary".

Exact sha256 matching is also reported, but is expected to find nothing: BRISC
re-encoded everything to 512x512 JPEG.

CPU only. No model, no GPU, no labels used for any fitting.

Usage
-----
    python docs/check_brisc_overlap.py

Writes docs/results/brisc_overlap.json and prints a summary.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image

BRISC_ROOT = Path(r"C:\Users\medha\Downloads\archive (1)\brisc2025")
INTERNAL_MANIFEST = Path("data/split_manifest.csv")
OUT = Path("docs/results/brisc_overlap.json")

PHASH_THRESHOLD = 5  # same as src/code.py::_phash_cluster_ids

BRISC_TO_INTERNAL = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "no_tumor": "notumor",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_of(path: Path) -> int:
    with Image.open(path) as im:
        return int(str(imagehash.phash(im.convert("L"))), 16)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main() -> None:
    # ---- internal pool -----------------------------------------------------
    man = pd.read_csv(INTERNAL_MANIFEST)
    print(f"internal manifest: {len(man)} rows")

    internal = []
    for row in man.itertuples():
        fp = Path(row.filepath)
        if not fp.exists():
            # manifest stores absolute paths from the original run
            fp = Path("data/brain_tumor") / row.class_name / fp.name
        internal.append(
            {
                "path": fp,
                "class_name": row.class_name,
                "split": row.split,
                "sha256": sha256_of(fp),
                "phash": phash_of(fp),
            }
        )
    print(f"hashed {len(internal)} internal images")

    # ---- BRISC -------------------------------------------------------------
    bman = pd.read_csv(BRISC_ROOT / "manifest.csv")
    bman = bman[(bman["task"] == "classification") & (~bman["is_mask"])]
    print(f"brisc manifest: {len(bman)} classification rows")

    brisc = []
    for row in bman.itertuples():
        fp = BRISC_ROOT / str(row.relative_path).replace("\\", "/")
        brisc.append(
            {
                "path": fp,
                "rel": str(row.relative_path),
                "class_name": BRISC_TO_INTERNAL[row.tumor_label],
                "brisc_split": row.split,
                "plane": row.plane_label,
                "sha256": row.sha256,
                "phash": phash_of(fp),
            }
        )
    print(f"hashed {len(brisc)} brisc images")

    # ---- exact sha256 ------------------------------------------------------
    internal_sha = {r["sha256"]: r for r in internal}
    exact = [b for b in brisc if b["sha256"] in internal_sha]

    # ---- phash near-duplicate ---------------------------------------------
    # Bucket internal hashes by the high 16 bits to cut the comparison count,
    # but still fall back to a full scan so nothing is missed. 7.2k x 6k = 43M
    # 64-bit popcounts is fine on CPU.
    matches = []
    per_split = Counter()
    per_class = Counter()
    per_plane = Counter()
    brisc_split_hit = Counter()
    brisc_split_total = Counter()

    for b in brisc:
        brisc_split_total[b["brisc_split"]] += 1
        best = None
        best_d = 99
        for i in internal:
            d = hamming(b["phash"], i["phash"])
            if d < best_d:
                best_d, best = d, i
                if d == 0:
                    break
        if best_d <= PHASH_THRESHOLD:
            matches.append(
                {
                    "brisc": b["rel"],
                    "brisc_class": b["class_name"],
                    "brisc_split": b["brisc_split"],
                    "plane": b["plane"],
                    "internal": str(best["path"]),
                    "internal_class": best["class_name"],
                    "internal_split": best["split"],
                    "distance": best_d,
                    "label_agrees": best["class_name"] == b["class_name"],
                }
            )
            per_split[best["split"]] += 1
            per_class[b["class_name"]] += 1
            per_plane[b["plane"]] += 1
            brisc_split_hit[b["brisc_split"]] += 1

    n = len(brisc)
    summary = {
        "phash_threshold": PHASH_THRESHOLD,
        "n_brisc": n,
        "n_internal": len(internal),
        "n_exact_sha256_matches": len(exact),
        "n_phash_near_duplicates": len(matches),
        "frac_brisc_near_duplicate_of_internal": len(matches) / n if n else 0.0,
        "matched_internal_split_counts": dict(per_split),
        "n_brisc_matching_internal_train": per_split.get("train", 0),
        "frac_brisc_matching_internal_train": per_split.get("train", 0) / n if n else 0.0,
        "matched_by_brisc_class": dict(per_class),
        "matched_by_plane": dict(per_plane),
        "brisc_split_hit": dict(brisc_split_hit),
        "brisc_split_total": dict(brisc_split_total),
        "n_label_disagreements_among_matches": sum(
            1 for m in matches if not m["label_agrees"]
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({"summary": summary, "matches": matches}, fh, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
