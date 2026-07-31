"""Emit a per-image overlap flag table other sessions can join against.

Reads docs/results/brisc_overlap.json (written by docs/check_brisc_overlap.py)
and writes docs/results/brisc_overlap_flags.csv with one row per BRISC
classification image.

Join key is `image_path`, matching the `image_path` column of the Contract 1
prediction caches at analysis/results/brisc/predictions/*.parquet.

Columns
-------
image_path              str   BRISC relative path, matches Contract 1
overlaps_internal       bool  near-duplicate (phash Hamming <= 5) of any internal image
exact_byte_duplicate    bool  identical sha256 to an internal image
internal_split          str   which internal split the match landed in, or "" if none
matched_internal_path   str   the internal file it matched, or ""
phash_distance          int   0-5, or -1 if no match
usable_as_external      bool  True only when overlaps_internal is False

`usable_as_external` is the strict flag. Use it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

BRISC_ROOT = Path(r"C:\Users\medha\Downloads\archive (1)\brisc2025")
SRC = Path("docs/results/brisc_overlap.json")
OUT = Path("docs/results/brisc_overlap_flags.csv")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    data = json.loads(SRC.read_text())
    matches = {m["brisc"]: m for m in data["matches"]}

    bman = pd.read_csv(BRISC_ROOT / "manifest.csv")
    bman = bman[(bman["task"] == "classification") & (~bman["is_mask"])]

    internal_sha = set()
    man = pd.read_csv("data/split_manifest.csv")
    for row in man.itertuples():
        fp = Path(row.filepath)
        if fp.exists():
            internal_sha.add(sha256_of(fp))

    rows = []
    for row in bman.itertuples():
        rel = str(row.relative_path)
        m = matches.get(rel)
        rows.append(
            {
                "image_path": rel,
                "overlaps_internal": m is not None,
                "exact_byte_duplicate": row.sha256 in internal_sha,
                "internal_split": m["internal_split"] if m else "",
                "matched_internal_path": m["internal"] if m else "",
                "phash_distance": m["distance"] if m else -1,
                "usable_as_external": m is None,
            }
        )

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  n={len(df)}")
    print(df[["overlaps_internal", "exact_byte_duplicate", "usable_as_external"]].sum())
    print(df["internal_split"].value_counts())


if __name__ == "__main__":
    main()
