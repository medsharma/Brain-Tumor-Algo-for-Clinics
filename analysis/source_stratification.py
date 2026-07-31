#!/usr/bin/env python
"""Per-source accuracy on the internal splits.

`analysis/EXTERNAL_VALIDATION_GAP.md` section 1 has flagged this since long
before the parallel sessions ran, and nobody had attempted it. Session E called
it "the cheapest unfinished work in this project" and it is: no new data, no new
inference, no new cohort.

Why it matters
--------------
The training pool is the Kaggle Nickparvar merge of three public datasets that
were collected separately, by different people, on different scanners. If the
model scores well on one of them and badly on another, that gap is the only
generalisation signal available without acquiring a real external cohort. BRISC
cannot supply one, because BRISC is ~80% of the same images republished.

This is a weaker signal than a real external cohort. Sources here are still
inside the training pool, so a model can be good at a source because it trained
on that source. Read the gap between sources, not the absolute numbers.

How a source is inferred
------------------------
There is no source column. The merge did not keep one. So the source is inferred
from a file property that survived the merge: PIL image mode.

  Figshare (Cheng et al.)  ->  single-channel grayscale ("L")
  Br35H                    ->  RGB, no-tumour
  SARTAJ                   ->  RGB, tumour

The grayscale fingerprint is strong. It reproduces Figshare's published per-class
counts almost exactly:

  class        grayscale here   Figshare published
  pituitary               930                  930
  meningioma              709                  708
  glioma                 1399                 1426

The RGB split into Br35H and SARTAJ is weaker and rests on the documented
construction of the merge: Br35H is a binary tumour/no-tumour dataset, and the
Nickparvar merge replaced SARTAJ's glioma images with Figshare ones because the
SARTAJ glioma class was documented as mislabelled. That is consistent with what
is on disk (glioma is 1399 grayscale against only 401 RGB).

Every number this script prints is labelled with the confidence of the
assignment. The 29 grayscale no-tumour images do not fit any source cleanly and
are reported as "unassigned" rather than forced into one.

Inference only. Reads prediction caches that already exist. No model is loaded,
no threshold is fitted, and BRISC is not read at all.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "split_manifest.csv"
INTERNAL_PRED = ROOT / "analysis" / "results" / "internal" / "predictions"
OUT_DIR = ROOT / "analysis" / "results" / "source_stratification"

MODELS = ("resnet50", "vit")
SEEDS = (42, 123, 7, 2024, 31)
NOTUMOR_INDEX = 3

# Figshare (Cheng et al. 2016), published per-class counts, for the fingerprint check.
FIGSHARE_PUBLISHED = {"glioma": 1426, "meningioma": 708, "pituitary": 930}


# --------------------------------------------------------------------------
# source inference
# --------------------------------------------------------------------------
def infer_sources(manifest: pd.DataFrame) -> pd.DataFrame:
    """Add `pil_mode`, `source` and `source_confidence` columns."""
    modes: List[str] = []
    for fp in manifest["filepath"]:
        with Image.open(str(ROOT / str(fp).replace("\\", "/"))) as im:
            modes.append(im.mode)
    out = manifest.copy()
    out["pil_mode"] = modes

    gray = out["pil_mode"] == "L"
    notumor = out["class_name"] == "notumor"

    source = np.where(
        gray & ~notumor, "figshare",
        np.where(~gray & notumor, "br35h",
                 np.where(~gray & ~notumor, "sartaj", "unassigned")),
    )
    out["source"] = source

    # Grayscale tumour -> strong (matches published counts).
    # RGB -> weaker, rests on the documented construction of the merge.
    out["source_confidence"] = np.where(
        out["source"] == "figshare", "strong",
        np.where(out["source"] == "unassigned", "none", "moderate"),
    )
    return out


def fingerprint_report(src: pd.DataFrame) -> Dict[str, object]:
    fig = src[src["source"] == "figshare"]["class_name"].value_counts().to_dict()
    deltas = {c: int(fig.get(c, 0)) - n for c, n in FIGSHARE_PUBLISHED.items()}
    return {
        "figshare_counts_here": {k: int(v) for k, v in fig.items()},
        "figshare_counts_published": FIGSHARE_PUBLISHED,
        "delta_vs_published": deltas,
        "max_abs_delta": int(max(abs(v) for v in deltas.values())),
        "source_totals": {k: int(v) for k, v in src["source"].value_counts().items()},
        "unassigned_note": (
            "Grayscale no-tumour images. Figshare has no no-tumour class, so these "
            "fit no source cleanly. Reported separately, never folded into a source."
        ),
    }


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959963985) -> tuple:
    """Wilson score interval. Behaves sanely at k=0, which bootstrap does not."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float((c - h) / d), float((c + h) / d))


def metrics_for(df: pd.DataFrame) -> Dict[str, object]:
    """Accuracy and tumour miss rate for one group of rows."""
    n = len(df)
    correct = int((df["pred_label"] == df["true_label"]).sum())
    tumour = df[df["true_label"] != NOTUMOR_INDEX]
    missed = int((tumour["pred_label"] == NOTUMOR_INDEX).sum())
    acc_lo, acc_hi = wilson(correct, n)
    miss_lo, miss_hi = wilson(missed, len(tumour))
    return {
        "n": n,
        "n_tumour": int(len(tumour)),
        "accuracy": correct / n if n else float("nan"),
        "accuracy_ci": [acc_lo, acc_hi],
        "tumour_miss_rate": missed / len(tumour) if len(tumour) else float("nan"),
        "tumour_miss_rate_ci": [miss_lo, miss_hi],
        "n_missed_tumours": missed,
        "mean_entropy": float(df["entropy"].mean()),
    }


def load_internal(model: str, seed: int, split: str) -> pd.DataFrame:
    return pd.read_parquet(INTERNAL_PRED / f"{model}_seed{seed}_{split}.parquet")


def run(split: str = "test") -> Dict[str, object]:
    manifest = pd.read_csv(MANIFEST)
    src = infer_sources(manifest)
    fp = fingerprint_report(src)

    # key on file stem, which both the manifest and the cache carry
    src["stem"] = src["filepath"].astype(str).str.replace("\\", "/", regex=False).str.split("/").str[-1]
    key = src.set_index("stem")[["source", "source_confidence"]]

    report: Dict[str, object] = {
        "split": split,
        "method": "PIL image mode fingerprint; see module docstring",
        "fingerprint": fp,
        "caveat": (
            "These sources are all inside the training pool. A source the model "
            "trained on is not an external cohort. Read the GAP between sources, "
            "not the absolute numbers."
        ),
        "by_model": {},
    }

    for model in MODELS:
        per_seed: Dict[str, object] = {}
        pooled_frames = []
        for seed in SEEDS:
            path = INTERNAL_PRED / f"{model}_seed{seed}_{split}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            df["stem"] = df["image_path"].astype(str).str.replace("\\", "/", regex=False).str.split("/").str[-1]
            df = df.join(key, on="stem")
            if df["source"].isna().any():
                raise AssertionError(f"{path.name}: {int(df['source'].isna().sum())} rows had no source")
            per_seed[str(seed)] = {s: metrics_for(g) for s, g in df.groupby("source")}
            pooled_frames.append(df)

        if not pooled_frames:
            continue
        pooled = pd.concat(pooled_frames, ignore_index=True)
        report["by_model"][model] = {
            "per_seed": per_seed,
            "pooled_over_seeds": {s: metrics_for(g) for s, g in pooled.groupby("source")},
            "pooled_note": (
                "Pooled over 5 seeds on the same images, so the seeds are not "
                "independent samples. The interval is narrower than the true "
                "uncertainty. Session E measured the same effect: 18 images cause "
                "every miss, so effective sample size is far below n."
            ),
        }
    return report


def to_markdown(rep: Dict[str, object]) -> str:
    L: List[str] = []
    A = L.append
    A("# Per-source accuracy on the internal test split")
    A("")
    A("The training pool is three datasets merged. This splits the internal test")
    A("split back apart and scores each source separately.")
    A("")
    A("**This is not external validation.** Every source here is inside the")
    A("training pool. The model trained on all three. Read the gap between")
    A("sources, not the absolute numbers.")
    A("")
    fp = rep["fingerprint"]
    A("## How the source was inferred")
    A("")
    A("There is no source column. The merge did not keep one. Source is inferred")
    A("from PIL image mode, which survived the merge.")
    A("")
    A("| source | rule | confidence |")
    A("|---|---|---|")
    A("| Figshare (Cheng et al.) | grayscale, tumour | strong |")
    A("| Br35H | RGB, no-tumour | moderate |")
    A("| SARTAJ | RGB, tumour | moderate |")
    A("| unassigned | grayscale, no-tumour | none, reported separately |")
    A("")
    A("The grayscale fingerprint reproduces Figshare's published per-class counts:")
    A("")
    A("| class | here | published | delta |")
    A("|---|---|---|---|")
    for c, pub in FIGSHARE_PUBLISHED.items():
        here = fp["figshare_counts_here"].get(c, 0)
        A(f"| {c} | {here} | {pub} | {here - pub:+d} |")
    A("")
    A(f"Largest disagreement is {fp['max_abs_delta']} images. That is a strong match.")
    A("")
    A("The RGB split into Br35H and SARTAJ is weaker. It rests on the documented")
    A("construction of the merge: Br35H is a binary tumour/no-tumour dataset, and")
    A("the merge replaced SARTAJ's glioma images with Figshare ones because the")
    A("SARTAJ glioma class was documented as mislabelled. What is on disk agrees:")
    A("glioma is mostly grayscale.")
    A("")
    A("Images on disk by inferred source: " + ", ".join(
        f"{k} {v}" for k, v in sorted(fp["source_totals"].items())))
    A("")
    for model, blk in rep["by_model"].items():
        A(f"## {model}, pooled over 5 seeds, internal {rep['split']} split")
        A("")
        A("| source | n | accuracy | 95% CI | tumour miss rate | 95% CI | missed |")
        A("|---|---|---|---|---|---|---|")
        for s, m in sorted(blk["pooled_over_seeds"].items()):
            acc = f"{100*m['accuracy']:.2f}%"
            aci = f"{100*m['accuracy_ci'][0]:.2f}-{100*m['accuracy_ci'][1]:.2f}"
            if m["n_tumour"]:
                mr = f"{100*m['tumour_miss_rate']:.2f}%"
                mci = f"{100*m['tumour_miss_rate_ci'][0]:.2f}-{100*m['tumour_miss_rate_ci'][1]:.2f}"
            else:
                mr, mci = "n/a", "n/a"
            A(f"| {s} | {m['n']} | {acc} | {aci} | {mr} | {mci} | {m['n_missed_tumours']} |")
        A("")
        A(f"*{blk['pooled_note']}*")
        A("")
    return "\n".join(L) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rep = run("test")
    (OUT_DIR / "source_stratification.json").write_text(
        json.dumps(rep, indent=2), encoding="utf-8")
    (OUT_DIR / "SOURCE_STRATIFICATION.md").write_text(
        to_markdown(rep), encoding="utf-8")

    fp = rep["fingerprint"]
    print("Figshare fingerprint vs published counts:", fp["delta_vs_published"])
    print("source totals:", fp["source_totals"])
    for model, blk in rep["by_model"].items():
        print(f"\n{model} pooled over seeds, internal test:")
        for s, m in sorted(blk["pooled_over_seeds"].items()):
            mr = "n/a" if not m["n_tumour"] else f"{100*m['tumour_miss_rate']:.2f}%"
            print(f"  {s:11s} n={m['n']:5d}  acc {100*m['accuracy']:6.2f}%  miss {mr}")
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
