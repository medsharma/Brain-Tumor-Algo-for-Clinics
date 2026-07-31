#!/usr/bin/env python3
"""
analysis/out_of_scope_rejection.py

Session B. Teaching the classifier to refuse images it should not judge.

The model sorts anything into one of four classes. A knee MRI, a chest X-ray, a
blank file, a photo of a document. In a clinic with nobody checking the input,
that returns a confident "no tumor" on an image the model cannot read. This
script builds the out-of-scope image set, fits a rejector, and measures it.

Subcommands
    build     Build data/out_of_scope/ and its manifest.
    score     Compute per-image scores for in-scope and out-of-scope images.
    evaluate  Fit the threshold, measure everything, write the report.

    python analysis/out_of_scope_rejection.py build
    python analysis/out_of_scope_rejection.py score --model resnet50 --seed 42
    python analysis/out_of_scope_rejection.py evaluate --model resnet50 --seed 42

WHY THIS DOES NOT REUSE src/code.py::evaluate_ood UNCHANGED
    src/code.py already has evaluate_ood(), run_ood_evaluation(), a
    _FlatImageDataset and an --ood-dir flag. They were never run because no
    out-of-scope image set existed. They are the right idea and the wrong shape
    for what is needed here, for four reasons:

      1. evaluate_ood returns a single AUROC over one flat directory. The
         headline number this work needs is rejection rate broken out per
         category. An overall AUROC hides that photographs are trivially
         rejected while blurred brain MRI is not.
      2. It scores entropy only. Mutual information, max softmax and
         feature-space distance are needed for comparison, and mutual
         information is the one expected to work best here.
      3. It never picks a threshold, so it cannot report a rejection rate or a
         false rejection rate. Those are the numbers that decide deployment.
      4. It does not measure false rejection on external in-scope data, which
         is the single most likely way a rejector turns out to be useless.

    src/code.py is frozen under the ownership rules in prompts/CONTRACTS.md, so
    it is left untouched. This script imports its model classes and transforms
    so preprocessing stays identical, and supersedes evaluate_ood for the
    reporting. _FlatImageDataset's per-file header check is reused in spirit.

THE RULE
    Nothing here fits on BRISC. Thresholds come from the internal validation
    split plus the out-of-scope fitting half. BRISC is used once, at the end,
    to measure how much valid outside brain MRI the rejector wrongly throws
    away.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "analysis"
OOD_DIR = ANALYSIS_DIR / "results" / "ood"
DATA_DIR = REPO_ROOT / "data" / "out_of_scope"
RAW_DIR = DATA_DIR / "_raw"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
SPLIT_MANIFEST = REPO_ROOT / "data" / "split_manifest.csv"

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(ANALYSIS_DIR))

BRISC_ROOT = Path(r"C:\Users\medha\Downloads\archive (1)\brisc2025")
CHECKPOINT_ROOT = Path(r"C:\Users\medha\OneDrive\Documents\MRI ALGO\results\20260703_155524")

BRISC_TO_INTERNAL = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "no_tumor": "notumor",
}
LABEL_INDEX = {"glioma": 0, "meningioma": 1, "pituitary": 2, "notumor": 3}
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]

SEED = 20260730
N_SOURCE_IMAGES = 120          # internal train images that corruptions are made from
N_SYNTHETIC = 120              # per synthetic subcategory
N_PER_EXTERNAL = 240           # cap per downloaded subcategory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ood")


# =============================================================================
# Category 1 — corruptions of images we already have
# =============================================================================
# Free, no download, and they cover the most likely real-world failures. Every
# corruption is generated from the internal TRAIN split, never the test split,
# so that the false rejection rate measured on internal test stays clean.

def _pil(a: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _arr(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"), dtype=np.float32)


def c_blur_heavy(im: Image.Image, rng: random.Random) -> Image.Image:
    r = 0.055 * min(im.size)
    return im.filter(ImageFilter.GaussianBlur(radius=r))


def c_noise_heavy(im: Image.Image, rng: random.Random) -> Image.Image:
    a = _arr(im)
    g = np.random.default_rng(rng.randrange(2**31))
    return _pil(a + g.normal(0, 0.30 * 255, a.shape))


def c_near_black(im: Image.Image, rng: random.Random) -> Image.Image:
    return _pil(_arr(im) * 0.05)


def c_near_white(im: Image.Image, rng: random.Random) -> Image.Image:
    return _pil(255.0 - (255.0 - _arr(im)) * 0.05)


def c_underexposed(im: Image.Image, rng: random.Random) -> Image.Image:
    return _pil(255.0 * (_arr(im) / 255.0) ** 3.5)


def c_overexposed(im: Image.Image, rng: random.Random) -> Image.Image:
    return _pil(255.0 * (_arr(im) / 255.0) ** 0.22)


def c_corner_crop(im: Image.Image, rng: random.Random) -> Image.Image:
    """A corner window. Shows background and at most the edge of the skull."""
    w, h = im.size
    fw, fh = int(0.32 * w), int(0.32 * h)
    cx = rng.choice([0, w - fw])
    cy = rng.choice([0, h - fh])
    return im.crop((cx, cy, cx + fw, cy + fh)).resize((w, h), Image.BILINEAR)


def c_rotated_90(im: Image.Image, rng: random.Random) -> Image.Image:
    return im.rotate(rng.choice([90, 270]), expand=True)


def c_upside_down(im: Image.Image, rng: random.Random) -> Image.Image:
    return im.rotate(180, expand=True)


def c_jpeg_q5(im: Image.Image, rng: random.Random) -> Image.Image:
    import io
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=5)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def c_thumbnail(im: Image.Image, rng: random.Random) -> Image.Image:
    """A thumbnail blown back up. A very common upload mistake."""
    w, h = im.size
    return im.resize((24, 24), Image.BILINEAR).resize((w, h), Image.NEAREST)


CORRUPTIONS = {
    "blur_heavy": c_blur_heavy,
    "noise_heavy": c_noise_heavy,
    "near_black": c_near_black,
    "near_white": c_near_white,
    "underexposed": c_underexposed,
    "overexposed": c_overexposed,
    "corner_crop": c_corner_crop,
    "rotated_90": c_rotated_90,
    "upside_down": c_upside_down,
    "jpeg_q5": c_jpeg_q5,
    "thumbnail": c_thumbnail,
}

# Orientation changes leave a readable brain in the frame. They belong in the
# out-of-scope set because the model was never trained on them and a rejection
# is the right behaviour, but they are the least clearly out-of-scope group and
# are reported separately. Horizontal flip is deliberately absent: it is a
# training augmentation, so a flipped scan is genuinely in scope.
ORIENTATION_SUBCATS = {"rotated_90", "upside_down"}


def build_category1(rng: random.Random) -> List[Dict[str, Any]]:
    df = pd.read_csv(SPLIT_MANIFEST)
    train = df[df["split"] == "train"].reset_index(drop=True)
    per_class = max(1, N_SOURCE_IMAGES // 4)
    picks = (
        train.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_class), random_state=SEED))
        .reset_index(drop=True)
    )

    # fit / eval halves are split by SOURCE image, so no source image appears on
    # both sides of the threshold-fitting boundary.
    order = list(range(len(picks)))
    rng.shuffle(order)
    half = len(order) // 2
    split_of = {i: ("fit" if k < half else "eval") for k, i in enumerate(order)}

    rows: List[Dict[str, Any]] = []
    for name, fn in CORRUPTIONS.items():
        out = DATA_DIR / "cat1_corruption" / name
        out.mkdir(parents=True, exist_ok=True)
        for i, row in picks.iterrows():
            src = REPO_ROOT / str(row["filepath"]).replace("\\", os.sep)
            try:
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    made = fn(im, rng)
            except Exception as exc:
                log.warning("cat1 %s: skipping %s (%s)", name, src.name, exc)
                continue
            dest = out / f"{name}_{i:04d}.png"
            made.save(dest)
            rows.append({
                "image_path": str(dest.relative_to(REPO_ROOT)).replace("\\", "/"),
                "category": "cat1_corruption",
                "subcategory": name,
                "source": "internal train split, corrupted in software",
                "license": "same as data/brain_tumor (internal training data)",
                "oos_split": split_of[i],
                "source_image": str(row["filepath"]).replace("\\", "/"),
            })
        log.info("cat1/%s: %d images", name, sum(r["subcategory"] == name for r in rows))

    # Synthetic images that have no source at all.
    for name, maker in (("solid_colour", "solid"), ("pure_noise", "noise")):
        out = DATA_DIR / "cat1_corruption" / name
        out.mkdir(parents=True, exist_ok=True)
        g = np.random.default_rng(SEED + (0 if maker == "solid" else 1))
        for i in range(N_SYNTHETIC):
            if maker == "solid":
                a = np.ones((512, 512, 3), np.float32) * g.integers(0, 256, 3)
            else:
                a = g.integers(0, 256, (512, 512, 3)).astype(np.float32)
            dest = out / f"{name}_{i:04d}.png"
            _pil(a).save(dest)
            rows.append({
                "image_path": str(dest.relative_to(REPO_ROOT)).replace("\\", "/"),
                "category": "cat1_corruption",
                "subcategory": name,
                "source": "generated in software",
                "license": "n/a, generated",
                "oos_split": "fit" if i < N_SYNTHETIC // 2 else "eval",
                "source_image": "",
            })
        log.info("cat1/%s: %d images", name, N_SYNTHETIC)

    return rows


# =============================================================================
# Category 3 — non-medical images
# =============================================================================
# The model should reject these near perfectly. If it does not, something is
# badly wrong. Rejecting a photo of a dog proves close to nothing about clinical
# safety, so these are reported as a floor, not as evidence.

def _make_documents(out: Path, n: int, rng: random.Random) -> int:
    """Pages of text, as if someone photographed a referral letter."""
    from PIL import ImageDraw, ImageFont
    out.mkdir(parents=True, exist_ok=True)
    words = ("patient referral clinic scan report history findings impression "
             "radiology department date signature diagnosis follow up review "
             "measurement contrast sequence protocol technician").split()
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    made = 0
    for i in range(n):
        w, h = rng.choice([(850, 1100), (760, 990), (1000, 720)])
        bg = rng.randint(225, 252)
        im = Image.new("RGB", (w, h), (bg, bg, rng.randint(bg - 6, bg)))
        d = ImageDraw.Draw(im)
        y = rng.randint(40, 90)
        while y < h - 50:
            line = " ".join(rng.choice(words) for _ in range(rng.randint(6, 12)))
            d.text((rng.randint(40, 80), y), line, fill=(rng.randint(10, 70),) * 3, font=font)
            y += rng.randint(20, 30)
        im.save(out / f"document_{i:04d}.png")
        made += 1
    return made


def _make_charts(out: Path, n: int, rng: random.Random) -> int:
    """Screenshots of plots. People paste these into the wrong box."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    made = 0
    for i in range(n):
        g = np.random.default_rng(SEED + i)
        fig, ax = plt.subplots(figsize=(rng.uniform(4, 7), rng.uniform(3, 5)), dpi=100)
        kind = rng.choice(["line", "bar", "scatter", "hist"])
        if kind == "line":
            for _ in range(rng.randint(1, 4)):
                ax.plot(np.cumsum(g.normal(size=60)))
        elif kind == "bar":
            k = rng.randint(4, 10)
            ax.bar(range(k), g.random(k))
        elif kind == "scatter":
            ax.scatter(g.random(80), g.random(80), c=g.random(80), s=18)
        else:
            ax.hist(g.normal(size=400), bins=25)
        ax.set_title(f"figure {i}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()
        fig.savefig(out / f"chart_{i:04d}.png")
        plt.close(fig)
        made += 1
    return made


def _copy_photographs(out: Path, n: int, rng: random.Random) -> Tuple[int, str, str]:
    """Real photographs from Imagenette, if it downloaded."""
    out.mkdir(parents=True, exist_ok=True)
    roots = list((RAW_DIR / "imagenette").rglob("*.JPEG")) + \
            list((RAW_DIR / "imagenette").rglob("*.jpeg")) + \
            list((RAW_DIR / "imagenette").rglob("*.jpg"))
    if not roots:
        return 0, "", ""
    rng.shuffle(roots)
    made = 0
    for src in roots:
        if made >= n:
            break
        try:
            with Image.open(src) as im:
                im.convert("RGB").save(out / f"photo_{made:04d}.png")
            made += 1
        except Exception:
            continue
    return (made,
            "Imagenette v2 (fast.ai), a 10-class subset of ImageNet, validation split",
            "fast.ai Imagenette: Apache-2.0. Underlying ImageNet photographs: "
            "research use only, see https://image-net.org/download.php")


def build_category3(rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base = DATA_DIR / "cat3_nonmedical"

    n = _make_documents(base / "document", N_SYNTHETIC, rng)
    rows += _rows_for(base / "document", "cat3_nonmedical", "document",
                      "generated in software, pages of printed text",
                      "n/a, generated", n)

    n = _make_charts(base / "chart_screenshot", N_SYNTHETIC, rng)
    rows += _rows_for(base / "chart_screenshot", "cat3_nonmedical", "chart_screenshot",
                      "generated in software, matplotlib figures",
                      "n/a, generated", n)

    n, src, lic = _copy_photographs(base / "photograph", N_PER_EXTERNAL, rng)
    if n:
        rows += _rows_for(base / "photograph", "cat3_nonmedical", "photograph", src, lic, n)
    else:
        log.warning("cat3/photograph: Imagenette not downloaded, no real photographs")
    return rows


def _rows_for(folder: Path, category: str, subcat: str, source: str,
              lic: str, n: int, group_by_patient: bool = False,
              rng: Optional[random.Random] = None) -> List[Dict[str, Any]]:
    """Manifest rows for one subcategory, split into a fit half and an eval half.

    The threshold is fitted on the fit half and reported on the eval half, so
    the rejection rates quoted are not the rates the threshold was tuned to.

    For images sliced out of patient volumes, the split is by patient. Two
    slices of the same head are near-duplicates, and letting them straddle the
    boundary would make the eval half look independent when it is not.
    """
    files = sorted(p for p in folder.glob("*.png"))
    if group_by_patient:
        groups = sorted({_patient_of(p) for p in files})
        r = rng or random.Random(SEED)
        shuffled = list(groups)
        r.shuffle(shuffled)
        fit_groups = set(shuffled[: len(shuffled) // 2])
        split_of = {g: ("fit" if g in fit_groups else "eval") for g in groups}
    else:
        split_of = {}

    rows = []
    for i, p in enumerate(files):
        oos_split = split_of[_patient_of(p)] if group_by_patient else ("fit" if i % 2 == 0 else "eval")
        rows.append({
            "image_path": str(p.relative_to(REPO_ROOT)).replace("\\", "/"),
            "category": category,
            "subcategory": subcat,
            "source": source,
            "license": lic,
            "oos_split": oos_split,
            "source_image": _patient_of(p) if group_by_patient else "",
        })
    return rows


def _patient_of(p: Path) -> str:
    """Patient identifier recovered from a TCIA-derived filename.

    Files are copied as ``<PatientID>_<sliceindex>.png``, so everything before
    the last underscore is the patient.
    """
    return p.stem.rsplit("_", 1)[0]


# =============================================================================
# Category 2 — non-brain medical images
# =============================================================================

MEDMNIST_SPECS = [
    ("organamnist", 128, "abdominal_ct_axial",
     "MedMNIST v2 OrganAMNIST, axial abdominal CT patches from the Liver Tumor "
     "Segmentation Benchmark (LiTS)", "CC BY 4.0"),
    ("organcmnist", 128, "abdominal_ct_coronal",
     "MedMNIST v2 OrganCMNIST, coronal abdominal CT patches from LiTS", "CC BY 4.0"),
    ("pneumoniamnist", 224, "chest_xray",
     "MedMNIST v2 PneumoniaMNIST, paediatric chest X-ray", "CC BY 4.0"),
    ("breastmnist", 224, "breast_ultrasound",
     "MedMNIST v2 BreastMNIST, breast ultrasound", "CC BY 4.0"),
]


def build_category2(rng: random.Random) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base = DATA_DIR / "cat2_nonbrain_medical"

    for flag, size, subcat, source, lic in MEDMNIST_SPECS:
        npz = RAW_DIR / f"{flag}_{size}.npz"
        if not npz.exists():
            log.warning("cat2/%s: %s not downloaded, skipping", subcat, npz.name)
            continue
        out = base / subcat
        out.mkdir(parents=True, exist_ok=True)
        with np.load(npz) as z:
            imgs = z["test_images"]
        idx = rng.sample(range(len(imgs)), min(N_PER_EXTERNAL, len(imgs)))
        for i, j in enumerate(idx):
            a = imgs[j]
            Image.fromarray(a.astype(np.uint8)).convert("RGB").save(out / f"{subcat}_{i:04d}.png")
        rows += _rows_for(out, "cat2_nonbrain_medical", subcat, source, lic, len(idx))
        log.info("cat2/%s: %d images", subcat, len(idx))

    # Non-brain MRI from TCIA. Same modality as the training data, different
    # body part. The hardest case in this category by a wide margin.
    for subcat, source, lic in (
        ("pelvic_mri", "TCIA Prostate-3T, axial T2 pelvic MRI", "CC BY 3.0"),
        ("extremity_mri", "TCIA Soft-tissue-Sarcoma, axial MRI of limbs", "CC BY 3.0"),
    ):
        src_dir = RAW_DIR / "tcia" / f"cat2_{subcat}"
        n = _copy_from_raw(src_dir, base / subcat, N_PER_EXTERNAL, rng)
        if n:
            rows += _rows_for(base / subcat, "cat2_nonbrain_medical", subcat, source, lic, n,
                              group_by_patient=True, rng=rng)
            log.info("cat2/%s: %d images", subcat, n)
        else:
            log.warning("cat2/%s: no images downloaded", subcat)
    return rows


# =============================================================================
# Category 4 — brain MRI outside the trained scope
# =============================================================================
# The hardest and by far the most clinically important. The model knows three
# tumor families plus no-tumor. Anything else, a metastasis, a schwannoma, a
# stroke, an abscess, gets forced into one of four wrong answers.

def build_category4(rng: random.Random) -> List[Dict[str, Any]]:
    base = DATA_DIR / "cat4_out_of_scope_brain"
    rows: List[Dict[str, Any]] = []
    for subcat, source, lic in (
        ("vestibular_schwannoma",
         "TCIA Vestibular-Schwannoma-MC-RC, routine clinical T1 contrast-enhanced "
         "and T2 brain MRI of vestibular schwannoma", "CC BY 4.0"),
    ):
        src_dir = RAW_DIR / "tcia" / f"cat4_{subcat}"
        n = _copy_from_raw(src_dir, base / subcat, N_PER_EXTERNAL, rng)
        if n:
            rows += _rows_for(base / subcat, "cat4_out_of_scope_brain", subcat, source, lic, n,
                              group_by_patient=True, rng=rng)
            log.info("cat4/%s: %d images", subcat, n)
        else:
            log.warning("cat4/%s: NOT OBTAINED", subcat)
    if not rows:
        log.warning(
            "CATEGORY 4 IS EMPTY. The most clinically important rejection case "
            "is untested. This must go in the report headline."
        )
    return rows


def _copy_from_raw(src_dir: Path, out: Path, n: int, rng: random.Random) -> int:
    if not src_dir.exists():
        return 0
    files = sorted(src_dir.glob("*.png"))
    if not files:
        return 0
    if len(files) > n:
        files = rng.sample(files, n)
        files.sort()
    out.mkdir(parents=True, exist_ok=True)
    made = 0
    for p in files:
        shutil.copyfile(p, out / p.name)     # keeps <PatientID>_<slice>.png
        made += 1
    return made


# =============================================================================
# build subcommand
# =============================================================================

def cmd_build(args: argparse.Namespace) -> None:
    rng = random.Random(SEED)
    np.random.seed(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    rows += build_category1(rng)
    rows += build_category2(rng)
    rows += build_category3(rng)
    rows += build_category4(rng)

    df = pd.DataFrame(rows)
    # split by source image where there is one, so a corrupted image and its
    # source never straddle the fit/eval boundary
    df.to_csv(MANIFEST_PATH, index=False)

    log.info("manifest: %d images -> %s", len(df), MANIFEST_PATH)
    print()
    print(df.groupby(["category", "subcategory"]).size().to_string())
    print()
    print(df.groupby(["category", "oos_split"]).size().to_string())

    if (df["category"] == "cat4_out_of_scope_brain").sum() == 0:
        print()
        print("WARNING: category 4 is empty. The most clinically important")
        print("rejection case, brain MRI showing something the model has no")
        print("class for, is untested.")


# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build data/out_of_scope and its manifest")
    b.set_defaults(func=cmd_build)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
