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
# score subcommand — one score cache for every image this work touches
# =============================================================================

def _torch():
    import torch
    return torch


def resolve_checkpoint(model_name: str, seed: int) -> Path:
    p = CHECKPOINT_ROOT / model_name / f"seed_{seed}" / f"best_{model_name}_seed{seed}.pth"
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    return p


def load_model(model_name: str, seed: int, device: Any) -> Any:
    """Load a trained checkpoint. Inference only, no optimizer, no backward pass."""
    import torch
    from code import BrainTumorResNet50, BrainTumorViT, NUM_CLASSES

    cls = {"vit": BrainTumorViT, "resnet50": BrainTumorResNet50}[model_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = cls(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(resolve_checkpoint(model_name, seed), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def split_backbone_and_head(model: Any) -> Tuple[Any, Any, str]:
    """Return (feature extractor, head, feature name).

    Both architectures put every active Dropout inside the classification head:
    ResNet-50's backbone has no dropout at all, and ViT-B/16's encoder is built
    with dropout p=0.0, which is the identity in train mode or eval mode. So the
    backbone output is deterministic, and MC Dropout can be done by running the
    backbone once and the head T times. That is numerically identical to
    src/code.py's predict_with_uncertainty and about T times cheaper. It is
    verified numerically by ``verify_mc_equivalence``, not assumed.
    """
    import torch.nn as nn

    bb = model.backbone
    if hasattr(bb, "fc") and isinstance(bb.fc, nn.Sequential):
        head = bb.fc
        bb.fc = nn.Identity()
        return bb, head, "resnet50_pool2048"
    if hasattr(bb, "heads") and isinstance(bb.heads, nn.Sequential):
        head = bb.heads
        bb.heads = nn.Identity()
        return bb, head, "vit_cls768"
    raise TypeError(f"unrecognised architecture: {type(model)}")


def activate_dropout(module: Any) -> None:
    import torch.nn as nn
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.train()


class PathDataset:
    """A list of image paths, preprocessed exactly as the model expects.

    Uses src/code.py's test transform so preprocessing here is identical to
    training and to the deployed application.
    """

    def __init__(self, paths: Sequence[Path]) -> None:
        from code import get_transforms
        from torch.utils.data import Dataset  # noqa: F401
        self.paths = list(paths)
        self.tf = get_transforms("test")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB")), i


def _make_loader(paths: Sequence[Path], batch_size: int, num_workers: int):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class _DS(PathDataset, Dataset):     # PathDataset first, so its __getitem__ wins
        pass

    ds = _DS(paths)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=False)


def verify_mc_equivalence(model: Any, paths: Sequence[Path], device: Any, T: int = 20) -> Dict[str, float]:
    """Check the run-the-head-T-times shortcut against the real thing.

    Same seed, same passes, so the two must agree to floating point noise. If
    they do not, the shortcut is wrong and every score built on it is wrong.
    """
    import torch

    loader = _make_loader(paths[:16], 16, 0)
    x, _ = next(iter(loader))
    x = x.to(device)

    torch.manual_seed(12345)
    ref = model.predict_with_uncertainty(x, T=T)
    ref_mean = ref["mean_probs"].cpu().numpy()
    ref_ent = ref["entropy"].cpu().numpy()

    bb, head, _ = split_backbone_and_head(model)
    try:
        model.eval()
        with torch.no_grad():
            feats = bb(x)
        activate_dropout(head)
        torch.manual_seed(12345)
        with torch.no_grad():
            probs = torch.stack([torch.softmax(head(feats), -1) for _ in range(T)], 0)
    finally:
        _restore_head(model, head)

    mean = probs.mean(0)
    ent = -(mean * torch.log2(mean + 1e-10)).sum(-1)
    d_mean = float(np.abs(mean.cpu().numpy() - ref_mean).max())
    d_ent = float(np.abs(ent.cpu().numpy() - ref_ent).max())
    log.info("MC shortcut check: max |dp| = %.3e, max |dH| = %.3e", d_mean, d_ent)
    return {"max_abs_prob_diff": d_mean, "max_abs_entropy_diff": d_ent}


def _restore_head(model: Any, head: Any) -> None:
    import torch.nn as nn
    bb = model.backbone
    if isinstance(getattr(bb, "fc", None), nn.Identity):
        bb.fc = head
    elif isinstance(getattr(bb, "heads", None), nn.Identity):
        bb.heads = head


def score_paths(
    model: Any,
    paths: Sequence[Path],
    device: Any,
    T: int = 20,
    batch_size: int = 32,
    num_workers: int = 0,
    label: str = "",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Score a list of images. Returns (per-image scores, penultimate features).

    Everything runs under torch.no_grad(). No optimizer is ever constructed.
    """
    import torch

    eps = 1e-10
    bb, head, feat_name = split_backbone_and_head(model)
    model.eval()
    activate_dropout(head)

    loader = _make_loader(paths, batch_size, num_workers)
    n = len(paths)
    all_feats: List[np.ndarray] = []
    rows = {k: np.zeros(n, np.float32) for k in
            ("entropy", "mutual_information", "max_softmax")}
    probs_out = np.zeros((n, 4), np.float32)

    torch.manual_seed(SEED)
    done = 0
    try:
        with torch.no_grad():
            for x, idx in loader:
                x = x.to(device)
                feats = bb(x)                                        # (B, D) deterministic
                p = torch.stack([torch.softmax(head(feats), -1) for _ in range(T)], 0)  # (T,B,C)
                mean = p.mean(0)                                     # (B, C)
                ent_of_mean = -(mean * torch.log2(mean + eps)).sum(-1)
                mean_of_ent = -(p * torch.log2(p + eps)).sum(-1).mean(0)

                i = idx.numpy()
                probs_out[i] = mean.cpu().numpy()
                rows["entropy"][i] = ent_of_mean.cpu().numpy()
                rows["mutual_information"][i] = (ent_of_mean - mean_of_ent).cpu().numpy()
                rows["max_softmax"][i] = mean.max(-1).values.cpu().numpy()
                all_feats.append(feats.cpu().numpy().astype(np.float32))

                done += len(i)
                if done % (batch_size * 20) < batch_size:
                    log.info("  %s: %d / %d", label, done, n)
    finally:
        _restore_head(model, head)

    df = pd.DataFrame({
        "image_path": [str(p) for p in paths],
        **{f"p_{c}": probs_out[:, k] for k, c in enumerate(CLASS_NAMES)},
        "pred_label": probs_out.argmax(1).astype(np.int16),
        "entropy": rows["entropy"],
        "mutual_information": rows["mutual_information"],
        "max_softmax": rows["max_softmax"],
    })
    df["p_tumor"] = df[["p_glioma", "p_meningioma", "p_pituitary"]].sum(axis=1)
    return df, np.concatenate(all_feats, 0)


# --- the four image sources -------------------------------------------------

def internal_paths(split: str) -> Tuple[List[Path], np.ndarray]:
    df = pd.read_csv(SPLIT_MANIFEST)
    df = df[df["split"] == split].reset_index(drop=True)
    paths = [REPO_ROOT / str(p).replace("\\", os.sep) for p in df["filepath"]]
    return paths, df["label"].to_numpy()


def oos_paths() -> pd.DataFrame:
    m = pd.read_csv(MANIFEST_PATH)
    m["abs_path"] = [REPO_ROOT / p.replace("/", os.sep) for p in m["image_path"]]
    return m


def brisc_paths() -> pd.DataFrame:
    """BRISC is the external in-scope set. Inference only, never fitted on.

    Uses the exact class mapping from prompts/CONTRACTS.md and asserts the
    per-class counts, because `no_tumor` silently failing to match `notumor`
    would produce plausible, meaningless numbers.
    """
    man = pd.read_csv(BRISC_ROOT / "manifest.csv")
    man = man[(man["task"] == "classification") & (~man["is_mask"].astype(bool))].copy()
    man["folder_class"] = [Path(p).parent.name for p in man["relative_path"]]
    unknown = set(man["folder_class"]) - set(BRISC_TO_INTERNAL)
    if unknown:
        raise ValueError(f"unexpected BRISC folder classes: {unknown}")
    man["true_label_name"] = man["folder_class"].map(BRISC_TO_INTERNAL)
    man["true_label"] = man["true_label_name"].map(LABEL_INDEX)

    expected = {
        ("train", "glioma"): 1147, ("train", "meningioma"): 1329,
        ("train", "pituitary"): 1457, ("train", "no_tumor"): 1067,
        ("test", "glioma"): 254, ("test", "meningioma"): 306,
        ("test", "pituitary"): 300, ("test", "no_tumor"): 140,
    }
    got = man.groupby(["split", "folder_class"]).size().to_dict()
    if got != expected:
        raise ValueError(
            "BRISC per-class counts do not match the contract.\n"
            f"  expected {expected}\n  got      {got}"
        )
    log.info("BRISC: %d images, per-class counts match the contract", len(man))
    man["abs_path"] = [BRISC_ROOT / str(p).replace("\\", os.sep) for p in man["relative_path"]]
    return man


def cmd_score(args: argparse.Namespace) -> None:
    import torch

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        log.warning("CUDA requested but not available in this torch build. Falling back to CPU.")
    torch.set_num_threads(args.threads)
    log.info("device=%s threads=%d model=%s seed=%d T=%d",
             device, args.threads, args.model, args.seed, args.mc_T)

    model = load_model(args.model, args.seed, device)

    out = OOD_DIR / "scores" / f"{args.model}_seed{args.seed}"
    feat_dir = OOD_DIR / "_features" / f"{args.model}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    val_paths, _ = internal_paths("val")
    eq = verify_mc_equivalence(model, val_paths, device, T=args.mc_T)
    if max(eq.values()) > 1e-4:
        raise RuntimeError(f"MC shortcut disagrees with predict_with_uncertainty: {eq}")
    (out / "mc_equivalence_check.json").write_text(json.dumps(eq, indent=2), encoding="utf-8")

    jobs: List[Tuple[str, List[Path], Optional[pd.DataFrame]]] = []
    for split in ("train", "val", "test"):
        p, y = internal_paths(split)
        jobs.append((f"internal_{split}", p, pd.DataFrame({"true_label": y})))
    oos = oos_paths()
    jobs.append(("out_of_scope", list(oos["abs_path"]),
                 oos[["category", "subcategory", "oos_split", "source", "license", "source_image"]]))
    if not args.skip_brisc:
        br = brisc_paths()
        jobs.append(("brisc", list(br["abs_path"]),
                     br[["true_label", "true_label_name", "split", "plane_label", "sequence"]]
                     .rename(columns={"split": "brisc_split", "plane_label": "plane"})))

    for name, paths, extra in jobs:
        dest = out / f"{name}.parquet"
        if dest.exists() and not args.force:
            log.info("%s: already scored, skipping (use --force to redo)", name)
            continue
        log.info("scoring %s: %d images", name, len(paths))
        df, feats = score_paths(model, paths, device, T=args.mc_T,
                                batch_size=args.batch_size, num_workers=args.workers, label=name)
        if extra is not None:
            df = pd.concat([df.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        df["group"] = name
        df.to_parquet(dest, index=False)
        np.save(feat_dir / f"{name}.npy", feats.astype(np.float16))
        log.info("%s -> %s  (features %s)", name, dest.name, feats.shape)

    log.info("scoring complete: %s", out)


# =============================================================================
# precheck subcommand — cache the cheap image statistics for every image
# =============================================================================

def _precheck_worker(path: str) -> Dict[str, float]:
    from input_validation import extract_precheck_features
    try:
        return extract_precheck_features(path)
    except Exception:
        return {"__unreadable__": 1.0}


def cache_precheck_features(paths: Sequence[Path], dest: Path, workers: int) -> pd.DataFrame:
    """Precheck features for a list of images, in parallel. Cached to parquet."""
    if dest.exists():
        return pd.read_parquet(dest)
    from concurrent.futures import ProcessPoolExecutor

    strs = [str(p) for p in paths]
    log.info("precheck features: %d images -> %s", len(strs), dest.name)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            feats = list(ex.map(_precheck_worker, strs, chunksize=16))
    else:
        feats = [_precheck_worker(s) for s in strs]

    df = pd.DataFrame(feats)
    df["unreadable"] = df.get("__unreadable__", pd.Series(np.zeros(len(df)))).fillna(0.0) > 0
    df.drop(columns=[c for c in ("__unreadable__",) if c in df.columns], inplace=True)
    df.insert(0, "image_path", strs)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    return df


def cmd_precheck(args: argparse.Namespace) -> None:
    dest_dir = OOD_DIR / "precheck_features"
    dest_dir.mkdir(parents=True, exist_ok=True)
    jobs: List[Tuple[str, List[Path]]] = []
    for split in ("train", "val", "test"):
        jobs.append((f"internal_{split}", internal_paths(split)[0]))
    jobs.append(("out_of_scope", list(oos_paths()["abs_path"])))
    if not args.skip_brisc:
        jobs.append(("brisc", list(brisc_paths()["abs_path"])))
    for name, paths in jobs:
        cache_precheck_features(paths, dest_dir / f"{name}.parquet", args.workers)
    log.info("precheck feature cache complete: %s", dest_dir)


# =============================================================================
# evaluate subcommand
# =============================================================================

# Candidate precheck rules: (feature, side, plain-words reason).
# "low" means the image is rejected when the feature falls BELOW the band,
# "high" when it rises above, "both" when it leaves the band either way.
RULE_CANDIDATES: List[Tuple[str, str, str]] = [
    ("min_side", "low", "the image is too small to be a diagnostic scan"),
    ("aspect_ratio", "high", "the image is far from square, and a brain slice is roughly square"),
    ("color_spread", "high", "the image is in colour, and MRI is not"),
    ("intensity_std", "low", "the image is nearly blank"),
    ("dynamic_range", "low", "the image has almost no contrast"),
    ("frac_dark", "both", "the image does not have the dark background a brain scan has"),
    ("frac_bright", "high", "too much of the image is blown out to white"),
    ("border_mean", "high", "the edges of the image are bright, and a brain scan has air at its edges"),
    ("hist_entropy", "low", "the image carries almost no detail"),
    ("laplacian_var", "low", "the image is too blurred to read"),
    ("fg_area_frac", "both", "no single object of brain-like size was found"),
    ("fg_centroid_offset", "high", "the main object is not near the centre, and a brain slice is centred"),
    ("fg_border_touch", "high", "the main object runs off the edge of the image, and a brain does not"),
    ("fg_solidity", "low", "the main object is not a compact rounded shape"),
    ("fg_n_components", "high", "the image contains many separate objects, and a brain slice has one"),
]


def fit_precheck_rules(
    in_scope: pd.DataFrame,
    oos_fit: pd.DataFrame,
    alpha: float,
    margin: float,
    min_catch: float,
) -> Dict[str, Dict[str, Any]]:
    """Choose an accept band per feature from in-scope data only.

    Each band is the [alpha, 1-alpha] quantile range of the in-scope fitting
    data, widened by ``margin`` of that range on each side. The widening is
    deliberate. A band shrink-wrapped to the internal training set is not
    detecting out-of-scope, it is detecting "not my training set", and it will
    reject the first new clinic's scanner it meets.

    A rule is kept only if it rejects at least ``min_catch`` of the out-of-scope
    fitting half. A rule that catches nothing costs false rejections for free.
    """
    rules: Dict[str, Dict[str, Any]] = {}
    for feat, side, reason in RULE_CANDIDATES:
        if feat not in in_scope.columns:
            continue
        v = in_scope[feat].to_numpy(np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        q_lo, q_hi = np.quantile(v, [alpha, 1.0 - alpha])
        span = max(float(q_hi - q_lo), 1e-9)
        lo = float(q_lo - margin * span) if side in ("low", "both") else None
        hi = float(q_hi + margin * span) if side in ("high", "both") else None

        o = oos_fit[feat].to_numpy(np.float64)
        caught = np.zeros(len(o), bool)
        if lo is not None:
            caught |= o < lo
        if hi is not None:
            caught |= o > hi
        catch_rate = float(np.nanmean(caught))
        if catch_rate < min_catch:
            log.info("  drop rule %-20s catches only %.1f%% of out-of-scope", feat, 100 * catch_rate)
            continue

        frr = 0.0
        if lo is not None:
            frr += float((v < lo).mean())
        if hi is not None:
            frr += float((v > hi).mean())
        rules[feat] = {
            "feature": feat, "low": lo, "high": hi, "reason": reason,
            "catch_rate_oos_fit": catch_rate, "false_reject_in_scope_fit": frr,
        }
        log.info("  keep rule %-20s band=[%s, %s] catches %.1f%% oos, costs %.2f%% in-scope",
                 feat,
                 "-inf" if lo is None else f"{lo:.4g}",
                 "+inf" if hi is None else f"{hi:.4g}",
                 100 * catch_rate, 100 * frr)
    return rules


def apply_rules_frame(df: pd.DataFrame, rules: Dict[str, Dict[str, Any]]) -> np.ndarray:
    """Boolean array: True where the precheck rejects the image."""
    rej = df.get("unreadable", pd.Series(np.zeros(len(df), bool))).to_numpy(bool).copy()
    for r in rules.values():
        v = df[r["feature"]].to_numpy(np.float64)
        bad = ~np.isfinite(v)
        if r["low"] is not None:
            bad |= v < r["low"]
        if r["high"] is not None:
            bad |= v > r["high"]
        rej |= bad
    return rej


def per_rule_hits(df: pd.DataFrame, rules: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
    out = {}
    for name, r in rules.items():
        v = df[r["feature"]].to_numpy(np.float64)
        bad = ~np.isfinite(v)
        if r["low"] is not None:
            bad |= v < r["low"]
        if r["high"] is not None:
            bad |= v > r["high"]
        out[name] = bad
    return out


# --- feature-space detectors -------------------------------------------------

def fit_mahalanobis(train_feats: np.ndarray, train_labels: np.ndarray, shrink: float = 0.01
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Class means and a shared inverse covariance, fitted on internal train only."""
    f = train_feats.astype(np.float64)
    means = np.stack([f[train_labels == c].mean(0) for c in range(len(CLASS_NAMES))])
    centred = f - means[train_labels]
    cov = np.cov(centred, rowvar=False)
    cov += shrink * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
    return means, np.linalg.inv(cov)


def mahalanobis_scores(feats: np.ndarray, means: np.ndarray, precision: np.ndarray) -> np.ndarray:
    f = feats.astype(np.float64)
    out = np.empty((len(f), len(means)))
    for c, mu in enumerate(means):
        d = f - mu
        out[:, c] = np.einsum("nd,de,ne->n", d, precision, d)
    return np.sqrt(np.maximum(out, 0.0)).min(1)


def fit_knn(train_feats: np.ndarray) -> np.ndarray:
    """L2-normalised training features. Distance to the k-th nearest is the score."""
    f = train_feats.astype(np.float32)
    return f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)


def knn_scores(feats: np.ndarray, bank: np.ndarray, k: int = 50, chunk: int = 512) -> np.ndarray:
    f = feats.astype(np.float32)
    f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
    out = np.empty(len(f), np.float32)
    for i in range(0, len(f), chunk):
        sim = f[i:i + chunk] @ bank.T                    # cosine similarity
        d = np.sqrt(np.maximum(2.0 - 2.0 * sim, 0.0))    # euclidean on the sphere
        out[i:i + chunk] = np.partition(d, k - 1, axis=1)[:, k - 1]
    return out


# --- metrics -----------------------------------------------------------------

SCORE_METHODS = ["entropy", "mutual_information", "max_softmax", "mahalanobis", "knn"]
HIGHER_IS_OOS = {"entropy": True, "mutual_information": True, "max_softmax": False,
                 "mahalanobis": True, "knn": True}


def auroc_aupr(in_scope: np.ndarray, out_scope: np.ndarray, higher_is_oos: bool
               ) -> Tuple[float, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    s = np.concatenate([in_scope, out_scope])
    y = np.concatenate([np.zeros(len(in_scope)), np.ones(len(out_scope))])
    if not higher_is_oos:
        s = -s
    ok = np.isfinite(s)
    return float(roc_auc_score(y[ok], s[ok])), float(average_precision_score(y[ok], s[ok]))


def threshold_at_frr(in_scope: np.ndarray, target_frr: float, higher_is_oos: bool) -> float:
    """Largest rejection power at a false rejection rate no worse than target."""
    v = in_scope[np.isfinite(in_scope)]
    if higher_is_oos:
        return float(np.quantile(v, 1.0 - target_frr))
    return float(np.quantile(v, target_frr))


def reject_mask(scores: np.ndarray, thr: float, higher_is_oos: bool) -> np.ndarray:
    return (scores > thr) if higher_is_oos else (scores < thr)


OVERLAP_CANDIDATES = [
    REPO_ROOT / "analysis" / "results" / "brisc" / "brisc_overlap_per_image.csv",
    REPO_ROOT.parent / "mri-A" / "analysis" / "results" / "brisc" / "brisc_overlap_per_image.csv",
    Path(r"C:\Users\medha\OneDrive\Documents\MRI ALGO") / "analysis" / "results" / "brisc" / "brisc_overlap_per_image.csv",
]


def load_brisc_overlap(brisc_scores: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Session A's per-image contamination flags, aligned to the BRISC score rows.

    Session A found that about 80% of BRISC 2025 is pixel-identical to images in
    `data/brain_tumor/`. Both datasets repackage the same public sources. So a
    false rejection rate measured on the full BRISC set is largely a rate
    measured on the training data, and it will look better than it is.

    This adds three columns:
      clean_vs_fitted : far from everything session B fitted on (internal train
                        and internal val). The primary honest number here.
      clean_vs_any    : far from every internal split. Strictest, but heavily
                        skewed towards no-tumor, so read it with that in mind.
      clean_vs_train  : session A's definition, kept so the two reports line up.
    """
    src = next((p for p in OVERLAP_CANDIDATES if p.exists()), None)
    if src is None:
        log.warning(
            "session A's brisc_overlap_per_image.csv was not found. BRISC false "
            "rejection can only be reported on the full, contaminated set."
        )
        return None
    ov = pd.read_csv(src)
    ov["key"] = [str(p).replace("\\", "/").lower() for p in ov["image_path"]]
    ov["clean_vs_fitted"] = (ov["d_train"] > 5) & (ov["d_val"] > 5)

    root = str(BRISC_ROOT).replace("\\", "/").lower().rstrip("/") + "/"
    keys = [str(p).replace("\\", "/").lower().replace(root, "") for p in brisc_scores["image_path"]]
    aligned = ov.set_index("key").reindex(keys)
    missing = int(aligned["d_train"].isna().sum())
    if missing:
        log.warning("overlap flags missing for %d of %d BRISC rows", missing, len(keys))
    log.info("BRISC contamination flags loaded from %s", src)
    return aligned.reset_index(drop=True)


def wilson(k: int, n: int) -> Tuple[float, float]:
    """95% Wilson interval for a proportion. Small counts need it."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.959963985
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# =============================================================================
# evaluate
# =============================================================================

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]
CATEGORY_ORDER = ["cat1_corruption", "cat2_nonbrain_medical", "cat3_nonmedical",
                  "cat4_out_of_scope_brain"]
CATEGORY_LABEL = {
    "cat1_corruption": "1. corrupted scans",
    "cat2_nonbrain_medical": "2. non-brain medical",
    "cat3_nonmedical": "3. non-medical",
    "cat4_out_of_scope_brain": "4. brain, outside scope",
}


@dataclass
class Bundle:
    """Everything loaded for one (model, seed)."""
    scores: Dict[str, pd.DataFrame]
    feats: Dict[str, np.ndarray]
    pre: Dict[str, pd.DataFrame]
    model: str
    seed: int


def load_bundle(model: str, seed: int, need_brisc: bool = True) -> Bundle:
    sdir = OOD_DIR / "scores" / f"{model}_seed{seed}"
    fdir = OOD_DIR / "_features" / f"{model}_seed{seed}"
    pdir = OOD_DIR / "precheck_features"
    groups = ["internal_train", "internal_val", "internal_test", "out_of_scope"]
    if need_brisc:
        groups.append("brisc")

    scores, feats, pre = {}, {}, {}
    for g in groups:
        sp, fp, pp = sdir / f"{g}.parquet", fdir / f"{g}.npy", pdir / f"{g}.parquet"
        for path in (sp, fp, pp):
            if not path.exists():
                raise FileNotFoundError(
                    f"missing {path}. Run the 'score' and 'precheck' subcommands first.")
        scores[g] = pd.read_parquet(sp)
        feats[g] = np.load(fp).astype(np.float32)
        p = pd.read_parquet(pp)
        if list(p["image_path"]) != list(scores[g]["image_path"]):
            p = p.set_index("image_path").loc[list(scores[g]["image_path"])].reset_index()
        pre[g] = p
    return Bundle(scores, feats, pre, model, seed)


def attach_method_scores(b: Bundle) -> Dict[str, Dict[str, np.ndarray]]:
    """Every method's score for every group. Statistics fitted on internal train only."""
    train_labels = b.scores["internal_train"]["true_label"].to_numpy()
    means, prec = fit_mahalanobis(b.feats["internal_train"], train_labels)
    bank = fit_knn(b.feats["internal_train"])

    out: Dict[str, Dict[str, np.ndarray]] = {}
    for g, df in b.scores.items():
        out[g] = {
            "entropy": df["entropy"].to_numpy(np.float64),
            "mutual_information": df["mutual_information"].to_numpy(np.float64),
            "max_softmax": df["max_softmax"].to_numpy(np.float64),
            "mahalanobis": mahalanobis_scores(b.feats[g], means, prec),
            "knn": knn_scores(b.feats[g], bank),
        }
        log.info("scores ready: %s (%d)", g, len(df))
    return out, means, prec, bank


def cmd_evaluate(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OOD_DIR.mkdir(parents=True, exist_ok=True)
    b = load_bundle(args.model, args.seed, need_brisc=not args.skip_brisc)
    methods, maha_means, maha_prec, knn_bank = attach_method_scores(b)

    oos = b.scores["out_of_scope"]
    fit_mask = (oos["oos_split"] == "fit").to_numpy()
    eval_mask = ~fit_mask

    report: Dict[str, Any] = {
        "schema_version": "1.0",
        "model": args.model,
        "seed": args.seed,
        "mc_T": args.mc_T,
        "entropy_units": "bits (log base 2), matching src/code.py",
        "n": {g: int(len(df)) for g, df in b.scores.items()},
        "n_out_of_scope_by_category": oos.groupby("category").size().to_dict(),
        "fitting_data": "internal train + internal val (in scope) and the out-of-scope fit half",
        "brisc_used_for": "measuring false rejection only, never for fitting",
    }

    # ---------------------------------------------------------------- precheck
    log.info("fitting precheck rules")
    in_scope_fit = pd.concat([b.pre["internal_train"], b.pre["internal_val"]], ignore_index=True)
    rules = fit_precheck_rules(
        in_scope_fit, b.pre["out_of_scope"][fit_mask].reset_index(drop=True),
        alpha=args.precheck_alpha, margin=args.precheck_margin, min_catch=args.precheck_min_catch,
    )
    pre_rej = {g: apply_rules_frame(df, rules) for g, df in b.pre.items()}
    report["precheck"] = {
        "alpha": args.precheck_alpha,
        "margin": args.precheck_margin,
        "min_catch": args.precheck_min_catch,
        "rules": {k: {kk: vv for kk, vv in v.items()} for k, v in rules.items()},
        "false_rejection_internal_val": float(pre_rej["internal_val"].mean()),
        "false_rejection_internal_test": float(pre_rej["internal_test"].mean()),
        "rejection_out_of_scope_all": float(pre_rej["out_of_scope"].mean()),
        "rejection_out_of_scope_eval": float(pre_rej["out_of_scope"][eval_mask].mean()),
        "rejection_by_category": {
            c: float(pre_rej["out_of_scope"][(oos["category"] == c).to_numpy()].mean())
            for c in sorted(oos["category"].unique())
        },
    }
    if "brisc" in b.pre:
        report["precheck"]["false_rejection_brisc"] = float(pre_rej["brisc"].mean())
    log.info("precheck: FRR val %.2f%%, test %.2f%%, brisc %s, catches %.1f%% of out-of-scope",
             100 * pre_rej["internal_val"].mean(), 100 * pre_rej["internal_test"].mean(),
             f"{100 * pre_rej['brisc'].mean():.2f}%" if "brisc" in b.pre else "n/a",
             100 * pre_rej["out_of_scope"].mean())

    # ------------------------------------------------- score methods, ranked
    # Selection happens on internal val against the out-of-scope FIT half only.
    log.info("ranking score methods")
    per_method: Dict[str, Dict[str, Any]] = {}
    for m in SCORE_METHODS:
        hi = HIGHER_IS_OOS[m]
        au_fit, ap_fit = auroc_aupr(methods["internal_val"][m],
                                    methods["out_of_scope"][m][fit_mask], hi)
        au_ev, ap_ev = auroc_aupr(methods["internal_test"][m],
                                  methods["out_of_scope"][m][eval_mask], hi)
        entry = {
            "auroc_selection_internal_val_vs_oos_fit": au_fit,
            "aupr_selection_internal_val_vs_oos_fit": ap_fit,
            "auroc_report_internal_test_vs_oos_eval": au_ev,
            "aupr_report_internal_test_vs_oos_eval": ap_ev,
        }
        if "brisc" in b.scores:
            au_br, ap_br = auroc_aupr(methods["brisc"][m], methods["out_of_scope"][m][eval_mask], hi)
            entry["auroc_brisc_vs_oos_eval"] = au_br
            entry["aupr_brisc_vs_oos_eval"] = ap_br
        thr = threshold_at_frr(methods["internal_val"][m], args.target_frr, hi)
        entry["threshold_at_target_frr"] = thr
        entry["target_frr"] = args.target_frr
        entry["rejection_oos_eval_at_threshold"] = float(
            reject_mask(methods["out_of_scope"][m][eval_mask], thr, hi).mean())
        entry["false_rejection_internal_test"] = float(
            reject_mask(methods["internal_test"][m], thr, hi).mean())
        if "brisc" in b.scores:
            entry["false_rejection_brisc"] = float(
                reject_mask(methods["brisc"][m], thr, hi).mean())
        per_method[m] = entry
        log.info("  %-19s AUROC(sel)=%.4f  AUROC(rep)=%.4f  reject_oos=%.1f%%  FRR_brisc=%s",
                 m, au_fit, au_ev, 100 * entry["rejection_oos_eval_at_threshold"],
                 f"{100 * entry.get('false_rejection_brisc', float('nan')):.1f}%")
    report["methods"] = per_method

    # Pre-registered selection rule: highest AUROC on the selection pair.
    # BRISC never enters this choice.
    chosen = max(SCORE_METHODS, key=lambda m: per_method[m]["auroc_selection_internal_val_vs_oos_fit"])
    if args.method:
        chosen = args.method
    hi = HIGHER_IS_OOS[chosen]
    log.info("chosen method: %s", chosen)

    # --------------------------------------- combined gate: precheck then score
    # The score threshold is set on the images that SURVIVE the precheck, so the
    # two stages together land on the overall false rejection budget.
    surv_val = ~pre_rej["internal_val"]
    remaining = max(1e-6, args.target_frr - float(pre_rej["internal_val"].mean()))
    frr_among_survivors = remaining / max(1e-9, float(surv_val.mean()))
    frr_among_survivors = float(np.clip(frr_among_survivors, 0.0, 0.95))
    thr = threshold_at_frr(methods["internal_val"][chosen][surv_val], frr_among_survivors, hi)
    log.info("combined budget: precheck spends %.2f%%, score threshold set at %.2f%% of survivors",
             100 * pre_rej["internal_val"].mean(), 100 * frr_among_survivors)

    def combined(group: str) -> np.ndarray:
        return pre_rej[group] | reject_mask(methods[group][chosen], thr, hi)

    rej = {g: combined(g) for g in b.scores}

    brisc_clean: Optional[np.ndarray] = None
    if "brisc" in b.scores:
        ovf = load_brisc_overlap(b.scores["brisc"])
        brisc_clean = (ovf["clean_vs_fitted"].fillna(False).to_numpy(bool)
                       if ovf is not None else np.ones(len(b.scores["brisc"]), bool))

    # --------------------------------------------------------------- deferral
    # Deferral is session A's signal and answers a different question: this is a
    # valid brain MRI, should a human double-check the answer. It is computed
    # here only to measure how much the two signals overlap.
    defer_thr, defer_src = _deferral_threshold(methods["internal_val"]["entropy"], args.defer_rate)
    defer = {g: methods[g]["entropy"] > defer_thr for g in b.scores}
    rej_ov, defer_ov = dict(rej), dict(defer)
    if brisc_clean is not None:
        rej_ov["brisc_clean"] = rej["brisc"][brisc_clean]
        defer_ov["brisc_clean"] = defer["brisc"][brisc_clean]
    report["deferral_comparison"] = _overlap_stats(rej_ov, defer_ov, oos, defer_thr, defer_src,
                                                   args.defer_rate)

    # --------------------------------------------------------------- headline
    accepted = ~rej["out_of_scope"]
    confident = accepted & ~defer["out_of_scope"]
    by_cat, by_sub = {}, {}
    for c in CATEGORY_ORDER:
        sel = (oos["category"] == c).to_numpy()
        if not sel.any():
            continue
        sel_ev = sel & eval_mask
        k, n = int(rej["out_of_scope"][sel_ev].sum()), int(sel_ev.sum())
        lo, hiw = wilson(k, n)
        by_cat[c] = {
            "n_total": int(sel.sum()),
            "n_eval_half": n,
            "rejection_rate_eval": k / n if n else float("nan"),
            "rejection_rate_eval_ci95": [lo, hiw],
            "rejection_rate_all": float(rej["out_of_scope"][sel].mean()),
            "accepted_rate_all": float(accepted[sel].mean()),
            "accepted_and_confident_rate_all": float(confident[sel].mean()),
            "precheck_alone_rate_all": float(pre_rej["out_of_scope"][sel].mean()),
        }
    for sc in sorted(oos["subcategory"].unique()):
        sel = (oos["subcategory"] == sc).to_numpy()
        sel_ev = sel & eval_mask
        k, n = int(rej["out_of_scope"][sel_ev].sum()), int(sel_ev.sum())
        lo, hiw = wilson(k, n)
        by_sub[sc] = {
            "category": oos.loc[sel, "category"].iloc[0],
            "n_total": int(sel.sum()),
            "n_eval_half": n,
            "rejection_rate_eval": k / n if n else float("nan"),
            "rejection_rate_eval_ci95": [lo, hiw],
            "rejection_rate_all": float(rej["out_of_scope"][sel].mean()),
            "accepted_and_confident_rate_all": float(confident[sel].mean()),
            "precheck_alone_rate_all": float(pre_rej["out_of_scope"][sel].mean()),
            "is_orientation_only": sc in ORIENTATION_SUBCATS,
        }

    report["chosen"] = {
        "method": chosen,
        "threshold": float(thr),
        "threshold_units": "bits" if chosen in ("entropy", "mutual_information") else
                           ("probability" if chosen == "max_softmax" else "distance"),
        "higher_is_out_of_scope": bool(hi),
        "precheck_enabled": True,
        "target_combined_frr_internal_val": args.target_frr,
        "score_frr_among_precheck_survivors": frr_among_survivors,
    }
    report["combined_rejector"] = {
        "rejection_rate_out_of_scope_all": float(rej["out_of_scope"].mean()),
        "rejection_rate_out_of_scope_eval": float(rej["out_of_scope"][eval_mask].mean()),
        "false_rejection_rate_internal_val": float(rej["internal_val"].mean()),
        "false_rejection_rate_internal_test": float(rej["internal_test"].mean()),
        "rejection_rate_by_category": by_cat,
        "rejection_rate_by_subcategory": by_sub,
    }
    if "brisc" in b.scores:
        br = b.scores["brisc"]
        ov = load_brisc_overlap(br)
        views: List[Tuple[str, np.ndarray, str]] = [
            ("full", np.ones(len(br), bool),
             "all 6,000 BRISC images. About 80% are pixel-identical to internal "
             "training data (session A). Reference only, not an external number.")
        ]
        if ov is not None:
            views += [
                ("clean_vs_fitted", ov["clean_vs_fitted"].fillna(False).to_numpy(bool),
                 "far from everything session B fitted on, internal train and val. "
                 "The primary honest false rejection number."),
                ("clean_vs_train", ov["clean_vs_train"].fillna(False).to_numpy(bool),
                 "session A's definition, far from internal train."),
                ("clean_vs_any", ov["clean_vs_any"].fillna(False).to_numpy(bool),
                 "far from every internal split. Strictest, but about 95% no-tumor, "
                 "so it mostly measures false rejection on healthy brains."),
            ]
        bl: Dict[str, Any] = {}
        for name, sel, note in views:
            n = int(sel.sum())
            if n == 0:
                continue
            k = int(rej["brisc"][sel].sum())
            lo, hiw = wilson(k, n)
            bl[name] = {
                "n": n,
                "note": note,
                "false_rejection_rate": k / n,
                "ci95": [lo, hiw],
                "precheck_alone": float(pre_rej["brisc"][sel].mean()),
                "by_class": {
                    c: float(rej["brisc"][sel & (br["true_label_name"] == c).to_numpy()].mean())
                    for c in CLASS_NAMES
                    if (sel & (br["true_label_name"] == c).to_numpy()).any()
                },
                "by_plane": {
                    str(p): float(rej["brisc"][sel & (br["plane"] == p).to_numpy()].mean())
                    for p in sorted(br["plane"].dropna().unique())
                    if (sel & (br["plane"] == p).to_numpy()).any()
                },
                "n_by_class": {
                    c: int((sel & (br["true_label_name"] == c).to_numpy()).sum())
                    for c in CLASS_NAMES
                },
            }
            log.info("BRISC %-16s n=%5d  false rejection %.2f%%", name, n, 100 * k / n)
        report["combined_rejector"]["brisc"] = bl
        headline = bl.get("clean_vs_fitted", bl["full"])
        report["combined_rejector"]["false_rejection_rate_brisc"] = headline["false_rejection_rate"]
        report["combined_rejector"]["false_rejection_rate_brisc_ci95"] = headline["ci95"]
        report["combined_rejector"]["false_rejection_rate_brisc_basis"] = (
            "clean_vs_fitted" if "clean_vs_fitted" in bl else "full (CONTAMINATED)")
        report["combined_rejector"]["false_rejection_rate_brisc_full_contaminated"] = \
            bl["full"]["false_rejection_rate"]
        # per-method BRISC numbers, on the same honest basis
        hsel = (ov["clean_vs_fitted"].fillna(False).to_numpy(bool)
                if ov is not None else np.ones(len(br), bool))
        for m in SCORE_METHODS:
            t = per_method[m]["threshold_at_target_frr"]
            per_method[m]["false_rejection_brisc_clean"] = float(
                reject_mask(methods["brisc"][m][hsel], t, HIGHER_IS_OOS[m]).mean())

    # ------------------------------------------------- wrongly accepted images
    n_extracted = _extract_wrongly_accepted(oos, accepted, confident, args.max_extract)
    report["wrongly_accepted"] = {
        "n_accepted": int(accepted.sum()),
        "n_accepted_and_confident": int(confident.sum()),
        "n_extracted": n_extracted,
        "path": "analysis/results/ood/wrongly_accepted/",
    }

    # ----------------------------------------------------------------- figures
    _figures(b, methods, rej, pre_rej, oos, fit_mask, eval_mask, chosen, thr, hi,
             per_method, brisc_clean)

    # ----------------------------------------------------------------- outputs
    (OOD_DIR / "ood_metrics.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8")

    np.savez_compressed(
        OOD_DIR / "rejector_stats.npz",
        mahalanobis_means=maha_means.astype(np.float32),
        mahalanobis_precision=maha_prec.astype(np.float32),
        knn_bank=knn_bank.astype(np.float16),
    )
    _write_rejector_config(report, chosen, thr, rules, args)
    _write_report_md(report, rules, per_method, args)
    log.info("wrote ood_metrics.json, rejector_config.json, rejector_stats.npz, OOD_RESULTS.md")


def _deferral_threshold(val_entropy: np.ndarray, defer_rate: float) -> Tuple[float, str]:
    """Session A's deferral threshold if published, otherwise a stand-in.

    A stand-in is honest here because the only use is measuring overlap, and it
    is labelled as a stand-in in the report.
    """
    cfg = REPO_ROOT / "analysis" / "results" / "safety" / "deployment_config.json"
    if cfg.exists():
        try:
            d = json.loads(cfg.read_text(encoding="utf-8"))
            t = float(d["entropy_defer_threshold"])
            if t > 0:
                return t, "session A deployment_config.json"
        except Exception as exc:
            log.warning("could not read session A's deployment config: %s", exc)
    return float(np.quantile(val_entropy, 1.0 - defer_rate)), \
        f"stand-in, internal val entropy at a {defer_rate:.0%} defer rate"


def _overlap_stats(rej: Dict[str, np.ndarray], defer: Dict[str, np.ndarray],
                   oos: pd.DataFrame, defer_thr: float, defer_src: str,
                   defer_rate: float) -> Dict[str, Any]:
    """How much do input validation and deferral say the same thing.

    They answer different questions. Input validation asks whether the model
    should judge the image at all. Deferral asks whether a human should
    double-check an in-scope judgement. If deferral already caught everything
    input validation catches, input validation adds nothing. It does not.
    """
    out: Dict[str, Any] = {
        "deferral_threshold_entropy_bits": defer_thr,
        "deferral_threshold_source": defer_src,
        "target_defer_rate": defer_rate,
        "note": ("Input validation and deferral answer different questions and "
                 "must not be reported as one number. This block measures only "
                 "how far the two signals happen to agree."),
    }
    for g in rej:
        r, d = rej[g], defer[g]
        both = int((r & d).sum())
        union = int((r | d).sum())
        out[g] = {
            "n": int(len(r)),
            "flagged_by_rejector": float(r.mean()),
            "flagged_by_deferral": float(d.mean()),
            "flagged_by_both": float(both / len(r)),
            "rejector_only": float((r & ~d).mean()),
            "deferral_only": float((~r & d).mean()),
            "flagged_by_neither": float((~r & ~d).mean()),
            "jaccard": (both / union) if union else float("nan"),
        }
    out["out_of_scope_by_category"] = {}
    for c in CATEGORY_ORDER:
        sel = (oos["category"] == c).to_numpy()
        if not sel.any():
            continue
        r, d = rej["out_of_scope"][sel], defer["out_of_scope"][sel]
        out["out_of_scope_by_category"][c] = {
            "rejector_catches": float(r.mean()),
            "deferral_alone_would_catch": float(d.mean()),
            "caught_only_because_of_input_validation": float((r & ~d).mean()),
            "missed_by_both": float((~r & ~d).mean()),
        }
    return out


def _extract_wrongly_accepted(oos: pd.DataFrame, accepted: np.ndarray,
                              confident: np.ndarray, cap: int) -> Dict[str, int]:
    """Copy out the out-of-scope images the rejector let through, and look at them."""
    root = OOD_DIR / "wrongly_accepted"
    if root.exists():
        shutil.rmtree(root)
    counts: Dict[str, int] = {}
    rng = random.Random(SEED)
    for c in sorted(oos["category"].unique()):
        sel = ((oos["category"] == c).to_numpy()) & accepted
        idx = list(np.nonzero(sel)[0])
        counts[c] = len(idx)
        if not idx:
            continue
        keep = rng.sample(idx, min(cap, len(idx)))
        d = root / c
        d.mkdir(parents=True, exist_ok=True)
        for i in sorted(keep):
            row = oos.iloc[i]
            src = Path(row["image_path"])
            tag = "CONFIDENT" if confident[i] else "deferred"
            try:
                shutil.copyfile(src, d / f"{row['subcategory']}__{tag}__{src.name}")
            except Exception as exc:
                log.warning("could not copy %s: %s", src, exc)
        _contact_sheet([Path(oos.iloc[i]["image_path"]) for i in sorted(keep)[:24]],
                       root / f"{c}_contact_sheet.png",
                       f"accepted out-of-scope: {CATEGORY_LABEL.get(c, c)}")
    (root / "README.md").write_text(
        "# Out-of-scope images the rejector accepted\n\n"
        "These are images the model was allowed to judge and should not have been.\n"
        "`__CONFIDENT__` in the filename means the image was also not deferred, so a\n"
        "clinic user would have seen a confident class with no warning at all.\n"
        "`__deferred__` means the uncertainty signal still flagged it for a human.\n\n"
        + "\n".join(f"- {c}: {n} accepted" for c, n in counts.items()) + "\n",
        encoding="utf-8")
    return counts


def _contact_sheet(paths: Sequence[Path], dest: Path, title: str, cols: int = 6,
                   tile: int = 150) -> None:
    if not paths:
        return
    rows = int(math.ceil(len(paths) / cols))
    sheet = Image.new("RGB", (cols * tile, rows * tile + 22), (16, 16, 16))
    from PIL import ImageDraw
    ImageDraw.Draw(sheet).text((6, 6), title, fill=(235, 235, 235))
    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                sheet.paste(im.convert("RGB").resize((tile, tile)),
                            (tile * (i % cols), 22 + tile * (i // cols)))
        except Exception:
            continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest)


def _style(ax) -> None:
    """Recessive grid and axes, so the data is the loudest thing on the page."""
    ax.grid(True, alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9a9a")
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors="#4a4a4a", labelsize=9)


def _figures(b: Bundle, methods, rej, pre_rej, oos, fit_mask, eval_mask,
             chosen: str, thr: float, hi: bool, per_method,
             brisc_clean: Optional[np.ndarray] = None) -> None:
    """Figures. Every BRISC series uses the uncontaminated subset only.

    Plotting the full BRISC set would draw the training data as if it were
    outside data, which is the exact mistake session A caught.
    """
    import matplotlib.pyplot as plt

    fig_dir = OOD_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_brisc = len(b.scores["brisc"]) if "brisc" in b.scores else 0
    if brisc_clean is None:
        brisc_clean = np.ones(n_brisc, bool)

    def sub(g: str, v: np.ndarray) -> np.ndarray:
        return v[brisc_clean] if g == "brisc" else v

    groups = [("internal_test", "in scope: internal test", PALETTE[0])]
    if "brisc" in b.scores:
        groups.append(("brisc", f"in scope: BRISC, uncontaminated (n={int(brisc_clean.sum())})",
                       PALETTE[5]))
    cats = [c for c in CATEGORY_ORDER if (oos["category"] == c).any()]

    # --- score distributions, one panel per method -------------------------
    fig, axes = plt.subplots(len(SCORE_METHODS), 1, figsize=(9.5, 2.5 * len(SCORE_METHODS)))
    for ax, m in zip(np.atleast_1d(axes), SCORE_METHODS):
        series = [(lbl, sub(g, methods[g][m]), col) for g, lbl, col in groups]
        for k, c in enumerate(cats):
            series.append((CATEGORY_LABEL[c],
                           methods["out_of_scope"][m][(oos["category"] == c).to_numpy()],
                           PALETTE[1 + k % 4]))
        allv = np.concatenate([s for _, s, _ in series])
        allv = allv[np.isfinite(allv)]
        lo, hival = np.percentile(allv, [0.2, 99.8])
        bins = np.linspace(lo, hival, 70)
        for lbl, v, col in series:
            ax.hist(np.clip(v, lo, hival), bins=bins, density=True, histtype="step",
                    linewidth=2.0, color=col, label=lbl)
        if m == chosen:
            ax.axvline(thr, color="#333333", linestyle="--", linewidth=1.6)
            ax.text(thr, ax.get_ylim()[1] * 0.92, "  threshold", fontsize=8, color="#333333")
        ax.set_ylabel("density", fontsize=9)
        ax.set_title(f"{m}{'   <- chosen' if m == chosen else ''}"
                     f"    AUROC {per_method[m]['auroc_report_internal_test_vs_oos_eval']:.3f}",
                     fontsize=10, loc="left")
        _style(ax)
    np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False, ncol=2)
    np.atleast_1d(axes)[-1].set_xlabel("score", fontsize=9)
    fig.suptitle("Score distributions: in scope versus out of scope", fontsize=12, y=0.999)
    fig.tight_layout()
    fig.savefig(fig_dir / "score_distributions.png", dpi=150)
    plt.close(fig)

    # --- ROC per method -----------------------------------------------------
    from sklearn.metrics import roc_curve
    fig, axes = plt.subplots(1, 2 if "brisc" in b.scores else 1,
                             figsize=(11 if "brisc" in b.scores else 6, 5), squeeze=False)
    panels = [("internal_test", "in scope = internal test")]
    if "brisc" in b.scores:
        panels.append(("brisc", "in scope = uncontaminated BRISC (the honest one)"))
    for ax, (g, title) in zip(axes[0], panels):
        for k, m in enumerate(SCORE_METHODS):
            s_in, s_out = sub(g, methods[g][m]), methods["out_of_scope"][m][eval_mask]
            s = np.concatenate([s_in, s_out])
            y = np.concatenate([np.zeros(len(s_in)), np.ones(len(s_out))])
            if not HIGHER_IS_OOS[m]:
                s = -s
            fpr, tpr, _ = roc_curve(y, s)
            from sklearn.metrics import auc as _auc
            ax.plot(fpr, tpr, linewidth=2.0, color=PALETTE[k % len(PALETTE)],
                    label=f"{m}  {_auc(fpr, tpr):.3f}")
        ax.plot([0, 1], [0, 1], color="#b0b0b0", linewidth=1, linestyle=":")
        ax.set_xlabel("false rejection rate (valid scans thrown away)", fontsize=9)
        ax.set_ylabel("out-of-scope rejection rate", fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        _style(ax)
    fig.suptitle("Separating in-scope brain MRI from out-of-scope images", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_per_method.png", dpi=150)
    plt.close(fig)

    # --- rejection rate per subcategory ------------------------------------
    subs = (oos.groupby(["category", "subcategory"]).size().reset_index(name="n")
            .sort_values(["category", "subcategory"]))
    labels, vals, pre_vals, cols = [], [], [], []
    for _, r in subs.iterrows():
        sel = (oos["subcategory"] == r["subcategory"]).to_numpy()
        labels.append(r["subcategory"].replace("_", " "))
        vals.append(100 * rej["out_of_scope"][sel].mean())
        pre_vals.append(100 * pre_rej["out_of_scope"][sel].mean())
        cols.append(PALETTE[1 + CATEGORY_ORDER.index(r["category"]) % 4])
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(labels) + 2.4))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=cols, height=0.66, label="full rejector")
    ax.barh(y, pre_vals, color="#00000000", edgecolor="#222222", height=0.66,
            linewidth=1.1, linestyle="-", label="precheck alone")
    for i, v in enumerate(vals):
        ax.text(min(v + 1.2, 101), i, f"{v:.0f}%", va="center", fontsize=8, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("rejected (%)", fontsize=9)
    ax.set_title("Rejection rate by out-of-scope subcategory\n"
                 "colour is the category; the outline is what the cheap precheck catches on its own",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "rejection_by_subcategory.png", dpi=150)
    plt.close(fig)

    # --- false rejection: the cost side ------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    names, fr, cols2 = [], [], []
    for g, lbl, col in groups:
        names.append(lbl.replace("in scope: ", "").replace(", uncontaminated", "\nuncontaminated"))
        fr.append(100 * sub(g, rej[g]).mean())
        cols2.append(col)
    names.append("out of scope (eval half)")
    fr.append(100 * rej["out_of_scope"][eval_mask].mean())
    cols2.append(PALETTE[2])
    ax.bar(names, fr, color=cols2, width=0.55)
    for i, v in enumerate(fr):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9, color="#333333")
    ax.set_ylabel("flagged by the rejector (%)", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_title("What the rejector throws away\n"
                 "the first bars are valid brain MRI it should keep; the last is what it should reject",
                 fontsize=10, loc="left")
    plt.setp(ax.get_xticklabels(), fontsize=8.5)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "false_rejection.png", dpi=150)
    plt.close(fig)
    log.info("figures -> %s", fig_dir)


def _write_rejector_config(report: Dict[str, Any], chosen: str, thr: float,
                           rules: Dict[str, Dict[str, Any]], args) -> None:
    """Contract 3. Session C reads this. The signature and keys stay stable."""
    from datetime import datetime, timezone
    comb = report["combined_rejector"]
    cfg = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": chosen,
        "threshold": float(thr),
        "threshold_units": report["chosen"]["threshold_units"],
        "higher_is_out_of_scope": bool(HIGHER_IS_OOS[chosen]),
        "mc_T": args.mc_T,
        "model": args.model,
        "seed": args.seed,
        "fitted_on": "internal_train+internal_val+out_of_scope",
        "precheck_enabled": True,
        "precheck_rules": {
            k: {"feature": v["feature"], "low": v["low"], "high": v["high"], "reason": v["reason"]}
            for k, v in rules.items()
        },
        "fitted_statistics_path": "analysis/results/ood/rejector_stats.npz",
        "measured": {
            "rejection_rate_by_category": {
                c: d["rejection_rate_eval"] for c, d in comb["rejection_rate_by_category"].items()
            },
            "false_rejection_rate_internal_test": comb["false_rejection_rate_internal_test"],
            "false_rejection_rate_brisc": comb.get("false_rejection_rate_brisc", None),
        },
        "notes": (
            "Input validation, not deferral. This decides whether the model may judge "
            "an image at all. Deferral decides whether a human should double-check an "
            "in-scope judgement. Do not merge the two numbers. "
            "Entropy and mutual information are in bits (log base 2), matching "
            "src/code.py. No threshold here was fitted on BRISC."
        ),
    }
    (OOD_DIR / "rejector_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.1f}%"


def _write_report_md(report: Dict[str, Any], rules, per_method, args) -> None:
    comb = report["combined_rejector"]
    ch = report["chosen"]
    by_cat = comb["rejection_rate_by_category"]
    by_sub = comb["rejection_rate_by_subcategory"]
    dc = report["deferral_comparison"]

    worst = sorted(by_sub.items(), key=lambda kv: kv[1]["rejection_rate_all"])[:6]
    cat4 = by_cat.get("cat4_out_of_scope_brain")

    L: List[str] = []
    A = L.append
    A("# Out-of-scope rejection: results")
    A("")
    A(f"Model {report['model']} seed {report['seed']}, MC Dropout T={report['mc_T']}. "
      f"Method `{ch['method']}` plus the image precheck.")
    A("")
    A("## The headline")
    A("")
    A(f"- Out-of-scope images the rejector lets through: "
      f"**{_pct(1 - comb['rejection_rate_out_of_scope_eval'])}** on the held-out half.")
    A(f"- Of all out-of-scope images, **{_pct(_confident_rate(by_cat))}** come back with a "
      f"confident class and no warning of any kind.")
    if cat4:
        A(f"- Brain MRI showing a tumor the model has no class for (category 4): "
          f"rejected **{_pct(cat4['rejection_rate_eval'])}** of the time.")
    A(f"- Valid brain MRI wrongly thrown away: **{_pct(comb['false_rejection_rate_internal_test'])}** "
      f"on internal test, **{_pct(comb.get('false_rejection_rate_brisc'))}** on the "
      f"uncontaminated part of BRISC.")
    A("")
    if "brisc" in comb:
        A("> **BRISC is not clean external data.** Session A found that about 80% of")
        A("> BRISC 2025 is pixel-identical to `data/brain_tumor/`, the training set.")
        A("> Both repackage the same public sources. Every BRISC number below is")
        A("> broken out by how much internal data the subset overlaps. The full-set")
        A("> number is reference only and must not be quoted as external validation.")
        A("")
    A("## Fraction still getting a confident class, by category")
    A("")
    A("`accepted` means the rejector let the image through. `confident` means it was")
    A("also not deferred, so a clinic user sees a class with no warning at all.")
    A("")
    A("| category | n | rejected (held-out half) | accepted | accepted and confident |")
    A("|---|---|---|---|---|")
    for c in CATEGORY_ORDER:
        d = by_cat.get(c)
        if not d:
            continue
        lo, hi = d["rejection_rate_eval_ci95"]
        A(f"| {CATEGORY_LABEL[c]} | {d['n_total']} | "
          f"{_pct(d['rejection_rate_eval'])} ({100*lo:.0f}-{100*hi:.0f}) | "
          f"{_pct(d['accepted_rate_all'])} | **{_pct(d['accepted_and_confident_rate_all'])}** |")
    A("")
    A("### By subcategory")
    A("")
    A("| subcategory | category | n | rejected | precheck alone | confident class |")
    A("|---|---|---|---|---|---|")
    for sc, d in sorted(by_sub.items(), key=lambda kv: (kv[1]["category"], kv[0])):
        star = " (orientation only)" if d["is_orientation_only"] else ""
        A(f"| {sc.replace('_', ' ')}{star} | {d['category'][:4]} | {d['n_total']} | "
          f"{_pct(d['rejection_rate_all'])} | {_pct(d['precheck_alone_rate_all'])} | "
          f"{_pct(d['accepted_and_confident_rate_all'])} |")
    A("")
    A("Weakest subcategories, lowest rejection first: "
      + ", ".join(f"{k.replace('_', ' ')} ({_pct(v['rejection_rate_all'])})" for k, v in worst) + ".")
    A("")
    A("## What it costs: valid brain MRI thrown away")
    A("")
    A(f"- Internal validation: {_pct(comb['false_rejection_rate_internal_val'])} "
      f"(this is the budget the threshold was set to, so it is not evidence)")
    A(f"- Internal test: {_pct(comb['false_rejection_rate_internal_test'])}")
    if "brisc" in comb:
        A("")
        A("### BRISC, split by how much of it the model already saw")
        A("")
        A("| BRISC subset | n | wrongly rejected | 95% CI | precheck alone |")
        A("|---|---|---|---|---|")
        for name, d in comb["brisc"].items():
            lo, hi = d["ci95"]
            star = " **<- headline**" if name == comb.get("false_rejection_rate_brisc_basis") else ""
            A(f"| `{name}`{star} | {d['n']} | {_pct(d['false_rejection_rate'])} | "
              f"{100*lo:.1f}-{100*hi:.1f} | {_pct(d['precheck_alone'])} |")
        A("")
        for name, d in comb["brisc"].items():
            A(f"- `{name}`: {d['note']}")
        A("")
        hd = comb["brisc"].get(comb.get("false_rejection_rate_brisc_basis"), comb["brisc"]["full"])
        A("By class on the headline subset: " + ", ".join(
            f"{k} {_pct(v)} (n={hd['n_by_class'][k]})" for k, v in hd["by_class"].items()) + ".")
        A("")
        A("By plane on the headline subset: " + ", ".join(
            f"{k} {_pct(v)}" for k, v in hd["by_plane"].items()) + ".")
    A("")
    A("A false rejection rate on the full BRISC set would have been flattering and")
    A("meaningless, because 80% of that set is the training data wearing new")
    A("filenames. The uncontaminated subset is the real domain-shift check. A")
    A("rejector that throws away a large share of it is not detecting out-of-scope,")
    A("it is detecting \"not my training set\", and it will reject every new clinic's")
    A("scanner on day one.")
    A("")
    A("Even the uncontaminated subset is a weak external check. It is what is left")
    A("after removing overlap, not a dataset chosen to be independent. Real external")
    A("validation on a cohort that does not descend from Figshare, SARTAJ or Br35H")
    A("is still outstanding for this whole project.")
    A("")
    A("## Methods compared")
    A("")
    A("Chosen by highest AUROC on internal validation against the out-of-scope fitting")
    A("half. BRISC never entered the choice.")
    A("")
    A("| method | AUROC (selection) | AUROC (held out) | reject out-of-scope | FRR internal test | FRR BRISC clean |")
    A("|---|---|---|---|---|---|")
    for m in SCORE_METHODS:
        d = per_method[m]
        mark = " **<-**" if m == ch["method"] else ""
        A(f"| `{m}`{mark} | {d['auroc_selection_internal_val_vs_oos_fit']:.4f} | "
          f"{d['auroc_report_internal_test_vs_oos_eval']:.4f} | "
          f"{_pct(d['rejection_oos_eval_at_threshold'])} | "
          f"{_pct(d['false_rejection_internal_test'])} | "
          f"{_pct(d.get('false_rejection_brisc_clean'))} |")
    A("")
    A("Score-only, at a matched false rejection rate on internal validation. The")
    A("combined rejector adds the precheck on top.")
    A("")
    A("## The image precheck")
    A("")
    A("Cheap statistics, no model, no GPU, about 70 ms per image. It runs first.")
    A("")
    A("| rule | accept band | catches out-of-scope | costs in-scope |")
    A("|---|---|---|---|")
    for k, v in rules.items():
        lo = "-inf" if v["low"] is None else f"{v['low']:.4g}"
        hi = "+inf" if v["high"] is None else f"{v['high']:.4g}"
        A(f"| `{k}` | [{lo}, {hi}] | {_pct(v['catch_rate_oos_fit'])} | "
          f"{_pct(v['false_reject_in_scope_fit'])} |")
    A("")
    p = report["precheck"]
    A(f"Precheck alone rejects {_pct(p['rejection_out_of_scope_all'])} of out-of-scope images "
      f"and costs {_pct(p['false_rejection_internal_test'])} on internal test"
      + (f", {_pct(p.get('false_rejection_brisc'))} on BRISC." if "false_rejection_brisc" in p else "."))
    A("")
    A("## Overlap with deferral")
    A("")
    A("These are two different questions. Input validation asks whether the model")
    A("should judge the image at all. Deferral asks whether a human should")
    A("double-check an in-scope judgement. Both are needed. Reporting them as one")
    A("number is an error.")
    A("")
    A(f"Deferral threshold used here: {dc['deferral_threshold_entropy_bits']:.4f} bits "
      f"({dc['deferral_threshold_source']}).")
    A("")
    A("| set | rejected | deferred | both | rejector only | neither | Jaccard |")
    A("|---|---|---|---|---|---|---|")
    for g in ("internal_test", "brisc_clean", "out_of_scope"):
        if g not in dc:
            continue
        d = dc[g]
        A(f"| {g} | {_pct(d['flagged_by_rejector'])} | {_pct(d['flagged_by_deferral'])} | "
          f"{_pct(d['flagged_by_both'])} | {_pct(d['rejector_only'])} | "
          f"{_pct(d['flagged_by_neither'])} | {d['jaccard']:.3f} |")
    A("")
    A("| out-of-scope category | rejector catches | deferral alone would catch | caught only by input validation | missed by both |")
    A("|---|---|---|---|---|")
    for c, d in dc["out_of_scope_by_category"].items():
        A(f"| {CATEGORY_LABEL[c]} | {_pct(d['rejector_catches'])} | "
          f"{_pct(d['deferral_alone_would_catch'])} | "
          f"{_pct(d['caught_only_because_of_input_validation'])} | "
          f"{_pct(d['missed_by_both'])} |")
    A("")
    A("## Wrongly accepted images")
    A("")
    wa = report["wrongly_accepted"]
    A(f"{wa['n_accepted']} out-of-scope images were accepted, of which "
      f"{wa['n_accepted_and_confident']} were not deferred either. Samples are in "
      f"`{wa['path']}`, with contact sheets per category.")
    A("")
    A("## Data")
    A("")
    A("| category | n |")
    A("|---|---|")
    for c, n in report["n_out_of_scope_by_category"].items():
        A(f"| {CATEGORY_LABEL.get(c, c)} | {n} |")
    A("")
    A("Acquisition and licences: `reproducibility/out_of_scope_data.md`.")
    A("")
    A("## Figures")
    A("")
    for f in ("score_distributions.png", "roc_per_method.png",
              "rejection_by_subcategory.png", "false_rejection.png"):
        A(f"- `analysis/results/ood/figures/{f}`")
    A("")
    (OOD_DIR / "OOD_RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def _confident_rate(by_cat: Dict[str, Any]) -> float:
    tot = sum(d["n_total"] for d in by_cat.values())
    if not tot:
        return float("nan")
    return sum(d["n_total"] * d["accepted_and_confident_rate_all"] for d in by_cat.values()) / tot


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build data/out_of_scope and its manifest")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("score", help="compute uncertainty scores and features for every image")
    s.add_argument("--model", default="resnet50", choices=["resnet50", "vit"])
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--mc-T", dest="mc_T", type=int, default=20)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--workers", type=int, default=0)
    s.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    s.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    s.add_argument("--skip-brisc", action="store_true")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_score)

    c = sub.add_parser("precheck", help="cache the cheap image statistics for every image")
    c.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    c.add_argument("--skip-brisc", action="store_true")
    c.set_defaults(func=cmd_precheck)

    e = sub.add_parser("evaluate", help="fit the threshold, measure everything, write the report")
    e.add_argument("--model", default="resnet50", choices=["resnet50", "vit"])
    e.add_argument("--seed", type=int, default=42)
    e.add_argument("--mc-T", dest="mc_T", type=int, default=20)
    e.add_argument("--method", default=None, choices=SCORE_METHODS,
                   help="override the automatic method choice")
    e.add_argument("--target-frr", type=float, default=0.05,
                   help="combined false rejection budget on internal validation")
    e.add_argument("--defer-rate", type=float, default=0.10,
                   help="stand-in defer rate, used only if session A has not published one")
    e.add_argument("--precheck-alpha", type=float, default=0.002)
    e.add_argument("--precheck-margin", type=float, default=0.20)
    e.add_argument("--precheck-min-catch", type=float, default=0.02)
    e.add_argument("--max-extract", type=int, default=60)
    e.add_argument("--skip-brisc", action="store_true")
    e.set_defaults(func=cmd_evaluate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
