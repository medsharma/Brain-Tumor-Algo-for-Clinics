#!/usr/bin/env python3
"""
analysis/external_validation_brisc.py  -  Session A, Phase 1 and Phase 2/3/5.

External validation of the trained brain-tumour classifiers on BRISC 2025, a
dataset the models have never seen.

THE RULE THIS FILE OBEYS
------------------------
Nothing here trains, fine-tunes, calibrates, threshold-fits or selects on BRISC
labels. Every forward pass runs under ``torch.no_grad()``. BRISC labels are read
only to score predictions after the fact. Any threshold or temperature used
anywhere in Session A is fitted in ``analysis/operating_point.py`` on the
internal validation split and applied to BRISC unchanged.

Subcommands
-----------
    cache    build the BRISC and internal prediction caches (Contract 1)
    metrics  compute the external validation metrics and figures (Phase 2/3)
    misses   extract confidently-wrong missed tumours (Phase 5)

Everything reuses ``src/code.py``: CLASS_NAMES, get_transforms("test"),
BrainTumorViT, BrainTumorResNet50, ManifestDataset, compute_calibration_metrics,
compute_risk_coverage_curve, bootstrap_ci, mcnemar_test.

MC-DROPOUT FAST PATH (important, and verified)
----------------------------------------------
``model.predict_with_uncertainty`` runs the whole network T=20 times. Both
architectures place every non-zero-probability Dropout layer in the
classification head only (ViT: ``backbone.heads``; ResNet-50: ``backbone.fc``).
The torchvision ViT encoder does contain Dropout modules but they are
constructed with p=0.0, and ATen's dropout kernel returns the input untouched
and draws no random numbers when p==0. The backbone is therefore deterministic.

So we run the backbone once per image and the head T times. ``--verify-fast-path``
asserts the result is bit-identical to ``predict_with_uncertainty`` under the
same RNG state (it is: ``torch.equal`` is True for both backbones), and the
verification is re-run automatically at the start of every cache build.

Measured speedup on this machine: 19.3x for ViT, 22.1x for ResNet-50. Without it
the 1.2 million forward passes take roughly 18 hours on CPU. With it, about one
hour. The numbers are the same numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CLASS_NAMES,
    DEVICE,
    NUM_CLASSES,
    REPO_ROOT,
    RESULTS_OUT_DIR,
    MODEL_CLASSES,
    get_transforms,
)

sys.path.insert(0, str(REPO_ROOT / "src"))
from code import (  # noqa: E402
    ManifestDataset,
    compute_calibration_metrics,
    mcnemar_test,
)

# ---------------------------------------------------------------------------
# Absolute paths. Git worktrees do not contain the checkpoints (they are
# gitignored, each is over GitHub's 100 MB limit), so these are absolute by
# necessity. Overridable from the command line.
# ---------------------------------------------------------------------------
MAIN_REPO = Path(r"C:\Users\medha\OneDrive\Documents\MRI ALGO")
CHECKPOINTS = MAIN_REPO / "results" / "20260703_155524"
BRISC = Path(r"C:\Users\medha\Downloads\archive (1)\brisc2025")

SEEDS: List[int] = [42, 123, 7, 2024, 31]
MODELS: List[str] = ["vit", "resnet50"]
MC_T: int = 20

# ---------------------------------------------------------------------------
# The class mapping trap. BRISC says "no_tumor", src/code.py says "notumor".
# A sorted-directory-name match silently mislabels every image. Map explicitly.
# ---------------------------------------------------------------------------
BRISC_TO_INTERNAL = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "no_tumor": "notumor",
}
LABEL_INDEX = {"glioma": 0, "meningioma": 1, "pituitary": 2, "notumor": 3}
NOTUMOR_IDX = LABEL_INDEX["notumor"]

# Contract table. A mismatch means the mapping or the dataset is wrong. Stop.
EXPECTED_COUNTS: Dict[str, Dict[str, int]] = {
    "train": {"glioma": 1147, "meningioma": 1329, "pituitary": 1457, "notumor": 1067},
    "test": {"glioma": 254, "meningioma": 306, "pituitary": 300, "notumor": 140},
}

BRISC_OUT = RESULTS_OUT_DIR / "brisc"
INTERNAL_OUT = RESULTS_OUT_DIR / "internal"


# ===========================================================================
# Manifests
# ===========================================================================

def build_brisc_manifest(brisc_root: Path = BRISC) -> pd.DataFrame:
    """Build the BRISC image table from the shipped manifest.csv.

    Not by walking directories: the manifest carries plane_label, sequence,
    sha256 and image dimensions, which the directory tree does not.
    """
    man_path = brisc_root / "manifest.csv"
    if not man_path.exists():
        raise FileNotFoundError(f"BRISC manifest not found: {man_path}")

    man = pd.read_csv(man_path)
    df = man[(man["task"] == "classification") & (~man["is_mask"])].copy()

    unmapped = sorted(set(df["tumor_label"]) - set(BRISC_TO_INTERNAL))
    if unmapped:
        raise ValueError(f"BRISC tumor_label values with no mapping: {unmapped}")

    df["class_name"] = df["tumor_label"].map(BRISC_TO_INTERNAL)
    if df["class_name"].isna().any():
        raise ValueError("BRISC_TO_INTERNAL produced NaN class names")
    df["label"] = df["class_name"].map(LABEL_INDEX).astype(int)

    # relative_path uses Windows separators in the shipped manifest.
    df["filepath"] = df["relative_path"].map(
        lambda p: str(brisc_root / Path(str(p).replace("\\", "/")))
    )
    df["image_path"] = df["relative_path"].map(lambda p: str(p).replace("\\", "/"))
    df["brisc_split"] = df["split"]
    df["plane"] = df["plane_label"]
    df = df.rename(columns={"sequence": "sequence_"})
    df["sequence"] = df["sequence_"]

    df = df.sort_values(["brisc_split", "class_name", "filename"]).reset_index(drop=True)
    # ManifestDataset filters on a "split" column; give every row the same tag.
    df["split"] = "brisc"
    return df[
        [
            "filepath", "image_path", "sha256", "class_name", "label",
            "brisc_split", "plane", "sequence", "split", "width", "height",
        ]
    ]


def assert_class_counts(df: pd.DataFrame) -> None:
    """Hard stop if the per-class counts do not match the contract table."""
    tab = pd.crosstab(df["brisc_split"], df["class_name"])
    print("\nBRISC per-class counts after BRISC_TO_INTERNAL mapping:")
    print(tab.to_string())
    print(f"  total = {len(df)}")

    problems: List[str] = []
    for split, expected in EXPECTED_COUNTS.items():
        for cls, n_exp in expected.items():
            n_got = int(tab.loc[split, cls]) if (split in tab.index and cls in tab.columns) else 0
            if n_got != n_exp:
                problems.append(f"{split}/{cls}: expected {n_exp}, got {n_got}")
    if len(df) != 6000:
        problems.append(f"total: expected 6000, got {len(df)}")
    if problems:
        raise AssertionError(
            "BRISC class counts do not match the contract table:\n  "
            + "\n  ".join(problems)
        )
    print("  OK - matches the contract table exactly.")


def verify_sha256(df: pd.DataFrame, n_sample: int = 50, seed: int = 0) -> Dict[str, Any]:
    """Hash a random sample of images and compare against the BRISC manifest."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n_sample, len(df)), replace=False)
    bad: List[str] = []
    for i in idx:
        row = df.iloc[int(i)]
        h = hashlib.sha256(Path(row["filepath"]).read_bytes()).hexdigest()
        if h != row["sha256"]:
            bad.append(row["image_path"])
    print(f"\nsha256 spot check: {len(idx)} images sampled, {len(bad)} mismatches.")
    if bad:
        raise AssertionError(f"sha256 mismatch on {len(bad)} images, e.g. {bad[:5]}")
    return {"n_checked": int(len(idx)), "n_mismatch": 0, "seed": seed}


def build_internal_manifest(split: str) -> pd.DataFrame:
    """Load the frozen internal split manifest and select one split."""
    man = pd.read_csv(REPO_ROOT / "data" / "split_manifest.csv")
    df = man[man["split"] == split].copy()
    if df.empty:
        raise ValueError(f"internal split '{split}' is empty")
    df["filepath"] = df["filepath"].map(
        lambda p: str(REPO_ROOT / Path(str(p).replace("\\", "/")))
    )
    df["image_path"] = man.loc[df.index, "filepath"].map(lambda p: str(p).replace("\\", "/"))
    df["sha256"] = ""
    df["internal_split"] = split
    df["plane"] = "unknown"
    df["sequence"] = "unknown"
    df = df.sort_values("filepath").reset_index(drop=True)
    return df[
        ["filepath", "image_path", "sha256", "class_name", "label",
         "internal_split", "plane", "sequence", "split"]
    ]


# ===========================================================================
# Preprocessing parity
# ===========================================================================

def assert_preprocessing_parity() -> Dict[str, Any]:
    """Assert the transform we use is exactly get_transforms("test").

    Any drift from training-time eval preprocessing invalidates every number in
    this file, so check the composed pipeline structurally rather than trusting
    that we called the right function.
    """
    from torchvision import transforms as tvt

    tf = get_transforms("test")
    steps = list(tf.transforms)
    kinds = [type(s).__name__ for s in steps]
    assert kinds == ["Resize", "ToTensor", "Normalize"], f"unexpected pipeline: {kinds}"

    resize: tvt.Resize = steps[0]  # type: ignore[assignment]
    norm: tvt.Normalize = steps[2]  # type: ignore[assignment]
    assert tuple(resize.size) == (224, 224), f"resize={resize.size}"
    assert tuple(np.round(norm.mean, 6)) == (0.485, 0.456, 0.406), f"mean={norm.mean}"
    assert tuple(np.round(norm.std, 6)) == (0.229, 0.224, 0.225), f"std={norm.std}"

    spec = {
        "pipeline": kinds,
        "resize": [224, 224],
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "source": "src/code.py get_transforms('test')",
    }
    print(f"preprocessing parity OK: {kinds} resize=(224,224) imagenet norm")
    return spec


# ===========================================================================
# Model loading and MC-Dropout inference
# ===========================================================================

def checkpoint_path(model_name: str, seed: int) -> Path:
    return CHECKPOINTS / model_name / f"seed_{seed}" / f"best_{model_name}_seed{seed}.pth"


def load_checkpoint(model_name: str, seed: int, device: torch.device = DEVICE) -> nn.Module:
    path = checkpoint_path(model_name, seed)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint missing: {path}")
    model = MODEL_CLASSES[model_name](num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _head(model: nn.Module, model_name: str) -> nn.Module:
    return model.backbone.heads if model_name == "vit" else model.backbone.fc


def _set_head(model: nn.Module, model_name: str, mod: nn.Module) -> None:
    if model_name == "vit":
        model.backbone.heads = mod
    else:
        model.backbone.fc = mod


def _activate_dropout(model: nn.Module) -> None:
    """Dropout modules back into train mode. Never the whole model."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_forward(model: nn.Module, model_name: str, x: torch.Tensor, T: int = MC_T) -> torch.Tensor:
    """T stochastic softmax samples, shape (T, B, C). Backbone runs once."""
    model.eval()
    _activate_dropout(model)

    head = _head(model, model_name)
    _set_head(model, model_name, nn.Identity())
    try:
        feats = model.backbone(x)
    finally:
        _set_head(model, model_name, head)

    return torch.stack([torch.softmax(head(feats), dim=-1) for _ in range(T)], dim=0)


@torch.no_grad()
def verify_fast_path(model: nn.Module, model_name: str, batch: int = 4, T: int = MC_T) -> float:
    """Assert mc_forward equals model.predict_with_uncertainty bit for bit."""
    x = torch.randn(batch, 3, 224, 224, device=DEVICE)
    torch.manual_seed(20260730)
    ref = model.predict_with_uncertainty(x, T=T)["all_probs"]
    torch.manual_seed(20260730)
    fast = mc_forward(model, model_name, x, T=T)
    max_abs = float((ref - fast).abs().max().item())
    if not torch.equal(ref, fast):
        raise AssertionError(
            f"fast MC path differs from predict_with_uncertainty for {model_name} "
            f"(max abs diff {max_abs:.3e}). Refusing to build a cache from it."
        )
    return max_abs


def entropy_nats(p: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return -(p * np.log(p + eps)).sum(axis=-1)


@torch.no_grad()
def predict_split(
    model: nn.Module,
    model_name: str,
    manifest_df: pd.DataFrame,
    split_tag: str,
    T: int = MC_T,
    batch_size: int = 32,
    num_workers: int = 0,
    rng_seed: int = 0,
    log_every: int = 25,
) -> Dict[str, np.ndarray]:
    """MC-Dropout over one manifest. Inference only, no grad, no optimizer."""
    ds = ManifestDataset(manifest_df, split_tag, transform=get_transforms("test"))
    if len(ds) != len(manifest_df):
        raise AssertionError(
            f"ManifestDataset dropped rows: {len(ds)} of {len(manifest_df)}"
        )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False, drop_last=False,
    )

    torch.manual_seed(rng_seed)

    mean_chunks: List[np.ndarray] = []
    mi_chunks: List[np.ndarray] = []
    lab_chunks: List[np.ndarray] = []
    t0 = time.time()

    for bi, (images, labels) in enumerate(loader, start=1):
        images = images.to(DEVICE, non_blocking=False)
        probs = mc_forward(model, model_name, images, T=T)          # (T, B, C)
        mean_p = probs.mean(dim=0)                                   # (B, C)
        # Mutual information = H(mean) - mean_t H(p_t). Both in nats.
        h_mean = -(mean_p * torch.log(mean_p + 1e-10)).sum(-1)
        h_each = -(probs * torch.log(probs + 1e-10)).sum(-1).mean(0)
        mean_chunks.append(mean_p.cpu().numpy())
        mi_chunks.append((h_mean - h_each).cpu().numpy())
        lab_chunks.append(labels.numpy())
        if bi % log_every == 0 or bi == len(loader):
            done = min(bi * batch_size, len(ds))
            rate = done / max(time.time() - t0, 1e-9)
            eta = (len(ds) - done) / max(rate, 1e-9)
            print(f"    {done}/{len(ds)}  {rate:.1f} img/s  eta {eta:5.0f}s", flush=True)

    mean_probs = np.concatenate(mean_chunks).astype(np.float64)
    mi = np.concatenate(mi_chunks).astype(np.float64)
    labels_np = np.concatenate(lab_chunks).astype(int)

    if not np.isfinite(mean_probs).all():
        raise AssertionError("non-finite probabilities in MC output")
    row_sums = mean_probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise AssertionError(f"probabilities do not sum to 1 (max dev {np.abs(row_sums-1).max():.2e})")

    return {
        "mean_probs": mean_probs,
        "mutual_information": mi,
        "labels_from_loader": labels_np,
    }


def to_cache_frame(
    manifest_df: pd.DataFrame,
    out: Dict[str, np.ndarray],
    split_col: str,
    T: int = MC_T,
) -> pd.DataFrame:
    """Assemble a Contract 1 dataframe."""
    p = out["mean_probs"]
    labels = manifest_df["label"].to_numpy()
    if not np.array_equal(labels, out["labels_from_loader"]):
        raise AssertionError("row order drift between manifest and DataLoader")

    ent = entropy_nats(p)
    df = pd.DataFrame({
        "image_path": manifest_df["image_path"].to_numpy(),
        "sha256": manifest_df["sha256"].to_numpy(),
        "true_label": labels.astype(np.int64),
        "true_label_name": manifest_df["class_name"].to_numpy(),
        split_col: manifest_df[split_col if split_col in manifest_df else "split"].to_numpy(),
        "plane": manifest_df["plane"].to_numpy(),
        "sequence": manifest_df["sequence"].to_numpy(),
        "p_glioma": p[:, 0],
        "p_meningioma": p[:, 1],
        "p_pituitary": p[:, 2],
        "p_notumor": p[:, 3],
        "pred_label": p.argmax(axis=1).astype(np.int64),
        "p_tumor": p[:, 0] + p[:, 1] + p[:, 2],
        "entropy": ent,
        "mutual_information": out["mutual_information"],
        "mc_T": np.int64(T),
    })
    # Convenience column: src/code.py reports predictive entropy in bits.
    # Contract 1 specifies nats. Both are here so nobody mixes units by accident.
    df["entropy_bits"] = ent / np.log(2.0)
    if df.isna().any().any():
        raise AssertionError("NaNs in cache frame")
    return df


# ===========================================================================
# Subcommand: cache
# ===========================================================================

def cmd_cache(args: argparse.Namespace) -> None:
    BRISC_OUT.mkdir(parents=True, exist_ok=True)
    (BRISC_OUT / "predictions").mkdir(parents=True, exist_ok=True)
    (INTERNAL_OUT / "predictions").mkdir(parents=True, exist_ok=True)

    print(f"device: {DEVICE}   torch {torch.__version__}")
    prep = assert_preprocessing_parity()

    brisc_df = build_brisc_manifest(args.brisc)
    assert_class_counts(brisc_df)
    sha_report = verify_sha256(brisc_df, n_sample=args.sha_sample)

    internal = {s: build_internal_manifest(s) for s in ("val", "test")}
    for s, d in internal.items():
        print(f"internal {s}: {len(d)} images  {d['class_name'].value_counts().to_dict()}")

    seeds = args.seeds or SEEDS
    models = args.models or MODELS

    if args.sanity:
        model = load_checkpoint(models[0], seeds[0])
        max_abs = verify_fast_path(model, models[0])
        print(f"fast-path verification ({models[0]}): bit-identical, max abs diff {max_abs:.1e}")
        sub = brisc_df.groupby("class_name", group_keys=False).head(50).reset_index(drop=True)
        print(f"\nSANITY RUN: {len(sub)} BRISC images, {models[0]} seed {seeds[0]}")
        out = predict_split(model, models[0], sub, "brisc", T=MC_T,
                            batch_size=args.batch_size, num_workers=args.workers,
                            rng_seed=seeds[0], log_every=5)
        frame = to_cache_frame(sub, out, "brisc_split")
        acc = float((frame["pred_label"] == frame["true_label"]).mean())
        tumour = frame["true_label"] != NOTUMOR_IDX
        miss = float((frame.loc[tumour, "pred_label"] == NOTUMOR_IDX).mean())
        print(f"\n  4-way accuracy       {acc:.4f}")
        print(f"  tumour miss rate     {miss:.4f}  (n_tumour={int(tumour.sum())})")
        print(f"  mean entropy (nats)  {frame['entropy'].mean():.4f}")
        print(f"  p range              [{frame[['p_glioma','p_meningioma','p_pituitary','p_notumor']].to_numpy().min():.4f}, "
              f"{frame[['p_glioma','p_meningioma','p_pituitary','p_notumor']].to_numpy().max():.4f}]")
        print("\n  predicted class distribution:")
        print(frame["pred_label"].map({v: k for k, v in LABEL_INDEX.items()}).value_counts().to_string())
        print("\nSanity run done. Re-run without --sanity for the full cache.")
        return

    provenance: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mc_T": MC_T,
        "device": str(DEVICE),
        "torch": torch.__version__,
        "checkpoints_dir": str(CHECKPOINTS),
        "brisc_dir": str(args.brisc),
        "preprocessing": prep,
        "sha256_spot_check": sha_report,
        "brisc_to_internal": BRISC_TO_INTERNAL,
        "label_index": LABEL_INDEX,
        "entropy_units": "nats (column 'entropy'); 'entropy_bits' is the same quantity in bits",
        "inference_only": "torch.no_grad throughout; no optimizer, no backward, no BRISC-fitted parameter",
        "fast_path": {},
        "runs": [],
    }

    for model_name in models:
        for seed in seeds:
            model = load_checkpoint(model_name, seed)
            max_abs = verify_fast_path(model, model_name)
            provenance["fast_path"][f"{model_name}_seed{seed}"] = {
                "bit_identical_to_predict_with_uncertainty": True,
                "max_abs_diff": max_abs,
            }

            jobs = [
                ("brisc", brisc_df, "brisc_split",
                 BRISC_OUT / "predictions" / f"{model_name}_seed{seed}.parquet"),
                ("val", internal["val"], "internal_split",
                 INTERNAL_OUT / "predictions" / f"{model_name}_seed{seed}_val.parquet"),
                ("test", internal["test"], "internal_split",
                 INTERNAL_OUT / "predictions" / f"{model_name}_seed{seed}_test.parquet"),
            ]
            for tag, mdf, split_col, out_path in jobs:
                if out_path.exists() and not args.overwrite:
                    print(f"  skip (exists): {out_path.name}")
                    continue
                print(f"\n[{model_name} seed {seed}] {tag}: {len(mdf)} images x T={MC_T}")
                t0 = time.time()
                res = predict_split(
                    model, model_name, mdf,
                    "brisc" if tag == "brisc" else tag,
                    T=MC_T, batch_size=args.batch_size, num_workers=args.workers,
                    rng_seed=seed,
                )
                frame = to_cache_frame(mdf, res, split_col)
                frame.to_parquet(out_path, index=False)
                acc = float((frame["pred_label"] == frame["true_label"]).mean())
                tumour = frame["true_label"] != NOTUMOR_IDX
                miss = float((frame.loc[tumour, "pred_label"] == NOTUMOR_IDX).mean())
                dt = time.time() - t0
                print(f"  -> {out_path.name}  n={len(frame)}  acc={acc:.4f}  "
                      f"miss={miss:.4f}  {dt:.0f}s")
                provenance["runs"].append({
                    "model": model_name, "seed": seed, "split": tag,
                    "n": int(len(frame)), "accuracy": acc, "tumor_miss_rate": miss,
                    "seconds": round(dt, 1), "path": str(out_path.relative_to(REPO_ROOT)),
                })
            del model

    (BRISC_OUT / "predictions" / "cache_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"\nprovenance -> {BRISC_OUT / 'predictions' / 'cache_provenance.json'}")


# ===========================================================================
# Cache readers, used by this file and by operating_point.py
# ===========================================================================

def load_brisc_cache(model_name: str, seed: int) -> pd.DataFrame:
    return pd.read_parquet(BRISC_OUT / "predictions" / f"{model_name}_seed{seed}.parquet")


def load_internal_cache(model_name: str, seed: int, split: str) -> pd.DataFrame:
    return pd.read_parquet(
        INTERNAL_OUT / "predictions" / f"{model_name}_seed{seed}_{split}.parquet"
    )


OVERLAP_CSV = BRISC_OUT / "brisc_overlap_per_image.csv"


def attach_overlap_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add the contamination columns to a BRISC prediction frame.

    Roughly 80 percent of BRISC 2025 is pixel-identical to images in the
    internal dataset, and a large share of that is in the internal TRAINING
    split. Any BRISC number computed without filtering on `clean_vs_train` is
    partly the model being scored on its own training data. Run
    `external_validation_brisc.py overlap` to produce the file.
    """
    if not OVERLAP_CSV.exists():
        raise FileNotFoundError(
            f"{OVERLAP_CSV} is missing. Run: python analysis/"
            "external_validation_brisc.py overlap"
        )
    ov = pd.read_csv(OVERLAP_CSV).set_index("image_path")
    out = df.copy()
    for col in ("d_train", "d_val", "d_test", "d_any", "clean_vs_train", "clean_vs_any"):
        out[col] = out["image_path"].map(ov[col])
    if out[["d_train", "clean_vs_train"]].isna().any().any():
        raise AssertionError("overlap flags did not join cleanly onto the cache")
    out["clean_vs_train"] = out["clean_vs_train"].astype(bool)
    out["clean_vs_any"] = out["clean_vs_any"].astype(bool)
    return out


def clean_subset(df: pd.DataFrame) -> pd.DataFrame:
    """The BRISC images the model genuinely never trained on."""
    return attach_overlap_flags(df).query("clean_vs_train").reset_index(drop=True)


PROB_COLS = ["p_glioma", "p_meningioma", "p_pituitary", "p_notumor"]


def ensemble_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Probability-averaged ensemble over seeds.

    Averages the MC-Dropout mean probabilities, then recomputes argmax, p_tumor
    and entropy from the averaged distribution. Mutual information is not
    meaningfully defined across seeds, so it is dropped.
    """
    base = frames[0]
    for f in frames[1:]:
        if not np.array_equal(base["image_path"].to_numpy(), f["image_path"].to_numpy()):
            raise AssertionError("row order differs between seed caches")
    p = np.mean([f[PROB_COLS].to_numpy() for f in frames], axis=0)
    out = base.drop(columns=["mutual_information"]).copy()
    out[PROB_COLS] = p
    out["pred_label"] = p.argmax(axis=1).astype(np.int64)
    out["p_tumor"] = p[:, :3].sum(axis=1)
    ent = entropy_nats(p)
    out["entropy"] = ent
    out["entropy_bits"] = ent / np.log(2.0)
    return out


# ===========================================================================
# Subcommand: metrics  (Phase 2 and Phase 3)
# ===========================================================================

def _figure_style():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "figures_common", REPO_ROOT / "figures" / "common.py")
    if spec is None or spec.loader is None:
        return lambda: None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.set_style


def _plots(brisc: pd.DataFrame, itest: pd.DataFrame, tag: str, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from code import compute_risk_coverage_curve as rc_curve

    set_style = _figure_style()
    set_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    PC = ["p_glioma", "p_meningioma", "p_pituitary", "p_notumor"]

    # --- reliability diagram, BRISC vs internal test -----------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, (name, d) in zip(axes, [("Internal test", itest), ("BRISC 2025 (external)", brisc)]):
        probs = d[PC].to_numpy()
        y = d["true_label"].to_numpy()
        conf = probs.max(axis=1)
        corr = (probs.argmax(axis=1) == y).astype(float)
        bins = np.linspace(0, 1, 16)
        accs = [corr[(conf > lo) & (conf <= hi)].mean()
                if ((conf > lo) & (conf <= hi)).sum() else 0.0
                for lo, hi in zip(bins[:-1], bins[1:])]
        ece = compute_calibration_metrics(probs, y, n_bins=15)["ece"]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
        ax.bar(bins[:-1], accs, width=1/15, align="edge", alpha=0.75,
               edgecolor="black", linewidth=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        ax.set_title(f"{name}\nECE = {ece:.4f}  n = {len(d)}")
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Reliability - {tag}", y=1.02)
    fig.tight_layout(); fig.savefig(out_dir / f"reliability_{tag}.png"); plt.close(fig)

    # --- risk-coverage ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for name, d, col in [("Internal test", itest, "steelblue"),
                         ("BRISC 2025", brisc, "crimson")]:
        r = rc_curve(d[PC].to_numpy(), d["true_label"].to_numpy())
        ax.plot(r["coverage"], r["risk"], lw=2, color=col,
                label=f"{name}  AURC = {r['aurc']:.4f}")
    ax.set_xlabel("Coverage (share of scans the tool answers on)")
    ax.set_ylabel("Risk (1 - accuracy)")
    ax.set_title(f"Risk-coverage - {tag}")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / f"risk_coverage_{tag}.png"); plt.close(fig)

    # --- confusion matrices ----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, (name, d) in zip(axes, [("Internal test", itest), ("BRISC 2025", brisc)]):
        cm = np.zeros((4, 4), dtype=int)
        for t, q in zip(d["true_label"], d["pred_label"]):
            cm[t, q] += 1
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i,j]}\n{cmn[i,j]*100:.1f}%", ha="center", va="center",
                        fontsize=7, color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(4)); ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{name}  (n = {len(d)})")
        ax.grid(False)
    fig.suptitle(f"Confusion - {tag}. Bottom-left 3 cells of the last column are missed tumours.",
                 fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.02)
    fig.savefig(out_dir / f"confusion_{tag}.png"); plt.close(fig)

    # --- entropy by correctness ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for ax, (name, d) in zip(axes, [("Internal test", itest), ("BRISC 2025", brisc)]):
        y, p, e = d["true_label"].to_numpy(), d["pred_label"].to_numpy(), d["entropy"].to_numpy()
        wrong = p != y
        missed = (y != NOTUMOR_IDX) & (p == NOTUMOR_IDX)
        bins = np.linspace(0, max(e.max(), 1e-6), 45)
        ax.hist(e[~wrong], bins=bins, alpha=0.6, density=True, label=f"Correct (n={int((~wrong).sum())})", color="steelblue")
        ax.hist(e[wrong], bins=bins, alpha=0.6, density=True, label=f"Wrong (n={int(wrong.sum())})", color="darkorange")
        if missed.any():
            for v in e[missed]:
                ax.axvline(v, color="crimson", lw=0.7, alpha=0.55)
            ax.plot([], [], color="crimson", lw=1, label=f"Missed tumour (n={int(missed.sum())})")
        ax.set_xlabel("Predictive entropy (nats)"); ax.set_ylabel("Density")
        ax.set_title(name); ax.legend(fontsize=7)
    fig.suptitle(f"Entropy on right vs wrong answers - {tag}. Overlap means deferral cannot separate them.",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / f"entropy_correct_vs_wrong_{tag}.png"); plt.close(fig)


def cmd_metrics(args: argparse.Namespace) -> None:
    import safety_metrics as sm

    fig_dir = BRISC_OUT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    nb = args.boot

    ov = pd.read_csv(OVERLAP_CSV)
    n_clean = int(ov["clean_vs_train"].sum())

    report: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "BRISC2025",
        "mc_T": MC_T,
        "n_bootstrap": nb,
        "binary_threshold": 0.5,
        "entropy_units": "nats",
        "CONTAMINATION_WARNING": (
            "About 80 percent of BRISC 2025 is pixel-identical to images in "
            "data/brain_tumor/, and 56 percent matches the internal TRAINING "
            "split specifically. `brisc_full` below is therefore NOT an external "
            "result and must never be quoted as one. `brisc_clean` is the only "
            "genuine external number: BRISC images whose nearest internal "
            f"training image is more than Hamming 5 away, n={n_clean}. See "
            "analysis/results/brisc/brisc_internal_overlap.json."
        ),
        "subsets": {
            "brisc_full": {"n": int(len(ov)), "status": "CONTAMINATED, reference only"},
            "brisc_clean": {"n": n_clean, "definition": "d_train > 5",
                            "status": "the genuine external test set"},
            "brisc_clean_strict": {"n": int((ov["d_train"] > 10).sum()),
                                   "definition": "d_train > 10",
                                   "status": "sensitivity check; too few tumours to "
                                             "carry the headline"},
        },
        "note": "Nothing here was fitted on BRISC. These are scores of a frozen model.",
        "models": {},
    }

    per_seed_preds: Dict[str, np.ndarray] = {}
    clean_mask_global: Optional[np.ndarray] = None

    for model_name in MODELS:
        entry: Dict[str, Any] = {"per_seed": {}, "ensemble5": {}}
        brisc_frames, itest_frames, val_frames = [], [], []

        for seed in SEEDS:
            b = attach_overlap_flags(load_brisc_cache(model_name, seed))
            t = load_internal_cache(model_name, seed, "test")
            v = load_internal_cache(model_name, seed, "val")
            brisc_frames.append(b); itest_frames.append(t); val_frames.append(v)
            bc = b[b["clean_vs_train"]]
            entry["per_seed"][str(seed)] = {
                "brisc_clean": sm.full_report(bc, 0.5, seed=seed, n_resamples=nb),
                "brisc_full": sm.full_report(b, 0.5, seed=seed, n_resamples=nb),
                "internal_test": sm.full_report(t, 0.5, seed=seed, n_resamples=nb),
            }
            e = entry["per_seed"][str(seed)]
            print(f"  {model_name} seed {seed}: "
                  f"CLEAN miss={e['brisc_clean']['tumor_miss_rate']['value']:.4f} "
                  f"acc={e['brisc_clean']['standard']['four_way_accuracy']['value']:.4f}  |  "
                  f"full(contaminated) miss={e['brisc_full']['tumor_miss_rate']['value']:.4f} "
                  f"acc={e['brisc_full']['standard']['four_way_accuracy']['value']:.4f}")

        be = ensemble_frames(brisc_frames)
        be = attach_overlap_flags(be)
        te = ensemble_frames(itest_frames)
        ve = ensemble_frames(val_frames)
        be_clean = be[be["clean_vs_train"]].reset_index(drop=True)
        be_strict = be[be["d_train"] > 10].reset_index(drop=True)
        per_seed_preds[model_name] = be["pred_label"].to_numpy()
        clean_mask_global = be["clean_vs_train"].to_numpy()

        entry["ensemble5"]["brisc_clean"] = sm.full_report(
            be_clean, 0.5, n_resamples=nb, subgroups=("plane", "brisc_split"))
        entry["ensemble5"]["brisc_full"] = sm.full_report(
            be, 0.5, n_resamples=nb, subgroups=("plane", "brisc_split"))
        entry["ensemble5"]["brisc_clean_strict"] = sm.full_report(be_strict, 0.5, n_resamples=nb)
        entry["ensemble5"]["internal_test"] = sm.full_report(te, 0.5, n_resamples=nb)
        entry["ensemble5"]["internal_val"] = sm.full_report(ve, 0.5, n_resamples=nb)

        # Contamination effect, measured directly: same checkpoints, the BRISC
        # images they trained on vs the ones they did not.
        be_dirty = be[~be["clean_vs_train"]].reset_index(drop=True)
        entry["contamination_effect"] = {
            "seen_in_training": {
                "n": int(len(be_dirty)),
                "accuracy": float((be_dirty["pred_label"] == be_dirty["true_label"]).mean()),
                "tumor_miss_rate": sm.tumor_miss_rate(be_dirty, n_resamples=nb),
            },
            "not_seen_in_training": {
                "n": int(len(be_clean)),
                "accuracy": float((be_clean["pred_label"] == be_clean["true_label"]).mean()),
                "tumor_miss_rate": sm.tumor_miss_rate(be_clean, n_resamples=nb),
            },
        }

        # The drop is the finding. Internal held-out test vs the CLEAN external set.
        for label, ext in (("clean", "brisc_clean"), ("full_contaminated", "brisc_full")):
            entry[f"drop_internal_to_{label}"] = {
                "four_way_accuracy": sm.drop_with_ci(
                    entry["ensemble5"]["internal_test"], entry["ensemble5"][ext],
                    ("standard", "four_way_accuracy")),
                "macro_f1": sm.drop_with_ci(
                    entry["ensemble5"]["internal_test"], entry["ensemble5"][ext],
                    ("standard", "macro_f1")),
                "macro_auc": sm.drop_with_ci(
                    entry["ensemble5"]["internal_test"], entry["ensemble5"][ext],
                    ("standard", "macro_auc")),
                "tumor_miss_rate": sm.drop_with_ci(
                    entry["ensemble5"]["internal_test"], entry["ensemble5"][ext],
                    ("tumor_miss_rate",)),
                "binary_sensitivity": sm.drop_with_ci(
                    entry["ensemble5"]["internal_test"], entry["ensemble5"][ext],
                    ("binary", "sensitivity")),
                "ece_15bin": {
                    "internal": entry["ensemble5"]["internal_test"]["standard"]["ece_15bin"],
                    "external": entry["ensemble5"][ext]["standard"]["ece_15bin"],
                    "change": (entry["ensemble5"][ext]["standard"]["ece_15bin"]
                               - entry["ensemble5"]["internal_test"]["standard"]["ece_15bin"]),
                },
                "note": "drop = internal minus external. For accuracy, F1, AUC and "
                        "sensitivity a POSITIVE drop means external is worse. For "
                        "tumour miss rate a NEGATIVE drop means external is worse, "
                        "because a higher miss rate is worse.",
            }
        report["models"][model_name] = entry
        _plots(be_clean, te, f"{model_name}_clean", fig_dir)
        _plots(be, te, f"{model_name}_full_contaminated", fig_dir)

    # ViT vs ResNet-50, paired on the same images, on the CLEAN subset.
    y_full = attach_overlap_flags(load_brisc_cache("vit", 42))
    m = clean_mask_global
    chi2, pval = mcnemar_test(y_full["true_label"].to_numpy()[m],
                              per_seed_preds["vit"][m], per_seed_preds["resnet50"][m])
    chi2f, pvalf = mcnemar_test(y_full["true_label"].to_numpy(),
                                per_seed_preds["vit"], per_seed_preds["resnet50"])
    report["vit_vs_resnet50_mcnemar"] = {
        "clean_subset": {"n": int(m.sum()), "chi2": chi2, "p_value": pval,
                         "significant": bool(pval < 0.05)},
        "full_contaminated": {"n": int(len(y_full)), "chi2": chi2f, "p_value": pvalf,
                              "significant": bool(pvalf < 0.05)},
        "note": "5-seed probability-averaged ensembles, paired on the same images",
    }
    print(f"\n  McNemar ViT vs ResNet-50, clean subset: chi2={chi2:.3f} p={pval:.4g}")

    out_path = BRISC_OUT / "brisc_metrics.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n-> {out_path}")

    write_brisc_results_md(report, BRISC_OUT / "BRISC_RESULTS.md")
    print(f"-> {BRISC_OUT / 'BRISC_RESULTS.md'}")


def _pct(d: Dict[str, Any], nd: int = 2) -> str:
    """'1.23% (0.80 to 1.78)' from a {value, ci_lo, ci_hi} block."""
    if d is None or not np.isfinite(d.get("value", np.nan)):
        return "n/a"
    return (f"{100*d['value']:.{nd}f}% ({100*d['ci_lo']:.{nd}f} to "
            f"{100*d['ci_hi']:.{nd}f})")


def write_brisc_results_md(rep: Dict[str, Any], path: Path) -> None:
    """Render BRISC_RESULTS.md from the metrics JSON so prose and numbers agree."""
    ov = json.loads((BRISC_OUT / "brisc_internal_overlap.json").read_text())
    L: List[str] = []
    A = L.append

    A("# External validation on BRISC 2025")
    A("")
    A("Session A. Generated by `analysis/external_validation_brisc.py metrics`.")
    A("")
    A("## Read this first")
    A("")
    A("**BRISC 2025 is not an external dataset for these models. Most of it is "
      "the training set.**")
    A("")
    n_tr = ov["by_threshold"][4]["n_matching_internal_TRAIN"]
    n_any = ov["by_threshold"][4]["n_matching_any_internal"]
    A(f"- {n_any} of 6,000 BRISC images ({100*n_any/6000:.1f}%) are near-duplicates "
      f"of images in `data/brain_tumor/`.")
    A(f"- {n_tr} of 6,000 ({100*n_tr/6000:.1f}%) match the internal **training** "
      f"split specifically.")
    A("- A pixel check on 40 random distance-0 pairs gave mean absolute difference "
      "0.0 and correlation 1.0 on all 40. They are the same image files under new "
      "names. Evidence figure: `overlap_evidence_pairs.png`.")
    A("")
    A("The contamination is not spread evenly. It sits almost entirely in the "
      "tumour classes:")
    A("")
    A("| BRISC class | images | match internal train | genuinely unseen |")
    A("|---|---|---|---|")
    for cls, n, mt in (("glioma", 1401, 965), ("meningioma", 1635, 1134),
                       ("pituitary", 1757, 1218), ("notumor", 1207, 49)):
        A(f"| {cls} | {n} | {mt} ({100*mt/n:.0f}%) | {n-mt} |")
    A("")
    A("So on the full BRISC set the model is largely being re-shown tumours it "
      "memorised, while the no-tumour class is a real test. That combination "
      "flatters the tumour miss rate specifically, which is the number that "
      "matters most.")
    A("")
    A(f"**Everything below is reported on the clean subset: n = "
      f"{rep['subsets']['brisc_clean']['n']}, the BRISC images whose nearest "
      f"internal training image is more than perceptual-hash Hamming distance 5 "
      f"away.** Full-BRISC numbers appear only in the contamination-effect table, "
      f"labelled as such.")
    A("")
    A("One limitation the clean subset does not fix: neither dataset ships patient "
      "identifiers, so other slices from the same patients may still be in "
      "training. **The clean subset is an upper bound on true external "
      "performance, not an unbiased estimate.**")
    A("")

    # ---- headline safety ------------------------------------------------
    A("## 1. Tumour miss rate: a real tumour shown to a clinician as no-tumour")
    A("")
    A("Argmax over the four MC-Dropout mean probabilities. 5-seed "
      "probability-averaged ensemble. Bootstrap 95% CI, 1,000 resamples.")
    A("")
    A("| backbone | clean subset | internal test | full BRISC (contaminated) |")
    A("|---|---|---|---|")
    for mn in MODELS:
        e = rep["models"][mn]["ensemble5"]
        A(f"| {mn} | **{_pct(e['brisc_clean']['tumor_miss_rate'])}** | "
          f"{_pct(e['internal_test']['tumor_miss_rate'])} | "
          f"{_pct(e['brisc_full']['tumor_miss_rate'])} |")
    A("")
    A("Per tumour class on the clean subset. An aggregate hides a single class "
      "failing.")
    A("")
    A("| backbone | glioma | meningioma | pituitary |")
    A("|---|---|---|---|")
    for mn in MODELS:
        pc = rep["models"][mn]["ensemble5"]["brisc_clean"]["per_class_miss_rate"]
        A(f"| {mn} | {_pct(pc['glioma'])} | {_pct(pc['meningioma'])} | "
          f"{_pct(pc['pituitary'])} |")
    A("")
    A("### What contamination was worth")
    A("")
    A("Same checkpoints, BRISC images split by whether the model trained on them.")
    A("")
    A("| backbone | seen in training | not seen | ")
    A("|---|---|---|")
    for mn in MODELS:
        c = rep["models"][mn]["contamination_effect"]
        A(f"| {mn} | acc {c['seen_in_training']['accuracy']:.4f}, miss "
          f"{100*c['seen_in_training']['tumor_miss_rate']['value']:.2f}% "
          f"(n={c['seen_in_training']['n']}) | acc "
          f"{c['not_seen_in_training']['accuracy']:.4f}, miss "
          f"{100*c['not_seen_in_training']['tumor_miss_rate']['value']:.2f}% "
          f"(n={c['not_seen_in_training']['n']}) |")
    A("")

    # ---- binary ---------------------------------------------------------
    A("## 2. Tumour vs no-tumour, the call a clinic acts on")
    A("")
    A("`p_tumor >= 0.5`. Clean subset.")
    A("")
    A("| backbone | accuracy | sensitivity | specificity | PPV | NPV |")
    A("|---|---|---|---|---|---|")
    for mn in MODELS:
        b = rep["models"][mn]["ensemble5"]["brisc_clean"]["binary"]
        A(f"| {mn} | {_pct(b['accuracy'])} | {_pct(b['sensitivity'])} | "
          f"{_pct(b['specificity'])} | {_pct(b['ppv'])} | {_pct(b['npv'])} |")
    A("")
    A("PPV and NPV here are at the clean subset's own tumour prevalence "
      f"({100*rep['models'][MODELS[0]]['ensemble5']['brisc_clean']['n_tumor']/rep['subsets']['brisc_clean']['n']:.0f}%), "
      "which is nothing like a rural clinic. Do not carry them across. "
      "`analysis/results/safety/OPERATING_POINT.md` projects them onto plausible "
      "clinic prevalences.")
    A("")

    # ---- standard -------------------------------------------------------
    A("## 3. Standard metrics, and the drop from internal to external")
    A("")
    A("| backbone | set | accuracy | macro F1 | macro AUC | ECE | Brier | AURC |")
    A("|---|---|---|---|---|---|---|---|")
    for mn in MODELS:
        for lbl, key in (("internal test", "internal_test"),
                         ("BRISC clean", "brisc_clean"),
                         ("BRISC full (contaminated)", "brisc_full")):
            s = rep["models"][mn]["ensemble5"][key]["standard"]
            A(f"| {mn} | {lbl} | {_pct(s['four_way_accuracy'])} | "
              f"{_pct(s['macro_f1'])} | {_pct(s['macro_auc'])} | "
              f"{s['ece_15bin']:.4f} | {s['brier']:.4f} | {s['aurc']:.4f} |")
    A("")
    A("The drop, internal held-out test minus BRISC clean. Positive means the "
      "external set is worse, except for miss rate where negative means worse.")
    A("")
    A("| backbone | metric | internal | external | drop | approx 95% CI |")
    A("|---|---|---|---|---|---|")
    for mn in MODELS:
        d = rep["models"][mn]["drop_internal_to_clean"]
        for k in ("four_way_accuracy", "macro_f1", "macro_auc",
                  "binary_sensitivity", "tumor_miss_rate"):
            v = d[k]
            A(f"| {mn} | {k} | {v['internal']:.4f} | {v['external']:.4f} | "
              f"{v['drop']:+.4f} | [{v['drop_ci_approx'][0]:+.4f}, "
              f"{v['drop_ci_approx'][1]:+.4f}] |")
        e = d["ece_15bin"]
        A(f"| {mn} | ece_15bin | {e['internal']:.4f} | {e['external']:.4f} | "
          f"{e['change']:+.4f} (change) | - |")
    A("")
    A("Accuracy at fixed coverage, clean subset:")
    A("")
    A("| backbone | 80% | 90% | 95% | 100% |")
    A("|---|---|---|---|---|")
    for mn in MODELS:
        s = rep["models"][mn]["ensemble5"]["brisc_clean"]["standard"]
        c = s["acc_at_coverage"]
        A(f"| {mn} | {c['80']:.4f} | {c['90']:.4f} | {c['95']:.4f} | "
          f"{s['four_way_accuracy']['value']:.4f} |")
    A("")

    # ---- uncertainty ----------------------------------------------------
    A("## 4. Does the uncertainty signal work?")
    A("")
    A("If entropy on wrong answers looks like entropy on right answers, "
      "defer-to-human cannot work, because the tool has no way to tell which "
      "cases to escalate. Entropy in nats.")
    A("")
    A("| backbone | H correct | H wrong | H missed tumours | error AUROC |")
    A("|---|---|---|---|---|")
    for mn in MODELS:
        en = rep["models"][mn]["ensemble5"]["brisc_clean"]["entropy"]
        mt = en["entropy_missed_tumors"]
        mt_s = (f"{mt['mean']:.3f} (n={mt['n']})" if mt.get("n") else "no misses")
        ea = en["error_detection_auroc"]
        A(f"| {mn} | {en['entropy_correct']['mean']:.3f} | "
          f"{en['entropy_wrong']['mean']:.3f} | {mt_s} | "
          f"{ea['value']:.3f} ({ea['ci_lo']:.3f} to {ea['ci_hi']:.3f}) |")
    A("")
    A("Overlap check. Share of wrong answers whose entropy sits below the median "
      "entropy of correct answers. High means deferral will not separate them.")
    A("")
    A("| backbone | wrong below median-correct | missed tumours below median-correct |")
    A("|---|---|---|")
    for mn in MODELS:
        en = rep["models"][mn]["ensemble5"]["brisc_clean"]["entropy"]
        fw = en["frac_wrong_below_median_correct_entropy"]
        fm = en["frac_missed_below_median_correct_entropy"]
        A(f"| {mn} | {100*fw:.1f}% | "
          f"{'n/a' if not np.isfinite(fm) else f'{100*fm:.1f}%'} |")
    A("")

    # ---- subgroups ------------------------------------------------------
    A("## 5. By plane")
    A("")
    A("BRISC is roughly a third each. Contrary to the assumption going in, the "
      "internal training data is **not** mostly axial: visual review of 40 random "
      "training images per class shows all three planes well represented in the "
      "tumour classes. The no-tumour class is the exception and is almost entirely "
      "axial. See `figures/domain_shift/`.")
    A("")
    A("| backbone | plane | n | miss rate | accuracy | sensitivity | specificity |")
    A("|---|---|---|---|---|---|---|")
    for mn in MODELS:
        bp = rep["models"][mn]["ensemble5"]["brisc_clean"].get("by_plane", {})
        for plane in ("axial", "coronal", "sagittal"):
            g = bp.get(plane)
            if not g or "tumor_miss_rate" not in g:
                A(f"| {mn} | {plane} | {g['n'] if g else 0} | - | - | - | - |")
                continue
            A(f"| {mn} | {plane} | {g['n']} | {_pct(g['tumor_miss_rate'])} | "
              f"{_pct(g['standard']['four_way_accuracy'])} | "
              f"{_pct(g['binary']['sensitivity'])} | "
              f"{_pct(g['binary']['specificity'])} |")
    A("")
    A("## 6. By BRISC's own split")
    A("")
    A("Reported so the numbers can be lined up against published BRISC results. "
      "Note both are still filtered to the clean subset, so neither matches a "
      "published BRISC test-set number directly.")
    A("")
    A("| backbone | brisc_split | n | miss rate | accuracy |")
    A("|---|---|---|---|---|")
    for mn in MODELS:
        bs = rep["models"][mn]["ensemble5"]["brisc_clean"].get("by_brisc_split", {})
        for sp in ("train", "test"):
            g = bs.get(sp)
            if not g or "tumor_miss_rate" not in g:
                continue
            A(f"| {mn} | {sp} | {g['n']} | {_pct(g['tumor_miss_rate'])} | "
              f"{_pct(g['standard']['four_way_accuracy'])} |")
    A("")

    # ---- per seed -------------------------------------------------------
    A("## 7. Per seed, clean subset")
    A("")
    A("| backbone | seed | miss rate | accuracy | macro F1 | ECE |")
    A("|---|---|---|---|---|---|")
    for mn in MODELS:
        for seed in SEEDS:
            r = rep["models"][mn]["per_seed"][str(seed)]["brisc_clean"]
            A(f"| {mn} | {seed} | {_pct(r['tumor_miss_rate'])} | "
              f"{_pct(r['standard']['four_way_accuracy'])} | "
              f"{_pct(r['standard']['macro_f1'])} | "
              f"{r['standard']['ece_15bin']:.4f} |")
    A("")
    mc = rep["vit_vs_resnet50_mcnemar"]
    A(f"McNemar, ViT vs ResNet-50, 5-seed ensembles paired on the clean subset "
      f"(n={mc['clean_subset']['n']}): chi2 = {mc['clean_subset']['chi2']:.3f}, "
      f"p = {mc['clean_subset']['p_value']:.4g}, "
      f"{'significant' if mc['clean_subset']['significant'] else 'not significant'} "
      f"at 0.05.")
    A("")

    # ---- confusion ------------------------------------------------------
    A("## 8. Confusion matrices, clean subset")
    A("")
    for mn in MODELS:
        A(f"**{mn}**  rows = true, columns = predicted, order "
          f"{', '.join(CLASS_NAMES)}")
        A("")
        A("| true \\ pred | " + " | ".join(CLASS_NAMES) + " |")
        A("|---|" + "---|" * 4)
        cm = rep["models"][mn]["ensemble5"]["brisc_clean"]["standard"]["confusion_matrix"]
        for i, row in enumerate(cm):
            A(f"| {CLASS_NAMES[i]} | " + " | ".join(str(x) for x in row) + " |")
        A("")
    A("The last column of the first three rows is the number that matters. Those "
      "are real tumours the tool would display as no-tumour.")
    A("")
    A("## 9. Method and provenance")
    A("")
    A(f"- MC Dropout, T = {rep['mc_T']}, matching training.")
    A("- Inference only, `torch.no_grad()` throughout. No optimizer, no backward "
      "pass, nothing fitted on BRISC.")
    A("- Preprocessing asserted identical to `get_transforms(\"test\")` in "
      "`src/code.py`: resize to 224x224, ImageNet normalisation.")
    A("- MC path verified bit-identical to `predict_with_uncertainty` on every "
      "checkpoint before any cache was written.")
    A(f"- Bootstrap percentile CIs, {rep['n_bootstrap']} resamples.")
    A("- Entropy in nats. `results/20260703_155524/` reports entropy in bits; "
      "the caches carry `entropy_bits` for that comparison.")
    A("- Checkpoints: `results/20260703_155524/{vit,resnet50}/seed_{42,123,7,2024,31}/`.")
    A("")

    path.write_text("\n".join(L), encoding="utf-8")


# ===========================================================================
# Subcommand: domain  (Phase 3)
# ===========================================================================

def _symmetry_features(path: str) -> Tuple[float, float, float]:
    """Three cheap, learning-free descriptors of a brain MRI slice.

    Returns (lr_symmetry, bbox_aspect, brain_fraction).

    lr_symmetry: correlation between the image and its left-right mirror, over
    the brain region. Axial and coronal slices are close to mirror-symmetric
    about the midline. Sagittal slices are not, because a sagittal cut shows a
    profile. This is the one plane cue that is robust without a model.

    bbox_aspect: width / height of the brain bounding box.
    brain_fraction: share of the frame occupied by brain.
    """
    from PIL import Image
    img = Image.open(path).convert("L").resize((160, 160))
    a = np.asarray(img, dtype=np.float64) / 255.0
    mask = a > max(0.08, float(np.percentile(a, 40)) * 0.35)
    if mask.sum() < 200:
        return float("nan"), float("nan"), float("nan")
    ys, xs = np.nonzero(mask)
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    flipped = a[:, ::-1]
    v1, v2 = a[mask], flipped[mask]
    if v1.std() < 1e-8 or v2.std() < 1e-8:
        sym = float("nan")
    else:
        sym = float(np.corrcoef(v1, v2)[0, 1])
    return sym, float(w / h), float(mask.mean())


def cmd_domain(args: argparse.Namespace) -> None:
    """Characterise the shift between the internal training data and BRISC."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    out_dir = BRISC_OUT / "figures" / "domain_shift"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style = _figure_style()
    set_style()

    brisc_df = build_brisc_manifest(args.brisc)
    internal = pd.read_csv(REPO_ROOT / "data" / "split_manifest.csv")
    internal["filepath_abs"] = internal["filepath"].map(
        lambda p: str(REPO_ROOT / Path(str(p).replace("\\", "/"))))

    res: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "brisc": {
            "n": int(len(brisc_df)),
            "sequence": brisc_df["sequence"].value_counts().to_dict(),
            "plane": brisc_df["plane"].value_counts().to_dict(),
            "plane_by_class": pd.crosstab(brisc_df["class_name"], brisc_df["plane"]).to_dict(),
            "source": "shipped manifest.csv metadata, not inferred",
        },
        "internal": {
            "n": int(len(internal)),
            "by_split_class": pd.crosstab(internal["split"], internal["class_name"]).to_dict(),
            "sequence": "UNKNOWN. The Kaggle Nickparvar merge (Br35H + SARTAJ + "
                        "Figshare) ships no sequence metadata. Visual inspection "
                        "shows a mix consistent with T1, T1-contrast and T2. "
                        "We do not claim a number.",
        },
    }

    # --- contact sheets, 40 images per class, internal training data -------
    rng = np.random.default_rng(0)
    for cls in CLASS_NAMES:
        sub = internal[(internal["class_name"] == cls) & (internal["split"] == "train")]
        pick = sub.iloc[rng.choice(len(sub), size=min(40, len(sub)), replace=False)]
        fig, axes = plt.subplots(5, 8, figsize=(14, 9))
        for ax, (_, row) in zip(axes.ravel(), pick.iterrows()):
            ax.imshow(Image.open(row["filepath_abs"]).convert("L"), cmap="gray")
            ax.set_title(Path(row["filepath"]).name[:18], fontsize=5)
            ax.axis("off")
        for ax in axes.ravel()[len(pick):]:
            ax.axis("off")
        fig.suptitle(f"Internal TRAIN data, class = {cls}, 40 random images", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"internal_contact_sheet_{cls}.png", dpi=110)
        plt.close(fig)

    # BRISC contact sheets, one per plane, for side-by-side comparison.
    for plane in ["axial", "coronal", "sagittal"]:
        sub = brisc_df[brisc_df["plane"] == plane]
        pick = sub.iloc[rng.choice(len(sub), size=min(24, len(sub)), replace=False)]
        fig, axes = plt.subplots(3, 8, figsize=(14, 5.6))
        for ax, (_, row) in zip(axes.ravel(), pick.iterrows()):
            ax.imshow(Image.open(row["filepath"]).convert("L"), cmap="gray")
            ax.set_title(row["class_name"], fontsize=6)
            ax.axis("off")
        for ax in axes.ravel()[len(pick):]:
            ax.axis("off")
        fig.suptitle(f"BRISC, plane = {plane} (ground-truth metadata label)", fontsize=12)
        fig.tight_layout()
        fig.savefig(out_dir / f"brisc_contact_sheet_{plane}.png", dpi=110)
        plt.close(fig)

    # --- symmetry heuristic ------------------------------------------------
    n_s = args.sample
    b_pick = (brisc_df.groupby("plane", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), n_s), random_state=0)))
    i_pick = internal[internal["split"] == "train"].sample(
        min(len(internal[internal["split"] == "train"]), n_s * 3), random_state=0)

    print(f"computing symmetry features: {len(b_pick)} BRISC, {len(i_pick)} internal ...")
    b_feat = pd.DataFrame([_symmetry_features(p) for p in b_pick["filepath"]],
                          columns=["sym", "aspect", "frac"])
    b_feat["plane"] = b_pick["plane"].to_numpy()
    i_feat = pd.DataFrame([_symmetry_features(p) for p in i_pick["filepath_abs"]],
                          columns=["sym", "aspect", "frac"])
    i_feat["class_name"] = i_pick["class_name"].to_numpy()

    res["symmetry_heuristic"] = {
        "method": "left-right mirror correlation over the brain region, on a "
                  "160x160 grayscale resize. No model, no fitting. Axial and "
                  "coronal slices are near mirror-symmetric about the midline; "
                  "sagittal slices are not.",
        "brisc_by_plane": {k: {kk: float(vv) for kk, vv in v.items()}
                           for k, v in b_feat.groupby("plane")["sym"].describe().T.to_dict().items()},
        "internal_train": {k: float(v) for k, v in i_feat["sym"].describe().to_dict().items()},
        "n_brisc": int(len(b_feat)), "n_internal": int(len(i_feat)),
    }

    # Sagittal share of internal data, estimated by the threshold that best
    # separates BRISC sagittal from BRISC axial+coronal. Reported WITH the
    # separation quality, because a weak separator makes the estimate weak.
    sag = b_feat.loc[b_feat["plane"] == "sagittal", "sym"].dropna().to_numpy()
    nonsag = b_feat.loc[b_feat["plane"] != "sagittal", "sym"].dropna().to_numpy()
    grid = np.linspace(0, 1, 501)
    accs = [((sag < t).sum() + (nonsag >= t).sum()) / (len(sag) + len(nonsag)) for t in grid]
    t_best = float(grid[int(np.argmax(accs))])
    acc_best = float(max(accs))
    from sklearn.metrics import roc_auc_score as _auc
    sep_auc = float(_auc(np.r_[np.ones(len(sag)), np.zeros(len(nonsag))],
                         np.r_[-sag, -nonsag]))
    isym = i_feat["sym"].dropna().to_numpy()
    res["internal_plane_estimate"] = {
        "method": "threshold on the symmetry score, calibrated on BRISC PLANE "
                  "metadata only (never BRISC tumour labels), then applied to "
                  "internal training images",
        "threshold": t_best,
        "separator_accuracy_on_brisc": acc_best,
        "separator_auroc_on_brisc": sep_auc,
        "estimated_sagittal_share_internal_train": float((isym < t_best).mean()),
        "uncertainty": ("The separator only distinguishes sagittal from "
                        "axial-or-coronal. It cannot tell axial from coronal, "
                        "because both are mirror-symmetric. The axial/coronal "
                        "split of the internal data is reported as UNKNOWN."),
    }

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for plane, col in [("axial", "steelblue"), ("coronal", "seagreen"), ("sagittal", "crimson")]:
        ax.hist(b_feat.loc[b_feat["plane"] == plane, "sym"].dropna(), bins=40, alpha=0.45,
                density=True, label=f"BRISC {plane}", color=col)
    ax.hist(isym, bins=40, alpha=0.9, density=True, histtype="step", lw=2.2,
            color="black", label="Internal train (plane unknown)")
    ax.axvline(t_best, color="grey", ls="--", lw=1, label=f"sagittal cutoff {t_best:.2f}")
    ax.set_xlabel("Left-right mirror symmetry"); ax.set_ylabel("Density")
    ax.set_title("Plane cue: internal training data vs BRISC by known plane")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "symmetry_internal_vs_brisc.png"); plt.close(fig)

    path = BRISC_OUT / "domain_shift.json"
    path.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res["internal_plane_estimate"], indent=2))
    print(f"-> {path}\n-> contact sheets in {out_dir}")


# ===========================================================================
# Subcommand: overlap  -  is BRISC actually external?
# ===========================================================================

def cmd_overlap(args: argparse.Namespace) -> None:
    """Look for images shared between BRISC and the internal training data.

    This check exists because the external result came out BETTER than the
    internal one, which is the classic signature of a leak. The internal data is
    the Kaggle Nickparvar merge of Br35H, SARTAJ and Figshare. BRISC 2025 is a
    curated collection. If BRISC re-used images from those same public sources,
    then part of "external validation" is the model being tested on its own
    training set, and every number in this session is inflated.

    Method: 64-bit perceptual hash (pHash) of every image on both sides, then
    exact-hash matching plus a Hamming-distance search. Hamming distance 0 means
    visually identical. Distance <= 5 is the same near-duplicate threshold
    `src/code.py` already uses for its leakage-safe splitting.
    """
    import imagehash
    from PIL import Image

    brisc_df = build_brisc_manifest(args.brisc)
    internal = pd.read_csv(REPO_ROOT / "data" / "split_manifest.csv")
    internal["filepath_abs"] = internal["filepath"].map(
        lambda p: str(REPO_ROOT / Path(str(p).replace("\\", "/"))))

    out_dir = BRISC_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_npz = out_dir / "phash_bits.npz"

    def hashes(paths: List[str], label: str) -> np.ndarray:
        bits = np.zeros((len(paths), 64), dtype=np.uint8)
        for i, p in enumerate(paths):
            if i % 2000 == 0:
                print(f"  {label}: {i}/{len(paths)}", flush=True)
            bits[i] = imagehash.phash(Image.open(p).convert("L")).hash.flatten().astype(np.uint8)
        return bits

    if cache_npz.exists() and not args.rehash:
        z = np.load(cache_npz)
        b_bits, i_bits = z["brisc"], z["internal"]
        print(f"  reusing cached pHashes from {cache_npz.name}")
    else:
        b_bits = hashes(brisc_df["filepath"].tolist(), "brisc")
        i_bits = hashes(internal["filepath_abs"].tolist(), "internal")
        np.savez_compressed(cache_npz, brisc=b_bits, internal=i_bits)

    # Hamming distance via matrix product: d = popcount(a XOR b).
    # (a != b).sum() == a@(1-b).T + (1-a)@b.T
    # float32 so the matmul goes through BLAS; values are small integers so the
    # representation is exact.
    a = b_bits.astype(np.float32)
    b = i_bits.astype(np.float32)
    dist = (a @ (1 - b).T + (1 - a) @ b.T).astype(np.int16)   # (6000, 7200)

    # Nearest internal image overall, and nearest within each internal split.
    # The split that matters is TRAIN: an image the model was fitted on is not
    # external test data, whatever dataset it was later republished in.
    per = pd.DataFrame({"image_path": brisc_df["image_path"].to_numpy()})
    split_arr = internal["split"].to_numpy()
    for sp in ("train", "val", "test"):
        cols = np.nonzero(split_arr == sp)[0]
        sub = dist[:, cols]
        per[f"d_{sp}"] = sub.min(axis=1)
        idx_local = sub.argmin(axis=1)
        per[f"match_{sp}"] = internal["filepath"].to_numpy()[cols[idx_local]]
    per["d_any"] = dist.min(axis=1)
    per["class_name"] = brisc_df["class_name"].to_numpy()
    per["brisc_split"] = brisc_df["brisc_split"].to_numpy()
    per["plane"] = brisc_df["plane"].to_numpy()
    thr = args.threshold
    per["clean_vs_train"] = per["d_train"] > thr
    per["clean_vs_any"] = per["d_any"] > thr
    per.to_csv(out_dir / "brisc_overlap_per_image.csv", index=False)

    rows: List[Dict[str, Any]] = []
    for t in (0, 1, 2, 3, 5, 8, 10):
        rows.append({
            "hamming_threshold": t,
            "n_matching_any_internal": int((per["d_any"] <= t).sum()),
            "share_matching_any_internal": float((per["d_any"] <= t).mean()),
            "n_matching_internal_TRAIN": int((per["d_train"] <= t).sum()),
            "share_matching_internal_TRAIN": float((per["d_train"] <= t).mean()),
        })
        print(f"  hamming <= {t:2d}:  any-internal {rows[-1]['n_matching_any_internal']:5d} "
              f"({100*rows[-1]['share_matching_any_internal']:5.2f}%)   "
              f"internal-TRAIN {rows[-1]['n_matching_internal_TRAIN']:5d} "
              f"({100*rows[-1]['share_matching_internal_TRAIN']:5.2f}%)")

    clean = per[per["clean_vs_train"]]
    res = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": (
            "BRISC 2025 is NOT external data for these checkpoints. "
            f"{int((per['d_train'] <= thr).sum())} of {len(per)} BRISC images "
            f"({100*(per['d_train'] <= thr).mean():.1f}%) are near-duplicates of "
            "images the model was trained on."
        ),
        "method": "64-bit pHash Hamming distance from every BRISC image to every "
                  "internal image, split by internal split. Distance 0 means the "
                  "pHash is identical; a 40-pair pixel check on distance-0 pairs "
                  "found mean absolute difference 0.0 and correlation 1.0, i.e. "
                  "the same image file content. Distance <= 5 is the near-duplicate "
                  "threshold src/code.py already uses for leakage-safe splitting.",
        "n_brisc": int(len(brisc_df)),
        "n_internal": int(len(internal)),
        "internal_split_sizes": internal["split"].value_counts().to_dict(),
        "nearest_distance_percentiles_any": {
            str(q): float(np.percentile(per["d_any"], q)) for q in (0, 1, 5, 25, 50, 75, 95, 100)},
        "nearest_distance_percentiles_train": {
            str(q): float(np.percentile(per["d_train"], q)) for q in (0, 1, 5, 25, 50, 75, 95, 100)},
        "by_threshold": rows,
        "reported_threshold": thr,
        "clean_subset_vs_train": {
            "definition": f"BRISC images whose nearest internal TRAIN image is more "
                          f"than Hamming {thr} away. These are the only BRISC images "
                          f"that constitute genuine external test data.",
            "n": int(len(clean)),
            "share_of_brisc": float(len(clean) / len(per)),
            "by_class": clean["class_name"].value_counts().to_dict(),
            "by_plane": clean["plane"].value_counts().to_dict(),
            "by_brisc_split": clean["brisc_split"].value_counts().to_dict(),
        },
        "clean_subset_vs_any": {
            "n": int(per["clean_vs_any"].sum()),
            "by_class": per.loc[per["clean_vs_any"], "class_name"].value_counts().to_dict(),
        },
    }
    (out_dir / "brisc_internal_overlap.json").write_text(
        json.dumps(res, indent=2, default=str), encoding="utf-8")
    print("\n" + res["verdict"])
    print(f"\nclean subset (unseen in training): n={len(clean)} "
          f"({100*len(clean)/len(per):.1f}%)  {clean['class_name'].value_counts().to_dict()}")
    print(f"-> {out_dir / 'brisc_internal_overlap.json'}")
    print(f"-> {out_dir / 'brisc_overlap_per_image.csv'}")


# ===========================================================================
# Subcommand: misses  (Phase 5)
# ===========================================================================

def cmd_misses(args: argparse.Namespace) -> None:
    """Extract true tumours the model called `notumor` with low entropy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import safety_metrics as sm

    out_dir = BRISC_OUT / "confident_misses"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name, seeds = args.model, (args.seeds or SEEDS)
    frames = [load_brisc_cache(model_name, s) for s in seeds]
    brisc = frames[0] if len(frames) == 1 else ensemble_frames(frames)

    # "Confident" means the tool would NOT have deferred it. The cutoff is the
    # internal-val entropy quantile for a 20 percent deferral budget, fitted on
    # internal val, applied to BRISC unchanged.
    vframes = [load_internal_cache(model_name, s, "val") for s in seeds]
    val = vframes[0] if len(vframes) == 1 else ensemble_frames(vframes)
    cut = float(np.quantile(val["entropy"].to_numpy(), 1.0 - args.defer_budget))

    y = brisc["true_label"].to_numpy()
    p = brisc["pred_label"].to_numpy()
    missed_all = brisc.loc[(y != NOTUMOR_IDX) & (p == NOTUMOR_IDX)].copy()
    conf = sm.confident_misses(brisc, cut).sort_values("entropy")

    brisc_man = build_brisc_manifest(args.brisc).set_index("image_path")
    for d in (missed_all, conf):
        d["plane"] = d["image_path"].map(brisc_man["plane"])
        d["abs_path"] = d["image_path"].map(brisc_man["filepath"])

    summary: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name, "seeds": list(seeds),
        "defer_budget_used_to_define_confident": args.defer_budget,
        "entropy_cutoff_nats_fitted_on_internal_val": cut,
        "n_tumors_in_brisc": int((y != NOTUMOR_IDX).sum()),
        "n_missed_tumors": int(len(missed_all)),
        "n_confidently_missed": int(len(conf)),
        "confident_share_of_misses": (float(len(conf) / len(missed_all)) if len(missed_all) else 0.0),
        "missed_by_class": missed_all["true_label_name"].value_counts().to_dict(),
        "confident_missed_by_class": conf["true_label_name"].value_counts().to_dict(),
        "missed_by_plane": missed_all["plane"].value_counts().to_dict(),
        "confident_missed_by_plane": conf["plane"].value_counts().to_dict(),
        "confident_missed_by_brisc_split": conf["brisc_split"].value_counts().to_dict(),
        "p_tumor_summary_confident": {k: float(v) for k, v in conf["p_tumor"].describe().to_dict().items()} if len(conf) else {},
        "entropy_summary_confident": {k: float(v) for k, v in conf["entropy"].describe().to_dict().items()} if len(conf) else {},
    }

    cols = ["image_path", "true_label_name", "brisc_split", "plane",
            "p_glioma", "p_meningioma", "p_pituitary", "p_notumor",
            "p_tumor", "entropy", "entropy_bits"]
    missed_all[cols].to_csv(out_dir / "all_missed_tumors.csv", index=False)
    conf[cols].to_csv(out_dir / "confident_missed_tumors.csv", index=False)

    # Contact sheets, every confident miss, labelled.
    per_sheet = 24
    n_sheets = 0
    for start in range(0, len(conf), per_sheet):
        chunk = conf.iloc[start:start + per_sheet]
        rows = int(np.ceil(len(chunk) / 6))
        fig, axes = plt.subplots(rows, 6, figsize=(16, 3.1 * rows), squeeze=False)
        for ax, (_, r) in zip(axes.ravel(), chunk.iterrows()):
            ax.imshow(Image.open(r["abs_path"]).convert("L"), cmap="gray")
            ax.set_title(
                f"true {r['true_label_name']} -> pred notumor\n"
                f"{r['plane']} | p_tumor {r['p_tumor']:.3f} | H {r['entropy']:.3f}",
                fontsize=7, color="crimson")
            ax.axis("off")
        for ax in axes.ravel()[len(chunk):]:
            ax.axis("off")
        fig.suptitle(
            f"BRISC confidently missed tumours [{model_name}, {len(seeds)} seed(s)] "
            f"sheet {n_sheets+1}. Entropy below {cut:.3f} nats, so the tool would "
            f"NOT have deferred these.", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"contact_sheet_{n_sheets+1:02d}.png", dpi=110)
        plt.close(fig)
        n_sheets += 1
    summary["n_contact_sheets"] = n_sheets

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\n-> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("metrics", help="Phase 2: external validation metrics and figures")
    m.add_argument("--boot", type=int, default=1000)
    m.set_defaults(func=cmd_metrics)

    d = sub.add_parser("domain", help="Phase 3: characterise the domain shift")
    d.add_argument("--brisc", type=Path, default=BRISC)
    d.add_argument("--sample", type=int, default=400)
    d.set_defaults(func=cmd_domain)

    o = sub.add_parser("overlap", help="check whether BRISC shares images with the training data")
    o.add_argument("--brisc", type=Path, default=BRISC)
    o.add_argument("--threshold", type=int, default=5)
    o.add_argument("--rehash", action="store_true", help="ignore the cached pHash matrix")
    o.set_defaults(func=cmd_overlap)

    x = sub.add_parser("misses", help="Phase 5: extract confidently wrong missed tumours")
    x.add_argument("--brisc", type=Path, default=BRISC)
    x.add_argument("--model", default="resnet50", choices=MODELS)
    x.add_argument("--seeds", nargs="+", type=int, default=None)
    x.add_argument("--defer-budget", type=float, default=0.20)
    x.set_defaults(func=cmd_misses)

    c = sub.add_parser("cache", help="build the BRISC and internal prediction caches")
    c.add_argument("--brisc", type=Path, default=BRISC)
    c.add_argument("--models", nargs="+", default=None, choices=MODELS)
    c.add_argument("--seeds", nargs="+", type=int, default=None)
    c.add_argument("--batch-size", type=int, default=32)
    c.add_argument("--workers", type=int, default=0)
    c.add_argument("--sha-sample", type=int, default=50)
    c.add_argument("--sanity", action="store_true", help="200-image dry run, one checkpoint")
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=cmd_cache)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
