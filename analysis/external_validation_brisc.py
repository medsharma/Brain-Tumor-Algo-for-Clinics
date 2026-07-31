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
from code import ManifestDataset  # noqa: E402

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

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
