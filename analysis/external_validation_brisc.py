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

    report: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "BRISC2025",
        "n": 6000,
        "mc_T": MC_T,
        "n_bootstrap": nb,
        "binary_threshold": 0.5,
        "entropy_units": "nats",
        "note": "Nothing here was fitted on BRISC. These are scores of a frozen model.",
        "models": {},
    }

    per_seed_preds: Dict[str, np.ndarray] = {}

    for model_name in MODELS:
        entry: Dict[str, Any] = {"per_seed": {}, "ensemble5": {}, "internal_test": {}}
        brisc_frames, itest_frames, val_frames = [], [], []

        for seed in SEEDS:
            b = load_brisc_cache(model_name, seed)
            t = load_internal_cache(model_name, seed, "test")
            v = load_internal_cache(model_name, seed, "val")
            brisc_frames.append(b); itest_frames.append(t); val_frames.append(v)
            entry["per_seed"][str(seed)] = {
                "brisc": sm.full_report(b, 0.5, seed=seed, n_resamples=nb),
                "internal_test": sm.full_report(t, 0.5, seed=seed, n_resamples=nb),
            }
            print(f"  {model_name} seed {seed}: BRISC miss="
                  f"{entry['per_seed'][str(seed)]['brisc']['tumor_miss_rate']['value']:.4f} "
                  f"acc={entry['per_seed'][str(seed)]['brisc']['standard']['four_way_accuracy']['value']:.4f}")

        be = ensemble_frames(brisc_frames)
        te = ensemble_frames(itest_frames)
        ve = ensemble_frames(val_frames)
        per_seed_preds[model_name] = be["pred_label"].to_numpy()

        entry["ensemble5"]["brisc"] = sm.full_report(
            be, 0.5, n_resamples=nb, subgroups=("plane", "brisc_split"))
        entry["ensemble5"]["internal_test"] = sm.full_report(te, 0.5, n_resamples=nb)
        entry["ensemble5"]["internal_val"] = sm.full_report(ve, 0.5, n_resamples=nb)

        # The drop is the finding. Internal test vs BRISC, same checkpoints.
        entry["drop_internal_to_brisc"] = {
            "four_way_accuracy": sm.drop_with_ci(
                entry["ensemble5"]["internal_test"], entry["ensemble5"]["brisc"],
                ("standard", "four_way_accuracy")),
            "macro_f1": sm.drop_with_ci(
                entry["ensemble5"]["internal_test"], entry["ensemble5"]["brisc"],
                ("standard", "macro_f1")),
            "macro_auc": sm.drop_with_ci(
                entry["ensemble5"]["internal_test"], entry["ensemble5"]["brisc"],
                ("standard", "macro_auc")),
            "tumor_miss_rate": sm.drop_with_ci(
                entry["ensemble5"]["internal_test"], entry["ensemble5"]["brisc"],
                ("tumor_miss_rate",)),
            "binary_sensitivity": sm.drop_with_ci(
                entry["ensemble5"]["internal_test"], entry["ensemble5"]["brisc"],
                ("binary", "sensitivity")),
            "ece_15bin": {
                "internal": entry["ensemble5"]["internal_test"]["standard"]["ece_15bin"],
                "external": entry["ensemble5"]["brisc"]["standard"]["ece_15bin"],
                "drop": (entry["ensemble5"]["internal_test"]["standard"]["ece_15bin"]
                         - entry["ensemble5"]["brisc"]["standard"]["ece_15bin"]),
            },
        }
        report["models"][model_name] = entry
        _plots(be, te, model_name, fig_dir)

    # ViT vs ResNet-50 on BRISC, paired on the same images.
    y_true = load_brisc_cache("vit", 42)["true_label"].to_numpy()
    chi2, pval = mcnemar_test(y_true, per_seed_preds["vit"], per_seed_preds["resnet50"])
    report["vit_vs_resnet50_brisc_mcnemar"] = {
        "chi2": chi2, "p_value": pval, "significant": bool(pval < 0.05),
        "note": "5-seed ensembles, paired on the same 6000 BRISC images",
    }
    print(f"\n  McNemar ViT vs ResNet-50 on BRISC: chi2={chi2:.3f} p={pval:.4g}")

    out_path = BRISC_OUT / "brisc_metrics.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\n-> {out_path}")


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

    def hashes(paths: List[str], label: str) -> np.ndarray:
        bits = np.zeros((len(paths), 64), dtype=np.uint8)
        for i, p in enumerate(paths):
            if i % 1000 == 0:
                print(f"  {label}: {i}/{len(paths)}", flush=True)
            h = imagehash.phash(Image.open(p).convert("L"))
            bits[i] = h.hash.flatten().astype(np.uint8)
        return bits

    b_bits = hashes(brisc_df["filepath"].tolist(), "brisc")
    i_bits = hashes(internal["filepath_abs"].tolist(), "internal")

    # Hamming distance via matrix product: d = popcount(a XOR b).
    # (a != b).sum() == a@(1-b).T + (1-a)@b.T
    # float32 so the matmul goes through BLAS; values are small integers so the
    # representation is exact.
    a = b_bits.astype(np.float32)
    b = i_bits.astype(np.float32)
    dist = (a @ (1 - b).T + (1 - a) @ b.T).astype(np.int16)   # (6000, 7200)
    nearest = dist.min(axis=1)
    nearest_idx = dist.argmin(axis=1)

    rows: List[Dict[str, Any]] = []
    for thr in (0, 1, 2, 3, 5, 8, 10):
        n_hit = int((nearest <= thr).sum())
        rows.append({"hamming_threshold": thr, "n_brisc_images_matched": n_hit,
                     "share_of_brisc": n_hit / len(brisc_df)})
        print(f"  hamming <= {thr:2d}: {n_hit:5d} of {len(brisc_df)} BRISC images "
              f"({100*n_hit/len(brisc_df):.2f}%)")

    matched = brisc_df.loc[nearest <= args.threshold].copy()
    matched["hamming"] = nearest[nearest <= args.threshold]
    matched["internal_match"] = internal["filepath"].to_numpy()[nearest_idx[nearest <= args.threshold]]
    matched["internal_split"] = internal["split"].to_numpy()[nearest_idx[nearest <= args.threshold]]
    matched["internal_class"] = internal["class_name"].to_numpy()[nearest_idx[nearest <= args.threshold]]

    out_dir = BRISC_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    matched[["image_path", "class_name", "brisc_split", "plane", "hamming",
             "internal_match", "internal_split", "internal_class"]].to_csv(
        out_dir / "brisc_internal_overlap.csv", index=False)

    res = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "64-bit pHash, minimum Hamming distance from each BRISC image "
                  "to any internal image. Distance 0 = visually identical. "
                  "Distance <= 5 is the near-duplicate threshold src/code.py "
                  "already uses for leakage-safe splitting.",
        "n_brisc": int(len(brisc_df)),
        "n_internal": int(len(internal)),
        "nearest_distance_percentiles": {
            str(q): float(np.percentile(nearest, q)) for q in (0, 1, 5, 25, 50, 75, 100)
        },
        "by_threshold": rows,
        "reported_threshold": args.threshold,
        "n_matched_at_reported_threshold": int(len(matched)),
        "matched_by_internal_split": matched["internal_split"].value_counts().to_dict(),
        "matched_by_brisc_split": matched["brisc_split"].value_counts().to_dict(),
    }
    (out_dir / "brisc_internal_overlap.json").write_text(
        json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res["nearest_distance_percentiles"], indent=2))
    print(f"-> {out_dir / 'brisc_internal_overlap.json'}")


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
