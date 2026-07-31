#!/usr/bin/env python3
"""
Brain Tumor MRI Classification — ViT-B/16 + ResNet-50 with MC Dropout UQ
==========================================================================
4-class task: Glioma · Meningioma · Pituitary · No-Tumor

Architecture highlights:
  - Backbones : ViT-B/16 (blocks 10–11 + head trainable)
                ResNet-50 (layer4 + head trainable; BN in frozen stages kept in eval)
  - UQ method : Monte Carlo Dropout (T=20 stochastic forward passes)
  - Splitting  : Leakage-safe 70/15/15 — patient-ID grouping → phash clustering → per-file
  - Metrics   : Accuracy, macro-F1, macro-AUC, ECE (15-bin), Brier, AURC
  - Stats     : Bootstrap 95% CI, per-seed McNemar's test, OOD-AUROC
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import csv
import json
import logging
import os
import re
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import chi2 as scipy_chi2
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# Fix 8 — np.trapz was removed in NumPy 2.0; np.trapezoid is the replacement.
# getattr fallback keeps us compatible with both NumPy 1.x and 2.x.
_trapz = getattr(np, "trapezoid", np.trapz)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# Section 1 — Setup & Reproducibility
# =============================================================================

def setup_reproducibility(seed: int = 42) -> None:
    """Fix all random seeds and disable non-deterministic CUDA kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    log.info("Reproducibility locked — seed=%d, cudnn.deterministic=True", seed)


DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBED_DIM:   int       = 768
NUM_CLASSES: int       = 4
CLASS_NAMES: List[str] = ["glioma", "meningioma", "pituitary", "notumor"]
SEEDS:       List[int] = [42, 123, 7, 2024, 31]

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# =============================================================================
# Section 2 — Leakage-Safe Manifest-Based Data Splitting
# =============================================================================

# Matches filename stems that begin with a recognisable patient/scan ID:
#   P001, pat_042, patient-7, 1234_slice3, etc.
_PATIENT_RE = re.compile(r"^([Pp](?:at(?:ient)?)?[\-_]?\d+|\d{3,})", re.IGNORECASE)


def _parse_patient_id(stem: str) -> Optional[str]:
    m = _PATIENT_RE.match(stem)
    return m.group(1).lower() if m else None


def _per_file_stratified_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> pd.Series:
    labels = df["label"].values
    idx    = np.arange(len(df))

    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=(1.0 - train_frac), random_state=seed
    )
    train_idx, rest_idx = next(sss1.split(idx, labels))

    test_frac_of_rest = (1.0 - train_frac - val_frac) / (1.0 - train_frac)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_frac_of_rest, random_state=seed
    )
    val_local, test_local = next(sss2.split(rest_idx, labels[rest_idx]))

    split_col = pd.Series([""] * len(df), index=df.index)
    split_col.iloc[train_idx]            = "train"
    split_col.iloc[rest_idx[val_local]]  = "val"
    split_col.iloc[rest_idx[test_local]] = "test"
    return split_col


def _patient_group_split(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> pd.Series:
    """All slices from one patient go to exactly one split (no leakage)."""
    rng       = np.random.default_rng(seed)
    split_col = pd.Series([""] * len(df), index=df.index)

    for label in df["label"].unique():
        mask     = df["label"] == label
        patients = df.loc[mask, "patient_id"].unique()
        rng.shuffle(patients)
        n       = len(patients)
        n_train = max(1, int(round(n * train_frac)))
        n_val   = max(1, int(round(n * val_frac)))
        n_train = min(n_train, n - 2)
        n_val   = min(n_val,   n - n_train - 1)

        train_pts = set(patients[:n_train])
        val_pts   = set(patients[n_train : n_train + n_val])

        for row_idx in df[mask].index:
            pid = df.at[row_idx, "patient_id"]
            if pid in train_pts:
                split_col[row_idx] = "train"
            elif pid in val_pts:
                split_col[row_idx] = "val"
            else:
                split_col[row_idx] = "test"

    return split_col


# Fix 2 — phash-based near-duplicate grouping
def _phash_cluster_ids(df: pd.DataFrame, hash_threshold: int = 5) -> pd.Series:
    """Cluster images by perceptual hash similarity; return cluster IDs per row.

    Uses Union-Find with path compression over all within-class image pairs whose
    pHash Hamming distance is ≤ ``hash_threshold`` (default 5 of 64 bits). Clusters
    are then used exactly as patient IDs in ``_patient_group_split`` to prevent
    near-duplicate slices from spanning train/val/test.

    Args:
        df:             DataFrame with columns ``filepath`` and ``label``.
        hash_threshold: Maximum Hamming distance to merge two images into one cluster.

    Returns:
        Series of string cluster IDs aligned with ``df.index``.

    Raises:
        ImportError: If ``imagehash`` is not installed.
    """
    try:
        import imagehash
    except ImportError:
        raise ImportError(
            "imagehash is required for phash-based deduplication. "
            "Install it with:  pip install imagehash"
        )

    log.info("Computing perceptual hashes for %d images (threshold=%d) …", len(df), hash_threshold)
    hashes: Dict[Any, Any] = {}
    for idx, row in df.iterrows():
        try:
            hashes[idx] = imagehash.phash(Image.open(row["filepath"]).convert("RGB"))
        except Exception as exc:
            log.warning("phash: skipping %s — %s", Path(row["filepath"]).name, exc)
            hashes[idx] = None

    # Union-Find with path compression
    indices = list(df.index)
    parent  = {i: i for i in indices}

    def find(x: Any) -> Any:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:          # path compression
            parent[x], x = root, parent[x]
        return root

    def union(x: Any, y: Any) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Only compare pairs within the same class — across-class near-duplicates
    # would indicate dataset corruption, not patient grouping.
    for label in df["label"].unique():
        group = df[df["label"] == label].index.tolist()
        for i, ii in enumerate(group):
            if hashes.get(ii) is None:
                continue
            for jj in group[i + 1:]:
                if hashes.get(jj) is None:
                    continue
                try:
                    if hashes[ii] - hashes[jj] <= hash_threshold:
                        union(ii, jj)
                except Exception:
                    pass

    cluster_col = pd.Series({i: str(find(i)) for i in indices}, name="patient_id")

    sizes      = cluster_col.value_counts()
    n_multi    = int((sizes > 1).sum())
    log.info(
        "phash clustering: %d clusters from %d images  "
        "(%d clusters contain >1 near-duplicate image)",
        len(sizes), len(df), n_multi,
    )
    log.info("Cluster-size distribution:\n%s", sizes.describe().to_string())
    return cluster_col


def _log_split_stats(df: pd.DataFrame) -> None:
    for split in ("train", "val", "test"):
        sub    = df[df["split"] == split]
        counts = sub.groupby("class_name").size().to_dict()
        log.info("[%s]  total=%d  %s", split, len(sub), counts)


def build_split_manifest(
    raw_root: str,
    manifest_path: str = "data/split_manifest.csv",
    train_frac: float  = 0.70,
    val_frac: float    = 0.15,
    seed: int          = 42,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Build (or load) a stratified 70/15/15 split manifest CSV.

    Split path (descending priority):
      1. Patient-ID grouping — if a patient/scan ID can be parsed from *every*
         filename stem. ID cluster sizes are logged so you can verify whether
         parsing actually produced multi-file groups.
      2. phash-cluster grouping — if ``imagehash`` is installed and IDs couldn't
         be parsed. Near-duplicate images are clustered and the cluster is kept
         within one split.
      3. Per-file stratified split — fallback with a loud leakage warning.

    Args:
        raw_root:      Directory with one sub-folder per class name.
        manifest_path: Output CSV path.
        train_frac:    Training fraction (default 0.70).
        val_frac:      Validation fraction (default 0.15); test = remainder.
        seed:          RNG seed.
        force_rebuild: Rebuild even if the CSV already exists.

    Returns:
        DataFrame: filepath, class_name, label, patient_id, split.
    """
    manifest_path = Path(manifest_path)
    if manifest_path.exists() and not force_rebuild:
        log.info("Loading existing manifest: %s", manifest_path)
        return pd.read_csv(manifest_path)

    raw_root = Path(raw_root)
    rows: List[Dict[str, Any]] = []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = raw_root / class_name
        if not class_dir.exists():
            log.warning("Class directory not found — skipping: %s", class_dir)
            continue
        for fp in sorted(class_dir.iterdir()):
            if fp.suffix.lower() not in _IMG_EXTS:
                continue
            rows.append({
                "filepath":   str(fp),
                "class_name": class_name,
                "label":      label_idx,
                "patient_id": _parse_patient_id(fp.stem),
            })

    if not rows:
        raise FileNotFoundError(f"No images found under {raw_root}")

    df        = pd.DataFrame(rows)
    n_missing = int(df["patient_id"].isna().sum())

    # Fix 1 — always log cluster-size distribution so caller can verify
    # whether the regex actually produced multi-file groupings or just
    # one-file "clusters" (which would give no leakage protection).
    if df["patient_id"].notna().any():
        id_sizes = df.groupby("patient_id").size()
        log.info(
            "Patient-ID cluster-size distribution (parsed IDs only):\n%s",
            id_sizes.describe().to_string(),
        )
        n_multi = int((id_sizes > 1).sum())
        if n_multi == 0:
            log.warning(
                "All %d parsed patient IDs are singletons — the regex matched "
                "but each ID covers exactly one file, so group-splitting offers "
                "no protection against leakage. Check filename format.",
                len(id_sizes),
            )
        else:
            log.info(
                "%d / %d IDs cover >1 file — multi-file patient grouping confirmed.",
                n_multi, len(id_sizes),
            )

    if n_missing == 0:
        log.info(
            "Patient IDs parsed from all %d filenames — using group-stratified split.",
            len(df),
        )
        df["split"] = _patient_group_split(df, train_frac, val_frac, seed)

    else:
        # Fix 2 — try phash clustering before falling back to per-file split
        log.warning(
            "WARNING: Patient IDs could not be parsed for %d / %d files.",
            n_missing, len(df),
        )
        phash_ok = False
        try:
            cluster_ids      = _phash_cluster_ids(df)
            df_phash         = df.copy()
            df_phash["patient_id"] = cluster_ids
            df["split"]      = _patient_group_split(df_phash, train_frac, val_frac, seed)
            # Carry the cluster IDs back onto the frame that actually gets
            # written. Without this the saved CSV's patient_id column is 100%
            # null even though clustering genuinely ran, so the manifest alone
            # cannot prove which split path was taken and the leakage claim is
            # unverifiable from the repo. Bug documented in
            # results/leakage_audit.md; does not affect split assignment.
            df["patient_id"] = df_phash["patient_id"]
            log.info("Using phash-cluster-stratified split.")
            phash_ok = True
        except ImportError:
            log.warning(
                "imagehash not installed (pip install imagehash) — "
                "falling back to per-file stratified split. "
                "Patient-level leakage cannot be ruled out."
            )
        except Exception as exc:
            log.warning(
                "phash clustering failed (%s) — "
                "falling back to per-file stratified split.",
                exc,
            )

        if not phash_ok:
            df["split"] = _per_file_stratified_split(df, train_frac, val_frac, seed)

    df.to_csv(manifest_path, index=False)
    log.info("Manifest written → %s", manifest_path)
    _log_split_stats(df)
    return df


# =============================================================================
# Section 3 — Transforms, Dataset & DataLoaders
# =============================================================================

_IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)


def get_transforms(split: str) -> transforms.Compose:
    """Augmentation pipeline for a given split.

    Choices for brain MRI:
      - RandomHorizontalFlip retained: left/right axial symmetry is valid.
      - RandomVerticalFlip removed: no meaningful vertical symmetry in MRI.
      - ColorJitter(saturation) removed: RGB-converted MRI carries no colour.
    """
    if split == "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            # No RandomVerticalFlip — brain MRI has no vertical symmetry
            transforms.RandomRotation(degrees=15),
            # No saturation — MRI-to-RGB conversion carries no spectral information
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


class ManifestDataset(Dataset):
    """PyTorch Dataset backed by a split-manifest DataFrame."""

    def __init__(
        self,
        df: pd.DataFrame,
        split: str,
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.df = df[df["split"] == split].reset_index(drop=True)
        if max_samples is not None and len(self.df) > 0:
            per_class = max(1, max_samples // NUM_CLASSES)
            self.df = (
                self.df
                .groupby("label", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), per_class), random_state=0))
                .reset_index(drop=True)
            )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["label"])


def build_dataloaders_from_manifest(
    manifest_df: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples_per_split: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """Construct train/val/test DataLoaders from a split-manifest DataFrame."""
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = ManifestDataset(
            manifest_df, split,
            transform=get_transforms(split),
            max_samples=max_samples_per_split,
        )
        if len(dataset) == 0:
            log.warning("Split '%s' has 0 samples — skipping.", split)
            continue
        effective_bs = min(batch_size, len(dataset))
        loaders[split] = DataLoader(
            dataset,
            batch_size=effective_bs,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=(DEVICE.type == "cuda"),
            drop_last=(split == "train" and len(dataset) >= batch_size),
            persistent_workers=(num_workers > 0),
        )
        log.info(
            "[%s]  %d samples  /  %d batches",
            split, len(dataset), len(loaders[split]),
        )
    return loaders


# =============================================================================
# Section 4 — Architecture
# =============================================================================

class BrainTumorViT(nn.Module):
    """ViT-B/16 for 4-class brain-tumour classification with MC Dropout UQ.

    Selective fine-tuning:
      Frozen    — patch embedding, encoder blocks 0–9, positional embedding.
      Trainable — encoder blocks 10–11, encoder.ln, custom head.

    Head: LayerNorm(768) → Dropout(p) → Linear(768→256) → GELU
          → Dropout(p) → Linear(256→4)

    The two Dropout layers are the stochasticity source for MC Dropout UQ.
    """

    def __init__(self, num_classes: int = 4, dropout_p: float = 0.3) -> None:
        super().__init__()
        backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

        for param in backbone.parameters():
            param.requires_grad = False

        for block_idx in (10, 11):
            for param in backbone.encoder.layers[block_idx].parameters():
                param.requires_grad = True

        for param in backbone.encoder.ln.parameters():
            param.requires_grad = True

        backbone.heads = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Dropout(p=dropout_p),
            nn.Linear(EMBED_DIM, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes),
        )
        self.backbone = backbone

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        log.info(
            "BrainTumorViT | trainable: %s / %s  (%.2f%%)",
            f"{n_train:,}", f"{n_total:,}", 100.0 * n_train / n_total,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Class logits for a batch, shape (B, num_classes)."""
        return self.backbone(x)

    @staticmethod
    def _activate_dropout(model: nn.Module) -> None:
        """Re-enable stochastic masking in every nn.Dropout layer after eval()."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        T: int = 20,
        epsilon: float = 1e-10,
    ) -> Dict[str, torch.Tensor]:
        """Monte Carlo Dropout inference — predictive mean and entropy.

        Performs *T* stochastic forward passes with Dropout active, then
        aggregates the resulting posterior samples to compute:

        **Predictive mean** (per class c):

        .. math::
            \\mu_c = \\frac{1}{T} \\sum_{t=1}^{T} p_{t,c}

        **Predictive entropy** (total uncertainty, in bits):

        .. math::
            H = -\\sum_{c=1}^{C} \\mu_c \\log_2(\\mu_c + \\varepsilon)

        High *H* indicates the model is uncertain about its prediction —
        useful for flagging ambiguous or out-of-distribution scans.

        Args:
            x:       Input batch, shape ``(B, 3, 224, 224)``, on the correct device.
            T:       Number of stochastic MC Dropout forward passes.
            epsilon: Numerical stability constant inside :math:`\\log_2`.

        Returns:
            Dictionary with the following keys:

            * ``"predictions"``  — ``(B,)``    hard class predictions (argmax of μ).
            * ``"mean_probs"``   — ``(B, C)``  predictive mean probabilities μ.
            * ``"entropy"``      — ``(B,)``    predictive entropy H in bits.
            * ``"std_probs"``    — ``(B, C)``  standard deviation across T passes.
            * ``"all_probs"``    — ``(T, B, C)`` full posterior sample matrix.

        Note:
            This method leaves the model in ``eval()`` mode with Dropout layers
            in ``train()`` mode.  Call ``model.eval()`` afterwards to restore
            fully deterministic behaviour, or ``model.train()`` before the next
            training step.
        """
        # eval() stabilises LayerNorm / encoder normalisation statistics;
        # _activate_dropout() then re-enables stochastic masking.
        self.eval()
        self._activate_dropout(self)

        sample_probs: List[torch.Tensor] = []
        for _ in range(T):
            logits: torch.Tensor = self.backbone(x)              # (B, C)
            probs:  torch.Tensor = torch.softmax(logits, dim=-1) # (B, C)
            sample_probs.append(probs)

        # Posterior sample matrix — shape (T, B, C)
        stacked: torch.Tensor = torch.stack(sample_probs, dim=0)

        # μ_c — predictive mean over T passes: (B, C)
        mean_probs: torch.Tensor = stacked.mean(dim=0)

        # H — predictive entropy in bits: (B,)
        entropy: torch.Tensor = -(
            mean_probs * torch.log2(mean_probs + epsilon)
        ).sum(dim=-1)

        # σ_c — spread of posterior samples per class: (B, C)
        std_probs: torch.Tensor = stacked.std(dim=0)

        return {
            "predictions": mean_probs.argmax(dim=-1),   # (B,)
            "mean_probs":  mean_probs,                   # (B, C)
            "entropy":     entropy,                      # (B,)
            "std_probs":   std_probs,                    # (B, C)
            "all_probs":   stacked,                      # (T, B, C)
        }


class BrainTumorResNet50(nn.Module):
    """ResNet-50 baseline with selective freezing for fair ViT comparison.

    Selective fine-tuning (analogous to ViT's block 10-11 strategy):
      Frozen    — conv1, bn1, layer1, layer2, layer3.
      Trainable — layer4 (last residual stage) and new classification head.

    Head mirrors BrainTumorViT:
      LayerNorm(2048) → Dropout(p) → Linear(2048→256) → GELU
      → Dropout(p) → Linear(256→4)

    BatchNorm note (Fix 3): when the model is put into train() mode, the
    frozen stages (conv1, bn1, layer1–3) are immediately forced back to eval()
    so their running statistics don't drift. layer4 and the head stay in their
    standard train/eval mode.
    """

    # Frozen backbone sub-modules whose BN stats must not drift during training
    _FROZEN_STAGE_NAMES = ("conv1", "bn1", "layer1", "layer2", "layer3")

    def __init__(self, num_classes: int = 4, dropout_p: float = 0.3) -> None:
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        for param in backbone.parameters():
            param.requires_grad = False

        for param in backbone.layer4.parameters():
            param.requires_grad = True

        in_features = backbone.fc.in_features  # 2048
        backbone.fc = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes),
        )
        self.backbone = backbone

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        log.info(
            "BrainTumorResNet50 | trainable: %s / %s  (%.2f%%)",
            f"{n_train:,}", f"{n_total:,}", 100.0 * n_train / n_total,
        )

    # Fix 3 — prevent BatchNorm drift in frozen stages
    def train(self, mode: bool = True) -> "BrainTumorResNet50":
        """Override train() to keep frozen BN layers in eval() mode.

        Without this override, calling model.train() would switch the frozen
        BatchNorm layers (conv1/bn1/layer1-3) into training mode, causing their
        running mean/variance to accumulate gradients from the new data and drift
        away from ImageNet statistics — even though their weights are frozen.
        """
        super().train(mode)
        if mode:
            for name in self._FROZEN_STAGE_NAMES:
                getattr(self.backbone, name).eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @staticmethod
    def _activate_dropout(model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        T: int = 20,
        epsilon: float = 1e-10,
    ) -> Dict[str, torch.Tensor]:
        self.eval()
        self._activate_dropout(self)

        sample_probs: List[torch.Tensor] = []
        for _ in range(T):
            logits = self.backbone(x)
            probs  = torch.softmax(logits, dim=-1)
            sample_probs.append(probs)

        stacked    = torch.stack(sample_probs, dim=0)
        mean_probs = stacked.mean(dim=0)
        entropy    = -(mean_probs * torch.log2(mean_probs + epsilon)).sum(dim=-1)
        std_probs  = stacked.std(dim=0)

        return {
            "predictions": mean_probs.argmax(dim=-1),
            "mean_probs":  mean_probs,
            "entropy":     entropy,
            "std_probs":   std_probs,
            "all_probs":   stacked,
        }


# =============================================================================
# Section 5 — Training & Evaluation Loops
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: AdamW,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Execute one full training epoch.

    Fix 6: The forward pass and loss are wrapped in torch.autocast(bfloat16)
    when running on CUDA, enabling Blackwell/Ampere tensor-core acceleration.
    bfloat16 has the same dynamic range as float32 so no GradScaler is needed.

    Returns:
        Tuple (mean_loss, accuracy) over all batches.
    """
    model.train()
    total_loss: float = 0.0
    n_correct:  int   = 0
    n_total:    int   = 0
    _use_amp = device.type == "cuda"

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Fix 6 — autocast for Blackwell/Ampere bfloat16 tensor cores
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=_use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)

        loss.backward()
        # Gradient clipping — critical for ViT fine-tuning numerical stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs          = images.size(0)
        total_loss += loss.item() * bs
        n_correct  += (logits.detach().argmax(dim=-1) == labels).sum().item()
        n_total    += bs

        if step % 20 == 0:
            log.info(
                "Epoch [%02d]  step [%d/%d]  batch_loss=%.4f",
                epoch, step, len(loader), loss.item(),
            )

    return total_loss / n_total, n_correct / n_total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> Tuple[float, float]:
    """Deterministic evaluation with Dropout disabled (model.eval()).

    Fix 6: autocast(bfloat16) is also applied during evaluation for speed.

    Returns:
        Tuple (mean_loss, accuracy).
    """
    model.eval()
    total_loss: float = 0.0
    n_correct:  int   = 0
    n_total:    int   = 0
    _use_amp = device.type == "cuda"

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Fix 6 — autocast for Blackwell/Ampere bfloat16 tensor cores
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=_use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        n_correct  += (logits.argmax(dim=-1) == labels).sum().item()
        n_total    += images.size(0)

    return total_loss / n_total, n_correct / n_total


def evaluate_with_uncertainty(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    T: int = 20,
) -> Dict[str, torch.Tensor]:
    """MC Dropout evaluation over an entire DataLoader.

    Returns:
        Dict with concatenated tensors: predictions, labels, entropy, mean_probs.
    """
    preds_list:   List[torch.Tensor] = []
    labels_list:  List[torch.Tensor] = []
    entropy_list: List[torch.Tensor] = []
    probs_list:   List[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        result = model.predict_with_uncertainty(images, T=T)

        preds_list.append(result["predictions"].cpu())
        labels_list.append(labels.cpu())
        entropy_list.append(result["entropy"].cpu())
        probs_list.append(result["mean_probs"].cpu())

    return {
        "predictions": torch.cat(preds_list),
        "labels":      torch.cat(labels_list),
        "entropy":     torch.cat(entropy_list),
        "mean_probs":  torch.cat(probs_list),
    }


# =============================================================================
# Section 6 — Calibration & Uncertainty Metrics
# =============================================================================

def compute_calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, float]:
    """Compute ECE (15-bin) and multi-class Brier score.

    Args:
        probs:  (N, C) predicted probabilities from MC Dropout mean.
        labels: (N,) integer true labels.
        n_bins: Equal-width confidence bins for ECE.

    Returns:
        Dict with ``"ece"`` and ``"brier"``.
    """
    N, C       = probs.shape
    confidence = probs.max(axis=1)
    preds      = probs.argmax(axis=1)
    correct    = (preds == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.sum() == 0:
            continue
        ece += np.abs(correct[in_bin].mean() - confidence[in_bin].mean()) * in_bin.sum() / N

    one_hot = np.eye(C)[labels]
    brier   = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    return {"ece": float(ece), "brier": brier}


def compute_risk_coverage_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    coverage_levels: Tuple[float, ...] = (0.80, 0.90, 0.95),
) -> Dict[str, Any]:
    """Compute the selective-prediction risk-coverage curve and AURC.

    Samples are ordered by MC-Dropout entropy ascending (most confident first).
    At each coverage fraction k/N, risk = 1 − accuracy on the k most-confident
    samples.  Fix 8: uses _trapz for NumPy 1.x / 2.x compatibility.

    Args:
        probs:           (N, C) MC Dropout mean probabilities.
        labels:          (N,) integer true labels.
        coverage_levels: Fixed coverage fractions at which to report accuracy.

    Returns:
        Dict with keys ``"coverage"``, ``"risk"``, ``"aurc"``,
        ``"acc_at_coverage"``.
    """
    entropy    = -(probs * np.log2(probs + 1e-10)).sum(axis=1)
    correct    = (probs.argmax(axis=1) == labels).astype(float)
    sorted_idx = np.argsort(entropy)
    correct_s  = correct[sorted_idx]

    N            = len(labels)
    coverage_arr = np.arange(1, N + 1) / N
    risk_arr     = 1.0 - np.cumsum(correct_s) / np.arange(1, N + 1)
    aurc         = float(_trapz(risk_arr, coverage_arr))   # Fix 8

    acc_at: Dict[float, float] = {}
    for level in coverage_levels:
        n_covered     = max(1, int(N * level))
        acc_at[level] = float(correct_s[:n_covered].mean())

    return {
        "coverage":        coverage_arr,
        "risk":            risk_arr,
        "aurc":            aurc,
        "acc_at_coverage": acc_at,
    }


# =============================================================================
# Section 7 — Statistical Analysis
# =============================================================================

def _acc_metric(y_true: np.ndarray, y_pred: np.ndarray, _: Any = None) -> float:
    return float((y_pred == y_true).mean())


def _auc_metric(y_true: np.ndarray, _: np.ndarray, y_probs: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))


def bootstrap_ci(
    metric: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Bootstrap percentile 95% CI for accuracy or macro-AUC.

    Returns:
        Tuple (point_estimate, lower_bound, upper_bound).
    """
    fn  = _acc_metric if metric == "accuracy" else _auc_metric
    rng = np.random.default_rng(seed)
    N   = len(y_true)

    point     = fn(y_true, y_pred, y_probs)
    estimates = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, N, size=N)
        try:
            estimates[i] = fn(
                y_true[idx],
                y_pred[idx],
                y_probs[idx] if y_probs is not None else None,
            )
        except ValueError:
            estimates[i] = np.nan

    valid = estimates[~np.isnan(estimates)]
    lo    = float(np.percentile(valid, 100 * alpha / 2))
    hi    = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


def mcnemar_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> Tuple[float, float]:
    """McNemar's test comparing two classifiers on the same test set.

    Continuity-corrected statistic: (|b − c| − 1)² / (b + c) ~ χ²(1).

    Returns:
        Tuple (chi2_statistic, p_value).
    """
    correct_a = preds_a == y_true
    correct_b = preds_b == y_true
    b = int(( correct_a & ~correct_b).sum())
    c = int((~correct_a &  correct_b).sum())

    if b + c == 0:
        return 0.0, 1.0

    stat = float((abs(b - c) - 1) ** 2 / (b + c))
    pval = float(1.0 - scipy_chi2.cdf(stat, df=1))
    return stat, pval


# =============================================================================
# Section 8 — OOD Detection via MC-Dropout Entropy
# =============================================================================

class _FlatImageDataset(Dataset):
    """Dataset for a flat directory of images (no class sub-structure).

    Fix 7: replaces the per-image loop in evaluate_ood with a batched DataLoader.
    """

    def __init__(self, img_dir: str, transform: transforms.Compose) -> None:
        self.transform = transform
        self.files: List[Path] = []
        for fp in sorted(Path(img_dir).iterdir()):
            if fp.suffix.lower() not in _IMG_EXTS:
                continue
            try:
                Image.open(fp)           # quick header check; PIL is lazy, no decode
                self.files.append(fp)
            except Exception as exc:
                log.warning("OOD dataset: skipping %s — %s", fp.name, exc)
        log.info("OOD dataset: %d valid images in %s", len(self.files), img_dir)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.transform(Image.open(self.files[idx]).convert("RGB")), 0


def evaluate_ood(
    model: nn.Module,
    in_dist_loader: DataLoader,
    ood_image_dir: str,
    device: torch.device,
    T: int = 20,
) -> Dict[str, Any]:
    """AUROC for separating in-distribution from OOD images via entropy.

    Higher entropy → predicted OOD. In-distribution = label 0, OOD = label 1.
    Fix 7: OOD images are processed in batches via _FlatImageDataset.

    Args:
        model:           Trained model with ``predict_with_uncertainty``.
        in_dist_loader:  DataLoader for the in-distribution test split.
        ood_image_dir:   Flat directory of OOD images (no class sub-folders).
        device:          Compute device.
        T:               MC Dropout passes.

    Returns:
        Dict with ``"ood_auroc"``, ``"n_ind"``, ``"n_ood"``.
    """
    def _entropies_from_loader(loader: DataLoader) -> List[float]:
        ents: List[float] = []
        for images, _ in loader:
            r = model.predict_with_uncertainty(images.to(device), T=T)
            ents.extend(r["entropy"].cpu().tolist())
        return ents

    # Fix 7 — batch OOD images with a DataLoader instead of looping one by one
    ood_dataset = _FlatImageDataset(ood_image_dir, get_transforms("test"))
    if len(ood_dataset) == 0:
        log.warning("OOD directory has no valid images: %s", ood_image_dir)
        return {"ood_auroc": float("nan"), "n_ind": 0, "n_ood": 0}

    ood_loader = DataLoader(
        ood_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    ind_ent = _entropies_from_loader(in_dist_loader)
    ood_ent = _entropies_from_loader(ood_loader)

    all_scores = np.array(ind_ent + ood_ent)
    all_labels = np.array([0] * len(ind_ent) + [1] * len(ood_ent))

    if len(np.unique(all_labels)) < 2:
        log.warning("OOD: need samples from both classes — AUROC undefined.")
        return {"ood_auroc": float("nan"), "n_ind": len(ind_ent), "n_ood": len(ood_ent)}

    auroc = float(roc_auc_score(all_labels, all_scores))
    log.info(
        "OOD AUROC (entropy)  =  %.4f   [n_ind=%d  n_ood=%d]",
        auroc, len(ind_ent), len(ood_ent),
    )
    return {"ood_auroc": auroc, "n_ind": len(ind_ent), "n_ood": len(ood_ent)}


def run_ood_evaluation(
    model: nn.Module,
    manifest_df: pd.DataFrame,
    ood_image_dir: str,
    device: torch.device = DEVICE,
    T: int = 20,
    results_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Standalone OOD evaluation entry point.

    Loads the in-distribution test split from ``manifest_df`` and compares
    its entropy distribution against images from ``ood_image_dir``.
    """
    in_loader = build_dataloaders_from_manifest(
        manifest_df, batch_size=32, num_workers=0
    ).get("test")
    if in_loader is None:
        raise RuntimeError("Manifest has no test split for in-distribution baseline.")

    result = evaluate_ood(model, in_loader, ood_image_dir, device, T=T)

    if results_dir is not None:
        with open(results_dir / "ood_results.json", "w") as f:
            json.dump(result, f, indent=2)

    return result


# =============================================================================
# Section 9 — Results Logging & Plotting
# =============================================================================

class ResultsLogger:
    """Writes all per-run artifacts to a self-contained directory.

    Artifacts per seed:
      config.json, epoch_metrics.csv, confusion_matrix.png,
      calibration.png, risk_coverage.png, summary.json
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self._epoch_rows: List[Dict[str, Any]] = []

    def log_config(self, cfg: Dict[str, Any]) -> None:
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2, default=str)

    def log_epoch(self, row: Dict[str, Any]) -> None:
        self._epoch_rows.append(row)

    def save_epoch_csv(self) -> None:
        if not self._epoch_rows:
            return
        path = self.run_dir / "epoch_metrics.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._epoch_rows[0].keys()))
            writer.writeheader()
            writer.writerows(self._epoch_rows)

    def save_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        cm  = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False)
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        fig.savefig(self.run_dir / "confusion_matrix.png", dpi=150)
        plt.close(fig)

    def save_calibration_plot(
        self, probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
    ) -> None:
        confidence = probs.max(axis=1)
        correct    = (probs.argmax(axis=1) == labels).astype(float)
        bins       = np.linspace(0.0, 1.0, n_bins + 1)
        acc_bins   = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            in_bin = (confidence > lo) & (confidence <= hi)
            acc_bins.append(correct[in_bin].mean() if in_bin.sum() > 0 else 0.0)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.bar(bins[:-1], acc_bins, width=1 / n_bins, align="edge",
               alpha=0.7, label="Accuracy per bin")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title("Reliability Diagram")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(self.run_dir / "calibration.png", dpi=150)
        plt.close(fig)

    def save_risk_coverage_plot(
        self, coverage: np.ndarray, risk: np.ndarray, aurc: float
    ) -> None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(coverage, risk, lw=2, color="steelblue")
        ax.fill_between(coverage, risk, alpha=0.15, color="steelblue")
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Risk  (1 − Accuracy)")
        ax.set_title(f"Risk-Coverage Curve   AURC = {aurc:.4f}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.run_dir / "risk_coverage.png", dpi=150)
        plt.close(fig)

    def save_summary(self, summary: Dict[str, Any]) -> None:
        path = self.run_dir / "summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        log.info("Summary → %s", path)


# =============================================================================
# Section 10 — Single-Seed Training Pipeline
# =============================================================================

def run_single_seed(
    manifest_df: pd.DataFrame,
    seed: int,
    model_class: Type[nn.Module],
    model_name: str,
    run_dir: Path,
    num_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    label_smoothing: float = 0.1,
    num_workers: int = 4,
    mc_T: int = 20,
    max_samples_smoke: Optional[int] = None,
    early_stop_patience: int = 5,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train and fully evaluate one model for one seed.

    Fix 5: training loop includes early stopping on val_loss with configurable
    patience (default 5 epochs with no improvement → stop).

    Returns:
        Tuple (model, metrics_dict).
    """
    setup_reproducibility(seed)
    logger = ResultsLogger(run_dir)
    cfg    = dict(
        seed=seed, model=model_name, num_epochs=num_epochs,
        batch_size=batch_size, lr=learning_rate, wd=weight_decay,
        label_smoothing=label_smoothing, mc_T=mc_T,
        early_stop_patience=early_stop_patience,
    )
    logger.log_config(cfg)

    loaders = build_dataloaders_from_manifest(
        manifest_df,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples_per_split=max_samples_smoke,
    )
    if "train" not in loaders or "val" not in loaders:
        raise RuntimeError("Manifest must contain both 'train' and 'val' splits.")

    model     = model_class(num_classes=NUM_CLASSES, dropout_p=0.3).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate, weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-7
    )

    checkpoint_path  = run_dir / f"best_{model_name}_seed{seed}.pth"
    best_val_loss    = float("inf")
    epochs_no_improve = 0  # Fix 5 — early-stopping counter

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, DEVICE, epoch
        )
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, DEVICE)
        scheduler.step(val_loss)

        logger.log_epoch(dict(
            epoch=epoch, train_loss=train_loss, train_acc=train_acc,
            val_loss=val_loss, val_acc=val_acc,
        ))
        log.info(
            "[%s | seed=%d]  Epoch [%02d/%02d]  "
            "train_loss=%.4f  train_acc=%.4f  val_loss=%.4f  val_acc=%.4f",
            model_name, seed, epoch, num_epochs,
            train_loss, train_acc, val_loss, val_acc,
        )

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":             val_loss,
                    "val_acc":              val_acc,
                },
                checkpoint_path,
            )
            log.info(
                "  ✓ Checkpoint saved  (epoch=%d  val_loss=%.4f  val_acc=%.4f)",
                epoch, val_loss, val_acc,
            )
        else:
            epochs_no_improve += 1
            # Fix 5 — early stopping
            if epochs_no_improve >= early_stop_patience:
                log.info(
                    "Early stopping at epoch %d — "
                    "no val_loss improvement for %d consecutive epochs.",
                    epoch, early_stop_patience,
                )
                break

    logger.save_epoch_csv()

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info(
        "Best weights restored — epoch %d  val_loss=%.4f  val_acc=%.4f",
        ckpt["epoch"], ckpt["val_loss"], ckpt["val_acc"],
    )

    metrics: Dict[str, Any] = {
        "model":         model_name,
        "seed":          seed,
        "best_val_loss": float(ckpt["val_loss"]),
        "best_val_acc":  float(ckpt["val_acc"]),
    }

    if "test" not in loaders:
        log.warning("No test split — skipping test evaluation.")
        logger.save_summary({**cfg, **metrics})
        return model, metrics

    test_loss, test_acc = evaluate(model, loaders["test"], criterion, DEVICE)
    log.info("Test (deterministic)  loss=%.4f  acc=%.4f", test_loss, test_acc)

    log.info("MC Dropout  T=%d on test set …", mc_T)
    mc = evaluate_with_uncertainty(model, loaders["test"], DEVICE, T=mc_T)

    y_true  = mc["labels"].numpy()
    y_pred  = mc["predictions"].numpy()
    y_probs = mc["mean_probs"].numpy()

    mc_acc    = float((y_pred == y_true).mean())
    macro_f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_auc = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
    mean_ent  = float(mc["entropy"].mean().item())
    log.info(
        "Test (MC T=%d)  acc=%.4f  macro-F1=%.4f  macro-AUC=%.4f  mean-H=%.4f bits",
        mc_T, mc_acc, macro_f1, macro_auc, mean_ent,
    )

    cal = compute_calibration_metrics(y_probs, y_true, n_bins=15)
    log.info("ECE=%.4f  Brier=%.4f", cal["ece"], cal["brier"])

    rc = compute_risk_coverage_curve(y_probs, y_true)
    log.info("AURC=%.4f", rc["aurc"])
    for level, acc_val in rc["acc_at_coverage"].items():
        log.info("  Accuracy @ %.0f%% coverage = %.4f", level * 100, acc_val)

    acc_pt, acc_lo, acc_hi = bootstrap_ci("accuracy", y_true, y_pred, seed=seed)
    auc_pt, auc_lo, auc_hi = bootstrap_ci("auc", y_true, y_pred, y_probs, seed=seed)
    log.info(
        "Bootstrap (1000) | acc=%.4f [%.4f, %.4f]  auc=%.4f [%.4f, %.4f]",
        acc_pt, acc_lo, acc_hi, auc_pt, auc_lo, auc_hi,
    )

    logger.save_confusion_matrix(y_true, y_pred)
    logger.save_calibration_plot(y_probs, y_true)
    logger.save_risk_coverage_plot(rc["coverage"], rc["risk"], rc["aurc"])

    metrics.update({
        "test_loss":    float(test_loss),
        "test_acc":     float(test_acc),
        "mc_acc":       mc_acc,
        "macro_f1":     macro_f1,
        "macro_auc":    macro_auc,
        "mean_entropy": mean_ent,
        "ece":          cal["ece"],
        "brier":        cal["brier"],
        "aurc":         rc["aurc"],
        "acc_at_80":    rc["acc_at_coverage"].get(0.80),
        "acc_at_90":    rc["acc_at_coverage"].get(0.90),
        "acc_at_95":    rc["acc_at_coverage"].get(0.95),
        "acc_ci_lo":    acc_lo,
        "acc_ci_hi":    acc_hi,
        "auc_ci_lo":    auc_lo,
        "auc_ci_hi":    auc_hi,
    })

    logger.save_summary({**cfg, **metrics})
    return model, metrics


# =============================================================================
# Section 11 — Multi-Seed Runner & ViT vs ResNet-50 Comparison
# =============================================================================

def run_multi_seed(
    manifest_df: pd.DataFrame,
    model_class: Type[nn.Module],
    model_name: str,
    results_dir: Path,
    seeds: Optional[List[int]] = None,
    **train_kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[int, nn.Module]]:
    """Train across multiple seeds and aggregate into a summary table."""
    if seeds is None:
        seeds = SEEDS

    all_metrics:    List[Dict[str, Any]] = []
    trained_models: Dict[int, nn.Module] = {}

    for seed in seeds:
        seed_dir = results_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        log.info("=" * 60)
        log.info("Seed %d  |  %s", seed, model_name.upper())
        log.info("=" * 60)
        model, metrics = run_single_seed(
            manifest_df=manifest_df,
            seed=seed,
            model_class=model_class,
            model_name=model_name,
            run_dir=seed_dir,
            **train_kwargs,
        )
        all_metrics.append(metrics)
        trained_models[seed] = model

    summary_df = pd.DataFrame(all_metrics)

    log.info("\n%s — Aggregate  (%d seeds)", model_name.upper(), len(seeds))
    log.info("─" * 50)
    for col in ("mc_acc", "macro_f1", "macro_auc", "ece", "brier", "aurc"):
        if col in summary_df.columns:
            vals = summary_df[col].dropna()
            log.info("  %-16s  %.4f ± %.4f", col, vals.mean(), vals.std())
    log.info("─" * 50)

    csv_path = results_dir / f"summary_{model_name}.csv"
    summary_df.to_csv(csv_path, index=False)
    log.info("Summary CSV → %s", csv_path)
    return summary_df, trained_models


def run_comparison(
    manifest_df: pd.DataFrame,
    results_dir: Path,
    seeds: Optional[List[int]] = None,
    **train_kwargs: Any,
) -> Tuple[Dict[str, Any], Dict[str, Dict[int, nn.Module]]]:
    """Train ViT and ResNet-50, aggregate, then run per-seed McNemar's tests.

    Fix 4: McNemar's test is run for every (ViT_seed_i, ResNet50_seed_i) pair
    rather than only the first seed.  The summary reports how many seeds show a
    significant difference in the same direction, giving a more robust picture
    of whether the performance gap is consistent across random initialisations.
    mc_T is taken from train_kwargs rather than the hardcoded default.

    Returns:
        Tuple (comparison_dict, {"vit": {seed: model}, "resnet50": {seed: model}}).
    """
    vit_dir = results_dir / "vit"
    rn_dir  = results_dir / "resnet50"
    vit_dir.mkdir(exist_ok=True)
    rn_dir.mkdir(exist_ok=True)

    vit_summary, vit_models = run_multi_seed(
        manifest_df, BrainTumorViT, "vit", vit_dir, seeds=seeds, **train_kwargs
    )
    rn_summary, rn_models = run_multi_seed(
        manifest_df, BrainTumorResNet50, "resnet50", rn_dir, seeds=seeds, **train_kwargs
    )

    # Fix 4 — per-seed McNemar using mc_T from caller, not hardcoded default
    mc_T = train_kwargs.get("mc_T", 20)
    active_seeds = seeds or SEEDS

    test_loaders = build_dataloaders_from_manifest(manifest_df, batch_size=32, num_workers=0)
    test_loader  = test_loaders.get("test")

    per_seed_mcnemar: List[Dict[str, Any]] = []

    if test_loader is None:
        log.warning("No test split — McNemar's tests skipped.")
    else:
        for seed in active_seeds:
            if seed not in vit_models or seed not in rn_models:
                continue

            mc_vit_s = evaluate_with_uncertainty(vit_models[seed], test_loader, DEVICE, T=mc_T)
            mc_rn_s  = evaluate_with_uncertainty(rn_models[seed],  test_loader, DEVICE, T=mc_T)

            y_true_s   = mc_vit_s["labels"].numpy()
            preds_vit_s = mc_vit_s["predictions"].numpy()
            preds_rn_s  = mc_rn_s["predictions"].numpy()

            stat_s, pval_s = mcnemar_test(y_true_s, preds_vit_s, preds_rn_s)

            vit_acc_s = float((preds_vit_s == y_true_s).mean())
            rn_acc_s  = float((preds_rn_s  == y_true_s).mean())
            direction = "vit_better" if vit_acc_s > rn_acc_s else "resnet50_better"

            per_seed_mcnemar.append({
                "seed":        seed,
                "chi2":        stat_s,
                "p_value":     pval_s,
                "significant": bool(pval_s < 0.05),
                "direction":   direction,
                "vit_acc":     vit_acc_s,
                "rn_acc":      rn_acc_s,
            })
            log.info(
                "McNemar [seed=%d]  χ²=%.4f  p=%.4f  dir=%s  "
                "(vit=%.4f  rn=%.4f)",
                seed, stat_s, pval_s, direction, vit_acc_s, rn_acc_s,
            )

        # Summary across seeds
        n_sig     = sum(r["significant"] for r in per_seed_mcnemar)
        sig_dirs  = [r["direction"] for r in per_seed_mcnemar if r["significant"]]
        if sig_dirs:
            dominant = max(set(sig_dirs), key=sig_dirs.count)
            consistent = len(set(sig_dirs)) == 1
            log.info(
                "McNemar summary: %d / %d seeds significant  |  "
                "dominant direction: %s  |  consistent: %s",
                n_sig, len(per_seed_mcnemar), dominant, consistent,
            )
        else:
            log.info(
                "McNemar summary: 0 / %d seeds show a significant difference "
                "(p < 0.05).",
                len(per_seed_mcnemar),
            )

    comparison = {
        "vit":             vit_summary.to_dict(orient="list"),
        "resnet50":        rn_summary.to_dict(orient="list"),
        "mcnemar_per_seed": per_seed_mcnemar,
    }
    with open(results_dir / "comparison_summary.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    log.info("Comparison summary → %s", results_dir / "comparison_summary.json")

    return comparison, {"vit": vit_models, "resnet50": rn_models}


# =============================================================================
# Section 12 — Entry Point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Brain Tumor MRI: ViT vs ResNet-50 with MC Dropout UQ"
    )
    parser.add_argument("--data-root",          type=str,   default="./data/brain_tumor")
    parser.add_argument("--manifest",           type=str,   default="data/split_manifest.csv")
    parser.add_argument("--results-dir",        type=str,   default="results")
    parser.add_argument("--epochs",             type=int,   default=30)
    parser.add_argument("--batch-size",         type=int,   default=32)
    parser.add_argument("--lr",                 type=float, default=1e-4)
    parser.add_argument("--weight-decay",       type=float, default=0.01)
    parser.add_argument("--label-smoothing",    type=float, default=0.1)
    parser.add_argument("--workers",            type=int,   default=4)
    parser.add_argument("--mc-T",               type=int,   default=20)
    parser.add_argument("--early-stop-patience",type=int,   default=5)
    parser.add_argument("--seeds",              type=int,   nargs="+", default=SEEDS)
    parser.add_argument("--model",              type=str,   default="both",
                        choices=["vit", "resnet50", "both"])
    parser.add_argument("--ood-dir",            type=str,   default=None)
    parser.add_argument("--smoke",              action="store_true",
                        help="1 epoch, seed=42, 8 samples/split, T=2.")
    parser.add_argument("--force-manifest",     action="store_true")
    args = parser.parse_args()

    log.info("Compute device: %s", DEVICE)

    smoke_samples: Optional[int] = None
    if args.smoke:
        log.info("SMOKE TEST — 1 epoch, 1 seed, 8 samples/split")
        args.epochs = 1
        args.seeds  = [42]
        args.mc_T   = 2
        smoke_samples = 8

    manifest_df = build_split_manifest(
        raw_root=args.data_root,
        manifest_path=args.manifest,
        seed=args.seeds[0],
        force_rebuild=args.force_manifest,
    )

    top_dir = Path(args.results_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    top_dir.mkdir(parents=True, exist_ok=True)

    train_kwargs: Dict[str, Any] = dict(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        num_workers=args.workers,
        mc_T=args.mc_T,
        max_samples_smoke=smoke_samples,
        early_stop_patience=args.early_stop_patience,
    )

    vit_models: Dict[int, nn.Module] = {}

    if args.model == "both":
        _, model_dicts = run_comparison(
            manifest_df, top_dir, seeds=args.seeds, **train_kwargs
        )
        vit_models = model_dicts["vit"]

    elif args.model == "vit":
        _, vit_models = run_multi_seed(
            manifest_df, BrainTumorViT, "vit", top_dir,
            seeds=args.seeds, **train_kwargs,
        )
    else:
        run_multi_seed(
            manifest_df, BrainTumorResNet50, "resnet50", top_dir,
            seeds=args.seeds, **train_kwargs,
        )

    if args.ood_dir and vit_models:
        first_seed = args.seeds[0]
        log.info("OOD evaluation (ViT, seed=%d) → %s", first_seed, args.ood_dir)
        run_ood_evaluation(
            vit_models[first_seed],
            manifest_df,
            args.ood_dir,
            device=DEVICE,
            T=args.mc_T,
            results_dir=top_dir,
        )

    log.info("All results saved under: %s", top_dir)


if __name__ == "__main__":
    main()
