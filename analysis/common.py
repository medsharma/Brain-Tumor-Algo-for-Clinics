#!/usr/bin/env python3
"""
analysis/common.py

Shared helpers for the analysis/ suite. Every analysis script imports model
classes, dataset/dataloader builders, and metric functions from the main
``src/code.py`` training script rather than reimplementing them, so analysis
always exercises the exact same architectures/preprocessing Session A trains
with.

This module adds only what src/code.py does not already provide:
  - locating the most recent *completed* multi-seed run under results/
  - flagging whether that run is a --smoke test (so downstream scripts can
    label their outputs as pipeline-validation only, not real findings)
  - loading a trained checkpoint into a fresh model instance
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = Path(__file__).resolve().parent
RESULTS_OUT_DIR = ANALYSIS_DIR / "results"

sys.path.insert(0, str(REPO_ROOT / "src"))

from code import (  # noqa: E402
    BrainTumorResNet50,
    BrainTumorViT,
    CLASS_NAMES,
    DEVICE,
    NUM_CLASSES,
    build_dataloaders_from_manifest,
    build_split_manifest,
    compute_calibration_metrics,
    get_transforms,
)

MODEL_CLASSES: Dict[str, type] = {
    "vit": BrainTumorViT,
    "resnet50": BrainTumorResNet50,
}


def find_latest_run(results_root: Path = REPO_ROOT / "results") -> Optional[Path]:
    """Return the most recent results/<timestamp>/ directory that has a
    comparison_summary.json (i.e. a completed run_comparison() call from
    src/code.py), or None if no completed run exists yet.
    """
    if not results_root.exists():
        return None
    candidates = sorted(
        (p for p in results_root.iterdir() if p.is_dir() and (p / "comparison_summary.json").exists()),
        key=lambda p: p.name,
    )
    return candidates[-1] if candidates else None


def is_smoke_run(seed_dir: Path) -> bool:
    """Heuristic: src/code.py's --smoke flag sets num_epochs=1 and mc_T=2."""
    cfg_path = seed_dir / "config.json"
    if not cfg_path.exists():
        return False
    with open(cfg_path) as f:
        cfg = json.load(f)
    return cfg.get("num_epochs") == 1 and cfg.get("mc_T") == 2


def latest_seed_dir(run_dir: Path, model_name: str) -> Path:
    """Latest (by seed dir mtime) seed_* directory for a given model under a run."""
    model_dir = run_dir / model_name
    seed_dirs = sorted((p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories under {model_dir}")
    return seed_dirs[0]


def all_seed_dirs(run_dir: Path, model_name: str) -> List[Path]:
    model_dir = run_dir / model_name
    return sorted((p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")))


def load_model_from_seed_dir(model_name: str, seed_dir: Path, device: torch.device = DEVICE) -> nn.Module:
    """Instantiate the right architecture and load its best checkpoint."""
    ckpt_files = list(seed_dir.glob(f"best_{model_name}_seed*.pth"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {seed_dir}")
    model_class = MODEL_CLASSES[model_name]
    model = model_class(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(ckpt_files[0], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt_files[0]


def get_manifest():
    manifest_path = REPO_ROOT / "data" / "split_manifest.csv"
    return build_split_manifest(
        raw_root=str(REPO_ROOT / "data" / "brain_tumor"),
        manifest_path=str(manifest_path),
        force_rebuild=False,
    )


def run_provenance(run_dir: Path, seed_dir: Path) -> Dict[str, Any]:
    """Standard provenance block every analysis output should embed, so the
    manuscript-merge session can tell exactly which checkpoint/run a given
    number came from and whether it's a smoke test.
    """
    smoke = is_smoke_run(seed_dir)
    cfg_path = seed_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    return {
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "seed_dir": str(seed_dir.relative_to(REPO_ROOT)),
        "is_smoke_test_run": smoke,
        "caveat": (
            "THIS IS A --smoke TEST CHECKPOINT (1 epoch, 8 samples/class, mc_T=2). "
            "Numbers below validate the analysis pipeline only and MUST NOT be "
            "quoted as real findings. Re-run this script once Session A's full "
            "multi-seed run completes."
            if smoke else
            "Real training run (not a smoke test)."
        ),
        "config": cfg,
    }
