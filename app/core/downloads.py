"""What the running app offers for download, so someone else can run it too.

The point of this module is a specific promise: **anything downloaded here,
dropped next to a copy of the app, reproduces the behaviour of the server you
downloaded it from.** Not a similar model. The same weights, the same
thresholds, the same answers.

That promise is why the file list is derived from the loaded deployment config
rather than from whatever happens to be sitting in a folder. If the server is
running a 5-seed ViT ensemble at threshold 0.590, those five files and that
config are what it offers. Change the config, restart, and the offer changes
with it. There is no way for the two to drift apart, because there is only one
source of truth.

Every entry carries a SHA-256. A truncated 344 MB download that still loads is
a real possibility, and a model that silently loads a corrupted tensor is worse
than one that refuses to start.

Nothing here writes anything. It reads the checkpoint files the engine already
depends on.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from . import paths
from .config import DeploymentConfig

#: SHA-256 is expensive on a 344 MB file, so it is computed once per file per
#: process and reused. Keyed by (path, size, mtime) so a replaced file is not
#: served with a stale hash.
_hash_cache: Dict[tuple, str] = {}


@dataclass(frozen=True)
class Downloadable:
    """One file the app will hand out."""

    key: str                 # url-safe identifier
    filename: str            # what the browser should save it as
    path: Path               # where it actually is
    kind: str                # "model" | "config"
    bytes: int
    sha256: str
    description: str

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "filename": self.filename,
            "kind": self.kind,
            "bytes": self.bytes,
            "size_human": human_size(self.bytes),
            "sha256": self.sha256,
            "description": self.description,
        }


def human_size(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f} GB"
    if n >= 1_000_000:
        return f"{n / 1e6:.0f} MB"
    if n >= 1_000:
        return f"{n / 1e3:.0f} KB"
    return f"{n} bytes"


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    stat = path.stat()
    cache_key = (str(path), stat.st_size, stat.st_mtime_ns)
    cached = _hash_cache.get(cache_key)
    if cached is not None:
        return cached

    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    digest = h.hexdigest()
    _hash_cache[cache_key] = digest
    return digest


def _safe_key(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "-" for c in name)


#: Where to look for stripped copies of the checkpoints, newest preference
#: first. Overridable for anyone laying the files out differently.
DOWNLOAD_DIR_ENV = "MRI_TRIAGE_DOWNLOAD_DIR"


def _download_dirs() -> List[Path]:
    import os

    dirs: List[Path] = []
    override = os.environ.get(DOWNLOAD_DIR_ENV, "").strip()
    if override:
        dirs.append(Path(override).expanduser())
    root = paths.repo_root()
    dirs.append(root / "models")
    dirs.append(root / "dist" / "BrainMRITriage" / "models")
    return dirs


def preferred_copy(original: Path) -> Path:
    """Serve the stripped copy of a checkpoint if one exists.

    A training checkpoint carries ``optimizer_state_dict`` next to the weights:
    459 MB on disk against 344 MB of actual model. Nobody downloading this to
    run inference needs Adam moments, and across the five-model ensemble it is
    575 MB of somebody's bandwidth for nothing.

    ``app/tools/prepare_clinic_install.py`` writes stripped copies and verifies
    they produce bit-identical predictions before accepting them. If one is
    sitting there, hand that out instead. Same weights, same answers, smaller
    download.

    Falls back to the original when no stripped copy exists, so this is an
    optimisation and never a missing file.
    """
    for directory in _download_dirs():
        candidate = directory / original.name
        try:
            if candidate.is_file() and candidate.stat().st_size < original.stat().st_size:
                return candidate
        except OSError:
            continue
    return original


def build_catalogue(cfg: DeploymentConfig) -> List[Downloadable]:
    """Every file needed to reproduce this server, and nothing else.

    Derived from the live config, so it cannot describe a different model from
    the one answering requests.
    """
    items: List[Downloadable] = []

    n = len(cfg.checkpoints)
    for i, checkpoint in enumerate(cfg.checkpoints, start=1):
        original = Path(checkpoint.path)
        if not original.is_file():
            continue
        path = preferred_copy(original)
        stripped = path != original
        items.append(Downloadable(
            key=_safe_key(original.name),
            filename=original.name,
            path=path,
            kind="model",
            bytes=path.stat().st_size,
            sha256=file_sha256(path),
            description=(
                f"Model {i} of {n}, {cfg.chosen_backbone} seed {checkpoint.seed}. "
                + ("All of them are needed: the answers are averaged across the set. "
                   if n > 1 else "This is the only model file needed. ")
                + ("Training leftovers removed, so it is smaller than the file on "
                   "the training machine and gives identical predictions."
                   if stripped else "")
            ).strip(),
        ))

    # The settings file. Without it the weights are just weights: no referral
    # threshold, no deferral cutoff, no temperature. The app refuses to start
    # rather than invent them, which is the correct behaviour and also means a
    # model-only download is useless on its own.
    cfg_path = Path(cfg.source_path)
    if cfg_path.is_file():
        items.append(Downloadable(
            key="deployment_config.json",
            filename="deployment_config.json",
            path=cfg_path,
            kind="config",
            bytes=cfg_path.stat().st_size,
            sha256=file_sha256(cfg_path),
            description=(
                "The safety settings: which model, the referral threshold, the "
                "deferral cutoff and the temperature. Required. The app will "
                "not start without it and will not guess these numbers."
            ),
        ))

    rejector = paths.repo_root() / "analysis" / "results" / "ood" / "rejector_config.json"
    if rejector.is_file():
        items.append(Downloadable(
            key="rejector_config.json",
            filename="rejector_config.json",
            path=rejector,
            kind="config",
            bytes=rejector.stat().st_size,
            sha256=file_sha256(rejector),
            description=(
                "The input check, which decides whether an image is a brain MRI "
                "the tool should judge at all. Without it the app falls back to "
                "unfitted limits that wrongly reject about 1 real scan in 12."
            ),
        ))

    return items


def find(catalogue: Iterable[Downloadable], key: str) -> Optional[Downloadable]:
    for item in catalogue:
        if item.key == key:
            return item
    return None


def manifest(cfg: DeploymentConfig, items: List[Downloadable]) -> dict:
    """A machine-readable description of the exact configuration on offer."""
    total = sum(i.bytes for i in items)
    return {
        "app_config": {
            "backbone": cfg.chosen_backbone,
            "seeds": list(cfg.chosen_seeds),
            "ensemble": bool(cfg.ensemble),
            "mc_passes": cfg.mc_T,
            "temperature": cfg.temperature,
            "tumor_threshold": cfg.tumor_threshold,
            "defer_threshold": cfg.entropy_defer_threshold,
            "defer_signal": str(cfg.raw.get("defer_signal", "entropy")),
            "entropy_units": cfg.entropy_units,
            "thresholds_fitted_on": cfg.thresholds_fitted_on,
        },
        "total_bytes": total,
        "total_size_human": human_size(total),
        "files": [i.to_dict() for i in items],
        "verify": (
            "Check each file's SHA-256 after downloading. A truncated model "
            "file can still load and then give wrong answers."
        ),
        "not_validated": (
            "This model has never been tested on data it did not train on. No "
            "clinician has reviewed a single output. Not for clinical use. Read "
            "MODEL_CARD.md before trusting anything it says."
        ),
    }


def manifest_json(cfg: DeploymentConfig, items: List[Downloadable]) -> str:
    return json.dumps(manifest(cfg, items), indent=2)
