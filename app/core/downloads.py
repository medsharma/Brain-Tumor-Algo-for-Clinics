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
    """One file the app will hand out.

    ``content`` is set for files that are generated rather than read from disk.
    The settings file is the only one: it has to describe the checkpoints being
    served, and those are not always the ones on this machine.
    """

    key: str                 # url-safe identifier
    filename: str            # what the browser should save it as
    path: Path               # where it actually is
    kind: str                # "model" | "config"
    bytes: int
    sha256: str
    description: str
    content: Optional[bytes] = None

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


def find_windows_package() -> Optional[Path]:
    """The built Windows app, zipped, if somebody has produced one.

    This is the download for a person who does not have Python and should not
    have to care what Python is. Unzip, double-click, upload a scan.

    Built by ``app/packaging/build_windows.bat``. Absent on a machine that has
    never run it, in which case the page falls back to offering the setup
    script, which needs Python.
    """
    import os

    override = os.environ.get("MRI_TRIAGE_PACKAGE_ZIP", "").strip()
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None

    for name in ("BrainMRITriage-windows.zip", "BrainMRITriage.zip"):
        candidate = paths.repo_root() / "release" / name
        if candidate.is_file():
            return candidate
    return None


def windows_package(cfg: DeploymentConfig) -> Optional[Downloadable]:
    path = find_windows_package()
    if path is None:
        return None
    return Downloadable(
        key="BrainMRITriage-windows.zip",
        filename=path.name,
        path=path,
        kind="package",
        bytes=path.stat().st_size,
        sha256=file_sha256(path),
        description=(
            "The whole application for Windows, including the models. No "
            "Python and no installer. Unzip it and double-click "
            "BrainMRITriage.exe."
        ),
    )


def find_windows_installer() -> Optional[Path]:
    """The small setup program, if one has been built.

    This is the download for someone who should not have to think at all: it
    fetches the application, installs it, makes a desktop shortcut and starts
    it. A few megabytes, so the browser download finishes in seconds and the
    progress bar appears immediately rather than after 2.4 GB of silence.

    Built by ``app/packaging/build_installer.py``, which bakes in the address to
    download from. Absent until somebody runs it.
    """
    import os

    override = os.environ.get("MRI_TRIAGE_INSTALLER_EXE", "").strip()
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None

    candidate = paths.repo_root() / "release" / "BrainMRITriageSetup.exe"
    return candidate if candidate.is_file() else None


#: Must match ``app/packaging/setup_gui.py``. The installer reads these back out
#: of its own file to learn where to download from.
INSTALLER_FOOTER_MAGIC = b"MRITRIAGE-SERVER-V1:"
INSTALLER_FOOTER_END = b":MRITRIAGE-END"


def stamp_installer(raw: bytes, base_url: str) -> bytes:
    """Write this server's address into the end of the setup program.

    The setup program has to download 2.4 GB from somewhere, and it is a
    compiled binary, so it cannot be regenerated per request the way the Python
    setup script is. Its build-time address is whatever machine it was built on.
    Hand that file to a clinic on another network and it tries to fetch the
    application from an address that means nothing there, which looks exactly
    like "the download is broken".

    So the address the browser actually used to reach us is appended here, and
    ``setup_gui.footer_server`` reads it back. PyInstaller locates its archive by
    searching backwards from the end of the file, so trailing bytes do not
    disturb the bootloader and the exe still starts normally.

    Any footer from a previous stamping is dropped first, so re-serving a file
    that has already been through here does not accumulate them.
    """
    cut = raw.rfind(INSTALLER_FOOTER_MAGIC)
    if cut != -1:
        raw = raw[:cut]
    return raw + INSTALLER_FOOTER_MAGIC + base_url.rstrip("/").encode("utf-8") + INSTALLER_FOOTER_END


#: The unstamped setup program, read once. Keyed by (path, size, mtime) so a
#: rebuilt exe is picked up without a restart.
_installer_cache: Dict[tuple, bytes] = {}


def _installer_bytes(path: Path) -> bytes:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    blob = _installer_cache.get(key)
    if blob is None:
        blob = path.read_bytes()
        _installer_cache.clear()   # only ever one build; do not accumulate
        _installer_cache[key] = blob
    return blob


def windows_installer(cfg: DeploymentConfig, base_url: str = "") -> Optional[Downloadable]:
    """The setup program, with ``base_url`` stamped in when one is given.

    ``base_url`` empty means "serve it as built", which is only right for a
    caller that has no request context. Every real download has one.
    """
    path = find_windows_installer()
    if path is None:
        return None

    description = (
        "Setup program for Windows. Downloads and installs the app, makes a "
        "desktop shortcut and opens it. Nothing else to do."
    )

    if not base_url:
        return Downloadable(
            key="BrainMRITriageSetup.exe",
            filename=path.name,
            path=path,
            kind="installer",
            bytes=path.stat().st_size,
            sha256=file_sha256(path),
            description=description,
        )

    # Held in memory rather than written next to the original, because the
    # right bytes depend on how this particular caller reached us and two
    # clinics on two networks must not race each other over one file on disk.
    #
    # Rebuilt per call rather than cached per address. `base_url` comes from the
    # Host header, so caching on it means anyone who can reach this server can
    # make it keep an extra 9 MB copy by sending a new Host, as many times as
    # they like. Only the raw file is cached, which is one fixed 9 MB; the
    # stamping is a concatenation and the hash of 9 MB is a few tens of
    # milliseconds, which is nothing against downloading it.
    blob = stamp_installer(_installer_bytes(path), base_url)
    return Downloadable(
        key="BrainMRITriageSetup.exe",
        filename=path.name,
        path=path,
        kind="installer",
        bytes=len(blob),
        sha256=hashlib.sha256(blob).hexdigest(),
        description=description,
        content=blob,
    )


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
    #
    # Generated rather than copied, because it has to describe the checkpoints
    # actually being served. The config on this machine records the SHA-256 of
    # the 459 MB training checkpoints; what gets handed out is the 344 MB
    # stripped copies. Serving the file verbatim ships a config whose hashes
    # match nothing in the download, and the receiving app then refuses to start
    # with "Model file does not match the deployment config".
    #
    # That refusal is correct and it is not a bug to work around. The config is
    # what makes it a safety check, so the config is what has to be right.
    cfg_path = Path(cfg.source_path)
    if cfg_path.is_file():
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        by_name = {i.filename: i for i in items if i.kind == "model"}
        for checkpoint in raw.get("checkpoints", []):
            name = Path(str(checkpoint.get("path", "")).replace("\\", "/")).name
            served = by_name.get(name)
            if served is not None:
                # relative, so the folder survives being copied or moved
                checkpoint["path"] = f"models/{name}"
                checkpoint["sha256"] = served.sha256
        raw["checkpoint_paths_are"] = "relative to the folder containing this file"
        blob = json.dumps(raw, indent=2).encode("utf-8")

        items.append(Downloadable(
            key="deployment_config.json",
            filename="deployment_config.json",
            path=cfg_path,
            kind="config",
            bytes=len(blob),
            sha256=hashlib.sha256(blob).hexdigest(),
            description=(
                "The safety settings: which model, the referral threshold, the "
                "deferral cutoff and the temperature. Required. The app will "
                "not start without it and will not guess these numbers."
            ),
            content=blob,
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
            "Development build. Use it as a second opinion, never as the only "
            "basis for a decision about a patient. It has not yet been tested "
            "on scans from a hospital or scanner outside its training sources, "
            "and no clinician has reviewed its output. It is not approved as a "
            "medical device anywhere and it is not for clinical use as the "
            "deciding factor. Read MODEL_CARD.md and LIMITATIONS.md before "
            "relying on it."
        ),
    }


def manifest_json(cfg: DeploymentConfig, items: List[Downloadable]) -> str:
    return json.dumps(manifest(cfg, items), indent=2)
