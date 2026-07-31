"""Where the app keeps its local data.

Everything stays on the laptop. There is no server, no bucket, no sync.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Override the data directory. Useful for tests and for putting the audit log
#: on an encrypted volume.
DATA_DIR_ENV = "MRI_CLINIC_DATA_DIR"

APP_DIR_NAME = "BrainMRITriage"


def data_dir() -> Path:
    """The per-user directory holding the audit log and the install key.

    Windows uses ``%LOCALAPPDATA%``, macOS uses ``~/Library/Application Support``,
    everything else uses ``~/.local/share``. Created on first use.
    """
    override = os.environ.get(DATA_DIR_ENV, "").strip()
    if override:
        path = Path(override).expanduser()
    elif sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        path = Path(base) / APP_DIR_NAME
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    else:
        base = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
        path = Path(base) / APP_DIR_NAME

    path.mkdir(parents=True, exist_ok=True)
    return path


def audit_dir() -> Path:
    path = data_dir() / "audit"
    path.mkdir(parents=True, exist_ok=True)
    return path


def repo_root() -> Path:
    """Where bundled read-only files live: configs, static assets.

    Two different answers depending on how the app is running.

    From a source checkout this is the repository root, three levels up from
    this file. Inside a PyInstaller build there is no repository: the bundled
    data was unpacked next to the executable under ``_internal``, and
    ``sys._MEIPASS`` is the only reliable way to find it. Guessing from
    ``__file__`` lands in the wrong place, which is how session B's rejector
    config went missing from the packaged build and silently downgraded the
    input check to unfitted limits.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parents[2]
