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
