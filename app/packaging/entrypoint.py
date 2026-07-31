"""PyInstaller entry point.

A separate file from ``app/launch.py`` because a frozen build needs its own
setup before anything else imports: the repository root has to be on the path,
and the bundled config has to be findable relative to the executable rather
than relative to a source checkout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bundle_root() -> Path:
    """Where the bundled data lives, frozen or not."""
    if getattr(sys, "frozen", False):
        # PyInstaller unpacks datas next to the executable in onedir mode.
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[2]


def main() -> int:
    root = _bundle_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # If the operator has not pointed the app at a config, look for one next to
    # the executable first. That is where the installer puts the real config
    # from session A, so a clinic install never falls back to the stub.
    if not os.environ.get("MRI_CLINIC_CONFIG"):
        candidates = [
            Path(sys.executable).parent / "deployment_config.json",
            root / "analysis" / "results" / "safety" / "deployment_config.json",
        ]
        for candidate in candidates:
            if candidate.is_file():
                os.environ["MRI_CLINIC_CONFIG"] = str(candidate)
                break

    from app.launch import main as launch_main

    return launch_main()


if __name__ == "__main__":
    raise SystemExit(main())
