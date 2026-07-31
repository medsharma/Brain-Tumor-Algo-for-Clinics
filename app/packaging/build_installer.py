#!/usr/bin/env python
"""Build BrainMRITriageSetup.exe, the small file a clinic downloads.

This is a different animal from the main build. The application bundle is 2.4 GB
because it contains PyTorch and five models. This is a few megabytes because it
contains neither: it is a download-and-unpack program with a progress bar, and
everything it needs is in the Python standard library.

Keeping it small is the entire point. Somebody clicking a link on a web page
gets a file in seconds and sees a progress window immediately, instead of
watching a browser download 2.4 GB with no feedback and no idea whether it is
working.

The server address is written into the exe at build time, because a compiled
binary cannot be rewritten per request the way the Python setup script is.
Set MRI_TRIAGE_SERVER_URL to the address people will actually reach:

    set MRI_TRIAGE_SERVER_URL=http://192.168.1.83:8765
    python app/packaging/build_installer.py

Rebuild it if that address changes. The installer accepts a URL as its first
argument too, which is the escape hatch when it does.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "app" / "packaging" / "setup_gui.py"
OUT_DIR = REPO_ROOT / "release"

#: Everything heavy. None of it is used by the setup program, and leaving any of
#: it in would turn a 10 MB download into a 300 MB one.
EXCLUDES = [
    "torch", "torchvision", "numpy", "scipy", "skimage", "sklearn", "cv2",
    "PIL", "pandas", "pyarrow", "matplotlib", "fastapi", "uvicorn", "starlette",
    "pydantic", "IPython", "notebook", "pytest", "_pytest", "imageio",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-url",
                    default=os.environ.get("MRI_TRIAGE_SERVER_URL", "").strip(),
                    help="address the installer downloads from")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter to build with (needs pyinstaller and tkinter)")
    args = ap.parse_args()

    server = args.server_url or "http://127.0.0.1:8765"
    print(f"building the installer against {server}")

    source = SOURCE.read_text(encoding="utf-8")
    if "@@SERVER_URL@@" not in source:
        raise SystemExit("setup_gui.py no longer has the @@SERVER_URL@@ placeholder")
    baked = source.replace("@@SERVER_URL@@", server)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        staged = tmp_path / "setup_gui.py"
        staged.write_text(baked, encoding="utf-8")

        command = [
            args.python, "-m", "PyInstaller",
            "--onefile",           # one file is the whole point
            "--windowed",          # no console window behind the progress bar
            "--name", "BrainMRITriageSetup",
            "--distpath", str(OUT_DIR),
            "--workpath", str(tmp_path / "build"),
            "--specpath", str(tmp_path),
            "--noconfirm",
        ]
        for module in EXCLUDES:
            command += ["--exclude-module", module]
        command.append(str(staged))

        result = subprocess.run(command)
        if result.returncode != 0:
            raise SystemExit("PyInstaller failed. Output is above.")

    exe = OUT_DIR / "BrainMRITriageSetup.exe"
    if not exe.is_file():
        raise SystemExit(f"expected {exe} and it is not there")

    size = exe.stat().st_size
    print(f"\n  {exe}")
    print(f"  {size / 1e6:.1f} MB")
    print(f"  downloads from {server}")
    if size > 60e6:
        print("\n  WARNING: that is far larger than expected for a downloader.")
        print("  Something heavy was bundled. Check the exclude list.")


if __name__ == "__main__":
    main()
