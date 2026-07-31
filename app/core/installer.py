"""Builds the one-file installer this server hands out.

The download list on its own is not a way to install anything. It is seven
files, a note about where each goes, and a hash to check by hand. That is fine
as reference material and useless as a setup path for a clinic.

This turns it into: download one small Python file, run it, get the same web
page running on your own machine, offline. The server's own address is baked in
when the script is generated, so the person running it configures nothing.

What the generated script does, in order:

1. checks the Python version
2. downloads the application source as a zip from the server it came from
3. downloads the model files and the settings files, verifying each SHA-256
4. installs the Python dependencies
5. starts the app and opens the browser

After that first run it needs no network at all, which is the point.

Nothing here executes anything. It writes a script and hands it over. Whoever
runs it can read it first, and it is deliberately written to be readable.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import List

from . import paths
from .downloads import Downloadable

#: Directories copied into the source zip, relative to the repo root.
_SOURCE_DIRS = ("app",)

#: Individual files the app reaches through ``importlib``, which is why they
#: cannot be discovered by walking imports.
_SOURCE_FILES = (
    "src/input_validation.py",
    "src/explain_runtime.py",
)

#: Never ship these. Tests and tools are not needed to run the app, caches are
#: noise, and the built package is gigabytes.
_SKIP_PARTS = {"__pycache__", "tests", "packaging", ".pytest_cache"}


def _include(rel: Path) -> bool:
    return not any(part in _SKIP_PARTS for part in rel.parts)


def build_source_zip() -> bytes:
    """Zip the application source. Small: a few hundred KB, no weights."""
    root = paths.repo_root()
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for directory in _SOURCE_DIRS:
            base = root / directory
            if not base.is_dir():
                continue
            for path in sorted(base.rglob("*")):
                if not path.is_file():
                    continue
                rel = path.relative_to(root)
                if not _include(rel):
                    continue
                zf.writestr(str(rel).replace("\\", "/"), path.read_bytes())

        for name in _SOURCE_FILES:
            path = root / name
            if path.is_file():
                zf.writestr(name, path.read_bytes())

        # `src` is a namespace package with no __init__.py. That works from a
        # checkout and is fragile everywhere else, so give the extracted copy a
        # real one rather than relying on PEP 420 on someone else's machine.
        zf.writestr("src/__init__.py", "")

    return buffer.getvalue()


def build_installer(base_url: str, items: List[Downloadable]) -> str:
    """Generate the setup script, with this server's address written into it."""
    manifest = [
        {"key": i.key, "filename": i.filename, "kind": i.kind,
         "sha256": i.sha256, "bytes": i.bytes}
        for i in items
    ]
    manifest_json = json.dumps(manifest, indent=4)
    total_gb = sum(i.bytes for i in items) / 1e9

    return _TEMPLATE.format(
        base_url=base_url.rstrip("/"),
        manifest=manifest_json,
        total_gb=f"{total_gb:.2f}",
    )


_TEMPLATE = '''#!/usr/bin/env python3
"""Set up Brain MRI Triage on this machine, then run it offline.

You downloaded this from {base_url}. That address is written into this file, so
there is nothing to configure. Run it and answer nothing:

    python setup_brain_mri_triage.py

It downloads about {total_gb} GB once. After that the tool needs no internet at
all, which is the whole point of it.

READ THIS BEFORE USING IT ON A PATIENT
--------------------------------------
This tool has never been tested on data it did not train on, and no clinician
has reviewed a single one of its outputs. It is a research prototype. It is not
a diagnosis and it has no regulatory clearance anywhere.

On the closest thing to unseen data available, 3 real tumours in 1,476 were
shown as "no tumour", and all three were shown confidently. No confidence score
catches those. If the patient has symptoms, refer them anyway.
"""
import hashlib
import io
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "{base_url}"
FILES = {manifest}

HERE = Path(__file__).resolve().parent
TARGET = HERE / "brain-mri-triage"


def say(message):
    print(message, flush=True)


def fetch(url, destination, expected_sha256=None):
    """Download with a progress line, then verify."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file() and expected_sha256:
        if sha256_of(destination) == expected_sha256:
            say("    already here and verified, skipping")
            return

    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with open(destination, "wb") as handle:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                handle.write(block)
                digest.update(block)
                done += len(block)
                if total:
                    pct = 100.0 * done / total
                    print("\\r    {{:.0f}} MB of {{:.0f}} MB  ({{:.0f}}%)".format(
                        done / 1e6, total / 1e6, pct), end="", flush=True)
        print()

    if expected_sha256 and digest.hexdigest() != expected_sha256:
        destination.unlink(missing_ok=True)
        raise SystemExit(
            "    DOWNLOAD CORRUPTED: {{}}\\n"
            "    The file did not match its checksum and has been deleted.\\n"
            "    Run this script again.".format(destination.name))


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if sys.version_info < (3, 10):
        raise SystemExit(
            "Python 3.10 or newer is needed. This is {{}}.".format(
                ".".join(str(n) for n in sys.version_info[:3])))

    say("")
    say("  Brain MRI Triage, local setup")
    say("  Downloading from " + BASE_URL)
    say("  Installing into " + str(TARGET))
    say("")

    TARGET.mkdir(parents=True, exist_ok=True)

    say("  [1/4] Application")
    source_zip = TARGET / "_source.zip"
    fetch(BASE_URL + "/api/source.zip", source_zip)
    with zipfile.ZipFile(source_zip) as zf:
        zf.extractall(TARGET)
    source_zip.unlink(missing_ok=True)
    say("    unpacked")

    say("")
    say("  [2/4] Model and settings files")
    for entry in FILES:
        say("  " + entry["filename"] + "  ({{:.0f}} MB)".format(entry["bytes"] / 1e6))
        if entry["kind"] == "model":
            destination = TARGET / "models" / entry["filename"]
        else:
            destination = TARGET / entry["filename"]
        fetch(BASE_URL + "/api/downloads/" + entry["key"], destination, entry["sha256"])

    # The app looks for the settings file at this path relative to itself. The
    # server already generated it to match the files it served, including their
    # checksums, so it is moved into place unchanged. Editing it here would
    # break the checksum it was downloaded under.
    settings_home = TARGET / "analysis" / "results" / "safety"
    settings_home.mkdir(parents=True, exist_ok=True)
    config_path = TARGET / "deployment_config.json"
    if config_path.is_file():
        (settings_home / "deployment_config.json").write_bytes(config_path.read_bytes())
        config_path.unlink(missing_ok=True)

    rejector = TARGET / "rejector_config.json"
    if rejector.is_file():
        ood_home = TARGET / "analysis" / "results" / "ood"
        ood_home.mkdir(parents=True, exist_ok=True)
        ood_home.joinpath("rejector_config.json").write_bytes(rejector.read_bytes())
        rejector.unlink(missing_ok=True)
    say("    settings written")

    say("")
    say("  [3/4] Python packages. This can take a few minutes.")
    requirements = TARGET / "app" / "requirements.txt"
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
        cwd=str(TARGET))
    if result.returncode != 0:
        raise SystemExit(
            "\\n  Installing the Python packages failed. The error is above.\\n"
            "  Everything else is already downloaded, so run this script again\\n"
            "  once that is fixed and it will pick up where it stopped.")

    say("")
    say("  [4/4] Starting")
    say("")
    say("  From now on, to start it again:")
    say("      cd " + str(TARGET))
    say("      python -m app")
    say("")
    say("  It needs no internet from here on.")
    say("")

    os.chdir(str(TARGET))
    subprocess.run([sys.executable, "-m", "app"], cwd=str(TARGET))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        say("\\n  Stopped.")
    except SystemExit as exc:
        if exc.code not in (0, None):
            say("\\n" + str(exc.code))
        raise
'''
