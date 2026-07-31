#!/usr/bin/env python
"""Prove the BUILT package works, by running it the way a clinic will.

Why this exists
---------------
`app/tests/` is thorough and it tests the source tree. It cannot see the one
class of bug that only exists after PyInstaller has run: a module that was
excluded from the bundle, or an import that static analysis could not follow.

Both happened here, on the same build, and neither was visible from the source
tests:

* `src/input_validation.py` and `src/explain_runtime.py` are reached with
  `importlib.import_module`, so PyInstaller did not collect them. The frozen app
  compiled, launched, and then refused to start because readiness found both
  adapters stubbed.
* `scipy` and `scikit-image` were on the exclude list under the comment
  "nothing here is imported by the app". `src/input_validation.py` imports them
  inside a function to find the brain-shaped region. The frozen app started,
  looked completely healthy, and **rejected 100% of real brain MRI** with "the
  file could not be read as an image". The validator fails closed on any
  exception, which is correct, and which turned a missing dependency into what
  looked like a data problem.

A packaging bug that rejects every scan is indistinguishable from a broken
scanner to the person at the laptop. Run this before any package leaves the
building.

Usage
-----
    python app/tools/smoke_test_package.py
    python app/tools/smoke_test_package.py --dist dist/BrainMRITriage --port 8899

Exit code 0 means the package is fit to copy onto a USB stick.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
BRISC = Path(r"C:\Users\medha\Downloads\archive (1)\brisc2025\classification_task\test")

# (folder, how many, what the app must NOT say)
CASES: Tuple[Tuple[str, int], ...] = (("glioma", 4), ("meningioma", 4),
                                      ("pituitary", 4), ("no_tumor", 4))


def post(port: int, path: Path) -> Dict[str, object]:
    boundary = uuid.uuid4().hex
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode(),
        b"Content-Type: image/jpeg\r\n\r\n", path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode()])
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/analyze", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def wait_for(port: int, proc: subprocess.Popen, timeout: int = 240) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"the app exited during startup with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=5):
                return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(3)
    raise SystemExit(f"the app did not answer on port {port} within {timeout}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dist", type=Path, default=REPO_ROOT / "dist" / "BrainMRITriage")
    ap.add_argument("--port", type=int, default=8899)
    args = ap.parse_args()

    exe = args.dist / "BrainMRITriage.exe"
    if not exe.is_file():
        raise SystemExit(f"{exe} not found. Build first.")
    if not (args.dist / "models").is_dir():
        raise SystemExit("no models/ folder. Run app/tools/prepare_clinic_install.py first.")

    print(f"starting {exe}")
    proc = subprocess.Popen([str(exe), "--no-browser", "--port", str(args.port)],
                            cwd=str(args.dist),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    failures: List[str] = []
    try:
        wait_for(args.port, proc)

        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/api/status", timeout=30) as r:
            status = json.loads(r.read())
        print(f"  state: {status['state']}")
        for adapter in ("validator", "explainer"):
            pass

        calls: Dict[str, int] = {}
        n_read = 0
        for folder, count in CASES:
            d = BRISC / folder
            if not d.is_dir():
                print(f"  skip {folder}: not on this machine")
                continue
            for p in sorted(d.iterdir())[:count]:
                out = post(args.port, p)
                key = str(out["call_key"])
                calls[key] = calls.get(key, 0) + 1
                if key != "cannot_read":
                    n_read += 1
                else:
                    failures.append(
                        f"{folder}/{p.name}: rejected as unreadable "
                        f"({out.get('validator_reason')})")
                if out.get("validator_is_stub"):
                    failures.append("the input validator is a STUB in the built package")
                if out.get("explainer_is_stub"):
                    failures.append("the heatmap explainer is a STUB in the built package")

        print(f"  calls: {calls}")
        print(f"  real brain MRI the package could read: {n_read} / "
              f"{sum(c for _, c in CASES)}")

        # a package that reads nothing is the exact bug this file exists to catch
        if n_read == 0:
            failures.append("the package rejected EVERY real brain MRI")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()

    if failures:
        print("\nPACKAGE IS NOT FIT TO SHIP:")
        for f in dict.fromkeys(failures):
            print(f"  - {f}")
        sys.exit(1)

    print("\nPackage smoke test passed. Safe to copy onto a USB stick.")


if __name__ == "__main__":
    main()
