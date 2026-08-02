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
import os
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


def check_dicom(port: int) -> List[str]:
    """Does the BUNDLE read DICOM, or only the source tree?

    ``pydicom`` is imported lazily, on purpose, so that a build without it
    degrades to the old "export a JPEG from your viewer" refusal instead of
    failing to start. That is the same property that let scipy and scikit-image
    go missing from an earlier build unnoticed: nothing static sees the import,
    the app looks healthy, and the failure only appears on a clinic laptop with
    a real file in front of a real patient.

    So: build a DICOM here, post it, and insist the package converted it.
    """
    try:
        import numpy as np
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    except ImportError:
        print("  skip DICOM check: pydicom is not installed in THIS interpreter")
        return []

    rows = cols = 224
    y, x = np.ogrid[:rows, :cols]
    inside = np.clip(1.0 - (((x - cols / 2) / (cols * 0.36)) ** 2
                            + ((y - rows / 2) / (rows * 0.42)) ** 2), 0.0, 1.0)
    array = (inside * 900 + 60).astype(np.uint16)

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "MR"
    ds.Rows, ds.Columns = rows, cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope, ds.RescaleIntercept = 1.0, 0.0
    ds.WindowCenter, ds.WindowWidth = 500, 900
    ds.PixelData = array.tobytes()
    ds.is_little_endian, ds.is_implicit_VR = True, False

    import io as _io
    buffer = _io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)

    temp = Path(os.environ.get("TEMP", ".")) / "_smoke_slice.dcm"
    temp.write_bytes(buffer.getvalue())
    try:
        out = post(port, temp)
    except Exception as exc:  # noqa: BLE001
        return [f"the package failed on a DICOM file: {exc}"]
    finally:
        temp.unlink(missing_ok=True)

    if out.get("source_format") != "dicom":
        return ["the package did not convert a DICOM file. pydicom is probably "
                "missing from the bundle, which silently sends every clinic back "
                "to exporting JPEGs by hand"]
    print(f"  DICOM: converted and called {out.get('call_key')!r}")
    return []


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
        if status["state"] == "refused":
            failures.append("the package refused to start")

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

        failures.extend(check_dicom(args.port))
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
