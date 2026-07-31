#!/usr/bin/env python
"""Pack the built app into one file somebody can download and double-click.

`dist/BrainMRITriage/` is 2,695 files. That is not something to hand to a clinic
over a web page. This makes it one zip.

The person on the other end has no Python, and should never have to find out
what Python is. Their whole job is: download this, right-click, Extract All,
double-click the .exe.

Compression choice
------------------
Most of the bulk is model tensors and compiled DLLs, both already close to
incompressible. Deflating 2.4 GB of them costs minutes of CPU and saves a few
percent. Those are stored; everything else is deflated. The result is 2.40 GB
built in 24 seconds instead of several minutes for a nearly identical size.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Already-compressed formats. Deflating these is CPU for nothing.
_STORE_SUFFIXES = {".pth", ".dll", ".pyd", ".zip", ".png", ".jpg", ".jpeg"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dist", type=Path, default=REPO_ROOT / "dist" / "BrainMRITriage")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "release" / "BrainMRITriage-windows.zip")
    args = ap.parse_args()

    if not (args.dist / "BrainMRITriage.exe").is_file():
        raise SystemExit(f"{args.dist} has no BrainMRITriage.exe. Build it first.")
    if not (args.dist / "models").is_dir():
        raise SystemExit(
            "No models/ folder. Run app/tools/prepare_clinic_install.py first, "
            "or the download will be an app with no model in it.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(args.dist.rglob("*")) if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    print(f"packing {len(files)} files, {total / 1e9:.2f} GB")

    t0 = time.time()
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for i, path in enumerate(files, 1):
            arc = Path("BrainMRITriage") / path.relative_to(args.dist)
            method = (zipfile.ZIP_STORED if path.suffix.lower() in _STORE_SUFFIXES
                      else zipfile.ZIP_DEFLATED)
            zf.write(path, str(arc).replace("\\", "/"), compress_type=method)
            if i % 500 == 0:
                print(f"  {i}/{len(files)}  {time.time() - t0:.0f}s", flush=True)

    size = args.out.stat().st_size
    digest = hashlib.sha256()
    with args.out.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)

    (args.out.with_suffix(".json")).write_text(json.dumps({
        "filename": args.out.name,
        "bytes": size,
        "sha256": digest.hexdigest(),
        "files": len(files),
        "built_from": str(args.dist),
    }, indent=2), encoding="utf-8")

    print(f"\n  {args.out}")
    print(f"  {size / 1e9:.2f} GB in {time.time() - t0:.0f}s")
    print(f"  sha256 {digest.hexdigest()}")
    print("\n  The running app offers this automatically on its download page.")


if __name__ == "__main__":
    main()
