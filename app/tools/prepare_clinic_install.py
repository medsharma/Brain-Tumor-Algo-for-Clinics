#!/usr/bin/env python
"""Turn a built app into something you can hand to a clinic on a USB stick.

`app/packaging/build_windows.bat` produces `dist/BrainMRITriage/`, which
deliberately contains no model weights. This finishes the job: it copies the
checkpoints the deployment config names, strips them, rewrites the config to
point at the copies with relative paths, and checks the result actually runs.

Why the checkpoints get stripped
--------------------------------
A training checkpoint carries `optimizer_state_dict` alongside the weights. A
clinic never trains, so that is dead weight: 459 MB on disk against 344 MB of
actual model. Across the 5-seed ensemble that is 575 MB of a USB stick spent on
Adam moments.

Stripping is **lossless**. Only `model_state_dict` is kept, and this tool
verifies afterwards that the stripped file produces bit-identical predictions to
the original. It is not quantisation and it is not pruning.

What this tool will NOT do
--------------------------
It will not convert weights to float16, even though that would halve the size
again. Session C measured the shipped pipeline as bit-identical to the validated
research pipeline, which is the strongest claim this project has. Half precision
moves the decision boundary and forfeits it, to save disk on a machine that has
plenty. If someone wants that trade later it needs its own measurement, not a
flag on a packaging script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "analysis" / "results" / "safety" / "deployment_config.json"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def strip_checkpoint(src: Path, dst: Path) -> Dict[str, object]:
    """Keep the weights, drop everything a clinic cannot use."""
    payload = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise SystemExit(f"{src.name}: not a training checkpoint, refusing to guess")

    state = payload["model_state_dict"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": state}, dst)

    before, after = src.stat().st_size, dst.stat().st_size
    return {
        "source": str(src), "output": str(dst),
        "bytes_before": before, "bytes_after": after,
        "saved_mb": round((before - after) / 1e6, 1),
        "sha256": sha256(dst),
        "dropped_keys": sorted(k for k in payload if k != "model_state_dict"),
    }


def verify_identical(original: Path, stripped: Path, backbone: str) -> float:
    """Same weights in, same numbers out. Refuse to ship if not."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from code import CLASS_NAMES, BrainTumorResNet50, BrainTumorViT  # noqa: E402

    cls = BrainTumorViT if backbone == "vit" else BrainTumorResNet50
    out = []
    for path in (original, stripped):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["model_state_dict"] if "model_state_dict" in payload else payload
        model = cls(num_classes=len(CLASS_NAMES))
        model.load_state_dict(state)
        model.eval()
        torch.manual_seed(0)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out.append(torch.softmax(model(x), dim=-1))
    diff = float((out[0] - out[1]).abs().max())
    if diff != 0.0:
        raise SystemExit(f"stripped checkpoint differs by {diff:.3e}. Refusing to ship.")
    return diff


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dist", type=Path, default=REPO_ROOT / "dist" / "BrainMRITriage",
                    help="the folder PyInstaller produced")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the bit-identity check (do not use for a real install)")
    args = ap.parse_args()

    if not args.dist.is_dir():
        raise SystemExit(f"{args.dist} does not exist. Build the app first:\n"
                         f"  python -m PyInstaller app/packaging/brain_mri_triage.spec --noconfirm")

    cfg = json.loads(args.config.read_text())
    models_dir = args.dist / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing {args.dist}")
    print(f"  backbone {cfg['chosen_backbone']}, seeds {cfg['chosen_seeds']}, "
          f"ensemble={cfg['ensemble']}\n")

    report: List[Dict[str, object]] = []
    new_checkpoints = []
    total_before = total_after = 0

    for entry in cfg["checkpoints"]:
        src = Path(entry["path"])
        if not src.is_file():
            raise SystemExit(f"checkpoint missing: {src}")
        dst = models_dir / src.name
        info = strip_checkpoint(src, dst)
        total_before += int(info["bytes_before"])
        total_after += int(info["bytes_after"])
        print(f"  {src.name:34s} {info['bytes_before']/1e6:6.0f} MB -> "
              f"{info['bytes_after']/1e6:6.0f} MB   saved {info['saved_mb']:5.1f} MB")

        if not args.skip_verify:
            d = verify_identical(src, dst, cfg["chosen_backbone"])
            print(f"    bit-identity check: max abs diff {d:.1e}  OK")

        report.append(info)
        e = dict(entry)
        e["path"] = str(Path("models") / src.name)   # relative, so the stick is portable
        e["sha256"] = info["sha256"]
        new_checkpoints.append(e)

    cfg["checkpoints"] = new_checkpoints
    cfg["checkpoint_paths_are"] = "relative to the folder containing this file"
    (args.dist / "deployment_config.json").write_text(json.dumps(cfg, indent=2),
                                                      encoding="utf-8")

    rejector = REPO_ROOT / "analysis" / "results" / "ood" / "rejector_config.json"
    if rejector.is_file():
        target = args.dist / "analysis" / "results" / "ood"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rejector, target / "rejector_config.json")
        print(f"\n  copied {rejector.name}")

    (args.dist / "install_report.json").write_text(
        json.dumps({"checkpoints": report,
                    "total_bytes_before": total_before,
                    "total_bytes_after": total_after,
                    "total_saved_mb": round((total_before - total_after) / 1e6, 1)},
                   indent=2), encoding="utf-8")

    print(f"\n  model files {total_before/1e9:.2f} GB -> {total_after/1e9:.2f} GB "
          f"(saved {(total_before-total_after)/1e6:.0f} MB, losslessly)")

    size = sum(f.stat().st_size for f in args.dist.rglob("*") if f.is_file())
    print(f"  whole install folder: {size/1e9:.2f} GB")
    print(f"\nReady. Copy {args.dist} to the clinic laptop and run BrainMRITriage.exe")


if __name__ == "__main__":
    main()
