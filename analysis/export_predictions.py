#!/usr/bin/env python3
"""
analysis/export_predictions.py

Dumps raw MC-Dropout predictions per (model, seed) to
analysis/results/<model>_seed<seed>_predictions.json, in the schema
figures/README.md documents as the single highest-leverage missing artifact
for the manuscript figure scripts (calibration_reliability.py,
risk_coverage.py, confusion_matrix.py, roc_curves.py all consume this glob).

This is a thin wrapper: all it does is call src/code.py::evaluate_with_uncertainty
on the test split for each trained checkpoint in the latest completed run and
serialize the result, tagged with analysis/common.py::run_provenance() so
downstream consumers can see at a glance whether a given file is smoke-test
or real.

Usage:
    python analysis/export_predictions.py [--mc-T 20]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import (  # noqa: E402
    CLASS_NAMES,
    DEVICE,
    RESULTS_OUT_DIR,
    all_seed_dirs,
    build_dataloaders_from_manifest,
    find_latest_run,
    get_manifest,
    load_model_from_seed_dir,
    run_provenance,
)
from code import evaluate_with_uncertainty  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc-T", type=int, default=20)
    args = parser.parse_args()

    run_dir = find_latest_run()
    if run_dir is None:
        print("ERROR: no completed run found under results/.", file=sys.stderr)
        sys.exit(1)
    print(f"Using run: {run_dir}")

    manifest_df = get_manifest()
    test_loader = build_dataloaders_from_manifest(manifest_df, batch_size=32, num_workers=0).get("test")
    if test_loader is None:
        print("ERROR: manifest has no test split.", file=sys.stderr)
        sys.exit(1)

    RESULTS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    for model_name in ("vit", "resnet50"):
        for seed_dir in all_seed_dirs(run_dir, model_name):
            seed = int(seed_dir.name.split("_")[1])
            print(f"\n=== {model_name} seed={seed} ===")
            model, ckpt_path = load_model_from_seed_dir(model_name, seed_dir, device=DEVICE)
            provenance = run_provenance(run_dir, seed_dir)

            r = evaluate_with_uncertainty(model, test_loader, DEVICE, T=args.mc_T)

            out = {
                "model": model_name,
                "seed": seed,
                "class_names": CLASS_NAMES,
                "mc_T": args.mc_T,
                "provenance": provenance,
                "y_true": r["labels"].tolist(),
                "y_pred": r["predictions"].tolist(),
                "mean_probs": r["mean_probs"].tolist(),
                "entropy": r["entropy"].tolist(),
            }
            out_path = RESULTS_OUT_DIR / f"{model_name}_seed{seed}_predictions.json"
            with open(out_path, "w") as f:
                json.dump(out, f)
            print(f"  -> {out_path}  (n={len(out['y_true'])}, smoke={provenance['is_smoke_test_run']})")
            written.append(str(out_path))

            del model

    print(f"\nWrote {len(written)} prediction file(s).")


if __name__ == "__main__":
    main()
