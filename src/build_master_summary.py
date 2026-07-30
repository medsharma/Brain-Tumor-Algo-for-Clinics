#!/usr/bin/env python3
"""Aggregate a completed --model both run's comparison_summary.json (+ optional
ood_results.json) into results/master_summary.json for the manuscript session.

Usage: python src/build_master_summary.py <run_dir>
  <run_dir> e.g. results/20260703_153000  (the timestamped top_dir main() creates)
  Run from the repo root — output path "results/master_summary.json" is CWD-relative.
"""
import json
import sys
from pathlib import Path

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]


def _per_seed_records(model_dict):
    seeds = model_dict["seed"]
    n = len(seeds)
    keys = list(model_dict.keys())
    return [{k: model_dict[k][i] for k in keys} for i in range(n)]


def main(run_dir: str) -> None:
    run_dir = Path(run_dir)
    comparison = json.loads((run_dir / "comparison_summary.json").read_text())

    vit_records = _per_seed_records(comparison["vit"])
    rn_records = _per_seed_records(comparison["resnet50"])

    ood_path = run_dir / "ood_results.json"
    ood = json.loads(ood_path.read_text()) if ood_path.exists() else None

    # figures/README.md's documented schema (and figures/risk_coverage.py's
    # master[model].get("aurc", []) lookup) expects "vit"/"resnet50" as
    # dict-of-lists at the top level, not nested per-seed records — this is
    # exactly comparison["vit"]/["resnet50"]'s existing shape, so pass it
    # through unchanged rather than reshaping it.
    master = {
        "class_names": CLASS_NAMES,
        "seeds": comparison["vit"]["seed"],
        "vit": comparison["vit"],
        "resnet50": comparison["resnet50"],
        "run_dir": str(run_dir),
        "per_seed_metrics": {
            "vit": vit_records,
            "resnet50": rn_records,
        },
        "mcnemar_per_seed": comparison["mcnemar_per_seed"],
        "ood": ood,
    }

    out_path = Path("results/master_summary.json")
    out_path.write_text(json.dumps(master, indent=2, default=str))
    print(f"Wrote {out_path} ({len(vit_records)} vit seeds, {len(rn_records)} resnet50 seeds, ood={'yes' if ood else 'no'})")


if __name__ == "__main__":
    main(sys.argv[1])
