#!/usr/bin/env python
"""Fit the deferral cutoff on the number the app actually computes.

The bug this exists to fix
--------------------------
The shipped cutoff was chosen from the prediction caches, where
``mutual_information`` is a **single model's** value: seed 42, 20 MC passes.

The app does something different. ``app/core/model.run_mc_dropout`` pools the
posterior samples of every model in the ensemble, so 5 seeds at T=20 contribute
100 samples, and the mutual information it reports therefore includes
seed-to-seed disagreement as well as dropout noise. That is a strictly larger
quantity.

So a cutoff fitted at the 10th percentile of single-model MI, applied to
ensemble MI, defers far more than 10%. Measured through the running app it was
deferring roughly a third of scans, and a user reported almost everything coming
back "UNCERTAIN". The threshold was not wrong by a little. It was fitted on the
wrong variable.

The fix is not to loosen the number until it looks better. It is to fit it on
the same quantity the app measures, by calling the app's own code.

Still fitted on internal validation only, and applied to everything else
unchanged. No BRISC label is read here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image                           # noqa: E402

from app.core import config as config_module     # noqa: E402
from app.core import model as model_module       # noqa: E402
from app.core import preprocess                  # noqa: E402

SAFETY = REPO_ROOT / "analysis" / "results" / "safety"
OVERLAP = REPO_ROOT / "analysis" / "results" / "brisc" / "brisc_overlap_per_image.csv"
BRISC = Path(r"C:\Users\medha\Downloads\archive (1)\brisc2025")
NOTUMOR = 3


def load_models(cfg):
    """Load exactly what the engine loads, hash checks and all."""
    modules = []
    for checkpoint in cfg.checkpoints:
        loaded = model_module.load_checkpoint(
            cfg.chosen_backbone, checkpoint.seed, Path(checkpoint.path),
            expected_sha256=getattr(checkpoint, "sha256", None),
        )
        modules.append(loaded.module)
    return modules


def score(modules, cfg, paths: List[Path], label: str) -> pd.DataFrame:
    """Exactly what the app computes, image by image."""
    rows = []
    transform = preprocess.build_transform()
    t0 = time.time()
    for i, path in enumerate(paths, 1):
        with Image.open(path) as image:
            tensor = preprocess.preprocess_pil(image, transform)
        result = model_module.run_mc_dropout(
            modules, tensor, T=cfg.mc_T, temperature=cfg.temperature,
            entropy_units=cfg.entropy_units, fast=True,
        )
        rows.append({
            "path": str(path),
            "p_tumor": result.p_tumor,
            "entropy": result.entropy,
            "mutual_information": result.mutual_information,
        })
        if i % 100 == 0 or i == len(paths):
            rate = i / max(time.time() - t0, 1e-9)
            print(f"    {label}: {i}/{len(paths)}  {rate:.1f} img/s  "
                  f"eta {(len(paths)-i)/max(rate,1e-9):5.0f}s", flush=True)
    return pd.DataFrame(rows)


def internal_paths(split: str) -> List[Path]:
    manifest = pd.read_csv(REPO_ROOT / "data" / "split_manifest.csv")
    rows = manifest[manifest["split"] == split]
    return [REPO_ROOT / str(p).replace("\\", "/") for p in rows["filepath"]]


def internal_labels(split: str) -> np.ndarray:
    manifest = pd.read_csv(REPO_ROOT / "data" / "split_manifest.csv")
    rows = manifest[manifest["split"] == split]
    return (rows["label"].to_numpy() != NOTUMOR)


def brisc_clean_paths(limit: int | None) -> tuple:
    flags = pd.read_csv(OVERLAP)
    clean = flags[flags["clean_vs_train"] == True]      # noqa: E712
    if limit:
        clean = clean.sample(n=min(limit, len(clean)), random_state=11)
    paths = [BRISC / str(p).replace("/", "\\") for p in clean["image_path"]]
    is_tumor = (clean["class_name"] != "notumor").to_numpy()
    return paths, is_tumor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defer-budget", type=float, default=0.10,
                    help="fraction of internal validation to send to a human")
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--brisc-limit", type=int, default=600)
    ap.add_argument("--write-config", action="store_true")
    args = ap.parse_args()

    cfg = config_module.load_config()
    print(f"config: {cfg.chosen_backbone}, seeds {list(cfg.chosen_seeds)}, "
          f"T={cfg.mc_T}, temperature {cfg.temperature:.4f}")
    print(f"current cutoff: {cfg.entropy_defer_threshold:.5f} "
          f"on '{cfg.raw.get('defer_signal', 'entropy')}'\n")

    modules = load_models(cfg)
    print(f"loaded {len(modules)} models\n")

    val_paths = internal_paths("val")
    if args.val_limit:
        val_paths = val_paths[:args.val_limit]
    print(f"scoring internal validation, {len(val_paths)} images")
    val = score(modules, cfg, val_paths, "val")

    # How wrong was the old number? Compare like for like.
    cached = pd.read_parquet(
        REPO_ROOT / "analysis/results/internal/predictions/vit_seed42_val.parquet")
    print("\n  mutual information, single model (cached) vs ensemble (what the app does)")
    print(f"    single-model  median {cached['mutual_information'].median():.5f}  "
          f"p90 {cached['mutual_information'].quantile(0.9):.5f}")
    print(f"    ensemble      median {val['mutual_information'].median():.5f}  "
          f"p90 {val['mutual_information'].quantile(0.9):.5f}")
    ratio = val["mutual_information"].median() / max(cached["mutual_information"].median(), 1e-12)
    print(f"    ensemble MI is about {ratio:.1f}x the single-model value")

    old_cut = float(cfg.entropy_defer_threshold)
    print(f"\n  the shipped cutoff {old_cut:.5f} defers "
          f"{100 * (val['mutual_information'] >= old_cut).mean():.1f}% of internal val")

    new_cut = float(np.quantile(val["mutual_information"], 1.0 - args.defer_budget))
    print(f"  refitted cutoff {new_cut:.5f} defers "
          f"{100 * (val['mutual_information'] >= new_cut).mean():.1f}%")

    report: Dict[str, object] = {
        "problem": ("the cutoff was fitted on single-model mutual information from "
                    "the prediction caches, but app/core/model.run_mc_dropout pools "
                    "samples across the whole ensemble and reports a larger value"),
        "fitted_on": "internal_val, using the app's own computation",
        "defer_budget": args.defer_budget,
        "old_cutoff": old_cut,
        "new_cutoff": new_cut,
        "single_model_mi_median": float(cached["mutual_information"].median()),
        "ensemble_mi_median": float(val["mutual_information"].median()),
        "internal_val": {
            "n": int(len(val)),
            "defer_rate_old": float((val["mutual_information"] >= old_cut).mean()),
            "defer_rate_new": float((val["mutual_information"] >= new_cut).mean()),
        },
    }

    # Check it on data the threshold was not fitted on.
    paths, is_tumor = brisc_clean_paths(args.brisc_limit)
    print(f"\nscoring BRISC clean subset, {len(paths)} images (never fitted on)")
    brisc = score(modules, cfg, paths, "brisc")
    brisc["is_tumor"] = is_tumor

    # What each budget actually costs, on data the cutoff was not fitted on.
    # One scoring pass, many candidate thresholds, so this is nearly free.
    print("\n  budget sweep. 'sent home' is the number that matters:")
    print(f"    {'budget':>7s} {'cutoff':>9s} {'val defer':>10s} "
          f"{'clean defer':>12s} {'sent home':>10s}")
    t_arr = brisc["is_tumor"].to_numpy()
    sweep = []
    for budget in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20):
        cut = float(np.quantile(val["mutual_information"], 1.0 - budget))
        d = brisc["mutual_information"].to_numpy() >= cut
        r = (~d) & (brisc["p_tumor"].to_numpy() >= cfg.tumor_threshold)
        c = (~d) & (~r)
        home = int((c & t_arr).sum())
        print(f"    {100*budget:6.0f}% {cut:9.5f} "
              f"{100*(val['mutual_information'] >= cut).mean():9.1f}% "
              f"{100*d.mean():11.1f}% {home:6d}/{int(t_arr.sum())}")
        sweep.append({"budget": budget, "cutoff": cut,
                      "clean_defer_rate": float(d.mean()),
                      "tumours_sent_home": home})
    report["budget_sweep"] = sweep

    for name, cut in (("old", old_cut), ("new", new_cut)):
        defer = brisc["mutual_information"].to_numpy() >= cut
        refer = (~defer) & (brisc["p_tumor"].to_numpy() >= cfg.tumor_threshold)
        clear = (~defer) & (~refer)
        t = brisc["is_tumor"].to_numpy()
        sent_home = int((clear & t).sum())
        print(f"\n  {name} cutoff {cut:.5f} on BRISC clean:")
        print(f"    UNCERTAIN {100*defer.mean():5.1f}%   "
              f"TUMOR {100*refer.mean():5.1f}%   NO TUMOR {100*clear.mean():5.1f}%")
        print(f"    tumours sent home {sent_home}/{int(t.sum())} "
              f"({100*sent_home/max(int(t.sum()),1):.2f}%)")
        report[f"brisc_clean_{name}"] = {
            "n": int(len(brisc)),
            "defer_rate": float(defer.mean()),
            "refer_rate": float(refer.mean()),
            "clear_rate": float(clear.mean()),
            "tumours_sent_home": sent_home,
            "n_tumour": int(t.sum()),
        }

    out = SAFETY / "defer_threshold_fit.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    if args.write_config:
        raw = json.loads(Path(cfg.source_path).read_text(encoding="utf-8"))
        raw["entropy_defer_threshold"] = new_cut
        raw.setdefault("defer_threshold_history", []).append({
            "previous": old_cut,
            "reason": ("fitted on single-model mutual information; the app pools "
                       "samples across the ensemble and reports a larger value, so "
                       "this over-deferred badly in real use"),
        })
        raw["defer_threshold_fitted_with"] = "app.core.model.run_mc_dropout on internal_val"
        Path(cfg.source_path).write_text(json.dumps(raw, indent=2), encoding="utf-8")
        print(f"updated {cfg.source_path}")


if __name__ == "__main__":
    main()
