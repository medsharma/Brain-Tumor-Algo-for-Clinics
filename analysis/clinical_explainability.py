#!/usr/bin/env python
"""Session D, parts 2 and 3: are the heatmaps pointing at the tumour?

Part 1 shipped `src/explain_runtime.py` and proved it numerically identical to
the validated research code. That says the app draws the same picture. It says
nothing about whether the picture is right.

This measures whether it is right, against ground truth. BRISC ships 4,793
pixel-level tumour masks. So "does the model look at the tumour" is a
measurement here, not an opinion.

What is measured
----------------
Part 2, localisation:
  * pointing game  - does the hottest pixel land inside the tumour mask
  * mass in mask   - what share of heatmap activation is inside the mask,
                     against the share of the image the mask occupies (chance)
  * IoU            - thresholded heatmap against the mask, swept over thresholds
  broken down by backbone, class, plane, tumour size, and whether the model's
  own prediction was right.

Part 3, faithfulness:
  * deletion / insertion curves, with a random-heatmap baseline.
    **Read the caveat on these.** A random heatmap is spatially scattered and a
    real one is spatially concentrated, so deleting each removes a very
    different kind of structure from the image. The comparison is confounded by
    concentration, not purely by importance, and the two curves can disagree
    with each other. They are reported, and they are not the headline.
  * model-randomisation sanity check (Adebayo et al. 2018): re-initialise the
    weights the explanation actually reads and confirm the heatmap changes. A
    saliency map that survives a randomised model is tracking image structure,
    not model reasoning. The two paths read different weights, so they are
    randomised differently. See `randomise_for_explanation`.
  * does localisation quality predict correctness? If not, the overlay is
    decoration and the UI must say so.

The contamination rule
----------------------
BRISC is ~80% the training set republished. Session A published per-image
overlap flags. Every headline number here is computed on the clean subset
(nearest internal *training* image further than Hamming 5) and the contaminated
number is reported beside it, labelled. A localisation score on an image the
model trained on is not evidence about new data.

Inference only. No backward pass reaches a weight: Grad-CAM takes gradients with
respect to a feature map, under `torch.enable_grad`, and no optimiser exists in
this file. Nothing is fitted on BRISC.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from code import (  # noqa: E402
    CLASS_NAMES,
    DEVICE,
    BrainTumorResNet50,
    BrainTumorViT,
    get_transforms,
)
from explain_runtime import generate_heatmap, overlay_heatmap  # noqa: E402

BRISC = Path(r"C:\Users\medha\Downloads\archive (1)\brisc2025")
CKPT_ROOT = ROOT / "results" / "20260703_155524"
OVERLAP_CSV = ROOT / "analysis" / "results" / "brisc" / "brisc_overlap_per_image.csv"
PRED_DIR = ROOT / "analysis" / "results" / "brisc" / "predictions"
OUT_DIR = ROOT / "analysis" / "results" / "explainability_clinical"

MODEL_CLASSES = {"resnet50": BrainTumorResNet50, "vit": BrainTumorViT}
NOTUMOR_INDEX = CLASS_NAMES.index("notumor")
IOU_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
INPUT_HW = 224


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def build_pairs() -> pd.DataFrame:
    """Every segmentation image/mask pair, joined to overlap flags."""
    rows: List[Dict[str, object]] = []
    for split in ("train", "test"):
        img_dir = BRISC / "segmentation_task" / split / "images"
        mask_dir = BRISC / "segmentation_task" / split / "masks"
        for img in sorted(img_dir.iterdir()):
            mask = mask_dir / (img.stem + ".png")
            if not mask.exists():
                raise AssertionError(f"no mask for {img.name}")
            rows.append({
                "stem": img.stem,
                "seg_split": split,
                "image": str(img),
                "mask": str(mask),
            })
    df = pd.DataFrame(rows)

    overlap = pd.read_csv(OVERLAP_CSV)
    overlap["stem"] = overlap["image_path"].str.replace("\\", "/", regex=False).str.split("/").str[-1].str.replace(".jpg", "", regex=False)
    keep = ["stem", "clean_vs_train", "clean_vs_any", "d_train", "class_name", "plane"]
    df = df.merge(overlap[keep], on="stem", how="left")

    missing = int(df["clean_vs_train"].isna().sum())
    if missing:
        # segmentation images that are not in the classification task
        print(f"  note: {missing} of {len(df)} mask pairs have no classification-task "
              f"counterpart, so no overlap flag. Dropped from clean-subset numbers.")
    return df


def attach_predictions(df: pd.DataFrame, model: str, seed: int) -> pd.DataFrame:
    path = PRED_DIR / f"{model}_seed{seed}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"prediction cache missing: {path}")
    pred = pd.read_parquet(path)
    pred["stem"] = pred["image_path"].str.replace("\\", "/", regex=False).str.split("/").str[-1].str.replace(".jpg", "", regex=False)
    cols = ["stem", "pred_label", "true_label", "p_tumor", "entropy"]
    out = df.merge(pred[cols], on="stem", how="left")
    out["pred_correct"] = out["pred_label"] == out["true_label"]
    out["missed_tumour"] = (out["true_label"] != NOTUMOR_INDEX) & (out["pred_label"] == NOTUMOR_INDEX)
    return out


def load_checkpoint(model_name: str, seed: int) -> torch.nn.Module:
    path = CKPT_ROOT / model_name / f"seed_{seed}" / f"best_{model_name}_seed{seed}.pth"
    model = MODEL_CLASSES[model_name](num_classes=len(CLASS_NAMES)).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def load_image_tensor(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return get_transforms("test")(img)


def load_mask(path: str, hw: int = INPUT_HW) -> np.ndarray:
    """Binary mask resized to the model input grid, nearest neighbour."""
    m = Image.open(path).convert("L").resize((hw, hw), Image.NEAREST)
    return (np.asarray(m) > 127).astype(np.float32)


# ---------------------------------------------------------------------------
# part 2 - localisation
# ---------------------------------------------------------------------------
def localisation_metrics(heat: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Pointing game, mass in mask, IoU sweep, for one image."""
    area = float(mask.mean())          # chance baseline for mass
    total = float(heat.sum())

    flat = int(np.argmax(heat))
    py, px = divmod(flat, heat.shape[1])
    pointing = bool(mask[py, px] > 0)

    mass_in = float((heat * mask).sum() / total) if total > 0 else float("nan")

    ious: Dict[str, float] = {}
    for t in IOU_THRESHOLDS:
        binm = heat >= t
        inter = float((binm & (mask > 0)).sum())
        union = float((binm | (mask > 0)).sum())
        ious[f"iou@{t:.1f}"] = inter / union if union > 0 else float("nan")

    return {
        "pointing_hit": float(pointing),
        "mass_in_mask": mass_in,
        "mask_area_frac": area,
        "mass_lift_over_chance": (mass_in / area) if area > 0 else float("nan"),
        "best_iou": float(np.nanmax(list(ious.values()))),
        **ious,
    }


@torch.no_grad()
def _noop() -> None:
    return None


def run_localisation(
    df: pd.DataFrame,
    model_name: str,
    seed: int,
    limit: Optional[int] = None,
    log_every: int = 250,
) -> pd.DataFrame:
    model = load_checkpoint(model_name, seed)
    work = df if limit is None else df.head(limit)
    recs: List[Dict[str, object]] = []

    import time
    t0 = time.time()
    for i, row in enumerate(work.itertuples(index=False), start=1):
        x = load_image_tensor(row.image).to(DEVICE)
        heat = generate_heatmap(model, x, model_name)
        mask = load_mask(row.mask)
        m = localisation_metrics(heat, mask)
        m.update({
            "stem": row.stem,
            "model": model_name,
            "seed": seed,
            "seg_split": row.seg_split,
            "class_name": row.class_name,
            "plane": row.plane,
            "clean_vs_train": row.clean_vs_train,
            "pred_correct": getattr(row, "pred_correct", None),
            "missed_tumour": getattr(row, "missed_tumour", None),
            "entropy": getattr(row, "entropy", None),
        })
        recs.append(m)
        if i % log_every == 0 or i == len(work):
            rate = i / max(time.time() - t0, 1e-9)
            print(f"    {i}/{len(work)}  {rate:.1f} img/s  eta {(len(work)-i)/max(rate,1e-9):5.0f}s",
                  flush=True)
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# part 3 - faithfulness
# ---------------------------------------------------------------------------
@torch.no_grad()
def _p_tumour(model: torch.nn.Module, x: torch.Tensor) -> float:
    """Deterministic tumour probability. Dropout off, single pass."""
    was = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        p = torch.softmax(model(x.unsqueeze(0) if x.dim() == 3 else x), dim=-1)[0]
        return float(p[:NOTUMOR_INDEX].sum())
    finally:
        for m, f in was:
            m.training = f


def deletion_insertion(
    model: torch.nn.Module,
    x: torch.Tensor,
    heat: np.ndarray,
    steps: int = 20,
) -> Dict[str, List[float]]:
    """Delete/insert pixels most-important first. Returns both curves."""
    order = np.argsort(heat.ravel())[::-1]          # most important first
    n = order.size
    per = max(1, n // steps)

    flat_x = x.clone().reshape(x.shape[0], -1)
    blurred = torch.zeros_like(flat_x)              # deletion target: zeroed input

    dele, ins = [], []
    for s in range(steps + 1):
        k = min(s * per, n)
        idx = torch.as_tensor(order[:k].copy(), device=x.device, dtype=torch.long)

        d = flat_x.clone()
        d[:, idx] = blurred[:, idx]
        dele.append(_p_tumour(model, d.reshape(x.shape)))

        j = blurred.clone()
        j[:, idx] = flat_x[:, idx]
        ins.append(_p_tumour(model, j.reshape(x.shape)))
    return {"deletion": dele, "insertion": ins}


def _auc(curve: Sequence[float]) -> float:
    trap = getattr(np, "trapezoid", np.trapz)
    return float(trap(curve, dx=1.0 / (len(curve) - 1)))


def randomise_for_explanation(
    model: torch.nn.Module, model_name: str, seed: int = 0
) -> torch.nn.Module:
    """Copy of the model with the weights the EXPLANATION uses re-initialised.

    Adebayo et al. 2018: a saliency map that survives randomising the model is
    tracking image structure, not model reasoning. The test only means something
    if you randomise the weights that particular explanation actually reads.

    The two paths read different weights, so they need different randomisation:

      resnet50 / Grad-CAM      gradients flow from the logit back through the
                               classification head to the layer4 feature map, so
                               randomising the head is a valid test.

      vit / attention rollout  reads encoder attention weights ONLY. It never
                               touches the head. Randomising the head leaves the
                               map bit-identical and returns correlation 1.000,
                               which looks like a catastrophic failure and is in
                               fact a vacuous test. The encoder blocks must be
                               randomised instead.

    An earlier version of this file randomised the head for both and reported
    corr 1.000 for ViT as if it were a result. It was not.
    """
    import copy
    m = copy.deepcopy(model)
    g = torch.Generator(device="cpu").manual_seed(seed)

    def reinit(module: torch.nn.Module) -> None:
        for mod in module.modules():
            if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)):
                w = torch.empty_like(mod.weight, device="cpu")
                if w.dim() >= 2:
                    torch.nn.init.kaiming_uniform_(w, a=5 ** 0.5, generator=g)
                else:
                    w.normal_(0.0, 0.02, generator=g)
                mod.weight.data.copy_(w.to(mod.weight.device))
                if getattr(mod, "bias", None) is not None:
                    mod.bias.data.zero_()

    if model_name == "vit":
        # rollout reads encoder self-attention; randomise the encoder blocks
        reinit(m.backbone.encoder.layers)
    else:
        reinit(m.backbone.fc)
    return m


def run_faithfulness(
    df: pd.DataFrame,
    model_name: str,
    seed: int,
    n_sample: int = 300,
    steps: int = 20,
    rng_seed: int = 20260731,
) -> Dict[str, object]:
    model = load_checkpoint(model_name, seed)
    rng = np.random.default_rng(rng_seed)

    pool = df.dropna(subset=["clean_vs_train"])
    pool = pool[pool["clean_vs_train"]]
    if len(pool) == 0:
        pool = df
    take = pool.sample(n=min(n_sample, len(pool)), random_state=rng_seed)

    real_del, real_ins, rand_del, rand_ins = [], [], [], []
    rand_corr: List[float] = []

    rmodel = randomise_for_explanation(model, model_name, seed=rng_seed)

    import time
    t0 = time.time()
    for i, row in enumerate(take.itertuples(index=False), start=1):
        x = load_image_tensor(row.image).to(DEVICE)
        heat = generate_heatmap(model, x, model_name)

        c = deletion_insertion(model, x, heat, steps=steps)
        real_del.append(_auc(c["deletion"]))
        real_ins.append(_auc(c["insertion"]))

        rheat = rng.random(heat.shape).astype(np.float32)
        c = deletion_insertion(model, x, rheat, steps=steps)
        rand_del.append(_auc(c["deletion"]))
        rand_ins.append(_auc(c["insertion"]))

        # model randomisation sanity check
        rh = generate_heatmap(rmodel, x, model_name)
        if heat.std() > 0 and rh.std() > 0:
            rand_corr.append(float(np.corrcoef(heat.ravel(), rh.ravel())[0, 1]))

        if i % 50 == 0 or i == len(take):
            rate = i / max(time.time() - t0, 1e-9)
            print(f"    {i}/{len(take)}  {rate:.2f} img/s  eta {(len(take)-i)/max(rate,1e-9):5.0f}s",
                  flush=True)

    def stat(v: Sequence[float]) -> Dict[str, float]:
        a = np.asarray(v, dtype=float)
        return {"mean": float(a.mean()), "std": float(a.std()), "n": int(a.size)}

    return {
        "n_sampled": int(len(take)),
        "subset": "clean_vs_train" if pool is not df else "all (no clean subset available)",
        "steps": steps,
        "deletion_auc_real": stat(real_del),
        "deletion_auc_random": stat(rand_del),
        "insertion_auc_real": stat(real_ins),
        "insertion_auc_random": stat(rand_ins),
        "model_randomisation_corr": stat(rand_corr),
        "reading": (
            "Deletion AUC lower is better: a faithful map kills the tumour "
            "probability fast when its own top pixels are removed. Insertion AUC "
            "higher is better. Compare each against the random baseline; the gap "
            "is the evidence. Model-randomisation correlation near 1 means the "
            "map ignores the trained weights and is tracking image structure."
        ),
    }


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959963985) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float((c - h) / d), float((c + h) / d))


def summarise(g: pd.DataFrame) -> Dict[str, object]:
    n = len(g)
    hits = int(g["pointing_hit"].sum())
    lo, hi = wilson(hits, n)
    return {
        "n": n,
        "pointing_accuracy": hits / n if n else float("nan"),
        "pointing_ci": [lo, hi],
        "mean_mask_area_frac": float(g["mask_area_frac"].mean()),
        "pointing_chance_baseline": float(g["mask_area_frac"].mean()),
        "mean_mass_in_mask": float(g["mass_in_mask"].mean()),
        "mean_mass_lift_over_chance": float(g["mass_lift_over_chance"].mean()),
        "mean_best_iou": float(g["best_iou"].mean()),
    }


def aggregate(loc: pd.DataFrame) -> Dict[str, object]:
    rep: Dict[str, object] = {}
    for model, g in loc.groupby("model"):
        clean = g[g["clean_vs_train"] == True]  # noqa: E712
        blk: Dict[str, object] = {
            "all_brisc_CONTAMINATED": summarise(g),
            "clean_vs_train": summarise(clean) if len(clean) else None,
            "by_class": {k: summarise(v) for k, v in clean.groupby("class_name")} if len(clean) else {},
            "by_plane": {k: summarise(v) for k, v in clean.groupby("plane")} if len(clean) else {},
        }
        if len(clean) and clean["pred_correct"].notna().any():
            cc = clean.dropna(subset=["pred_correct"])
            blk["by_correctness"] = {
                str(bool(k)): summarise(v) for k, v in cc.groupby("pred_correct")
            }
            right = cc[cc["pred_correct"]]["pointing_hit"]
            wrong = cc[~cc["pred_correct"]]["pointing_hit"]
            if len(right) and len(wrong):
                blk["does_localisation_predict_correctness"] = {
                    "pointing_acc_when_right": float(right.mean()),
                    "pointing_acc_when_wrong": float(wrong.mean()),
                    "difference": float(right.mean() - wrong.mean()),
                    "n_right": int(len(right)),
                    "n_wrong": int(len(wrong)),
                    "reading": (
                        "If these two are close, a clinician glancing at the "
                        "overlay learns nothing about whether to trust the call, "
                        "and the heatmap is decoration."
                    ),
                }
        if len(clean):
            q = clean["mask_area_frac"]
            try:
                clean = clean.assign(size_bucket=pd.qcut(
                    q, 4, labels=["smallest", "small", "large", "largest"], duplicates="drop"))
                blk["by_tumour_size"] = {
                    str(k): summarise(v) for k, v in clean.groupby("size_bucket", observed=True)
                }
            except ValueError:
                blk["by_tumour_size"] = {}
        rep[model] = blk
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["resnet50", "vit"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="cap images, for a dry run")
    ap.add_argument("--faithful-n", type=int, default=300)
    ap.add_argument("--skip-faithfulness", action="store_true")
    ap.add_argument("--skip-localisation", action="store_true",
                    help="reuse localization_per_image.csv from a previous run")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE}   torch {torch.__version__}")

    pairs = build_pairs()
    print(f"segmentation pairs: {len(pairs)}")
    print(f"  clean_vs_train: {int(pairs['clean_vs_train'].sum())}"
          f"   contaminated: {int((~pairs['clean_vs_train'].fillna(False)).sum())}")

    all_loc, faith = [], {}
    for model_name in args.models:
        df = attach_predictions(pairs, model_name, args.seed)

        if not args.skip_localisation:
            print(f"\n[{model_name} seed {args.seed}] localisation")
            all_loc.append(run_localisation(df, model_name, args.seed, limit=args.limit))

        if not args.skip_faithfulness:
            print(f"[{model_name} seed {args.seed}] faithfulness")
            faith[model_name] = run_faithfulness(df, model_name, args.seed,
                                                 n_sample=args.faithful_n)

    loc_csv = OUT_DIR / "localization_per_image.csv"
    if args.skip_localisation:
        if not loc_csv.exists():
            raise SystemExit(f"--skip-localisation needs {loc_csv}, which does not exist")
        loc = pd.read_csv(loc_csv)
        print(f"\nreused localisation from {loc_csv.name} ({len(loc)} rows)")
    else:
        loc = pd.concat(all_loc, ignore_index=True)
        loc.to_csv(loc_csv, index=False)

    report = {
        "seed": args.seed,
        "n_pairs": int(len(pairs)),
        "localisation": aggregate(loc),
        "faithfulness": faith,
        "contamination_note": (
            "BRISC is ~80% the training set republished (session A). Headline "
            "numbers are the clean_vs_train subset. The contaminated all-BRISC "
            "number is kept beside it and labelled, never quoted alone."
        ),
    }
    (OUT_DIR / "localization_metrics.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n=== pointing accuracy, clean subset ===")
    for model, blk in report["localisation"].items():
        c = blk["clean_vs_train"]
        if not c:
            continue
        print(f"  {model:9s} n={c['n']:5d}  pointing {100*c['pointing_accuracy']:5.1f}%"
              f"  chance {100*c['pointing_chance_baseline']:5.1f}%"
              f"  mass-in-mask {100*c['mean_mass_in_mask']:5.1f}%"
              f"  lift {c['mean_mass_lift_over_chance']:.2f}x")
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
