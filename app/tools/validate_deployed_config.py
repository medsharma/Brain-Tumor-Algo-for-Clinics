"""Prove the shipped app computes the same thing the research pipeline computed.

The brief asks a specific question: does the shipped configuration hold the
same tumour miss rate as the research pipeline. There are two ways to answer
it, and only one of them is worth much.

The weak way is to measure the miss rate twice and check the two numbers are
close. That leaves you comparing two noisy estimates and arguing about what
"close" means.

The strong way is to show the two pipelines produce **identical predicted
probabilities** on the same images. If they do, then the miss rate is
identical at every possible threshold, by construction, and there is nothing
left to argue about. That is what this tool checks.

It compares, on the same checkpoint and the same images:

- ``src.code.BrainTumorResNet50.predict_with_uncertainty`` — the research path,
  T full forward passes through the whole network.
- ``app.core.model.run_mc_dropout`` — the deployed path, trunk computed once
  and the head run T times.

Both are seeded identically, so a match means bit-identical, not merely close.

It then runs the deployed path across all of BRISC and caches the per-image
tumour probability and entropy, so the miss rate at session A's threshold can
be computed the instant A publishes it.

**No threshold is chosen here.** BRISC is the exam. Nothing in this file fits,
calibrates, or selects anything on BRISC labels. Labels are read only to count
outcomes after the fact.

Usage::

    python -m app.tools.validate_deployed_config --parity-images 400
    python -m app.tools.validate_deployed_config --full-brisc
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core import model as model_module  # noqa: E402
from app.core import preprocess  # noqa: E402

BRISC_ROOT = Path(r"C:\Users\medha\Downloads\archive (1)\brisc2025")
BRISC_CLASSIFICATION = BRISC_ROOT / "classification_task"
CHECKPOINT_ROOT = Path(
    r"C:\Users\medha\OneDrive\Documents\MRI ALGO\results\20260703_155524"
)

# The class mapping trap from CONTRACTS.md. BRISC's folder is "no_tumor" with
# an underscore; the model's class is "notumor" without one. Matching by
# sorting directory names produces wrong labels and raises nothing.
BRISC_TO_INTERNAL = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "no_tumor": "notumor",
}
LABEL_INDEX = {"glioma": 0, "meningioma": 1, "pituitary": 2, "notumor": 3}

EXPECTED_COUNTS = {
    ("train", "glioma"): 1147, ("train", "meningioma"): 1329,
    ("train", "pituitary"): 1457, ("train", "no_tumor"): 1067,
    ("test", "glioma"): 254, ("test", "meningioma"): 306,
    ("test", "pituitary"): 300, ("test", "no_tumor"): 140,
}


def checkpoint_path(backbone: str, seed: int) -> Path:
    return CHECKPOINT_ROOT / backbone / f"seed_{seed}" / f"best_{backbone}_seed{seed}.pth"


# --------------------------------------------------------------------------
# BRISC listing
# --------------------------------------------------------------------------

def list_brisc() -> list[dict]:
    """Every BRISC classification image, with correctly mapped labels."""
    rows: list[dict] = []
    counts: dict[tuple[str, str], int] = {}

    for split in ("train", "test"):
        for brisc_class, internal_class in BRISC_TO_INTERNAL.items():
            folder = BRISC_CLASSIFICATION / split / brisc_class
            if not folder.is_dir():
                raise SystemExit(f"BRISC folder missing: {folder}")
            files = sorted(folder.glob("*.jpg"))
            counts[(split, brisc_class)] = len(files)
            for path in files:
                rows.append({
                    "path": path,
                    "brisc_split": split,
                    "true_label_name": internal_class,
                    "true_label": LABEL_INDEX[internal_class],
                })

    for key, expected in EXPECTED_COUNTS.items():
        actual = counts.get(key, 0)
        if actual != expected:
            raise SystemExit(
                f"BRISC count mismatch for {key}: expected {expected}, found {actual}. "
                f"Refusing to continue. Either the dataset differs from the one "
                f"CONTRACTS.md describes, or the class mapping is wrong."
            )
    print(f"BRISC listing verified: {len(rows)} images, per-class counts match CONTRACTS.md")
    return rows


def stratified_sample(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[dict]] = {}
    for row in rows:
        by_class.setdefault(row["true_label"], []).append(row)

    per_class = max(1, n // len(by_class))
    sampled: list[dict] = []
    for label in sorted(by_class):
        group = by_class[label]
        picks = rng.choice(len(group), size=min(per_class, len(group)), replace=False)
        sampled.extend(group[i] for i in picks)
    return sampled


# --------------------------------------------------------------------------
# Parity: research path against deployed path
# --------------------------------------------------------------------------

def check_pipeline_parity(rows: list[dict], backbone: str, seed: int, T: int) -> dict:
    """Same checkpoint, same images, both code paths. Are the outputs identical?"""
    try:
        from src.code import BrainTumorResNet50 as ResearchResNet
        from src.code import BrainTumorViT as ResearchViT
    except ImportError as exc:
        raise SystemExit(f"src/code.py is not importable: {exc}")

    research_class = {"resnet50": ResearchResNet, "vit": ResearchViT}[backbone]

    path = checkpoint_path(backbone, seed)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload

    research = research_class()
    research.load_state_dict(state_dict, strict=True)
    research.eval()

    deployed = model_module.load_checkpoint(
        backbone, seed, path, None, verify_hash=False
    ).module

    transform = preprocess.build_transform()

    max_prob_delta = 0.0
    max_entropy_delta = 0.0
    label_mismatches = 0
    p_tumor_deltas: list[float] = []

    for index, row in enumerate(rows):
        with Image.open(row["path"]) as image:
            tensor = preprocess.preprocess_pil(image, transform)

        torch.manual_seed(20260731 + index)
        with torch.no_grad():
            research_out = research.predict_with_uncertainty(tensor, T=T)
        research_probs = research_out["mean_probs"][0].numpy()

        torch.manual_seed(20260731 + index)
        deployed_out = model_module.run_mc_dropout(
            [deployed], tensor, T=T, temperature=1.0, entropy_units="bits", fast=True
        )
        deployed_probs = np.asarray(deployed_out.mean_probs)

        max_prob_delta = max(max_prob_delta, float(np.abs(research_probs - deployed_probs).max()))

        research_entropy = float(research_out["entropy"][0].item())
        max_entropy_delta = max(max_entropy_delta, abs(research_entropy - deployed_out.entropy))

        if int(research_probs.argmax()) != deployed_out.pred_index:
            label_mismatches += 1

        research_p_tumor = float(research_probs[:3].sum())
        p_tumor_deltas.append(abs(research_p_tumor - deployed_out.p_tumor))

    return {
        "backbone": backbone,
        "seed": seed,
        "mc_T": T,
        "n_images": len(rows),
        "max_abs_probability_delta": max_prob_delta,
        "max_abs_entropy_delta": max_entropy_delta,
        "max_abs_p_tumor_delta": float(max(p_tumor_deltas)),
        "predicted_label_mismatches": label_mismatches,
        "identical": max_prob_delta == 0.0 and label_mismatches == 0,
    }


# --------------------------------------------------------------------------
# Full BRISC run through the deployed path
# --------------------------------------------------------------------------

def run_full_brisc(rows: list[dict], backbone: str, seeds: list[int], T: int, out: Path) -> dict:
    """Score every BRISC image with the exact deployed inference path.

    Caches p_tumor and entropy per image so the miss rate at any threshold
    session A publishes can be computed in a second, without re-running.
    """
    modules = [
        model_module.load_checkpoint(
            backbone, seed, checkpoint_path(backbone, seed), None, verify_hash=False
        ).module
        for seed in seeds
    ]
    transform = preprocess.build_transform()

    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "relative_path", "brisc_split", "true_label", "true_label_name",
            "p_glioma", "p_meningioma", "p_pituitary", "p_notumor",
            "p_tumor", "entropy_bits", "entropy_nats", "mutual_information", "pred_label",
        ])

        for index, row in enumerate(rows):
            with Image.open(row["path"]) as image:
                tensor = preprocess.preprocess_pil(image, transform)

            result = model_module.run_mc_dropout(
                modules, tensor, T=T, temperature=1.0, entropy_units="bits", fast=True
            )
            writer.writerow([
                row["path"].relative_to(BRISC_ROOT).as_posix(),
                row["brisc_split"],
                row["true_label"],
                row["true_label_name"],
                *[f"{p:.8f}" for p in result.mean_probs],
                f"{result.p_tumor:.8f}",
                f"{result.entropy_bits:.8f}",
                f"{result.entropy_nats:.8f}",
                f"{result.mutual_information:.8f}",
                result.pred_index,
            ])

            if (index + 1) % 500 == 0:
                rate = (index + 1) / (time.perf_counter() - started)
                print(f"  {index + 1}/{len(rows)}  {rate:.1f} img/s")

    elapsed = time.perf_counter() - started
    return {
        "n_images": len(rows),
        "seconds": round(elapsed, 1),
        "images_per_second": round(len(rows) / elapsed, 2),
        "cache_path": str(out),
    }


def miss_rate_curve(cache: Path) -> dict:
    """Tumour miss rate against tumour threshold, from the cached scores.

    A tumour miss is a real tumour called no-tumour. It is the number that
    outranks everything else in this project.

    This is a description of the cached predictions, not a threshold choice.
    Session A picks the threshold on the internal validation split. Reading a
    threshold off this curve would be circular and would make every BRISC
    number meaningless.
    """
    import collections

    rows = list(csv.DictReader(open(cache, encoding="utf-8")))
    truth_tumor = np.array([row["true_label_name"] != "notumor" for row in rows])
    p_tumor = np.array([float(row["p_tumor"]) for row in rows])
    entropy = np.array([float(row["entropy_bits"]) for row in rows])

    curve = []
    for threshold in [round(0.05 * i, 2) for i in range(1, 20)]:
        called_tumor = p_tumor >= threshold
        misses = int((truth_tumor & ~called_tumor).sum())
        n_tumor = int(truth_tumor.sum())
        false_alarms = int((~truth_tumor & called_tumor).sum())
        n_healthy = int((~truth_tumor).sum())
        curve.append({
            "tumor_threshold": threshold,
            "tumor_miss_rate": round(misses / n_tumor, 5),
            "misses": misses,
            "sensitivity": round(1 - misses / n_tumor, 5),
            "specificity": round(1 - false_alarms / n_healthy, 5),
        })

    four_way = np.array([int(row["pred_label"]) for row in rows])
    true_label = np.array([int(row["true_label"]) for row in rows])

    return {
        "n": len(rows),
        "n_tumor": int(truth_tumor.sum()),
        "n_no_tumor": int((~truth_tumor).sum()),
        "four_way_accuracy": round(float((four_way == true_label).mean()), 5),
        "entropy_bits_percentiles": {
            str(p): round(float(np.percentile(entropy, p)), 4) for p in (5, 25, 50, 75, 95)
        },
        "miss_rate_by_threshold": curve,
        "warning": (
            "This curve describes cached BRISC predictions. It must not be used "
            "to pick a threshold. Thresholds are fitted on the internal "
            "validation split and applied to BRISC unchanged."
        ),
    }


# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "vit"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--mc-T", type=int, default=20)
    parser.add_argument("--parity-images", type=int, default=400)
    parser.add_argument("--full-brisc", action="store_true")
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "app" / "benchmarks"
    )
    args = parser.parse_args(argv)

    model_module.configure_cpu_threads()
    rows = list_brisc()

    payload: dict = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backbone": args.backbone,
        "seeds": args.seeds,
        "mc_T": args.mc_T,
    }

    if args.parity_images > 0:
        sample = stratified_sample(rows, args.parity_images)
        print(f"\npipeline parity on {len(sample)} BRISC images "
              f"(research path against deployed path)")
        parity = check_pipeline_parity(sample, args.backbone, args.seeds[0], args.mc_T)
        payload["pipeline_parity"] = parity
        print(f"  max |probability difference| : {parity['max_abs_probability_delta']:.3e}")
        print(f"  max |entropy difference|     : {parity['max_abs_entropy_delta']:.3e}")
        print(f"  predicted label mismatches   : {parity['predicted_label_mismatches']}")
        print(f"  IDENTICAL                    : {parity['identical']}")

    if args.full_brisc:
        cache = args.out_dir / f"brisc_deployed_{args.backbone}_seeds{'-'.join(map(str, args.seeds))}.csv"
        print(f"\nscoring all {len(rows)} BRISC images through the deployed path")
        payload["full_brisc_run"] = run_full_brisc(
            rows, args.backbone, args.seeds, args.mc_T, cache
        )
        payload["brisc_summary"] = miss_rate_curve(cache)
        print(f"  four-way accuracy: {payload['brisc_summary']['four_way_accuracy']:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / "deployed_config_validation.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
