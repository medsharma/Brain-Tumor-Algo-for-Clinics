"""Measure what the shipped configuration actually costs, and what it costs to cut.

Two questions, both of which have to be answered with numbers rather than
guesses.

1. **How slow is it on a laptop CPU?** MC Dropout at T=20 across a 5-seed
   ensemble is 100 forward passes, which the brief flags as probably too slow.
   Measure before deciding anything.

2. **If T is reduced, does the deferral rule still hold?** This is the part
   that is easy to get wrong. Accuracy barely moves when T drops, because the
   predictive mean is stable. Entropy is a different matter: it is an estimate
   from T samples, and fewer samples means a noisier estimate. The deferral
   rule is a threshold on entropy, so a noisier estimate means scans flipping
   between "answered" and "sent to a human" for no reason but the dice. So
   this measures the *stability of the deferral decision*, not accuracy.

Everything here runs on the **internal validation split**, never on BRISC.
Nothing is fitted. No threshold is chosen. The deferral analysis sweeps a
range of candidate thresholds and reports how the decision behaves at each,
which is a description of the estimator, not a choice of operating point.
Session A owns the operating point.

Usage::

    python -m app.tools.benchmark_cpu --latency-images 40 --stability-images 300
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core import model as model_module  # noqa: E402
from app.core import preprocess  # noqa: E402

CHECKPOINT_ROOT = Path(
    r"C:\Users\medha\OneDrive\Documents\MRI ALGO\results\20260703_155524"
)
SEEDS = (42, 123, 7, 2024, 31)
MANIFEST = REPO_ROOT / "data" / "split_manifest.csv"

#: Candidate deferral thresholds, in bits. Swept, not chosen. The full range
#: of 4-class predictive entropy is 0 to 2 bits.
CANDIDATE_THRESHOLDS = tuple(round(0.1 * i, 2) for i in range(1, 20))


def checkpoint_path(backbone: str, seed: int) -> Path:
    return CHECKPOINT_ROOT / backbone / f"seed_{seed}" / f"best_{backbone}_seed{seed}.pth"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_internal_val(limit: int, seed: int = 0) -> list[Path]:
    """A class-balanced sample of the internal validation split."""
    frame = pd.read_csv(MANIFEST)
    val = frame[frame["split"] == "val"]
    per_class = max(1, limit // val["label"].nunique())
    sampled = pd.concat(
        [
            group.sample(min(len(group), per_class), random_state=seed)
            for _, group in val.groupby("label")
        ]
    ).reset_index(drop=True)
    return [REPO_ROOT / row for row in sampled["filepath"]]


def to_tensors(paths: list[Path]) -> list[torch.Tensor]:
    transform = preprocess.build_transform()
    tensors = []
    for path in paths:
        with Image.open(path) as image:
            tensors.append(preprocess.preprocess_pil(image, transform))
    return tensors


# --------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------

@dataclass
class LatencyResult:
    label: str
    backbone: str
    n_models: int
    mc_T: int
    path: str
    n_images: int
    median_ms: float
    mean_ms: float
    p95_ms: float
    forward_passes: int


def time_configuration(
    modules: list[torch.nn.Module],
    tensors: list[torch.Tensor],
    T: int,
    label: str,
    backbone: str,
    fast: bool,
) -> LatencyResult:
    forward = model_module.mc_forward_cached if fast else model_module.mc_forward_full

    # Warm up. The first pass pays for lazy allocation and thread pool setup,
    # and reporting that as the clinic's latency would be wrong.
    for module in modules:
        forward(module, tensors[0], min(T, 3), 1.0)

    timings: list[float] = []
    for tensor in tensors:
        started = time.perf_counter()
        for module in modules:
            forward(module, tensor, T, 1.0)
        timings.append((time.perf_counter() - started) * 1000.0)

    ordered = sorted(timings)
    return LatencyResult(
        label=label,
        backbone=backbone,
        n_models=len(modules),
        mc_T=T,
        path="cached-trunk" if fast else "full-network",
        n_images=len(timings),
        median_ms=round(statistics.median(timings), 1),
        mean_ms=round(statistics.mean(timings), 1),
        p95_ms=round(ordered[int(0.95 * (len(ordered) - 1))], 1),
        forward_passes=len(modules) * T,
    )


# --------------------------------------------------------------------------
# Entropy stability under reduced T
# --------------------------------------------------------------------------

def entropy_bits(probs: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Predictive entropy in bits, matching ``src/code.py``'s log2 convention."""
    return -(probs * np.log2(probs + epsilon)).sum(axis=-1)


def stability_analysis(
    module: torch.nn.Module,
    tensors: list[torch.Tensor],
    n_passes: int = 40,
) -> dict:
    """How stable the entropy estimate, and so the deferral decision, really is.

    The question a clinic cares about is not "how far is T=10 from T=20". It is
    "if I run this tool twice on the same scan, does it give me the same
    answer". So every T is measured the same way: draw ``n_passes`` Monte Carlo
    passes once per image, cut them into **disjoint** blocks of size T, and
    measure how much independent blocks disagree with each other.

    Disjoint is the whole point. Comparing a T=10 estimate against a T=20
    estimate that *contains* those same ten passes makes them look far more
    similar than two honest runs would be, and would understate the
    instability. Cutting one 40-pass draw into disjoint blocks gives eight
    independent T=5 runs, four independent T=10 runs, and two independent T=20
    runs, all from the same images and all directly comparable.
    """
    all_probs: list[np.ndarray] = []
    for tensor in tensors:
        stacked = model_module.mc_forward_cached(module, tensor, n_passes, 1.0)
        all_probs.append(stacked[:, 0, :].numpy())       # (n_passes, C)

    probs = np.stack(all_probs, axis=0)                   # (N, n_passes, C)
    best_estimate = entropy_bits(probs.mean(axis=1))      # (N,) all passes pooled

    results: dict = {
        "n_images": len(tensors),
        "n_passes_drawn": n_passes,
        "method": (
            "One draw of n_passes per image, cut into disjoint blocks of size "
            "T. Every T is compared the same way: independent block against "
            "independent block."
        ),
        "best_estimate_entropy_mean": round(float(best_estimate.mean()), 4),
        "best_estimate_entropy_sd": round(float(best_estimate.std()), 4),
        "best_estimate_entropy_percentiles": {
            str(p): round(float(np.percentile(best_estimate, p)), 4)
            for p in (5, 25, 50, 75, 95)
        },
        "by_T": {},
    }

    for T in (5, 10, 20):
        n_blocks = n_passes // T
        blocks = np.stack(
            [entropy_bits(probs[:, i * T:(i + 1) * T, :].mean(axis=1)) for i in range(n_blocks)]
        )                                                 # (n_blocks, N)

        # Mean absolute difference over every pair of independent runs.
        pairs = [(i, j) for i in range(n_blocks) for j in range(i + 1, n_blocks)]
        pairwise_delta = np.concatenate([np.abs(blocks[i] - blocks[j]) for i, j in pairs])

        flips: dict[str, float] = {}
        worst_threshold = None
        worst_rate = 0.0
        for threshold in CANDIDATE_THRESHOLDS:
            defer = blocks > threshold
            disagreement = float(
                np.mean([(defer[i] != defer[j]).mean() for i, j in pairs])
            )
            flips[f"{threshold:.1f}"] = round(disagreement, 4)
            if disagreement > worst_rate:
                worst_rate = disagreement
                worst_threshold = threshold

        results["by_T"][str(T)] = {
            "independent_runs_per_image": n_blocks,
            "mean_abs_delta_between_runs_bits": round(float(pairwise_delta.mean()), 4),
            "p95_abs_delta_between_runs_bits": round(float(np.percentile(pairwise_delta, 95)), 4),
            "max_abs_delta_between_runs_bits": round(float(pairwise_delta.max()), 4),
            "deferral_disagreement_by_threshold": flips,
            "worst_threshold": worst_threshold,
            "worst_disagreement_rate": round(worst_rate, 4),
        }

    return results


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency-images", type=int, default=40)
    parser.add_argument("--stability-images", type=int, default=300)
    parser.add_argument("--skip-vit", action="store_true")
    parser.add_argument("--skip-ensemble", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "app" / "benchmarks" / "cpu_benchmark.json",
    )
    args = parser.parse_args(argv)

    threads = model_module.configure_cpu_threads()
    print(f"torch threads: {threads}, device: cpu")

    latency_paths = load_internal_val(args.latency_images)
    latency_tensors = to_tensors(latency_paths)
    print(f"latency sample: {len(latency_tensors)} internal val images")

    results: list[LatencyResult] = []

    print("\nloading resnet50 seed 42")
    resnet = model_module.load_checkpoint(
        "resnet50", 42, checkpoint_path("resnet50", 42), None, verify_hash=False
    ).module

    for T in (5, 10, 20):
        result = time_configuration(
            [resnet], latency_tensors, T, f"resnet50 x1 seed, T={T}", "resnet50", fast=True
        )
        results.append(result)
        print(f"  {result.label:32s} median {result.median_ms:8.1f} ms")

    naive = time_configuration(
        [resnet], latency_tensors, 20, "resnet50 x1 seed, T=20, naive", "resnet50", fast=False
    )
    results.append(naive)
    print(f"  {naive.label:32s} median {naive.median_ms:8.1f} ms")

    if not args.skip_vit:
        print("\nloading vit seed 42")
        vit = model_module.load_checkpoint(
            "vit", 42, checkpoint_path("vit", 42), None, verify_hash=False
        ).module
        for T in (5, 20):
            result = time_configuration(
                [vit], latency_tensors, T, f"vit x1 seed, T={T}", "vit", fast=True
            )
            results.append(result)
            print(f"  {result.label:32s} median {result.median_ms:8.1f} ms")
        del vit

    if not args.skip_ensemble:
        print("\nloading resnet50 5-seed ensemble")
        ensemble = [resnet]
        for seed in SEEDS[1:]:
            path = checkpoint_path("resnet50", seed)
            if path.is_file():
                ensemble.append(
                    model_module.load_checkpoint(
                        "resnet50", seed, path, None, verify_hash=False
                    ).module
                )
        result = time_configuration(
            ensemble, latency_tensors, 20,
            f"resnet50 x{len(ensemble)} seeds, T=20", "resnet50", fast=True,
        )
        results.append(result)
        print(f"  {result.label:32s} median {result.median_ms:8.1f} ms")
        del ensemble

    print(f"\nentropy stability on {args.stability_images} internal val images")
    stability_tensors = to_tensors(load_internal_val(args.stability_images, seed=1))
    stability = stability_analysis(resnet, stability_tensors)

    print(
        f"  entropy across images: mean {stability['best_estimate_entropy_mean']:.3f} "
        f"sd {stability['best_estimate_entropy_sd']:.3f} bits"
    )
    for T, block in stability["by_T"].items():
        print(
            f"  T={T:>2}  run-to-run mean|delta| "
            f"{block['mean_abs_delta_between_runs_bits']:.4f} bits, "
            f"worst deferral disagreement {block['worst_disagreement_rate']:.1%} "
            f"at threshold {block['worst_threshold']}"
        )

    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_threads": threads,
        },
        "notes": (
            "Measured on the internal validation split. Nothing is fitted and "
            "no threshold is chosen here. BRISC is not touched."
        ),
        "latency": [asdict(result) for result in results],
        "entropy_stability": stability,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
