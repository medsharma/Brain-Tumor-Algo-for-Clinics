# Session A status

**Owner:** external validation on BRISC 2025, safety metrics, operating point.
**Branch:** `session/A`. **Last update:** 2026-07-30, cache build running.

---

## Headline

Not available yet. The full cache build is running. Nothing is reported until it
finishes and the numbers are checked.

---

## Phase 1: prediction caches - RUNNING

Everything that must be true before the caches are trustworthy has been checked
and passed:

| Check | Result |
|---|---|
| Class mapping `no_tumor` -> `notumor` applied explicitly | done |
| Per-class counts vs the contract table | **exact match**, 6000 images |
| sha256 spot check, 50 sampled images vs BRISC `manifest.csv` | 50/50 match |
| Preprocessing identical to `get_transforms("test")` | asserted structurally |
| MC fast path vs `predict_with_uncertainty` | **bit-identical**, both backbones |
| Sanity run, 200 BRISC images, ViT seed 42 | 4-way acc 0.965, miss rate 0.013 |

BRISC composition, straight from `manifest.csv`:

| brisc_split | glioma | meningioma | pituitary | notumor | total |
|---|---|---|---|---|---|
| train | 1147 | 1329 | 1457 | 1067 | 5000 |
| test | 254 | 306 | 300 | 140 | 1000 |

All 6000 are T1. Planes: axial 1993, coronal 1981, sagittal 2026.

Internal splits: val 1109 images, test 1112 images.

**Not yet published. Do not build against these paths until this file says
PUBLISHED.**

---

## What other sessions need to know right now

1. **There is no GPU.** `torch` is a CPU-only build. See `handoff/ISSUES.md`.
   Budget CPU time, not GPU time.
2. **Use the MC fast path.** `mc_forward()` in
   `analysis/external_validation_brisc.py` is 20x faster than looping
   `predict_with_uncertainty` and is bit-identical. See `handoff/ISSUES.md`.
3. **Entropy units.** Cache column `entropy` is nats per Contract 1.
   `entropy_bits` is the same number in bits, which is what
   `results/20260703_155524/` reports. Do not mix them.
4. **`pip install pyarrow`** if you cannot read parquet. It was missing.

---

## Blocked on

Nothing. Nobody is blocking me.

## Blocking

Sessions B, C, E are waiting on the prediction caches. ETA about one hour from
the start of the build.
