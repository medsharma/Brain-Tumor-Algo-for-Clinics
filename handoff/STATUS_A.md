# Session A status

**Owner:** external validation on BRISC 2025, safety metrics, operating point.
**Branch:** `session/A`. **Last update:** 2026-07-31, contamination found.

---

## Headline, and it is bad

**BRISC 2025 is not external data. About 80 percent of it is the training set.**

4,791 of 6,000 BRISC images have a perceptual-hash distance of 0 to an image in
`data/brain_tumor/`. A pixel check on 40 random such pairs found mean absolute
difference exactly 0.0 and correlation exactly 1.0 on all 40. They are the same
image files, republished under new names. Both datasets draw on the same public
sources.

The tell was that the model scored **higher** on "external" BRISC (0.9728) than
on its own internal held-out test split (0.9613). That does not happen on real
new data.

Full detail, evidence, and what each session must change:
**`handoff/ISSUES.md`, top entry.** Read it before writing any claim about
external validation.

**The project does not currently have an external validation result.** Getting
one requires a dataset that does not descend from Figshare, SARTAJ or Br35H.

---

## What I am doing about it

Reporting both, never mixing them:

- **Full BRISC, n=6000** - kept, labelled contaminated, never quoted as external.
- **Clean subset** - BRISC images whose nearest internal *training* image is more
  than Hamming 5 away. The only genuinely unseen images. Every headline number
  Session A publishes will be from this subset, with its size stated.

Per-image contamination flags:
`analysis/results/brisc/brisc_overlap_per_image.csv`
(`d_train`, `d_val`, `d_test`, `d_any`, `clean_vs_train`, `clean_vs_any`).

**Join the prediction caches against this file and filter on `clean_vs_train`
before you report anything.**

---

## Phase 1: prediction caches - RUNNING

All pre-flight checks passed:

| Check | Result |
|---|---|
| Class mapping `no_tumor` -> `notumor` applied explicitly | done |
| Per-class counts vs the contract table | **exact match**, 6000 images |
| sha256 spot check, 50 sampled images vs BRISC `manifest.csv` | 50/50 match |
| Preprocessing identical to `get_transforms("test")` | asserted structurally |
| MC fast path vs `predict_with_uncertainty` | **bit-identical**, both backbones |
| Sanity run, 200 BRISC images, ViT seed 42 | 4-way acc 0.965, miss 0.013 |

BRISC composition, from `manifest.csv`. All 6000 are T1.

| brisc_split | glioma | meningioma | pituitary | notumor | total |
|---|---|---|---|---|---|
| train | 1147 | 1329 | 1457 | 1067 | 5000 |
| test | 254 | 306 | 300 | 140 | 1000 |

Planes: axial 1993, coronal 1981, sagittal 2026.
Internal splits: val 1109, test 1112.

**Not yet published.** Caches are still being written. This file will say
PUBLISHED when all 30 parquet files are pushed.

---

## What other sessions need to know right now

1. **BRISC is contaminated.** See above and `handoff/ISSUES.md`. This is the
   one that changes what you are allowed to claim.
2. **There is no GPU.** `torch` is a CPU-only build.
3. **Use the MC fast path.** `mc_forward()` in
   `analysis/external_validation_brisc.py` is 20x faster than looping
   `predict_with_uncertainty` and is bit-identical.
4. **Entropy units.** Cache column `entropy` is nats per Contract 1.
   `entropy_bits` is the same number in bits, which is what
   `results/20260703_155524/` reports. Do not mix them.
5. **`pip install pyarrow`** if you cannot read parquet.

---

## Blocked on

Nothing.

## Blocking

Sessions B, C, E on the prediction caches.
