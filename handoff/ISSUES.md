# Cross-session issues

Append only. Never edit or delete another session's entry. Prefix every entry
with your session letter and a UTC timestamp.

---

## [A] 2026-07-31T01:20Z - STOP. BRISC 2025 IS NOT EXTERNAL DATA. IT IS THE TRAINING SET.

**Read this before you write a single claim about external validation.**

About **80 percent of BRISC 2025 is pixel-identical to images in
`data/brain_tumor/`**, the dataset these models were trained on.

Evidence:

1. 64-bit pHash, every BRISC image against every internal image.
   4,791 of 6,000 BRISC images (79.9%) have Hamming distance **0**.
   The median nearest-distance across all 6,000 BRISC images is **0**.
2. pHash is not degenerate on this data. On a 1,500-image sample, 99.4% of
   BRISC hashes and 95.5% of internal hashes are distinct. So distance 0 is
   not a collision artifact.
3. Pixel check on 40 random distance-0 pairs, both resized to 256x256:
   **mean absolute difference 0.0, correlation 1.0, on all 40.** Same class
   on all 40. These are the same images, republished under new filenames.
   Example: `brisc2025_train_01346_me_ax_t1.jpg` is `Tr-me_1026.jpg`.

Both datasets draw on the same public sources (Figshare / SARTAJ / Br35H).
BRISC 2025 re-packaged them. Nobody did anything wrong. But the consequence is
absolute:

**Running these checkpoints on the full BRISC set and calling the result
"external validation" is testing the model on its own training data.** The
headline number it produces is meaningless. It looked great, which is exactly
what a leak looks like.

Confirming signature: ViT seed 42 scores **0.9728** four-way accuracy on the
full BRISC set and only **0.9613** on the internal held-out test split. A model
does not beat its own held-out test set on genuinely new data.

### What Session A is doing about it

Not deleting anything. Reporting both, clearly separated:

* **Full BRISC (n=6000)** - kept for reference and labelled contaminated.
  Never quoted as external performance.
* **Clean subset** - BRISC images whose nearest internal **training** image is
  more than Hamming 5 away. Those are the only genuinely unseen images.
  This is the real external result and it is what every headline number in
  Session A will be based on. Per-image flags are published in
  `analysis/results/brisc/brisc_overlap_per_image.csv`
  (columns `d_train`, `d_val`, `d_test`, `d_any`, `clean_vs_train`).

### What YOU must do

* **Session E**: the model card, README and LIMITATIONS must not claim external
  validation on BRISC 2025. Say plainly that BRISC overlaps the training data by
  about 80 percent and that external validation is still outstanding. This is a
  headline limitation, not a footnote.
* **Session C**: any accuracy shown in the app must come from the clean subset
  or the internal test split, never from full BRISC.
* **Session B**: BRISC is not a clean in-distribution reference for tuning a
  rejector. Use the internal splits, and the clean BRISC subset if you need more.
* **Everyone**: the prediction caches in `analysis/results/brisc/predictions/`
  still cover all 6,000 images and are correct as caches. Join them against
  `brisc_overlap_per_image.csv` and filter on `clean_vs_train` before you report
  anything.

I did not break the no-training-on-BRISC rule. The contamination predates this
work: it is in the relationship between the two public datasets.

---

## [A] 2026-07-30T00:00Z - There is no GPU. torch is a CPU-only build.

`torch.__version__ == "2.12.1+cpu"`, `torch.cuda.is_available() == False`.
The "GPU contention" section of `prompts/CONTRACTS.md` does not apply. Nobody is
competing for a GPU because there is not one. There are 24 CPU cores and torch
uses all of them by default.

**How to apply:** if you run inference, you are on CPU. Budget accordingly.
Measured single-image throughput on this machine at 224x224, batch 32, using the
fast MC path described below: ViT-B/16 about 17 img/s, ResNet-50 about 51 img/s.
Naive full-network MC Dropout at T=20 is 20x slower than that.

---

## [A] 2026-07-30T00:00Z - MC-Dropout fast path: backbone once, head T times.

Session A does not call `model.predict_with_uncertainty` in a loop over 1.2
million forward passes. It runs the backbone once per image and the
classification head T=20 times.

This is exact, not an approximation. Every Dropout layer with p > 0 lives in the
head (`backbone.heads` for ViT, `backbone.fc` for ResNet-50). The torchvision ViT
encoder contains Dropout modules but they are built with p=0.0, and ATen's
dropout kernel returns the input untouched and consumes no RNG when p == 0. So
the backbone is deterministic and the RNG stream is identical either way.

Verified, not assumed: `verify_fast_path()` in
`analysis/external_validation_brisc.py` asserts `torch.equal(reference, fast)`
under the same seed, for every checkpoint, before any cache is written. It is
bit-identical (max abs diff 0.0) for both backbones.

**How to apply:** if you need MC-Dropout inference on CPU, reuse `mc_forward()`
from `analysis/external_validation_brisc.py`. It is 19x faster for ViT and 22x
faster for ResNet-50 and gives the same numbers.

---

## [A] 2026-07-30T00:00Z - Entropy units. Contract says nats, src/code.py uses bits.

Contract 1 specifies the `entropy` column as nats. `src/code.py`
`predict_with_uncertainty` computes entropy with `log2`, so every entropy number
in `results/20260703_155524/` (including `mean_entropy` in the summary CSVs) is
in **bits**.

Both are in the prediction caches so nobody mixes them up:

- `entropy` - nats, as Contract 1 requires. Use this for anything cross-session.
- `entropy_bits` - the same quantity in bits. Use this only to compare against
  the internal training-run numbers in `results/20260703_155524/`.

`entropy_bits = entropy / ln(2)`.

`mutual_information` is in **nats**, consistent with `entropy`.

I did not change the contract. Objection registered here per the contract rules.

---

## [A] 2026-07-30T00:00Z - Two extra columns in the Contract 1 caches.

The caches contain every Contract 1 column with the exact specified names, plus
two additions: `entropy_bits` (see above) and, for BRISC only, nothing else.
Additive only. Nothing was renamed or removed. If you select columns by name you
are unaffected.

---

## [A] 2026-07-30T00:00Z - `pyarrow` was not installed. I installed it.

Contract 1 requires parquet. Neither `pyarrow` nor `fastparquet` was present in
the environment. I ran `pip install pyarrow` (installed 25.0.0). If your session
cannot read the caches with `pd.read_parquet`, that is why, and the fix is the
same command.
