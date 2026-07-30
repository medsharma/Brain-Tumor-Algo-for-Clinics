# Leakage Audit — `build_split_manifest()`

*(Note: the repo was reorganized after this audit ran — `code.py` and
`split_manifest.csv` now live at `src/code.py` and `data/split_manifest.csv`
respectively. The log excerpts and paths below are preserved verbatim as
they were actually printed/run at the time; do not edit them to match the
new layout.)*

Run: 2026-07-03, `code.py`, `data/brain_tumor/` (7,200 images, 4 classes, 1,800/class).

## Dataset identity

Filenames (`Te-gl_1.jpg`, `Tr-me_995.jpg`, `Te-aug-me_1.jpg`, ...) match the public
Kaggle "Brain Tumor MRI Dataset" (Masoud Nickparvar / Sartaj Bhuvaji derivative),
which ships with `Te-`/`Tr-` (test/train) prefixes baked into filenames from the
original authors' split — **not** a per-patient DICOM export. There is no
patient/scan ID encoded anywhere in these names. The meningioma class additionally
contains augmented duplicates (`Te-aug-me_*`, `Tr-aug_*`: 103 + 100 = 203 files) —
this class is a known source of near-duplicate leakage in this dataset in the wider
literature, since the augmentation pipeline used to rebalance meningioma produces
images that are near-identical (rotated/brightness-jittered) to originals that may
land in a different split under a naive random shuffle.

## Step 1 — Patient-ID parsing

`_parse_patient_id()` requires a stem to start with `p[at[ient]]-<digits>` or
`\d{3,}` at position 0. Actual stems start with `Te-`/`Tr-`/`Te-aug-`, so:

```
WARNING: Patient IDs could not be parsed for 7200 / 7200 files.
```

**0 / 7200 (0%) parseable.** This dataset cannot use patient-ID grouping —
expected, since it's a public slice-level dataset with no patient metadata.

## Step 2 — phash fallback

`imagehash` **is installed** (v4.3.2), so the code took the phash-clustering path
rather than falling back to the leakage-prone per-file split. Full log:

```
Computing perceptual hashes for 7200 images (threshold=5) …
phash clustering: 4784 clusters from 7200 images  (1107 clusters contain >1 near-duplicate image)
Cluster-size distribution:
count    4784.000000
mean        1.505017
std         1.472721
min         1.000000
25%         1.000000
50%         1.000000
75%         1.000000
max        28.000000
Using phash-cluster-stratified split.
```

**Diagnosis: phash clustering is genuinely active and producing multi-file
clusters — this is NOT a degenerate all-singleton fallback.**

- 4,784 clusters from 7,200 images → 2,416 images (33.6%) belong to a cluster of
  size ≥2.
- 1,107 / 4,784 clusters (23.1%) contain more than one image — these are the
  near-duplicate groups that a naive per-file split would have been free to
  split across train/val/test.
- Largest cluster: 28 near-duplicate images kept together in one split.
- This confirms the dataset does contain a meaningful amount of near-duplicate
  content (consistent with the known augmented-meningioma issue above), and the
  grouped split is doing real leakage-prevention work, not a no-op.

Resulting split sizes (grouped by cluster, stratified per class):

| split | n    | glioma | meningioma | notumor | pituitary |
|-------|------|--------|------------|---------|-----------|
| train | 4979 | 1254   | 1268       | 1211    | 1246      |
| val   | 1109 | 269    | 266        | 298     | 276       |
| test  | 1112 | 277    | 266        | 291     | 278       |

≈ 69.2% / 15.4% / 15.4% — close to the nominal 70/15/15 target; the small
deviation is expected because whole clusters (not individual files) are the
unit being assigned to a split.

## Bug found (documentation-only, not fixed)

`build_split_manifest()` writes the **original** `df` to `split_manifest.csv`,
not `df_phash` — so `df["split"]` is correctly set via the phash-grouped
assignment, but the saved CSV's `patient_id` column is **100% null**, even
though clustering was genuinely used to compute `split`. Verified directly:

```python
>>> pd.read_csv("split_manifest.csv")["patient_id"].isna().all()
True
```

Practical effect: **the manifest CSV alone cannot be used to prove after the
fact which split path was taken** — you have to re-run with logging (as done
here) or inspect cluster IDs in-memory. This run's console log is the audit
trail for this claim. I have not patched this (out of scope for this session —
flagging for whoever owns `code.py` next; a one-line fix is
`df["patient_id"] = df_phash["patient_id"]` before the `to_csv` call in the
phash branch).

## Verdict

- Patient-ID grouping: **not applicable** (0% parseable — expected for this
  public dataset).
- phash-cluster grouping: **active and effective** (imagehash installed,
  1,107 real multi-file clusters found and kept intact).
- Per-file fallback: **not triggered.** The split now in `split_manifest.csv`
  is leakage-safe at the near-duplicate-cluster level, not per-file.
- This is a materially different (and stricter) split than a prior
  80/20 per-file split with no grouping, which could not have accounted for
  the 1,107 near-duplicate clusters identified here and likely leaked
  near-duplicate slices across train/test — consistent with the previously
  reported 98.9% accuracy being inflated by leakage.
