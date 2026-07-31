# Data provenance and licensing

Where every image in this project came from, who owns it, and what we are
allowed to do with it. Written to be checkable by someone who does not trust us.

Last updated: 2026-07-31.

---

## The short version

Both datasets in this project trace back to the same three public collections.
That is the single most important fact on this page, and it is the reason the
BRISC 2025 evaluation is a weaker test than it looks. See
[The independence problem](#the-independence-problem) below.

| | Training data | Evaluation data |
|---|---|---|
| Name | Brain Tumor MRI Dataset (Nickparvar) | BRISC 2025 |
| Images | 7,200 | 6,000 |
| License | CC0 1.0 Public Domain | CC BY 4.0 |
| Redistribution allowed | Yes | Yes, with attribution |
| Underlying sources | Br35H, SARTAJ, Figshare (Cheng) | Br35H, SARTAJ, Figshare (Cheng) |
| Independent of each other | **No** | **No** |

---

## Training data: Kaggle "Brain Tumor MRI Dataset" (Nickparvar)

- **Source:** <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>
- **Author:** Masoud Nickparvar, 2023.
- **Local copy:** `data/brain_tumor/{glioma,meningioma,pituitary,notumor}/`
- **Count in this repo:** 7,200 images, 1,800 per class.
- **DOI:** none. The Kaggle listing does not mint a DOI. Cite as
  Nickparvar, M. (2023), *Brain Tumor MRI Dataset*, Kaggle, Version 2.

### License: CC0 1.0 Universal (Public Domain Dedication)

The Kaggle listing states CC0: Public Domain. Under CC0 the author waives
copyright and neighbouring rights worldwide, to the extent legally possible.

**What that permits:** copying, modification, redistribution, commercial use,
with no attribution requirement and no share-alike obligation.

**Verdict for this repo: redistributing the 7,200 images in a public repository
is permitted.** This resolves the placeholder that previously sat in
`reproducibility/README.md`.

### Three caveats we are not going to bury

1. **CC0 is asserted by the Kaggle uploader, not by the original data
   custodians.** Nickparvar aggregated three upstream collections. Whether the
   uploader had standing to place the combined work in the public domain depends
   on the upstream licenses, which are:
   - **Figshare / Cheng et al. brain tumor dataset** — CC BY 4.0. Requires
     attribution. CC0 does not remove that obligation from the upstream work.
   - **SARTAJ "Brain Tumor Classification (MRI)"** — Kaggle listing, no clearly
     stated upstream license chain.
   - **Br35H "Brain Tumor Detection 2020"** — Kaggle listing, no clearly stated
     upstream license chain.

   We attribute all three regardless of whether CC0 obliges us to. That is the
   conservative reading and it costs us nothing.

2. **The image count does not match the current Kaggle listing.** The listing
   describes 7,023 images. This repo holds 7,200 (1,800 exactly per class). The
   filename conventions (`Te-gl_1.jpg`, `Tr-me_995.jpg`, and the author-supplied
   `Te-aug-me_*` / `Tr-aug_*` meningioma augmentations) match, and 7,200 matches
   the Version 2 Training+Testing folder count. We believe this copy is Version 2
   and the live listing has since changed. **This has not been confirmed against
   a fresh download.** Anyone reproducing should verify their own copy against
   `data/split_manifest.csv`, which lists all 7,200 filepaths.

3. **The labels carry an inherited correction we never checked.** SARTAJ's
   glioma class had documented mislabelling. The aggregator's fix was to drop
   SARTAJ's glioma images and substitute Figshare ones, a correction now widely
   noted in the literature that uses this dataset. It is very likely the right
   call. We took it on trust, as does everyone else using this dataset.

4. **No patient-level provenance survives.** The merge is slice-level. Scanner
   make, field strength, sequence parameters, acquisition site, scan date and
   all patient demographics are absent and are not recoverable from the files.
   `results/leakage_audit.md` documents that patient IDs are parseable for
   0 / 7,200 filenames. Consequences are in [LIMITATIONS.md](../LIMITATIONS.md).

---

## Evaluation data: BRISC 2025

- **Source:** <https://www.kaggle.com/datasets/briscdataset/brisc2025>
- **Paper:** Fateh, A., Rezvani, Y., Moayedi, S., Rezvani, S., Fateh, F.,
  Fateh, M., Abolghasemi, V. *BRISC: Annotated Dataset for Brain Tumor
  Segmentation and Classification.* arXiv:2506.14318 (v1 2025-06-17, v5
  2026-01-28). Peer-reviewed version: *Scientific Data*, 2026,
  `s41597-026-06753-y`. Open access mirror: PMC12982668.
- **Local copy:** `C:\Users\medha\Downloads\archive (1)\brisc2025`
  (not committed to this repo; see [Redistribution](#redistribution-of-brisc)).
- **Downloaded:** present on the analysis machine as of 2026-07-30. The exact
  download date was not recorded at the time. Integrity is pinned by the
  manifest hashes below instead.

### License: CC BY 4.0

The *Scientific Data* article carries CC BY 4.0, which the publisher applies to
the article and its associated data descriptor content. CC BY 4.0 permits use,
sharing, adaptation, distribution and reproduction in any medium, including
commercially, **provided you give appropriate credit, link the license, and
indicate if changes were made.**

**Verdict for this use: permitted.** We run inference on the images and report
aggregate metrics. We do not redistribute the images. Attribution is given here,
in `MODEL_CARD.md`, and in `README.md`.

### Redistribution of BRISC

We do **not** redistribute BRISC images in this repository. Not because CC BY
forbids it, it does not, but because the repo already carries 160 MB of training
data and there is no reason to mirror a dataset that is one Kaggle download away.
Anyone reproducing downloads it themselves and verifies against the hashes below.

### Integrity hashes

Recorded from the `.sha256` sidecar files shipped with the dataset:

```
manifest.csv    4eaa0c3b88ce1fd246d90382c1dcd1010cc126fff1bd3a4a415159661ca8b605
manifest.json   ada767e102e9e7ee788b50188712dfe19ba825c250c362d831b560902c249e3a
```

Verify your copy:

```bash
cd "<your brisc2025 dir>"
sha256sum -c manifest.csv.sha256 manifest.json.sha256
```

`manifest.csv` itself carries a per-image `sha256` column covering all 6,000
classification images plus the segmentation masks. Every prediction cache under
`analysis/results/brisc/predictions/` carries that hash per row, so any
individual prediction can be traced back to a byte-exact source image. That is
the integrity chain: sidecar hash pins the manifest, manifest pins each image,
prediction cache pins each image to a prediction.

### Composition, counted directly from `manifest.csv`

Classification task, masks excluded. 6,000 images, all T1-weighted contrast-enhanced.

| | glioma | meningioma | pituitary | no_tumor | total |
|---|---|---|---|---|---|
| BRISC train | 1,147 | 1,329 | 1,457 | 1,067 | 5,000 |
| BRISC test | 254 | 306 | 300 | 140 | 1,000 |
| **total** | **1,401** | **1,635** | **1,757** | **1,207** | **6,000** |

| plane | train | test | total |
|---|---|---|---|
| axial | 1,595 | 398 | 1,993 |
| coronal | 1,676 | 305 | 1,981 |
| sagittal | 1,729 | 297 | 2,026 |

Two notes for anyone comparing against the paper:

- The paper reports plane counts of 1,937 / 1,976 / 2,087. The shipped manifest
  gives 1,993 / 1,981 / 2,026. The totals agree at 6,000. We use the manifest,
  because that is what the files actually are. We have not chased the
  discrepancy with the authors.
- The bundled README implies uniform image size. It is not uniform: 4,881 of
  6,000 are 512x512, and the remaining 1,119 span 174x230 up to 1365x1365. All
  are resized to 224x224 before inference, so this does not affect results, but
  "512x512 dataset" is not an accurate description.

**All 6,000 are unseen by the model in the sense that none were used for
training, so all 6,000 are valid to score.** BRISC's own 1,000-image test split
is also reported separately so numbers are comparable to published BRISC results.

Note the class balance: no_tumor is only 14% of BRISC's own test split. Any
specificity or false-alarm number computed on that split rests on 140 images.

---

## The independence problem

**BRISC is not an independent external cohort, and this project should stop
describing it as one without qualification.**

The BRISC authors state, in the *Scientific Data* paper, that BRISC images were
collated from:

- the Cheng / Figshare brain tumor dataset,
- SARTAJ ("Brain Tumor Classification (MRI)"),
- Br35H ("Brain Tumor Detection 2020"),

aggregated through the Kaggle "Brain Tumor MRI Dataset" (Nickparvar) collection.

That is the training set. Same three upstream sources, same aggregator.

The authors also write: *"While complete subject-level independence cannot be
guaranteed due to the source limitations, this conservative approach ensures no
obvious same-patient images cross splits; multiple images from the same subject
may therefore be present within a single split."* Their de-duplication was
internal to BRISC. Nobody de-duplicated BRISC against this project's training
split.

### What that changes

A BRISC score is **not** evidence that the model works on scans from another
hospital, another scanner, or another population. At best it is evidence that
the model works on a cleaner, expert-re-annotated, differently-preprocessed cut
of its own source pool.

That is still worth having. It tests robustness to re-encoding, re-cropping,
label correction and a different class balance, and it is a genuinely harder test
than the internal held-out split. It is not external validation in the sense a
regulator or a clinical reviewer means it.

### What we measured about it

`docs/check_brisc_overlap.py` matches every BRISC image against every internal
image by perceptual hash, at Hamming distance ≤ 5. That is the same threshold
`src/code.py::_phash_cluster_ids` uses to decide two images are near-duplicates
and must not land on opposite sides of a train/test boundary. So a match here
means the internal split logic itself would have called these the same image.

Results: see `docs/results/brisc_overlap.json` and the summary in
[MODEL_CARD.md](../MODEL_CARD.md).

Where the overlap is material, the model card reports BRISC metrics **twice**:
on all 6,000, and on the non-overlapping subset only. The non-overlapping number
is the one that means something.

---

## What would actually close the external validation gap

Not BRISC. In rough order of cost:

1. **Sub-source stratification inside the existing training data.** Identify
   which of Br35H / SARTAJ / Figshare each of the 7,200 images came from, and
   report per-source accuracy. Cheapest possible generalisation signal, no new
   data. Still not attempted. Flagged since the original
   `analysis/EXTERNAL_VALIDATION_GAP.md` and still open.
2. **A public brain MRI collection with no lineage back to Br35H, SARTAJ or
   Figshare.** TCIA collections are the obvious candidate. This needs someone to
   check provenance carefully, because the field re-mixes the same handful of
   public datasets constantly, which is exactly how we ended up here.
3. **A prospectively collected institutional cohort** with recorded scanner make,
   field strength, protocol and demographics. IRB, data-sharing agreements, real
   money, real time. This is what a regulator will eventually want.

---

## Attribution block

Reproduce this wherever the work is presented.

> Training data: Nickparvar, M. (2023). *Brain Tumor MRI Dataset*. Kaggle
> (CC0 1.0). Aggregated from Br35H (Brain Tumor Detection 2020), SARTAJ (Brain
> Tumor Classification (MRI)), and the Figshare brain tumor dataset of Cheng et
> al.
>
> Evaluation data: Fateh, A., Rezvani, Y., Moayedi, S., Rezvani, S., Fateh, F.,
> Fateh, M., & Abolghasemi, V. (2026). *BRISC: Annotated Dataset for Brain Tumor
> Segmentation and Classification*. Scientific Data. arXiv:2506.14318.
> Licensed CC BY 4.0. No changes were made to the images; they were resized at
> inference time only.
