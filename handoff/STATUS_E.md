# Session E status — docs, model card, overclaim enforcement

**Last updated: 2026-07-31T01:20Z**

## Headline for the other sessions

**4,787 of BRISC's 6,000 images are byte-identical files to images in our
training pool. 3,353 of them are in our train split. BRISC is not external
validation and cannot be turned into it.**

Full detail, per-session instructions, and the join table are in
`handoff/ISSUES.md`, entry 3. Please read it before you publish any BRISC number
or write the word "external" anywhere.

The one-line version: BRISC has 4,793 tumor images and the model has seen 4,735
of them. The tumor miss rate cannot be measured on this dataset.

## Published for other sessions to use

- `docs/results/brisc_overlap_flags.csv` — one row per BRISC image. Join on
  `image_path`, which matches Contract 1's `image_path` exactly. Use
  `usable_as_external` (True for 1,198 images) as the strict filter.
- `docs/results/brisc_overlap.json` — full match list with distances and the
  internal file each BRISC image matched.
- `docs/check_brisc_overlap.py`, `docs/make_brisc_overlap_flags.py` — the code,
  so nobody has to take my word for it. CPU only, no GPU used, no labels fitted.

## Done

- Worktree `../mri-E` on `session/E` created. It did not exist; I made it.
- BRISC provenance traced to the peer-reviewed source, license confirmed CC BY 4.0.
- Kaggle Nickparvar license resolved: CC0 1.0. Redistribution permitted.
  Written up with the caveats in `docs/DATA_PROVENANCE.md`.
- BRISC / internal overlap measured and verified by re-hashing actual files.
- `docs/DATA_PROVENANCE.md` written.

## Next

- `MODEL_CARD.md`, `README.md`, `LIMITATIONS.md`.
- Zenodo deposit instructions, environment verification.
- Repo-wide overclaim grep, session C UI text review.
- `patient_id` fix, late in the run.

## Blocked on

Numbers from A, B, C and D. Every slot for a number I do not have is marked
`[PENDING: ...]` and will be gone before I finish.

## Numbers other sessions may need from me

- Internal held-out test split, n=1,112, phash-cluster grouped, from
  `results/master_summary.json` (`results/20260703_155524`):
  - ResNet-50 four-way accuracy, mean over 5 seeds: 0.964 (range 0.960 to 0.967)
  - ViT-B/16 four-way accuracy, mean over 5 seeds: 0.962 (range 0.957 to 0.964)
  - These are **internal** numbers on a split of the same pool. They are not a
    real-world performance estimate and must not be presented as one.
- BRISC overlap, all measured: 4,787 byte-identical, 4,802 near-duplicate or
  identical, 3,353 in our train split, 1,198 with no match, 58 unseen tumor
  images.
