# analysis/ — Handoff

---

> ## Update, 2026-07-31 — two statements below are superseded. Original preserved.
>
> Added by session E. **Nothing below has been deleted.**
>
> ### 1. "No external cohort exists for this project" is superseded
>
> A candidate was found after this was written: **BRISC 2025** (Fateh et al.,
> *Scientific Data* 2026, arXiv:2506.14318), 6,000 T1 images. The checkpoints were
> run on it, inference only.
>
> **It is not an independent cohort.** 4,787 of its 6,000 images are
> byte-identical files to images in `data/split_manifest.csv`, and 3,353 are in
> the train split. Its own paper states it was collated from the same three
> sources via the same Kaggle merge used for training. Of its 4,793 tumor-bearing
> images, the model has seen 4,735.
>
> **So the conclusion of `EXTERNAL_VALIDATION_GAP.md` is unchanged: this project
> has no external validation.** The statement is now measured rather than
> inferred from a failed search.
>
> Full write-up: [`../docs/DATA_PROVENANCE.md`](../docs/DATA_PROVENANCE.md).
> Reproduce: `python docs/check_brisc_overlap.py`.
> Session A's BRISC analysis: `analysis/results/brisc/`.
>
> ### 2. "OOD evaluation was not run, no image set was available" is superseded
>
> Session B is building an out-of-scope image set and an input rejector. See
> `analysis/results/ood/`, `src/input_validation.py` and
> `reproducibility/out_of_scope_data.md`.
>
> ### 3. "Status: complete" was true for this directory's original scope
>
> It is no longer the whole picture. `analysis/` now also carries session A's
> external validation and safety work, session B's rejection analysis, and
> session D's clinical explainability work, none of which existed when this was
> written.
>
> ### What has not changed
>
> Everything this document says about the 5-seed x 2-backbone run, the
> calibration, explainability, power and baseline analyses, and the smoke-test
> provenance is still accurate. The `[PLACEHOLDER: ...]` markers it points at in
> `reproducibility/README.md` have all been resolved; the ones in
> `manuscript/manuscript.md` have not, and that file is a historical snapshot
> now. See [`../manuscript/README.md`](../manuscript/README.md).

---

## Original document. Preserved unchanged.

**Status: complete.** The full 5-seed × 2-backbone training run
(`results/20260703_155524/`, finalized 2026-07-09) is real, not a smoke
test, and every script in this directory has been (re-)run against it:

- `analysis/calibration_comparison.py` — temperature-scaling calibration,
  all 5 seeds × 2 models, against the real checkpoints. Output:
  `analysis/results/calibration/calibration_comparison.{json,md}` +
  reliability PNGs, no `is_smoke_test_run` flags remaining.
- `analysis/explainability.py` — Grad-CAM (ResNet-50) / attention rollout
  (ViT-B/16), all 5 seeds × 2 models, against the real checkpoints. Output:
  `analysis/results/explainability/explainability_summary.{json,md}` +
  per-class heatmap PNGs under `analysis/results/explainability/{resnet50,vit}/`.
- `analysis/power_analysis.py` — already run against the real per-seed
  McNemar results; see `analysis/results/power_analysis.md`.
- `analysis/baseline_cnn.py` — SimpleCNN-from-scratch baseline, already a
  real (non-smoke) run; see `analysis/results/baseline/baseline_comparison.md`.

`results/20260620_165427/` is the earlier smoke-test run (1 epoch, seed 42
only) and is not the source of any number cited in `manuscript/manuscript.md`.

OOD entropy-detection evaluation (`src/code.py --ood-dir`) was not run — no
out-of-distribution image set was available in this project. Documented as
not-performed in the manuscript, not left as an open placeholder.

External validation status: **no external cohort exists** for this project
— see `analysis/EXTERNAL_VALIDATION_GAP.md` for what was checked and why.
This is a fixed limitation of the current work, not a pending item.

Remaining gaps before submission are all outside this directory's scope:
manuscript prose (citations, discussion framing) and non-technical metadata
(funding, COI, ethics approval, protocol/registration, dataset DOI
confirmation) — see the `[PLACEHOLDER: ...]` markers still in
`manuscript/manuscript.md` and `reproducibility/README.md`.
