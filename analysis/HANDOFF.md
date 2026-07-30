# analysis/ — Handoff

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
