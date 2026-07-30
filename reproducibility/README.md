# Reproducibility package

## Environment

```bash
# conda
conda env create -f reproducibility/environment.yml
conda activate mri-algo

# or pip
pip install -r reproducibility/requirements.txt
```

Verified against Python 3.10.11 on Windows 11 (CPU-only torch build — see
**Hardware** below for why that is *not* what training should run on).

## Seeds

Five seeds, fixed in `src/code.py::SEEDS` and used for every multi-seed run,
bootstrap CI, and per-seed McNemar's test in the manuscript:

```
42, 123, 7, 2024, 31
```

`setup_reproducibility(seed)` fixes `random`, `numpy`, `torch` (CPU + CUDA),
sets `cudnn.deterministic = True`, `cudnn.benchmark = False`, and
`PYTHONHASHSEED`, for each seed independently.

## Run commands

**1. Build the leakage-safe split manifest** (patient-ID grouping →
phash-cluster grouping → per-file stratified fallback; 70/15/15):

```bash
python src/code.py --data-root ./data/brain_tumor --manifest data/split_manifest.csv --force-manifest
```

**2. Pipeline smoke test** (1 epoch, seed 42 only, 8 samples/class, MC-Dropout
T=2 — validates the code path end-to-end in minutes, produces no numbers
that belong in the manuscript):

```bash
python src/code.py --smoke
```

**3. Full multi-seed training + evaluation** (both backbones, all 5 seeds,
MC-Dropout T=20, bootstrap 95% CIs, per-seed McNemar's test):

```bash
python src/code.py \
  --data-root ./data/brain_tumor \
  --manifest data/split_manifest.csv \
  --results-dir results \
  --epochs 30 --batch-size 32 --lr 1e-4 --weight-decay 0.01 \
  --label-smoothing 0.1 --workers 4 --mc-T 20 --early-stop-patience 5 \
  --seeds 42 123 7 2024 31 --model both
```

Writes `results/<timestamp>/{vit,resnet50}/seed_<seed>/` (checkpoint,
`config.json`, `summary.json`, per-run diagnostic PNGs) and
`results/<timestamp>/comparison_summary.json` (the aggregate + per-seed
McNemar table figures/mcnemar_table.py consumes).

**4. Optional OOD evaluation** (entropy-based AUROC vs. an out-of-distribution
image directory):

```bash
python src/code.py --model both --seeds 42 123 7 2024 31 --ood-dir /path/to/ood_images
```

**5. Analysis suite** (temperature-scaling calibration comparison; runs
against the latest completed `results/<timestamp>/comparison_summary.json`
automatically):

```bash
python analysis/calibration_comparison.py
```

**6. Manuscript figures** — see `figures/README.md` for the full command
list. Now running against the real completed run
(`results/master_summary.json` and `analysis/results/*_predictions.json`),
not the `figures/fixtures/` mock data.

## Hardware

Training and all evaluation (MC-Dropout T=20, calibration, explainability)
ran on a CPU-only machine (Windows 11, torch 2.12.1+cpu, `DEVICE =
torch.device("cuda" if torch.cuda.is_available() else "cpu")` fell back to
CPU throughout). **[PLACEHOLDER: add exact CPU model/core count if the
manuscript's reporting venue requires it — not recorded during the run.]**
No GPU was used for the full 5-seed × 2-backbone training run reported in
this manuscript.

## Data and code availability

- **Code:** this repository. **[PLACEHOLDER: public repository URL / DOI —
  fill in before submission.]**
- **Trained model checkpoints:** **[PLACEHOLDER: repository or archive URL —
  fill in once final multi-seed checkpoints exist.]**
- **Data:** 4-class brain tumor MRI dataset (glioma / meningioma / pituitary
  / no-tumor), 1,800 images per class (7,200 total) under
  `data/brain_tumor/<class>/` — the public Kaggle "Brain Tumor MRI Dataset"
  (Nickparvar, M., 2023, https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset;
  a combination of the Br35H, SARTAJ, and Figshare brain tumor sources).
  **Version 2** of that dataset is 7,200 images (Training + Testing folders),
  matching this copy exactly. Filename conventions (`Te-gl_1.jpg`, plus
  author-supplied `Te-aug-me_*` meningioma augmentations) confirm the match.
  **[PLACEHOLDER: confirm the license terms from the Kaggle listing before
  submission — the page surfaces no formal DOI, so cite as Nickparvar 2023,
  Version 2.]**
- **Splits:** `data/split_manifest.csv`, deterministic given `--data-root` and
  seed (patient-ID or phash-cluster grouped to prevent leakage); regenerate
  with `--force-manifest`.

## What is and isn't real yet

The full 5-seed × 2-backbone run completed on 2026-07-09
(`results/20260703_155524/`, `results/master_summary.json`,
`results/20260703_155524/comparison_summary.json`) — 30-epoch budget with
early stopping (patience=5), MC-Dropout T=20 at eval, bootstrap 95% CIs
(n=1000), per-seed McNemar's test. These are real findings, not a smoke
test; `results/20260620_165427/` is the earlier smoke-test run (1 epoch,
seed 42 only, 8 samples/class) and is not cited anywhere in the manuscript.
`analysis/calibration_comparison.py` and `analysis/explainability.py` have
been re-run against the real checkpoints in `results/20260703_155524/`
(all 5 seeds, both architectures) — see `analysis/results/calibration/` and
`analysis/results/explainability/`, no longer smoke-test-flagged.

OOD entropy-detection evaluation (`--ood-dir`) was not run — no
out-of-distribution image set was available for this project. This is
reported in the manuscript as not-performed, not as a missing result.
