# Reproducibility package

## Environment

```bash
# pip — this is the verified path
pip install -r reproducibility/requirements.txt

# conda — interpreter only, pins still come from requirements.txt
conda env create -f reproducibility/environment.yml
conda activate mri-algo
```

### What is verified and what is not

Checked 2026-07-31, session E.

**Verified.** `pip install -r reproducibility/requirements.txt` resolves cleanly
against PyPI on Python 3.10.11, Windows 11. Every pinned version in that file
matches what is actually installed and what actually produced the reported
results, with one exception: `seaborn` is not installed and is not needed by
anything under `figures/` or `analysis/`. The file already says so.

**Fixed, because it was broken.** The previous `environment.yml` asked conda for
`pytorch=2.12.1` and `torchvision=0.27.1` from the `pytorch` channel. That
channel's newest build is 2.5.1 and it was last updated in March 2025, so the
solve step could never have succeeded. The file now installs the interpreter
through conda and everything else through pip, which is PyTorch's own current
guidance.

**Not verified.** The conda path end to end. Conda is not installed on this
machine, so `conda env create` has not been run against the corrected file. It
is straightforward and should work. It has not been proven to.

**Not verified.** That a fresh install reproduces the published numbers
bit-for-bit. Nobody has retrained from scratch on a second machine. The
checkpoints are not deposited anywhere yet, so this cannot currently be checked
by anyone at all.

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
ran on CPU. Windows 11 (10.0.26200), torch 2.12.1+cpu. `DEVICE =
torch.device("cuda" if torch.cuda.is_available() else "cpu")` fell back to CPU
throughout. **No GPU was used for the 5-seed x 2-backbone run.**

Machine, recorded 2026-07-31 by session E:

- Intel Core Ultra 9 275HX, 24 cores / 24 logical processors, 2.7 GHz base
- 32 GB RAM

Caveat, and it matters: this is the machine the repository sits on **now**. The
training run finished 2026-07-09 and the hardware was not recorded at the time.
It is very likely the same machine and it has not been verified. Treat the spec
above as "the machine this was reproduced and audited on", not as a certified
record of the training host.

## Data and code availability

- **Code:** <https://github.com/medsharma/Brain-Tumor-Algo-for-Clinics>.
  No DOI. If one is needed, mint it from the Zenodo deposit described in
  `reproducibility/CHECKPOINT_DEPOSIT.md`, which can archive the repository at
  the same time.
- **Code license: not yet chosen.** This is an open question, not an oversight
  waiting to be typed in. See `docs/OPEN_QUESTIONS.md`. Until it is resolved,
  the repository is public but has no grant of rights, which means the default
  applies: all rights reserved, and nobody may legally reuse the code.
- **Trained model checkpoints: not deposited anywhere.** 3.4 GB across 10 files,
  each over GitHub's 100 MB limit, currently existing only on the original
  machine. **Nobody outside that machine can verify any number in this project.**
  Deposit instructions and a DOI slot: `reproducibility/CHECKPOINT_DEPOSIT.md`.
  DOI: `[PENDING: Zenodo DOI]`. sha256 for all ten:
  `MODEL_CARD.md`, "Checkpoint hashes".
- **Training data:** 4-class brain tumor MRI dataset (glioma / meningioma /
  pituitary / no-tumor), 1,800 images per class (7,200 total) under
  `data/brain_tumor/<class>/`. The public Kaggle "Brain Tumor MRI Dataset"
  (Nickparvar, M., 2023,
  <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>),
  a merge of the Br35H, SARTAJ and Figshare (Cheng et al.) sources.

  **License: CC0 1.0 Public Domain, as stated on the Kaggle listing.
  Redistribution of the 7,200 images in this public repository is permitted.**
  This resolves the placeholder that previously sat here. Three caveats apply
  and they are written out in `docs/DATA_PROVENANCE.md`: CC0 was asserted by
  the aggregator rather than the original custodians, the Figshare/Cheng
  upstream is CC BY 4.0 and we attribute all three sources regardless, and the
  live Kaggle listing now describes 7,023 images against this copy's 7,200
  (believed to be Version 2, not confirmed against a fresh download).

  No DOI is minted by Kaggle. Cite as Nickparvar 2023, Version 2.
- **Evaluation data:** BRISC 2025 (Fateh et al., *Scientific Data* 2026,
  arXiv:2506.14318). **License CC BY 4.0**, use permitted with attribution. Not
  redistributed here. Integrity hashes and composition:
  `docs/DATA_PROVENANCE.md`.

  **BRISC is not an independent external cohort.** Its own paper states it was
  collated from the same three sources via the same Kaggle merge used for
  training, and 4,787 of its 6,000 images are byte-identical files to training
  -pool images. Reproduce that check with `python docs/check_brisc_overlap.py`.
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

OOD entropy-detection evaluation (`--ood-dir`) was not run at the time this
section was written, because no out-of-distribution image set existed for the
project. **Superseded as of 2026-07-31:** session B is building an out-of-scope
image set and a rejector. Current state: `analysis/results/ood/` and
`reproducibility/out_of_scope_data.md`.

## External validation, and what changed on 2026-07-31

Added by session E.

The original version of this document, `analysis/EXTERNAL_VALIDATION_GAP.md` and
`analysis/HANDOFF.md` all stated that no external dataset existed for this
project. A candidate was then found and evaluated: BRISC 2025, 6,000 T1 images,
expert-annotated, published 2025.

It does not qualify. The BRISC paper states its images were collated from
Cheng/Figshare, SARTAJ and Br35H via the Kaggle Nickparvar merge, which is the
training data. Direct measurement:

- 4,787 of 6,000 BRISC images are **byte-identical files** (matching sha256) to
  images in `data/split_manifest.csv`
- 4,802 of 6,000 match at perceptual-hash Hamming distance ≤ 5, of which 4,791
  are at distance 0
- 3,353 match an image in the **train** split
- BRISC holds 4,793 tumor-bearing images; the model has seen 4,735 of them

Reproduce:

```bash
python docs/check_brisc_overlap.py       # ~15 min, CPU only
python docs/make_brisc_overlap_flags.py  # per-image join table
```

Outputs land in `docs/results/`. Full write-up in `docs/DATA_PROVENANCE.md`.

**The conclusion of the original gap document is unchanged. This project has no
external validation. It now has evidence for that statement rather than an
absence of evidence against it.**

## Reproducing the safety numbers in the model card

```bash
python docs/internal_safety_metrics.py
```

Reads the cached predictions in `analysis/results/*_predictions.json` (from run
`20260703_155524`) and writes `docs/results/internal_safety_metrics.json`: tumor
miss rate with Wilson intervals, per-class miss rate, binary sensitivity and
specificity, deferral behaviour at 5/10/20%, and the confidence distribution of
missed tumors. No GPU, no model loaded, a few seconds.

Session A owns the canonical safety analysis under `analysis/results/safety/`.
If the two disagree, A's is canonical.
