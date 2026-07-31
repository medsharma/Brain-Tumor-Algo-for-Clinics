# Session E status — docs, model card, overclaim enforcement

**Last updated: 2026-07-31T02:05Z**

## Headline

**4,787 of BRISC's 6,000 images are byte-identical files to images in our
training pool. 3,353 are in the train split.** Session A found the same thing
independently, by pixel comparison. We agree. Detail in `handoff/ISSUES.md`.

BRISC has 4,793 tumor images and the model has seen 4,735. **The tumor miss rate
cannot be estimated on this dataset.**

**Internal held-out test split, n=1,112, 821 tumor images, 5 seeds:
ResNet-50 misses 1.00% of tumors (95% CI 0.74 to 1.35). ViT misses 1.14%.**
That is the honest headline for the project and it is on data far easier than
anything real.

## Published, use freely

| file | what |
|---|---|
| `MODEL_CARD.md` | the document someone reads before trusting this. Leads with the miss rate. |
| `README.md` | for an external reader. Leads with "not for clinical use" and the miss rate. |
| `LIMITATIONS.md` | blunt list of everything not done. |
| `docs/DATA_PROVENANCE.md` | both dataset licenses resolved, BRISC independence analysis. |
| `docs/OVERCLAIM_AUDIT.md` | every overclaim found and what happened to it. |
| `docs/OPEN_QUESTIONS.md` | decisions needing a human, starting with: this repo has no license. |
| `reproducibility/CHECKPOINT_DEPOSIT.md` | Zenodo instructions for the 3.4 GB of checkpoints. |
| `docs/results/internal_safety_metrics.json` | miss rate, per-class, deferral, confident-miss data. |
| `docs/results/brisc_overlap_flags.csv` | per-image overlap flags, strict definition. |
| `docs/overclaim_grep.sh` | run before any release. |

## Numbers other sessions may want

Internal held-out test split, n=1,112 (821 tumor, 291 no-tumor), MC-Dropout T=20,
pooled over 5 seeds. **Internal. Not a real-world estimate. Never present as one.**

| metric | ResNet-50 | ViT-B/16 |
|---|---|---|
| tumor miss rate | 1.00% (0.74–1.35) | 1.14% (0.86–1.52) |
| ...glioma | 1.88% | 2.24% |
| ...meningioma | 1.13% | 1.20% |
| ...pituitary | 0.00% | 0.00% |
| binary sensitivity | 99.00% | 98.86% |
| binary specificity | 98.28% | 98.01% |
| four-way accuracy | 0.964 | 0.962 |
| misses caught by deferring top 5% | 73% | 36% |
| missed tumors called no-tumor at ≥90% confidence | 10% | 36% |

**Two things in there that matter more than the accuracy row:**

1. **The backbones are not interchangeable on safety.** McNemar says their
   accuracy is indistinguishable. Their failure behaviour is not. ResNet-50's
   uncertainty flags 73% of its own missed tumors at 5% deferral; ViT's flags
   36%. Choose ResNet-50.
2. **Confident misses exist and no threshold catches them.** ViT calls a third
   of its missed tumors "no tumor" at 90%+ confidence.

Reproduce: `python docs/internal_safety_metrics.py`. Session A owns the
canonical safety analysis; if A disagrees, A wins.

## Done

- BRISC provenance traced to the peer-reviewed source. License CC BY 4.0.
- Kaggle Nickparvar license resolved: CC0 1.0. Redistribution permitted.
- BRISC/internal overlap measured and verified by re-hashing actual files.
- `MODEL_CARD.md`, `README.md`, `LIMITATIONS.md` written.
- All four `[PLACEHOLDER: ...]` markers in `reproducibility/README.md` resolved.
- **`environment.yml` was broken and is fixed.** It pinned `pytorch=2.12.1` from
  the pytorch conda channel, whose newest build is 2.5.1 and which stopped
  publishing in March 2025. `conda env create` could never have solved. Now
  installs the interpreter via conda and pins via pip. The pip path is verified;
  the conda path is marked unverified because conda is not installed here.
- Pipeline smoke test run end to end. Passes.
- **`patient_id` bug fixed. Split confirmed byte-identical.** Verified in two
  steps: regenerated from unmodified code first to prove determinism, then with
  the fix. 0 split disagreements either time. Manifest now carries 4,784 cluster
  IDs and 0 clusters straddle a split boundary, so the leakage claim is checkable
  from a fresh clone. Nothing invalidated, nothing needs re-running.
- `analysis/EXTERNAL_VALIDATION_GAP.md` and `analysis/HANDOFF.md` corrected with
  dated headers, originals preserved verbatim.
- `manuscript/README.md` snapshot notice written. Nothing pushed anywhere near
  the publication repo.
- Repo-wide overclaim sweep run. Results in `docs/OVERCLAIM_AUDIT.md`.

## Blocked on

- **Session C.** Nothing pushed yet. UI text review is an exit criterion and I
  cannot do it against a branch that does not exist. Polling. The standard C's
  text will be held to is written out at the end of `docs/OVERCLAIM_AUDIT.md`,
  so C can self-check before I get there.
- **Session B.** Nothing pushed yet. `MODEL_CARD.md` and `LIMITATIONS.md` have
  `[PENDING: session B ...]` slots for out-of-scope rejection and the category-4
  case.
- **Session A.** BRISC metrics on the clean subset, deployment config, confident
  -miss case list.
- **Session D.** Localisation failure cases for the model card.

Every `[PENDING: ...]` marker will be gone before I finish. Count them any time
with `grep -rn "PENDING:" --include=*.md .`

## Two notes for whoever reads this after

**A's smoke alarm belongs in every future protocol.** ViT seed 42 scored 0.9728
on "external" BRISC and 0.9613 on its own internal test split. A model does not
beat its own held-out set on genuinely new data. That one comparison would have
caught this on day one, for free.

**The cheapest unfinished work in this project is still sub-source
stratification.** Identify which of Br35H / SARTAJ / Figshare each of the 7,200
training images came from and report per-source accuracy. No new data needed.
Flagged in `analysis/EXTERNAL_VALIDATION_GAP.md` section 1 well before this
session and still not attempted. It is the only generalisation signal available
without acquiring a cohort.
