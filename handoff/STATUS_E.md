# Session E status — docs, model card, overclaim enforcement

**Last updated: 2026-07-31T00:54Z**

## Headline for the other sessions

**BRISC is not an independent external cohort.** Its own paper says it was
collated from Cheng/Figshare + SARTAJ + Br35H via the Kaggle Nickparvar merge.
That is the dataset we trained on. Full detail and what I am asking of each
session is in `handoff/ISSUES.md`, first entry. Please read it before you write
the word "external" in anything user-facing.

## Done

- Worktree `../mri-E` on `session/E` created. (It did not exist; I made it.)
- Read `prompts/CONTRACTS.md`, `MISSION.md`, `results/leakage_audit.md`,
  `analysis/EXTERNAL_VALIDATION_GAP.md`, `analysis/HANDOFF.md`,
  `reproducibility/README.md`, `results/master_summary.json`.
- BRISC provenance and license researched and confirmed from the peer-reviewed
  source, not the bundled README.
- Kaggle Nickparvar license question researched.
- `handoff/ISSUES.md` created with the BRISC finding and advance notice of the
  `patient_id` fix.

## In flight

- `docs/check_brisc_overlap.py` running. Measures phash near-duplicate overlap
  between all 6,000 BRISC images and all 7,200 internal images, bucketed by
  internal split. Answers "how much of BRISC did the model already see."

## Next

- `MODEL_CARD.md`, `README.md`, `LIMITATIONS.md` structure and all
  results-independent prose.
- `docs/DATA_PROVENANCE.md`, Zenodo deposit instructions, environment check.
- Repo-wide overclaim grep.
- `patient_id` fix, late.

## Blocked on

Nothing yet. Numbers from A, B, C and D. Every slot for a number I do not have
is marked `[PENDING: ...]` and will be zero by the time I finish.

## Numbers other sessions may need from me

- BRISC / internal train overlap fraction: pending, script running.
- Internal test-split accuracy from `results/master_summary.json`
  (`results/20260703_155524`, n=1,112, held out, phash-cluster grouped):
  ResNet-50 mean 0.9642 across 5 seeds; ViT-B/16 mean 0.9615 across 5 seeds.
  These are **internal** numbers. Do not present them as real-world performance.
