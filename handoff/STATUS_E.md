# Session E status — docs, model card, overclaim enforcement

**Last updated: 2026-07-31T04:00Z. Exit criteria met. Still watching for A, B
and D so I can fill in their numbers.**

## The two headlines

**1. BRISC is not external validation, and it never could have been.** 4,787 of
its 6,000 images are byte-identical files to images in our training pool. 3,353
are in the train split. Session A found the same thing independently by pixel
comparison. BRISC has 4,793 tumor images and the model has seen 4,735, so the
tumor miss rate cannot be estimated on it at any threshold.

**2. The misses are systematic, not random, and that changes decisions.** All 88
missed-tumor events across all 10 checkpoints on the internal test split come
from **18 distinct images out of 821**. Nine images account for 81%. One image,
`Te-gl_74.jpg`, is missed by every checkpoint this project has ever trained, at
93% confidence.

Two consequences worth acting on:

- **Ensembling will not fix the miss rate.** The errors are correlated across
  seeds *and* across architectures. Averaging members does not cancel them.
  Session A: this is directly relevant to your single-seed-vs-ensemble choice.
- **Every pooled confidence interval in this project is optimistic**, mine
  included. The effective sample size behind the miss rate is closer to a dozen
  hard cases than to 821 images.

## Published, use freely

| file | what |
|---|---|
| `MODEL_CARD.md` | read before trusting any output. Leads with the miss rate. |
| `README.md` | for an external reader. Leads with "not for clinical use". |
| `LIMITATIONS.md` | blunt list of everything not done. |
| `docs/FOR_CLINICIANS.md` | plain words, no maths, for a non-technical reader. |
| `docs/DATA_PROVENANCE.md` | both licenses resolved, BRISC independence analysis. |
| `docs/OVERCLAIM_AUDIT.md` | every overclaim found and what happened to it. |
| `docs/OPEN_QUESTIONS.md` | decisions needing a human. This repo has no license. |
| `reproducibility/CHECKPOINT_DEPOSIT.md` | Zenodo instructions for 3.4 GB of checkpoints. |
| `docs/results/internal_safety_metrics.json` | miss rate, per-class, deferral, confident misses. |
| `docs/results/confident_misses.json` | the 18 images, named, with per-seed confidences. |
| `docs/results/confident_miss_examples/` | those images, copied, so a human can look. |
| `docs/results/brisc_overlap_flags.csv` | per-image overlap flags, strict definition. |
| `docs/overclaim_grep.sh` | run before any release. |

## Numbers other sessions may want

Internal held-out test split, n=1,112 (821 tumor, 291 no-tumor), MC-Dropout T=20,
pooled over 5 seeds, Wilson intervals. **Internal. Not a real-world estimate.**

| metric | ResNet-50 | 95% CI | ViT-B/16 | 95% CI |
|---|---|---|---|---|
| tumor miss rate | 1.00% | 0.74–1.35 | 1.14% | 0.86–1.52 |
| ...glioma | 1.88% | 1.28–2.74 | 2.24% | 1.58–3.16 |
| ...meningioma | 1.13% | 0.68–1.85 | 1.20% | 0.74–1.95 |
| ...pituitary | 0.00% | 0.00–0.28 | 0.00% | 0.00–0.28 |
| binary sensitivity | 99.00% | 98.65–99.26 | 98.86% | 98.48–99.14 |
| binary specificity | 98.28% | 97.48–98.83 | 98.01% | 97.15–98.61 |
| four-way accuracy | 0.964 | — | 0.962 | — |
| misses caught deferring top 5% | 73% | — | 36% | — |
| missed tumors at ≥90% confidence | 10% | — | 36% | — |

**Choose ResNet-50.** McNemar says the backbones are indistinguishable on
accuracy. They are not indistinguishable on failure behaviour, and that is the
difference that matters clinically.

Reproduce: `python docs/internal_safety_metrics.py`, `python docs/confident_misses.py`.
Session A owns the canonical safety analysis; if A disagrees, A wins.

## Exit criteria

- [x] `MODEL_CARD.md` complete, every number with a CI and a dataset name.
- [x] `README.md` for an external reader.
- [x] `LIMITATIONS.md`.
- [x] Overclaim grep run repo-wide, every hit resolved or justified in
      `docs/OVERCLAIM_AUDIT.md`.
- [x] Session C's UI text reviewed. Nine corrections sent via `handoff/ISSUES.md`.
- [x] BRISC provenance and license recorded and confirmed (CC BY 4.0).
- [x] Kaggle license resolved (CC0 1.0, redistribution permitted).
- [x] Zenodo deposit instructions written.
- [x] `patient_id` bug fixed, manifest regenerated, **split confirmed identical**.
- [x] `EXTERNAL_VALIDATION_GAP.md` and `HANDOFF.md` corrected, originals preserved.
- [x] **Zero `[PENDING: ...]` markers in any published document.** Verify with
      `grep -rn "PENDING:" --include=*.md . | grep -v prompts | grep -v handoff`
- [x] Nothing pushed to the publication repo. One remote configured, and it is
      the clinical repo.

## What is still open, and it is on me to say so

Sessions A, B and D had not published their results when I finished. Rather than
leave placeholders that could survive into a final document, every slot now
carries an explicit statement of what does not exist and where it will appear.
No number was invented anywhere.

**A, B, D:** when you publish, the slots are labelled and easy to find:

```
grep -n "Not available\|Untested\|not been finalised\|not systematically measured" MODEL_CARD.md LIMITATIONS.md README.md
```

Three of them are stronger as statements than they would be as numbers, and I
would push back before replacing them:

- **Plane subgroups are effectively unmeasurable here.** The clean BRISC subset
  is 1,198 images, 95% no-tumor. Split three ways that is roughly 19 tumors per
  plane.
- **A BRISC miss rate is not a generalisation estimate**, whatever it turns out
  to be, and the model card will not present it as one.
- **The category-4 case** (brain MRI with pathology the model has no class for)
  is fully in-distribution to any uncertainty score. No entropy threshold catches
  it, because the model is not uncertain, it is wrong.

## Two notes for whoever reads this after

**A's smoke alarm belongs in every future protocol.** ViT seed 42 scored 0.9728
on "external" BRISC and 0.9613 on its own internal test split. A model does not
beat its own held-out set on genuinely new data. That one comparison would have
caught this on day one, for free.

**The cheapest unfinished work in this project is still sub-source
stratification.** Identify which of Br35H / SARTAJ / Figshare each of the 7,200
training images came from and report per-source accuracy. No new data needed.
Flagged in `analysis/EXTERNAL_VALIDATION_GAP.md` section 1 long before this
session and still not attempted. It is the only generalisation signal available
without acquiring a cohort.

**And one hour of a radiologist's time is worth more than another training run.**
The 18 images that cause every miss are sitting in one folder. Nobody clinically
qualified has ever looked at them, or at any label in this project.

---

# Integration update, 2026-07-30T23:40Z. Every slot is filled.

A, B and D published. Every `Not available` / `Untested` / `had not published`
marker in `MODEL_CARD.md`, `LIMITATIONS.md` and `README.md` now carries a real
number with its dataset and its n. Verified: the grep from the exit criteria
returns nothing.

## Three places where my own text was wrong, corrected in place

**1. Plane subgroups are measurable after all.** I wrote that they were
"effectively unmeasurable" because the clean subset was 1,198 images at 95%
no-tumor. That was `clean_vs_any`, the strictest subset. Session A's canonical
subset is `clean_vs_train`: 2,634 images, roughly 490 tumours per plane. The
breakdown is now in the card, and it found something: the miss rate is flat
across planes but **specificity is not**. ResNet-50 drops to 71.4% on sagittal.

**2. Same error on the miss rate.** I wrote "the clean subset contains 58
tumors, which cannot support a rate". Also `clean_vs_any`. The canonical subset
has 1,476 tumours.

**3. "Ensembling will not fix the miss rate" was too strong.** My correlated-
error finding is real: 18 images cause all 88 internal misses and they are
missed across seeds *and* architectures. But the inference did not hold. On the
clean BRISC subset the 5-seed ensemble roughly halved the miss rate, 0.68% to
0.27%. Corrected in the README with the reasoning shown, not silently.

The lesson for whoever reads this: I was reasoning from the strictest subset in
two places, and it made me argue against measuring things that were measurable.
"Not enough data" is a claim that needs checking like any other.

## The finding I would put in front of a clinician first

**It is not the miss rate. It is specificity.**

| | internal test | BRISC clean subset |
|---|---|---|
| tumour miss rate | 1.00% | 0.27% |
| specificity | 98.3% | **90.7%** |

The number this project watches hardest holds up on unseen data. The one nobody
was watching fell 8 points, and for ResNet-50 nearly 17. **Roughly 1 healthy
person in 11 gets a false alarm**, and in a rural setting that is travel, money,
time and fear.

The tool is safe in the direction it was designed to be safe in and expensive in
the other one. Both are now in the README and the card, in that order.

## What I added beyond filling slots

- **Per-source performance** in the model card. Sources inferred from a PIL
  image-mode fingerprint that reproduces Figshare's published class counts to
  within one image on two of three classes. SARTAJ's miss rate is ~27x
  Figshare's. This was the "cheapest unfinished work in this project" I flagged
  last run, and it is done.
- **Three new entries in `docs/OPEN_QUESTIONS.md`**: the 41% deferral trade, the
  SARTAJ model-vs-labels ambiguity, and the fact that nobody clinically
  qualified has looked at anything.

## What has not changed and must not be softened

The lead paragraph still says nobody knows whether this works on new patients.
That is still true. 2,634 images that survived removing byte-identical overlap
are not an external cohort: same sources, same preprocessing, and patient-level
overlap cannot be excluded because neither dataset ships patient identifiers.

**The app's "NOT FOR CLINICAL USE" banner nearly disappeared during this
integration**, because readiness state flipped to `clinical` the moment the
config files stopped being stubs. Session C fixed it with a fail-safe check.
Worth knowing that the overclaim almost arrived through the state machine rather
than through prose.
