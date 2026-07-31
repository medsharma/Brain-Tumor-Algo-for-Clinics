# Limitations

Everything this project has not done, cannot currently do, or does not know.

No hedging in this document. If something is untested it says untested. If
something is a hole it says hole. Where a limitation has a number attached, the
number is here.

Last updated: 2026-07-31. Companion to [MODEL_CARD.md](MODEL_CARD.md).

---

## The three that matter most

**1. There is no external validation. There never has been.**

BRISC 2025 was brought in to close this gap. It does not. 4,787 of its 6,000
images are byte-identical files to images in the training pool, and 3,353 are in
the train split specifically. Of BRISC's 4,793 tumor-bearing images, the model has
already seen 4,735. See [docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md).

This project has one dataset. Every number it has ever produced comes from that
one dataset.

**2. The model misses about 1 tumor in 100, on the easiest data it will ever see.**

Internal held-out test split, 821 tumor images, 5 seeds. ResNet-50 misses 1.00%
(95% CI 0.74% to 1.35%), ViT-B/16 misses 1.14%. Gliomas are missed at roughly
twice the rate of meningiomas. This is measured on a held-out split of the
training pool. Performance on genuinely new data is unknown and there is no
reason to assume it is better.

**3. Some missed tumors are confident misses, and no confidence threshold catches
those.**

ViT-B/16 calls 36% of its missed tumors "no tumor" with 90%+ confidence.
ResNet-50 is better at 10%, but not zero. Deferring the most uncertain 20% of all
cases still leaves 3 missed tumors (ResNet-50) or 17 (ViT) out of 821. The
uncertainty machinery is real and it helps. It is not a safety net that catches
everything, and it must never be described as one.

**4. The misses are systematic, so the confidence intervals in this project are
optimistic.**

Across all 10 checkpoints there are 88 missed-tumor events on the internal test
split, and they come from **18 distinct images out of 821**. Nine images account
for 81% of all misses. One image, `Te-gl_74.jpg`, is missed by every checkpoint
this project has ever trained, at 93% confidence.

Two consequences:

- **Ensembling will not fix it.** These errors are correlated across seeds and
  across architectures, so averaging members does not cancel them.
- **Every pooled confidence interval in this repository is narrower than the
  evidence justifies.** A Wilson interval over 821 tumor images assumes 821
  independent observations. Pooled across seeds, the effective sample size behind
  the miss rate is closer to a dozen hard cases. Treat the stated intervals as a
  floor on the uncertainty, not a description of it.

Nobody clinically qualified has checked whether those 18 images are even labelled
correctly. They are collected in `docs/results/confident_miss_examples/` and it
is about an hour of a radiologist's time.

---

## Clinical evidence: none

- **No reader study.** No radiologist has used this tool on real cases. Nobody
  has measured whether it helps a clinician, harms a clinician, or changes
  nothing. Every claim about clinical usefulness in this repo is a hypothesis.
- **No prospective data.** The tool has never been run in a clinic, on a live
  case, in real time.
- **No patient outcomes. Ever.** Not one. No evidence that anything this tool
  does changes what happens to a person.
- **No comparison against a human baseline.** We do not know the miss rate of the
  radiologists this tool would supplement, so we cannot say whether 1% is good,
  bad, or irrelevant in context.
- **No measurement of the referral pathway it would load.** At realistic clinic
  prevalence, most flags from this tool would be false positives (about 54% true
  at 2% prevalence). The cost of that to a rural referral system is unmeasured.

## Regulatory: nothing

- **No clearance.** Not FDA, not CE/MDR, not CDSCO, not any national authority.
- **No submission.** Nothing has been filed anywhere.
- **No regulatory pathway chosen.** The target country has not been selected, so
  the applicable device class and evidence requirements are unknown.
- **No quality management system.** No ISO 13485, no IEC 62304 lifecycle
  documentation, no risk file per ISO 14971.
- **No IRB or ethics approval**, because no human-subjects work has been done.
- **No post-market surveillance plan.** If deployed, model drift would go
  unnoticed.

## Data: the deepest problem

- **One dataset, one merge, three public sources.** Br35H, SARTAJ and the Figshare
  Cheng collection, merged by a third party on Kaggle.
- **No sub-source labels.** We cannot tell which of the three sources any given
  image came from, so **sub-source heterogeneity inside the training data is
  still unmeasured.** We cannot check whether the model performs uniformly across
  the three, or whether it has keyed on source-specific preprocessing artefacts
  rather than tumor features. This was flagged in section 1 of
  `analysis/EXTERNAL_VALIDATION_GAP.md` as the cheapest available generalisation
  signal. It requires no new data. **It has still not been attempted.** It should
  be the next thing anyone does.
- **No scanner metadata.** No make, model, vendor, field strength, coil or
  protocol, for any image.
- **No acquisition site.** No hospital, country or year.
- **No sequence composition for training data.** Unknown what fraction is T1, T2,
  FLAIR, or contrast-enhanced.
- **No patient IDs.** 0 of 7,200 filenames carry one. The data is slice-level.
  The split is grouped by perceptual-hash near-duplicate cluster instead, which
  is the best available substitute and is not equivalent. **Slices from the same
  patient that are not visually near-identical can still land on opposite sides
  of the train/test boundary.** Residual leakage cannot be ruled out and cannot
  be measured.
- **No demographics at all.** No age, sex, ethnicity, race or geography.
  Therefore **no fairness analysis, no subgroup analysis, and no bias audit has
  been or can be performed on this data.** For a tool explicitly aimed at
  under-served populations, this is a serious hole, and no amount of modelling
  work closes it. Only different data closes it.
- **Labels were never verified by a radiologist for this project.** They were
  inherited. Our own overlap check found 5 images where the BRISC authors' expert
  re-annotation contradicts the label we trained on, 3 of them where we say tumor
  and the radiologists say no tumor.
- **Class balance is artificial.** Exactly 1,800 per class, including 203
  author-supplied augmented meningioma duplicates. Real clinic prevalence is
  nothing like 25% per tumor class, and no prevalence correction has been applied
  or validated.

## Evaluation: one dataset, several holes

- **Only one external dataset was even attempted, and it failed the independence
  test.** BRISC is one source. Two independent sources would be a minimum for
  any generalisation claim. We have zero.
- **Internal test split is 1,112 images.** Specificity rests on 291 no-tumor
  images. A false-alarm rate estimated on 291 images is not precise.
- **The 5 seeds score the same 1,112 images.** Pooled confidence intervals across
  seeds are optimistic because the reads are correlated. Per-seed ranges are the
  honest spread.
- **Calibration was measured in-distribution only.** Calibration is known to
  degrade under distribution shift. Untested here, because there is no shifted
  data to test on.
- **The deferral threshold has not been validated against clinician workload.**
  Deferring 20% of scans may be operationally impossible in the target setting,
  and 20% is where the miss rate gets tolerable.
- **No test-retest or repeatability measurement.** MC-Dropout is stochastic. The
  same image run twice can give different confidences. The size of that variation
  has not been characterised, and a clinician seeing two different numbers for
  one scan is a trust problem.
- **Latency measured, but not on target hardware.** About 130 ms per image on
  CPU (Windows 11, 8 threads, ResNet-50, single seed, MC-Dropout T=20), plus
  0.5 s model load. That was measured on a 24-core development laptop, **not on
  the low-spec hardware a rural clinic would actually have.** Memory footprint,
  thermal behaviour and battery cost on target hardware are all unmeasured.

## Model scope

- **Four classes only.** Glioma, meningioma, pituitary, no tumor.
- **No "abnormal but unknown" output.** Every input is forced into one of the
  four. There is no way for the model to say "something is wrong here and it is
  not one of my classes".
- **Metastases are not a class and are untested.** Metastases are among the most
  common intracranial tumors. They can be classified as "no tumor".
- **Other tumor types are not classes and are untested.** Lymphoma, acoustic
  neuroma, craniopharyngioma, ependymoma, medulloblastoma, and the rest.
- **Non-tumor pathology is untested.** Stroke, haemorrhage, MS, hydrocephalus,
  trauma, infection. A haemorrhage can be correctly labelled "no tumor" and that
  answer is dangerous.
- **Paediatric patients are untested**, and cannot be identified in the training
  data because no ages exist.
- **Post-operative and post-treatment anatomy is untested.**
- **Single-slice only.** The model reads one 2D slice. It does not read a volume
  and cannot use context from adjacent slices, which is how a radiologist
  actually works.
- **The heatmap is not a segmentation.** It shows where the model looked. It has
  no calibrated relationship to tumor boundary and must not be measured.

## Out-of-scope input handling

- **The rejector has no published measurements.** Session B had not published
  `analysis/results/ood/rejector_config.json` when this was written, so **there
  is currently no measured evidence that the tool can refuse an input it should
  not judge.** The application refuses to start without one, which is the correct
  behaviour, but it means the end-to-end system has never been evaluated.
- **The category-4 case is untested and is the most serious hole in the
  project.** A genuine brain MRI carrying pathology the model has no class for, a
  stroke, a bleed, an abscess, a metastasis, is fully in-distribution to any
  uncertainty score. It is a brain, it is an MRI, and it looks like the training
  data. The model will confidently assign one of four classes and "no tumor" is
  one of them. **No entropy threshold catches this, because the model is not
  uncertain, it is wrong.** No labelled data for this category exists in the
  project.
- **Any false-rejection rate measured on BRISC will be optimistic**, because 80%
  of BRISC is training data and a rejector accepts its own training data
  happily.

## Explainability

- **Nobody has ever checked whether the heatmap points at the tumor.** Heatmaps
  were generated for all 5 seeds and both backbones
  (`analysis/results/explainability/`), including for misclassified cases. No
  measurement of overlap between the highlighted region and the actual tumor
  exists, because the internal dataset has no segmentation masks. BRISC ships
  masks for 4,793 images, so this is measurable in principle and has not been
  measured.
- **No published localisation failure analysis.** Session D had published the
  runtime generator and its consistency tests, but not the clinical analysis,
  when this was written.

The general limitation stands regardless of what that analysis finds: **a plausible heatmap
is not evidence of a correct answer.** It is a sanity check that can catch gross
errors. Nothing more has been demonstrated.

## Application and deployment

- **Not packaged.** No PyInstaller build, no installer, no desktop shortcut.
  Running it currently means having Python and the full dependency set
  installed, which is not a realistic ask for a rural clinic.
- **The app cannot run in its real configuration yet**, because the safety
  config, the input check and the explainer are still being finalised. It
  refuses to start rather than run on stubs, which is correct, but it means the
  end-to-end system has never run in the configuration it would ship in.
- **The deployed configuration has not been re-validated against the safety
  analysis.** `app/DEPLOYED_CONFIG_VALIDATION.md` does not exist yet.
- **An entropy unit mismatch was found and is unresolved at time of writing.**
  `src/code.py` computes predictive entropy in bits (log base 2); the prediction
  cache contract specifies nats. They differ by a factor of 1.443. If the
  deferral threshold and the app disagree about units, **the app defers far less
  often than intended, and the difference surfaces as confident NO TUMOR
  calls.** The app currently assumes bits, which errs toward over-deferring, and
  says so loudly at startup. Tracked as issue C-1 in `handoff/ISSUES.md`. This
  is exactly the class of bug that kills people quietly, and it was caught by
  reading a contract carefully rather than by any test.
- **No deployment has happened.** No pilot site, no clinic, no user.
- **No user testing.** Nobody outside this project has used the interface.
- **No training material for clinicians.** A tool this easy to over-trust needs
  explicit instruction on how to not over-trust it, and none exists.
- **No incident reporting route** for a clinician who thinks the tool got
  something dangerously wrong.

## Reproducibility

- **Checkpoints are not in the repository.** Each exceeds GitHub's 100 MB limit.
  Until the archive deposit is made, nobody outside this machine can reproduce or
  verify any number in this project.
  **No DOI exists yet.** See
  [reproducibility/CHECKPOINT_DEPOSIT.md](reproducibility/CHECKPOINT_DEPOSIT.md)
  for how to make the deposit.
- **Training ran entirely on CPU** and took a long time. Reproducing the full
  5-seed, 2-backbone run is expensive for anyone who wants to check the work.
- **The exact CPU model was not recorded** during the original run.
- **The training data copy in this repo has 7,200 images; the current Kaggle
  listing describes 7,023.** We believe this is Version 2 versus a later
  revision. Not confirmed against a fresh download.

## Things previously claimed that are not true

Recorded so they do not come back.

- **98.9% accuracy.** From an earlier iteration using a naive per-file random
  split. Inflated by near-duplicate leakage. Describes nothing real. Do not quote
  it, including as a "previous result".
- **"External validation on 6,000 images."** BRISC is 80% the same files as the
  training data. It is not external validation.
- **"No external dataset exists for this project"**, as stated in
  `analysis/EXTERNAL_VALIDATION_GAP.md` and `analysis/HANDOFF.md`. Superseded:
  a candidate was found and evaluated, and it turned out not to be external. The
  conclusion is unchanged; the reasoning is now better evidenced.

## What would close the biggest gaps, cheapest first

1. **Sub-source stratification of the existing 7,200 images.** No new data
   needed. Identify Br35H vs SARTAJ vs Figshare per image, report per-source
   accuracy. Open since the original gap document and still the cheapest real
   generalisation signal available.
2. **A public dataset with no lineage to Br35H, SARTAJ or Figshare.** Harder
   than it sounds. The field re-mixes the same few public collections
   constantly, which is exactly how BRISC happened. Provenance must be checked
   against the source papers, not the dataset listing.
3. **A reader study**, even a small retrospective one.
4. **A prospective institutional cohort with scanner metadata and
   demographics.** IRB, data-sharing agreements, real time and money. This is
   what unlocks fairness analysis, external validation and a regulatory
   pathway simultaneously, and there is no shortcut around it.
