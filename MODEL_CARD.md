# Model card: brain MRI triage classifier

Read this before trusting anything this tool says.

**Version:** 20260703_155524
**Card last updated:** 2026-07-31
**Status: research prototype. Not cleared, not validated for clinical use, not
suitable for use on a patient today.**

---

## The one paragraph that matters

On a held-out split of the only dataset this model has ever seen, it calls a real
tumor "no tumor" about **1 time in 100**. That is the error that sends a sick
person home. It has never been tested on a scan from a different hospital, a
different scanner, or a different patient population, because no such test data
exists for this project. The dataset we believed was an independent external test
turned out to be 80% the same image files as the training data. So the honest
answer to "does this work on new patients" is: **nobody knows, and this model card
cannot tell you.**

---

## Intended use

**What it is for:** triage support in rural and under-resourced clinics that have
an MRI scanner but no on-site radiologist, where the realistic alternative is a
scan sitting unread for days or weeks.

**What that means in practice.** A clinician runs a brain MRI slice through the
tool. The tool says tumor or no tumor, gives a likely tumor family, says how
confident it is, and shows a heatmap of where it looked. A clinician uses that to
decide who gets referred first. A radiologist still reads the scan. The tool moves
that person up the queue; it does not decide anything.

**Intended users:** clinicians and trained health workers. Not patients. Not
untrained staff. Not an automated pipeline with nobody looking.

**Human in the loop is required, always.** There is no configuration of this tool
that is safe to run unattended, and there will not be one until somebody runs a
reader study and a prospective trial.

---

## Out of scope use

Every line here is out of scope because it has not been tested, not because we
think it would fail. Untested is the point.

- **Not primary diagnosis.** It has never been compared against a radiologist on
  real cases. There is no evidence it agrees with one.
- **Not autonomous.** No result should reach a patient without a human reading
  the scan.
- **Not validated for metastases.** The model has four output classes and none of
  them is "metastasis". A brain metastasis will be forced into glioma,
  meningioma, pituitary, or no tumor. **It can be called no tumor.**
- **Not validated for any tumor type outside the three trained families.**
  Lymphoma, acoustic neuroma, craniopharyngioma, ependymoma, medulloblastoma,
  abscess, and everything else are all out of scope and all can come back as
  "no tumor".
- **Not validated for non-tumor pathology.** Stroke, haemorrhage, multiple
  sclerosis, hydrocephalus, traumatic injury. A bleeding patient can be labelled
  "no tumor" with high confidence, and that label would be correct and useless.
- **Not validated on paediatric patients.** The training data carries no ages.
  Nobody can say whether any child's scan was in it. Paediatric brain tumors are
  differently distributed and often differently located. Treat any paediatric
  output as unvalidated.
- **Not validated on any named scanner.** Not one scanner make, model or field
  strength is recorded anywhere in the training data. There is no scanner this
  tool is known to work on.
- **Not validated on sequences other than T1.** Evaluation data is T1-weighted
  contrast-enhanced only. Training data sequence composition is unknown and
  unrecoverable. T2, FLAIR, DWI, and non-contrast studies are untested.
- **Not validated on any acquisition plane outside axial, coronal and sagittal
  as they appear in these public datasets.**
- **Not a segmentation tool.** The heatmap shows where the model looked. It is
  not a tumor boundary and must never be measured.
- **Not for treatment planning, surgical planning, response assessment, or any
  decision beyond "should this scan be looked at sooner".**
- **Not validated for non-brain imaging or non-MRI images.** Session B's input
  check exists to reject these. See [Out-of-scope rejection](#out-of-scope-input-rejection).

---

## Training data

**Source:** the Kaggle "Brain Tumor MRI Dataset" (Nickparvar, 2023), which is
itself a merge of three public collections: Br35H, SARTAJ, and the Figshare
dataset of Cheng et al.

**Size:** 7,200 images, exactly 1,800 per class.

**License:** CC0 1.0 Public Domain. Redistribution is permitted. Full licensing
analysis, including the caveat that CC0 was asserted by the aggregator rather
than the original custodians, is in [docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md).

### What is unknown about this data, and cannot be recovered

State this plainly to anyone who asks:

- **Acquisition sites are unknown.** Which hospitals, which countries, which
  years. Not recorded, not recoverable.
- **Scanners are unknown.** No make, model, vendor, field strength, coil, or
  protocol.
- **Sequence composition is unknown.** We do not know what fraction is T1, T2,
  FLAIR, or contrast-enhanced.
- **Patient identity is unknown.** Patient IDs are parseable for 0 of 7,200
  filenames (`results/leakage_audit.md`). The data is slice-level. Multiple
  slices of the same patient cannot be identified as such, only guessed at by
  image similarity.
- **Ground-truth labels were never verified by a radiologist for this project.**
  We inherited them. Two specific reasons that matters:
  - The SARTAJ component had **documented mislabelling in its glioma class**.
    The Kaggle aggregator's published fix was to discard SARTAJ's glioma images
    and substitute Figshare ones. That correction is widely noted in the
    literature using this dataset. It is a known-good fix, but it is a fix
    somebody else made, that we inherited without checking.
  - Our own overlap check found 5 images where the BRISC authors' expert
    re-annotation disagrees with the label we trained on, **3 of them in the
    direction of "we called it a tumor, the radiologists called it no tumor"**.
    Five out of 4,802 matched images is not a crisis. It is a reminder that
    nobody clinically qualified has ever looked at our labels.

### No demographic data exists, so no fairness analysis was possible

There is **no age, no sex, no ethnicity, no race, no geography, and no
socioeconomic marker** attached to any of the 7,200 training images. Not
anonymised. Not aggregated. Absent.

**Consequence: this project cannot report subgroup performance for any group, and
no fairness or bias audit has been performed or can be performed on this data.**

This is a serious limitation and it deserves to be stated where people will read
it rather than buried in an appendix. The stated mission of this tool is to serve
rural and under-served populations. Those are exactly the populations most likely
to be under-represented in public datasets assembled from whatever was available,
and exactly the populations most likely to be harmed if the model works worse on
them. We have built a tool aimed at under-served people, on data that makes it
impossible to check whether it works for under-served people.

Nothing about the model architecture fixes this. Only different data fixes it.
Closing it requires a cohort that carries demographics, which means a prospective
collection with an IRB, which means this is a funding and time problem, not a
code problem.

### Split

70/15/15, grouped so that near-duplicate images cannot land on opposite sides of
a split boundary. Grouping is by perceptual-hash cluster (Hamming distance ≤ 5),
because no patient IDs exist to group by. 4,784 clusters across 7,200 images;
1,107 clusters contain more than one image; the largest holds 28.

| split | n | glioma | meningioma | pituitary | notumor |
|---|---|---|---|---|---|
| train | 4,979 | 1,254 | 1,268 | 1,246 | 1,211 |
| val | 1,109 | 269 | 266 | 276 | 298 |
| test | 1,112 | 277 | 266 | 278 | 291 |

Grouping at the cluster level is stronger than a per-file random split but weaker
than patient-level grouping, which is impossible here. **Slices from one patient
that are not visually near-identical can still be split across train and test.**
The leakage risk is reduced, not eliminated. Audit: `results/leakage_audit.md`.

An earlier iteration of this project reported 98.9% accuracy under a naive
per-file random split. That number was inflated by near-duplicate leakage. It is
recorded here only so that nobody quotes it. It does not describe this model and
it never described anything real.

---

## Evaluation data

### Internal held-out test split

1,112 images, composition in the table above. Held out at the near-duplicate
cluster level from the same merged pool the model trained on.

**What this measures:** that the model did not simply memorise its training
images.
**What this does not measure:** anything about a different hospital, scanner,
population, or preprocessing pipeline.

### BRISC 2025, and why it is not the external test it was meant to be

BRISC 2025 (Fateh et al., *Scientific Data* 2026, arXiv:2506.14318, CC BY 4.0)
was brought into this project to be the independent external test set. 6,000
T1-weighted contrast-enhanced images, three planes, expert-reviewed annotations.

It is not independent. The BRISC paper states its images were collated from
Cheng/Figshare, SARTAJ and Br35H, aggregated via the same Kaggle Nickparvar
collection this model trained on.

We measured how bad that is by hashing every image in both sets:

| | count | of 6,000 |
|---|---|---|
| **Byte-identical file (same sha256) to a training-pool image** | **4,787** | **79.8%** |
| Near-duplicate or identical (phash Hamming ≤ 5) | 4,802 | 80.0% |
| No match at all | 1,198 | 20.0% |

Of the 4,802 matches, 4,791 are at Hamming distance zero. These are the same
files, not similar files.

Where they land in our splits:

| BRISC image matches our... | count | of 6,000 |
|---|---|---|
| **train split, the model was fitted on these** | **3,353** | **55.9%** |
| val split, thresholds and temperature were fitted here | 716 | 11.9% |
| test split, our internal test numbers came from here | 733 | 12.2% |
| nothing | 1,198 | 20.0% |

The overlap is concentrated in exactly the wrong place:

| BRISC class | overlapping | total | rate |
|---|---|---|---|
| pituitary | 1,757 | 1,757 | **100.0%** |
| glioma | 1,376 | 1,401 | 98.2% |
| meningioma | 1,602 | 1,635 | 98.0% |
| no_tumor | 67 | 1,207 | 5.6% |

**BRISC contains 4,793 tumor-bearing images. The model has seen 4,735 of them.
58 are unseen. In BRISC's own 1,000-image test split, 2 tumor images are unseen.**

**Therefore: the tumor miss rate cannot be estimated on BRISC.** Not on the full
6,000, where 99% of the tumors are training data. Not on the clean 1,198, where
there are 58 tumors and any rate computed on them carries a confidence interval
too wide to act on.

Any BRISC number in this document is reported with its overlap status attached.
Code: `docs/check_brisc_overlap.py`. Per-image flags:
`docs/results/brisc_overlap_flags.csv`. Full analysis:
[docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md).

**This project has one dataset. It has always had one dataset.**

---

## Performance

Every number below carries the dataset it came from. A number without a dataset
name is not a number, it is a rumour.

Deployment configuration selected by session A:
`[PENDING: session A deployment_config.json — chosen backbone, seed(s), ensemble
or not, temperature, thresholds]`.

Two backbones were trained, ViT-B/16 and ResNet-50, 5 seeds each
(42, 123, 7, 2024, 31), MC-Dropout with T=20 stochastic passes at evaluation.

### Tumor miss rate — the number that matters most

A real tumor called "no tumor". The error that sends a sick person home.

**Dataset: internal held-out test split. n = 1,112 images, of which 821 carry a
tumor. Same merged Kaggle pool as training, held out at near-duplicate-cluster
level. Not an external result.**

| backbone | miss rate, mean over 5 seeds | per-seed range | pooled count | 95% CI (Wilson) |
|---|---|---|---|---|
| **ResNet-50** | **1.00%** | 0.61% to 1.34% | 41 / 4,105 | **0.74% to 1.35%** |
| ViT-B/16 | 1.14% | 1.10% to 1.22% | 47 / 4,105 | 0.86% to 1.52% |

The pooled denominator is 5 seeds scoring the same 821 tumor images. That is five
correlated reads of one test set, not 4,105 independent images, so the pooled CI
is optimistic. **The per-seed range is the honest spread.** For ResNet-50 that is
5 to 11 missed tumors out of 821, depending only on the random seed.

**Both intervals above are narrower still than the evidence justifies**, for a
reason that only became clear when the misses were examined individually: 9
images out of 821 account for 81% of all missed-tumor events across all 10
checkpoints. See [Known failure modes](#1-a-handful-of-specific-images-defeat-every-model-this-project-has-trained).
Treat every confidence interval on this page as a floor on the uncertainty.

**Plain reading: roughly 1 tumor in 100 is missed, on the easiest data this model
will ever see.**

**Dataset: BRISC 2025.** `[PENDING: session A BRISC tumor miss rate, all 6,000]`
— and see the overlap section above for why that number is not a
generalisation estimate.
**Dataset: BRISC 2025, unseen subset only, n = 1,198, of which 58 carry a tumor.**
`[PENDING: session A]` — expect this to be reported as not estimable, or with a
CI wide enough that it constrains nothing.

### Tumor miss rate by tumor type

Dataset: internal held-out test split, pooled over 5 seeds.

95% CIs are Wilson score intervals, which behave correctly at 0 where a normal
approximation does not.

| tumor type | ResNet-50 | 95% CI | ViT-B/16 | 95% CI |
|---|---|---|---|---|
| **glioma** | **1.88%** (26/1,385) | 1.28 to 2.74 | **2.24%** (31/1,385) | 1.58 to 3.16 |
| meningioma | 1.13% (15/1,330) | 0.68 to 1.85 | 1.20% (16/1,330) | 0.74 to 1.95 |
| pituitary | 0.00% (0/1,390) | 0.00 to 0.28 | 0.00% (0/1,390) | 0.00 to 0.28 |

"0.00%" for pituitary does not mean the model never misses a pituitary tumor. It
means it missed none of 1,390 evaluations, which bounds the true rate at roughly
0.28% or below. Zero observed is not zero.

Gliomas are missed roughly twice as often as meningiomas and drive most of the
total. No pituitary tumor was missed in 1,390 evaluations, which is unsurprising:
they sit in a fixed anatomical location that is easy to key on. The flip side is
that a model keying on location will do badly on a tumor that turns up somewhere
unusual, and nothing here tests that.

**Gliomas are also the most aggressive of the three families.** The class this
tool misses most is the class where delay costs the most.

### Binary tumor vs no tumor

Dataset: internal held-out test split, n = 1,112. 821 tumor, 291 no-tumor.

| backbone | sensitivity | 95% CI | specificity | 95% CI | false alarms |
|---|---|---|---|---|---|
| ResNet-50 | 99.00% | 98.65 to 99.26 | 98.28% | 97.48 to 98.83 | 25 / 1,455 |
| ViT-B/16 | 98.86% | 98.48 to 99.14 | 98.01% | 97.15 to 98.61 | 29 / 1,455 |

Pooled over 5 seeds, Wilson intervals. Per-seed ranges: ResNet-50 sensitivity
98.66 to 99.39, specificity 97.59 to 98.63; ViT sensitivity 98.78 to 98.90,
specificity 97.25 to 98.97.

Specificity rests on **291 distinct no-tumor images** scored 5 times. The
denominator of 1,455 is not 1,455 independent images, so the interval above is
narrower than the evidence really justifies. A false-alarm rate estimated on 291
images is not a precise quantity.

**Dataset: BRISC 2025:** `[PENDING: session A binary sensitivity and specificity,
with overlap stratification]`.

### Four-way classification accuracy

Dataset: internal held-out test split, n = 1,112, MC-Dropout T=20, 5 seeds.

| backbone | mean accuracy | per-seed range | bootstrap 95% CI (per seed) |
|---|---|---|---|
| ResNet-50 | 0.964 | 0.960 to 0.967 | ≈0.948 to 0.977 |
| ViT-B/16 | 0.962 | 0.957 to 0.964 | ≈0.946 to 0.975 |

Macro-AUC ≈0.994 for both. Per-seed McNemar tests found no significant difference
between the two backbones in any of 5 seeds.

**This 0.96 is an internal number on a held-out split of the training pool. It is
not a real-world accuracy and it must never be quoted as one.** It is the least
important number on this page and it is the one people will want to quote.

**Dataset: BRISC 2025:** `[PENDING: session A four-way accuracy, stratified by
overlap]`.

### Plane subgroups

**Dataset: BRISC 2025** (the only dataset with plane labels; the training data has
none): `[PENDING: session A per-plane metrics, axial / coronal / sagittal, with
overlap stratification]`.

Note in advance that plane subgroup results on BRISC inherit the overlap problem.
The clean subset holds 1,198 images spread across 3 planes and is 95% no-tumor,
so a per-plane miss rate on clean data is not going to be estimable either.

### Calibration

Dataset: internal held-out test split, n = 1,112.

MC-Dropout mean-probability expected calibration error (ECE, 15-bin): 0.064 to
0.081 (ViT), 0.068 to 0.077 (ResNet-50). That is moderate, not good. The raw
model is overconfident.

Post-hoc temperature scaling on the deterministic single-pass softmax, with T
fitted on the **validation** split and applied unchanged to test, cuts ECE to
0.018 to 0.021 (ViT) and 0.011 to 0.020 (ResNet-50). Fitted T averages 0.62 for
both, well below 1, confirming overconfidence.

**Calibration was measured in-distribution only.** Calibration is known to
degrade under distribution shift, and this has not been tested, because there is
no shifted data to test it on. Do not assume the confidence numbers hold on a new
scanner.

### Out-of-scope input rejection

`[PENDING: session B rejector_config.json — method, threshold, rejection rate by
category, false rejection rate on internal test and on BRISC]`.

Note for whoever fills this in: a false-rejection rate measured on BRISC is
measured on data that is 80% training images, and a rejector will happily accept
its own training data. That number will look better than it is.

---

## Uncertainty and deferral

### How it works

Monte Carlo Dropout. Dropout layers (p=0.3) are left active at inference and the
image is passed through the network 20 times. The 20 softmax outputs are
averaged. Spread across the 20 passes is the model's uncertainty. Predictive
entropy of the averaged distribution is the score used to decide whether to defer
to a human.

Only the dropout modules are switched to training mode. The rest of the network
stays in eval mode, so batch-norm statistics are not disturbed.

**Units matter here and they are currently inconsistent.** `src/code.py` computes
predictive entropy with `torch.log2`, so its entropy is in **bits**. The
prediction-cache contract specifies **nats**. The two differ by a factor of
1.443. Session C found this and it is unresolved at the time of writing. If a
deferral threshold fitted in one unit is applied in the other, **the tool defers
far less often than intended, and the difference surfaces as confident NO TUMOR
calls.** Tracked as issue C-1 in `handoff/ISSUES.md`. Anyone reading an entropy
number out of this project should check which unit it is in first.

The deferral figures in the next section are unaffected, because they use
quantile thresholds (defer the most uncertain X%), which do not depend on the
unit.

### What threshold

`[PENDING: session A entropy_defer_threshold from deployment_config.json, fitted
on the internal validation split]`.

The numbers below use a simpler rule for illustration: defer the most uncertain
X% of cases by entropy. A fitted absolute threshold will behave differently.

### What it catches, and what it does not

Dataset: internal held-out test split, pooled over 5 seeds. 41 missed tumors
(ResNet-50) and 47 (ViT) before any deferral.

| backbone | defer | misses before | misses remaining | caught | residual miss rate |
|---|---|---|---|---|---|
| ResNet-50 | top 5% | 41 | 11 | 73% | 0.28% |
| ResNet-50 | top 10% | 41 | 6 | 85% | 0.16% |
| ResNet-50 | top 20% | 41 | 3 | 93% | 0.09% |
| ViT-B/16 | top 5% | 47 | 30 | 36% | 0.78% |
| ViT-B/16 | top 10% | 47 | 24 | 49% | 0.66% |
| ViT-B/16 | top 20% | 47 | 17 | 64% | 0.52% |

**Deferral helps. It does not fix the problem.** Even at 20% deferral, which
means a fifth of all scans go to a human and the tool's value proposition is
badly dented, ResNet-50 still misses 3 tumors and ViT still misses 17.

**The two backbones are not interchangeable on safety, even though accuracy says
they are.** McNemar found no significant accuracy difference in any seed. But
ResNet-50's uncertainty flags 73% of its own missed tumors at 5% deferral, and
ViT's flags 36%. If a backbone is chosen on accuracy alone, that difference is
invisible, and it is the difference that matters clinically. **Choose ResNet-50.**

### Confident misses: the failure the confidence score cannot save you from

Of the missed tumors, how confident was the model in the wrong "no tumor" call?

| backbone | median confidence on missed tumors | ≥0.8 confident | ≥0.9 confident |
|---|---|---|---|
| ResNet-50 | 0.691 | 29% (12/41) | 10% (4/41) |
| ViT-B/16 | 0.865 | 66% (31/47) | 36% (17/47) |

**ViT calls a third of its missed tumors "no tumor" with 90%+ confidence.** No
uncertainty threshold catches those. They arrive on the clinician's screen
looking clean, and the interface will say the model is sure.

This is the most important thing on this page after the miss rate itself. A tool
that is wrong and knows it is manageable. A tool that is wrong and confident is
the one that hurts someone.

---

## Known failure modes

Concrete cases where this tool is expected to fail. Not hypotheticals.

### 1. A handful of specific images defeat every model this project has trained

This is the dominant failure and it is more concentrated than the 1% miss rate
suggests.

Across all 10 checkpoints (2 backbones x 5 seeds) on the internal test split
there are 88 missed-tumor events. They come from **18 distinct images out of
821.** Nine of those 18 images account for **81% of all misses.**

| image | true class | missed by | max p(no tumor) |
|---|---|---|---|
| `Te-gl_74.jpg` | glioma | **10 of 10 checkpoints** | 0.933 |
| `Te-gl_372.jpg` | glioma | 9 of 10 | 0.944 |
| `Te-gl_97.jpg` | glioma | 9 of 10 | 0.941 |
| `Tr-me_202.jpg` | meningioma | 8 of 10 | 0.942 |
| `Te-me_279.jpg` | meningioma | 8 of 10 | 0.881 |
| `Tr-me_897.jpg` | meningioma | 8 of 10 | 0.864 |
| `Te-gl_351.jpg` | glioma | 7 of 10 | 0.833 |
| `Te-gl_143.jpg` | glioma | 6 of 10 | 0.929 |
| `Te-gl_72.jpg` | glioma | 6 of 10 | 0.929 |

Every one of these is a glioma or a meningioma. Not one is a pituitary tumor.
Nine of the 18 are missed by **both** architectures. `Te-gl_74.jpg` is missed by
every checkpoint this project has ever produced, at 93% confidence.

Copies of the worst cases are in `docs/results/confident_miss_examples/`, named
with the true class, the backbone, how many seeds missed it, and the confidence.
Full list with per-seed entropies: `docs/results/confident_misses.json`.
Reproduce: `python docs/confident_misses.py`.

**Three consequences, and they matter more than the headline rate.**

**These failures are systematic, not random.** Different random initialisations,
different architectures, and different training runs all fail on the same
images. Whatever is wrong is in the data or in the difficulty of the
presentation, not in the luck of one training run.

**Ensembling will not fix this.** An ensemble averages out errors that are
independent between members. These errors are correlated across seeds *and*
across architectures. A 5-seed ResNet-50 ensemble will still miss `Te-gl_74`.
Anyone choosing an ensemble should expect it to buy accuracy, not safety.

**The confidence interval on the miss rate is optimistic.** The Wilson interval
of 0.74% to 1.35% treats 821 tumor images as independent observations. For a
single checkpoint that is fair. Across seeds it is not, because the same nine
images drive four fifths of the misses. The effective sample size behind the
miss rate is closer to a dozen hard cases than to 821 images, so **the true
uncertainty is wider than any interval in this document.**

One further thing worth someone's time: three of these images are in phash
clusters, and 18 missed images span only 14 clusters. **Nobody clinically
qualified has looked at these 18 images to check the label is even correct.**
Given that our own overlap check already found 3 images where expert
re-annotation disagreed with our training label in the tumor-to-no-tumor
direction, that is a cheap and worthwhile hour of a radiologist's time.

### 2. Anything outside the four classes

The model has no way to say "abnormal but not one of my three tumor types". Every
input is forced into glioma, meningioma, pituitary, or no tumor. Metastases,
lymphoma, abscesses, strokes, bleeds and demyelinating disease all land somewhere,
and "no tumor" is one of the places they can land. This is untested, because no
labelled data for these conditions exists in this project.

`[PENDING: session B category-4 result — brain MRI carrying pathology the model
has no class for]`. If B could not source that data, this stays an open,
untested, clinically serious hole and is recorded as such in
[LIMITATIONS.md](LIMITATIONS.md).

### 3. Inputs that are not brain MRI at all

`[PENDING: session B out-of-scope rejection results by category — non-brain MRI,
non-MRI medical images, natural images, corrupted or blank images]`.

### 4. Heatmaps that look plausible and are not

`[PENDING: session D localisation failure cases — where the heatmap highlights
skull, background, or an unrelated region while the classification happens to be
correct, and cases where the model is right for the wrong reason]`.

A heatmap is a trust signal and trust signals cut both ways. A confident wrong
answer with a plausible-looking heatmap over roughly the right area is harder for
a clinician to catch than a confident wrong answer with an obviously silly
heatmap. The heatmap is a sanity check, not evidence.

### 5. Everything that has not been tested

Different scanner. Different field strength. Different sequence. Different
population. Paediatric patients. Post-operative anatomy. Motion artefact.
Non-contrast studies. Each of these is a plausible failure mode and none of them
has been measured.

---

## Ethical and safety considerations

### The cost of a missed tumor

A missed glioma is a delayed diagnosis of an aggressive cancer. In a rural clinic
the tool's output may be the only read a scan gets for days or weeks, so the
failure is not "the model was wrong and a doctor corrected it", it is "the model
was wrong and nobody looked again". At a 1% miss rate, a clinic running 500 scans
a year with a 15% tumor prevalence sends home roughly one person a year with a
tumor it saw and dismissed.

That is the whole risk in one sentence. Everything else on this page is detail.

### The cost of a false alarm

This is not free either, and it is the failure mode that people building these
tools routinely wave away.

In the setting this tool is aimed at, a referral means real travel to a distant
specialist. That is time off work, transport cost, possibly an overnight stay,
possibly for a family member too. For a subsistence household a false referral can
cost weeks of income. It also consumes a specialist slot that a genuinely sick
person needed.

The false-alarm rate on internal data is 1.7% (ResNet-50) to 2.0% (ViT). That
sounds small. It stops sounding small once you apply it to a realistic clinic
population, where tumors are rare.

Take ResNet-50's internal sensitivity 99.0% and specificity 98.3%, and assume
they hold in the field, which is itself unproven:

| tumor prevalence in scanned patients | share of flagged scans that really have a tumor |
|---|---|
| 15% | about 91% |
| 5% | about 75% |
| 2% | about 54% |
| 1% | about 37% |

At the low end, **most of the people this tool sends on a long journey to a
specialist do not have a tumor.** Nobody has measured the real prevalence in a
target clinic, so this is arithmetic rather than evidence. It is arithmetic every
deployment plan has to face before the first patient, and it is the reason the
false-alarm rate cannot be treated as the unimportant error.

A tool that floods a fragile referral pathway with false alarms gets switched off,
and then it helps nobody. **Both errors are real. Neither is a rounding error.**

### Automation bias

A confident, well-designed interface showing a heatmap and a percentage is
persuasive. It will be more persuasive than it deserves to be, especially to a
clinician who is not a radiologist and is under time pressure. The confident-miss
data above says the interface will sometimes be confidently, articulately wrong.
UI copy that oversells is therefore a patient-safety issue, not a marketing
question.

### Deploying to under-served populations first

The mission is to start where the need is highest. That is defensible. It also
means the first people exposed to an unvalidated tool are people with the least
ability to seek a second opinion or to complain, and the very people we cannot
check the model works for, because no demographic data exists. That tension does
not resolve with a better argument. It resolves with a reader study and a
prospective cohort that carries demographics.

### Data

All data used is public and already de-identified. No patient consent process
exists for it, and none was obtainable, because patient identity is not recorded.
No IRB approval was sought or is held for this work. Nothing here has been near a
real patient.

---

## Version and provenance

**Code commit:** `8cc94a86ec9f64e9f14056e56a4d7dc349bef9c0`, branch `session/E`,
repository <https://github.com/medsharma/Brain-Tumor-Algo-for-Clinics>.

**Training run:** `results/20260703_155524`, completed 2026-07-09.

**Configuration:** 30 epochs max with early stopping (patience 5), batch size 32,
lr 1e-4, weight decay 0.01, label smoothing 0.1, dropout p=0.3, MC-Dropout T=20.
ViT-B/16 with the last 2 encoder blocks plus head unfrozen; ResNet-50 with layer4
plus head unfrozen. Both initialised from ImageNet weights. Preprocessing: resize
to 224x224, ImageNet mean/std normalisation.

**Environment:** Python 3.10.11, torch 2.12.1+cpu, torchvision 0.27.1+cpu,
Windows 11. **All training and evaluation ran on CPU.** No GPU was used.

**Seeds:** 42, 123, 7, 2024, 31.

**Split manifest:** `data/split_manifest.csv`, regenerable with
`python src/code.py --data-root ./data/brain_tumor --manifest data/split_manifest.csv --force-manifest`.

### Checkpoint hashes

Checkpoints exceed GitHub's 100 MB limit and are not in the repository. Archive
deposit instructions are in
[reproducibility/CHECKPOINT_DEPOSIT.md](reproducibility/CHECKPOINT_DEPOSIT.md).
**No DOI exists.** The deposit has not been made, so the hashes below are
currently the only way to identify these files, and the files themselves exist
in one place.

sha256, relative to `results/20260703_155524/`:

| file | sha256 | bytes |
|---|---|---|
| `resnet50/seed_42/best_resnet50_seed42.pth` | `26c0b3dfeedbd70565119517f8e6ec704366501efb06a70152cb98c7cdec0f2c` | 220,460,701 |
| `resnet50/seed_123/best_resnet50_seed123.pth` | `eec2672c4011ef7ebdc5d235108a4f4fa3a23b09e2c151bbe40e32b268be4582` | 220,461,139 |
| `resnet50/seed_7/best_resnet50_seed7.pth` | `7c3c8f94216c12ba36618395365772830caeb1145147bdeb4872b6ba0341cf09` | 220,460,263 |
| `resnet50/seed_2024/best_resnet50_seed2024.pth` | `a2b80b941724c47f246f29cb09af92660d78dd19eb0f53c506a21808ce815568` | 220,461,641 |
| `resnet50/seed_31/best_resnet50_seed31.pth` | `a722a51f67632b10d40f9b07bac4493a5bf8116d5da004f61f91d1afafb2316f` | 220,460,701 |
| `vit/seed_42/best_vit_seed42.pth` | `803b48ca9403cab5c2d752593701e063250715d20088f49506a69f9d4e9716ba` | 459,097,175 |
| `vit/seed_123/best_vit_seed123.pth` | `4f77571c6b62ae06c931cf65a8b9564c911a2faa9132a743ad961d8d25b33f44` | 459,097,433 |
| `vit/seed_7/best_vit_seed7.pth` | `e2ba3e74705f6e0109f8ff4d89efeff4a0cf3cf8757b50d916139db4d12de9ed` | 459,096,917 |
| `vit/seed_2024/best_vit_seed2024.pth` | `2f2990ff42dc179ce90ee374106508f09f1f73432ff003f7e950c05e16c5b0fd` | 459,097,691 |
| `vit/seed_31/best_vit_seed31.pth` | `28c6ae7da2c00c957538026bedc614f18972c3a0c0dcff5a383fb44564f08c2c` | 459,097,175 |

### Where the numbers on this page come from

| section | source |
|---|---|
| internal miss rate, sensitivity, specificity, per-class, deferral, confident misses | `docs/results/internal_safety_metrics.json`, computed by `docs/internal_safety_metrics.py` from `analysis/results/*_predictions.json` |
| internal four-way accuracy, AUC, ECE, Brier | `results/master_summary.json` |
| temperature scaling | `analysis/results/calibration/calibration_comparison.md` |
| BRISC overlap | `docs/results/brisc_overlap.json`, `docs/check_brisc_overlap.py` |
| split composition and leakage audit | `data/split_manifest.csv`, `results/leakage_audit.md` |
| BRISC performance, deferral threshold, deployment config | session A, `analysis/results/safety/`, `analysis/results/brisc/` |
| out-of-scope rejection | session B, `analysis/results/ood/` |
| localisation failures | session D, `analysis/results/explainability_clinical/` |

---

## What would have to be true before this is used on a patient

None of these are done.

1. External validation on a cohort with **no lineage back to Br35H, SARTAJ or
   Figshare.** BRISC does not qualify.
2. A reader study. Radiologists using the tool on real cases, measuring whether
   it helps, harms, or does nothing.
3. Prospective evaluation in a clinic resembling a target site, with the real
   local prevalence, measuring the real false-referral burden.
4. Demographic and subgroup analysis, which requires data that carries
   demographics.
5. A regulatory pathway for the specific country of deployment. None exists,
   none has been started.
6. A monitoring plan for after deployment, because model performance drifts and
   nobody would currently notice.

Until then this is a research prototype and should be described as one, every
time, by everyone.

---

## Contact and reporting

Issues, and any suspected clinical safety problem, to the repository:
<https://github.com/medsharma/Brain-Tumor-Algo-for-Clinics/issues>.

**Companion documents:** [docs/FOR_CLINICIANS.md](docs/FOR_CLINICIANS.md) (plain
words, for a non-technical reader) · [README.md](README.md) ·
[LIMITATIONS.md](LIMITATIONS.md) ·
[docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md) ·
[MISSION.md](MISSION.md)
