# Out-of-scope rejection: results

Model resnet50 seed 42, MC Dropout T=20. **Shipped method: the image precheck only.** The model-score stage is measured below and switched OFF. `mahalanobis` scored best on every internal metric and then wrongly rejected 27.7% of genuinely unseen brain MRI, so it is not shipped.

## The headline

**The score-based rejector was built, measured, and rejected.** It is the
most likely way this workstream turns out to be useless, and it was:

| BRISC subset | mahalanobis wrongly rejects |
|---|---|
| full (mostly training data republished) | 12.3% |
| clean vs train (genuinely unseen) | 27.7% |
| clean vs any (strictest) | 57.8% |

False rejection RISES as images get less like the training set. That is a
'not my training set' detector, not an out-of-scope detector, and a clinic
with a different scanner would have most of its scans refused on day one.

The precheck does not do this. Its false rejection is flat across the same
three subsets: 0.22%, 0.15%, 0.00%.

- Out-of-scope images the rejector lets through: **41.4%** on the held-out half.
- Of all out-of-scope images, **0.0%** come back with a confident class and no warning of any kind.
- Valid brain MRI wrongly thrown away: **0.1%** on internal test, **0.0%** on the uncontaminated part of BRISC.

> **BRISC is not clean external data.** Session A found that about 80% of
> BRISC 2025 is pixel-identical to `data/brain_tumor/`, the training set.
> Both repackage the same public sources. Every BRISC number below is
> broken out by how much internal data the subset overlaps. The full-set
> number is reference only and must not be quoted as external validation.

## Fraction still getting a confident class, by category

`accepted` means the rejector let the image through. `confident` means it was
also not deferred, so a clinic user sees a class with no warning at all.

| category | n | rejected (held-out half) | accepted | accepted and confident |
|---|---|---|---|---|
| 1. corrupted scans | 1560 | 52.2% (49-56) | 48.1% | **0.0%** |
| 3. non-medical | 240 | 100.0% (97-100) | 0.0% | **0.0%** |

### By subcategory

| subcategory | category | n | rejected | precheck alone | confident class |
|---|---|---|---|---|---|
| blur heavy | cat1 | 120 | 8.3% | 8.3% | 0.0% |
| corner crop | cat1 | 120 | 50.0% | 50.0% | 0.0% |
| jpeg q5 | cat1 | 120 | 0.0% | 0.0% | 0.0% |
| near black | cat1 | 120 | 100.0% | 100.0% | 0.0% |
| near white | cat1 | 120 | 100.0% | 100.0% | 0.0% |
| noise heavy | cat1 | 120 | 100.0% | 100.0% | 0.0% |
| overexposed | cat1 | 120 | 30.8% | 30.8% | 0.0% |
| pure noise | cat1 | 120 | 100.0% | 100.0% | 0.0% |
| rotated 90 (orientation only) | cat1 | 120 | 0.0% | 0.0% | 0.0% |
| solid colour | cat1 | 120 | 100.0% | 100.0% | 0.0% |
| thumbnail | cat1 | 120 | 1.7% | 1.7% | 0.0% |
| underexposed | cat1 | 120 | 84.2% | 84.2% | 0.0% |
| upside down (orientation only) | cat1 | 120 | 0.0% | 0.0% | 0.0% |
| chart screenshot | cat3 | 120 | 100.0% | 100.0% | 0.0% |
| document | cat3 | 120 | 100.0% | 100.0% | 0.0% |

Weakest subcategories, lowest rejection first: jpeg q5 (0.0%), rotated 90 (0.0%), upside down (0.0%), thumbnail (1.7%), blur heavy (8.3%), overexposed (30.8%).

## What it costs: valid brain MRI thrown away

- Internal validation: 0.5% (this is the budget the threshold was set to, so it is not evidence)
- Internal test: 0.1%

### BRISC, split by how much of it the model already saw

| BRISC subset | n | wrongly rejected | 95% CI | precheck alone |
|---|---|---|---|---|
| `full` | 6000 | 0.2% | 0.1-0.4 | 0.2% |
| `clean_vs_fitted` **<- headline** | 1924 | 0.0% | 0.0-0.2 | 0.0% |
| `clean_vs_train` | 2634 | 0.2% | 0.1-0.4 | 0.2% |
| `clean_vs_any` | 1198 | 0.0% | 0.0-0.3 | 0.0% |

- `full`: all 6,000 BRISC images. About 80% are pixel-identical to internal training data (session A). Reference only, not an external number.
- `clean_vs_fitted`: far from everything session B fitted on, internal train and val. The primary honest false rejection number.
- `clean_vs_train`: session A's definition, far from internal train.
- `clean_vs_any`: far from every internal split. Strictest, but about 95% no-tumor, so it mostly measures false rejection on healthy brains.

By class on the headline subset: glioma 0.0% (n=229), meningioma 0.0% (n=271), pituitary 0.0% (n=277), notumor 0.0% (n=1147).

By plane on the headline subset: axial 0.0%, coronal 0.0%, sagittal 0.0%.

A false rejection rate on the full BRISC set would have been flattering and
meaningless, because 80% of that set is the training data wearing new
filenames. The uncontaminated subset is the real domain-shift check. A
rejector that throws away a large share of it is not detecting out-of-scope,
it is detecting "not my training set", and it will reject every new clinic's
scanner on day one.

Even the uncontaminated subset is a weak external check. It is what is left
after removing overlap, not a dataset chosen to be independent. Real external
validation on a cohort that does not descend from Figshare, SARTAJ or Br35H
is still outstanding for this whole project.

## Methods compared

Chosen by highest AUROC on internal validation against the out-of-scope fitting
half. BRISC never entered the choice.

| method | AUROC (selection) | AUROC (held out) | reject out-of-scope | FRR internal test | FRR BRISC clean |
|---|---|---|---|---|---|
| `entropy` | 0.8834 | 0.8685 | 71.6% | 7.1% | 26.2% |
| `mutual_information` | 0.8354 | 0.8425 | 59.7% | 6.6% | 24.5% |
| `max_softmax` | 0.8808 | 0.8660 | 70.8% | 6.9% | 25.8% |
| `mahalanobis` **<-** | 0.9397 | 0.9535 | 74.2% | 3.7% | 36.7% |
| `knn` | 0.9137 | 0.8966 | 74.0% | 6.1% | 36.3% |

Score-only, at a matched false rejection rate on internal validation. The
combined rejector adds the precheck on top.

## The image precheck

Cheap statistics, no model, no GPU, about 70 ms per image. It runs first.

| rule | accept band | catches out-of-scope | costs in-scope |
|---|---|---|---|
| `color_spread` | [-inf, 0.01718] | 25.7% | 0.0% |
| `intensity_std` | [0.04075, +inf] | 22.2% | 0.0% |
| `dynamic_range` | [0.2385, +inf] | 23.0% | 0.0% |
| `frac_dark` | [-0.1529, 0.9173] | 13.4% | 0.0% |
| `frac_bright` | [-inf, 0.1565] | 19.4% | 0.0% |
| `border_mean` | [-inf, 0.3967] | 32.2% | 0.0% |
| `hist_entropy` | [1.621, +inf] | 31.7% | 0.0% |
| `fg_area_frac` | [-0.09287, 0.833] | 2.7% | 0.0% |
| `fg_centroid_offset` | [-inf, 0.584] | 11.3% | 0.1% |
| `fg_border_touch` | [-inf, 0.3333] | 14.7% | 0.2% |
| `fg_n_components` | [-inf, 8.2] | 2.8% | 0.1% |

Precheck alone rejects 58.3% of out-of-scope images and costs 0.1% on internal test, 0.2% on BRISC.

## Overlap with deferral

These are two different questions. Input validation asks whether the model
should judge the image at all. Deferral asks whether a human should
double-check an in-scope judgement. Both are needed. Reporting them as one
number is an error.

Deferral threshold used here: 0.0378 bits (session A deployment_config.json).

| set | rejected | deferred | both | rejector only | neither | Jaccard |
|---|---|---|---|---|---|---|
| internal_test | 0.1% | 100.0% | 0.1% | 0.0% | 0.0% | 0.001 |
| brisc_clean | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.000 |
| out_of_scope | 58.3% | 100.0% | 58.3% | 0.0% | 0.0% | 0.583 |

| out-of-scope category | rejector catches | deferral alone would catch | caught only by input validation | missed by both |
|---|---|---|---|---|
| 1. corrupted scans | 51.9% | 100.0% | 0.0% | 0.0% |
| 3. non-medical | 100.0% | 100.0% | 0.0% | 0.0% |

## Wrongly accepted images

750 out-of-scope images were accepted, of which 0 were not deferred either. Samples are in `analysis/results/ood/wrongly_accepted/`, with contact sheets per category.

## Data

| category | n |
|---|---|
| 1. corrupted scans | 1560 |
| 3. non-medical | 240 |

Acquisition and licences: `reproducibility/out_of_scope_data.md`.

## Figures

- `analysis/results/ood/figures/score_distributions.png`
- `analysis/results/ood/figures/roc_per_method.png`
- `analysis/results/ood/figures/rejection_by_subcategory.png`
- `analysis/results/ood/figures/false_rejection.png`

