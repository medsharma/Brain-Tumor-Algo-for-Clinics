# Session B status - refusing scans the model should not judge

**Written 2026-07-30T22:55Z, during integration.** Session B never wrote a
status file before it stopped. This is the first one, and it covers the work
finished on GPU after the five parallel sessions ended.

## Headline

**The clever rejector was built, measured, and thrown away. The dumb one
shipped.**

Prompt B warned that the most likely way this workstream turns out useless is a
rejector that detects "not my training set" instead of "not a brain MRI". That
is exactly what happened, and it was caught by measuring false rejection on
outside brain MRI.

Mahalanobis distance in penultimate features won the pre-registered selection
rule. AUROC 0.94, best of five methods, and only 3.8% false rejection on
internal test. Every internal number said ship it.

Then it met legitimate outside brain MRI:

| BRISC subset | mahalanobis wrongly rejects |
|---|---|
| full (mostly training data republished) | 12.3% |
| clean vs train (genuinely unseen) | **27.7%** |
| clean vs any (strictest) | **57.8%** |

**False rejection rises as images get less like the training set.** That is the
signature. Internal data could never have revealed it, because the Mahalanobis
statistics were fitted on internal train. A clinic with a different scanner
would have had most of its scans refused on day one.

The cheap image precheck does not do this. Its false rejection is **flat**
across the same three subsets:

| BRISC subset | precheck wrongly rejects |
|---|---|
| full | 0.22% |
| clean vs train | 0.15% |
| clean vs any | 0.00% |

Flat is what a genuine image-property check looks like.

**Shipped: `precheck_only`.** The mahalanobis threshold is still recorded in
`rejector_config.json` and can be turned back on with `--enable-score-stage` by
a human who disagrees. Nothing was fitted on a BRISC label. The decision was
taken on a measurement, and it is written down so it can be reversed.

## What the precheck catches, and what it misses

Catches 58.3% of the out-of-scope set overall.

| category | rejected |
|---|---|
| non-medical photographs and screenshots | **100%** |
| corrupted or tampered scans | 52% |

Per-subcategory, worst first. **These are the holes:**

| subcategory | rejected |
|---|---|
| JPEG quality 5 | **0.0%** |
| rotated 90 degrees | **0.0%** |
| upside down | **0.0%** |
| thumbnail | 1.7% |
| heavy blur | **8.3%** |
| overexposed | 30.8% |
| corner crop | 50.0% |
| underexposed | 84.2% |

**Heavy blur at 8.3% is the one that should worry a clinician.** A motion-
corrupted or badly acquired scan is a real and common event in a rural clinic,
and the precheck waves it straight through to a model that will return a
confident four-way answer.

Rotation and flips are less alarming: those images are still brain MRI, so the
right question is whether the model is orientation-robust, which is session A's
territory, not a rejection question.

**Mitigating fact:** 0.0% of out-of-scope images come back with a confident
class and no warning of any kind. What the precheck misses, the entropy
deferral mostly catches. The two stages are different mechanisms and neither
alone is sufficient. Do not merge their numbers.

## The biggest unknown in this workstream, stated plainly

**Category 4 is empty and untested.**

Brain MRI showing something the model has no class for: a metastasis, a rarer
tumour type, a stroke, an abscess, a haemorrhage. The model has four classes and
will force any of these into one of them.

No permissively licensed set was obtained. So **the most clinically important
rejection case in this project has never been tested.** Rejecting a photo of a
document at 100% proves close to nothing about clinical safety.

Session E: this belongs in the model card as a headline limitation, not a
footnote. Session A's category-4 note is the same point from the other side: a
brain MRI with unknown pathology is fully in-distribution to any uncertainty
score, so no entropy threshold catches it either. The model is not uncertain.
It is wrong.

## What is published

| path | what |
|---|---|
| `src/input_validation.py` | Contract 3 signature, unchanged. C imports it. |
| `analysis/results/ood/rejector_config.json` | Contract 3. `method: precheck_only`. |
| `analysis/results/ood/OOD_RESULTS.md` | full results |
| `analysis/results/ood/ood_metrics.json` | every number |
| `analysis/results/ood/rejector_stats.npz` | fitted mahalanobis stats, kept for the override |
| `reproducibility/out_of_scope_data.md` | how to rebuild the set |

The out-of-scope images themselves are gitignored. Rebuild with
`python analysis/out_of_scope_rejection.py build`.

## Two things that were broken and are fixed

**1. The precheck was rejecting 7.95% of real brain MRI.** The app was running
on `DEFAULT_PRECHECK_RULES`, the hand-set thresholds, because `evaluate` had
never been run and no `rejector_config.json` existed. One rule, `fg_solidity`,
caused 7.30 of those 7.95 points while catching 0% of out-of-scope images. The
fitting routine drops it automatically. Session C's
`test_a_real_brain_mri_is_not_rejected` was failing on 3 of 8 real scans and now
passes.

**2. B's committed score parquets held absolute paths into the `mri-B`
worktree**, which no longer exists, so nothing downstream could join against
them. Regenerated with `score --force`.

## Caveat a reader should not miss

The rejector is fitted and measured on **resnet50 seed 42**, but session A now
ships **vit ensemble5**. The precheck itself is model-free, so the shipped
`precheck_only` path is unaffected. The mahalanobis statistics in
`rejector_stats.npz` are for a model the clinic will not be running, and would
need refitting before anyone re-enables the score stage.

## Blocked on

Nothing. Category 4 needs data acquisition and a licence check, which is a human
task.
