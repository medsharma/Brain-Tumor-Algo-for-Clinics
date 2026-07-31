# Open questions that need a human decision

Things this project cannot resolve by writing more code or reading more papers.
Each needs someone to make a call.

Last updated: 2026-07-31.

---

## 1. What license does this code carry?

**Status: unresolved. The repository is public with no license file.**

Public with no license means all rights reserved. Anyone who clones this and uses
it is technically infringing. That is almost certainly not the intent, and it
also blocks the Zenodo checkpoint deposit, which needs a license field.

The realistic options:

| license | what it means here |
|---|---|
| **MIT / BSD-3** | Anyone can use it, including commercially, including a company that closes it. Maximum adoption, zero leverage. |
| **Apache-2.0** | Same as MIT plus an explicit patent grant and a requirement to state changes. The usual default for anything that might touch a regulated product. |
| **AGPL-3.0** | Anyone using it over a network must publish their changes. Keeps derivatives open. Most companies will not touch it. |
| **CC BY-NC** | Non-commercial only. Not a software license; do not use it for code. |

**Recommendation: Apache-2.0.** The patent grant matters for anything with a
regulatory future, and the "state your changes" clause matters for a medical
tool where a modified fork behaving differently is a safety issue.

Whatever is chosen must match the Zenodo deposit and must be compatible with the
training data license (CC0, so no constraint) and BRISC (CC BY 4.0, but BRISC
images are not redistributed, so also no constraint).

**Blocked until decided:** `reproducibility/CHECKPOINT_DEPOSIT.md` step 3,
`README.md` licensing section.

---

## 2. Which country is this for?

**Status: unresolved, and it blocks the entire regulatory question.**

Device classification, evidence requirements, and whether this is even a
regulated device at all depend on jurisdiction. A triage aid that does not
provide a diagnosis is treated very differently in the EU (MDR, likely Class IIa
or IIb for anything driving triage) than in the US (FDA, possibly a Class II
device needing 510(k), possibly enforcement discretion as CDS), than in India
(CDSCO), than in most of sub-Saharan Africa (frequently no specific software
device pathway at all).

Nothing about the regulatory path can be written down until this is answered.
"Rural and under-resourced clinics" is a description of a setting, not a
jurisdiction.

---

## 3. Is anyone going to fund a prospective cohort?

**Status: unresolved, and it is the bottleneck on almost every open limitation.**

The following all resolve with one prospective, IRB-approved, demographically
annotated cohort, and resolve with nothing else:

- external validation on genuinely independent data
- fairness and subgroup analysis
- calibration under distribution shift
- real-world prevalence, and therefore the real false-referral burden
- any regulatory submission

Without it the project can improve its engineering indefinitely and its evidence
base not at all. This is a funding and partnership question, not a technical one.

---

## 4. Who owns clinical accountability in a pilot?

**Status: unresolved.**

If this is deployed to a clinic and a flagged-as-normal scan turns out to be a
glioma, who is responsible? The clinic? The developer? Nobody, because it was
"just research"?

This needs an answer before the first pilot site, not after. It also determines
what the software must log, which is a concrete engineering requirement that
session C's audit logging should be designed against.

---

## 5. Should the ViT checkpoints be published at all?

**Status: a judgement call, worth making deliberately.**

The measured evidence says ResNet-50 is the safer deployment choice. Not on
accuracy, where the two are statistically indistinguishable, but on failure
behaviour: ResNet-50's uncertainty flags 73% of its own missed tumors at 5%
deferral against ViT's 36%, and ViT calls 36% of its missed tumors "no tumor"
with 90%+ confidence against ResNet-50's 10%.

Publishing both is honest and supports the comparison in the manuscript.
Publishing both also means someone can pick up the ViT weights, see 0.962
accuracy, and deploy the less safe model without ever reading why. If they are
published, the model card warning has to travel with them, which is why the
Zenodo description in `reproducibility/CHECKPOINT_DEPOSIT.md` carries it.

---

## 6. The Kaggle image-count discrepancy

**Status: minor, but unresolved and worth ten minutes.**

This repo holds 7,200 training images. The current Kaggle listing describes
7,023. We believe this copy is Version 2 and the listing has since been revised.
Filename conventions match. It has not been confirmed against a fresh download.

Someone should download the current version, diff the filename list against
`data/split_manifest.csv`, and record the answer in `docs/DATA_PROVENANCE.md`. If
the datasets differ in content rather than count, that affects every claim about
what the model trained on.
