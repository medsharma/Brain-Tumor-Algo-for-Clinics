# Overclaim audit

Every place in this repository where the language could outrun the evidence, what
was done about it, and how to re-run the check.

Run the sweep yourself:

```bash
bash docs/overclaim_grep.sh          # whole repo
bash docs/overclaim_grep.sh --ui     # app/ only, the highest-stakes text
```

Last full sweep: 2026-07-31, session E. Re-run before any push to `main`, any
release, and any change to user-facing text.

---

## The rule being enforced

From `MISSION.md`: *"Be honest about what the tool can and cannot do. Overclaiming
gets people hurt and kills trust."*

Operationally, every claim in this repository must satisfy one of:

1. **Cut it**, because no evidence supports it.
2. **Qualify it**, with the dataset name, the n, and the conditions attached in
   the same sentence.

A number without its dataset is not a number. It is a rumour that happens to have
digits in it.

---

## What was cut or rewritten

### 1. BRISC 2025 as "external validation". The big one.

**Found:** the entire project was organised around BRISC being an independent
external cohort. Session prompts, `analysis/HANDOFF.md`, and every session's
plan assumed it.

**Evidence against:** 4,787 of BRISC's 6,000 images are byte-identical files to
images in `data/split_manifest.csv`. 3,353 are in the train split. BRISC's own
paper states it was collated from the same three sources via the same Kaggle
merge used for training. Of BRISC's 4,793 tumor-bearing images, the model has
seen 4,735.

**Action:** the claim is dead. It appears nowhere in `MODEL_CARD.md`,
`README.md`, `LIMITATIONS.md` or `docs/DATA_PROVENANCE.md` except as a
documented failure. Session A independently reached the same conclusion by pixel
comparison and now routes all BRISC reporting through a clean subset. Historical
documents that assumed otherwise carry dated correction headers with their
original text preserved.

**Reproduce:** `python docs/check_brisc_overlap.py`.

### 2. "No external image set was available" as a settled fact

**Found:** `reproducibility/README.md` and `analysis/HANDOFF.md` stated the OOD
evaluation could not be run because no out-of-distribution image set existed.

**Action:** superseded. Session B is building one. Both documents now say so and
point at `analysis/results/ood/`.

### 3. `environment.yml` implying a verified conda install

**Found:** `reproducibility/README.md` said "Verified against Python 3.10.11 on
Windows 11" directly under both a conda and a pip command, implying both were
checked. The conda file pinned `pytorch=2.12.1` from the `pytorch` channel,
whose newest build is 2.5.1 and which stopped publishing in March 2025. That
solve could never have succeeded.

**Action:** file fixed, and `reproducibility/README.md` now has an explicit
"What is verified and what is not" section. The pip path is verified. The conda
path is marked unverified, because conda is not installed on this machine.

### 4. Reproducibility placeholders reading as administrative

**Found:** four `[PLACEHOLDER: ...]` markers in `reproducibility/README.md`,
including the Kaggle dataset license. The repo is public and redistributes 7,200
images from that dataset, so "license TBC" was a live legal question dressed as
a formatting task.

**Action:** all four resolved. Kaggle training data is CC0 1.0, redistribution
permitted, with three caveats written out in `docs/DATA_PROVENANCE.md` rather
than buried. BRISC is CC BY 4.0. Hardware recorded. Checkpoint archive is
explicitly **not done**, with instructions and a DOI slot in
`reproducibility/CHECKPOINT_DEPOSIT.md`.

**One new honest gap opened rather than closed:** this public repository has **no
code license at all**, which means all rights reserved and nobody may legally
reuse it. That was not previously written down anywhere. See
`docs/OPEN_QUESTIONS.md`.

### 5. Accuracy as the headline metric

**Found:** the natural headline everywhere was "0.96 four-way accuracy". It is
the number a reader remembers and the one that says least about patient safety.

**Action:** `MODEL_CARD.md` and `README.md` both lead with the **tumor miss
rate**. Four-way accuracy appears well below it, labelled as an internal number
that must never be quoted as real-world performance. The relegation is
deliberate.

### 6. Backbone equivalence

**Found:** the manuscript concludes the two backbones are equivalent, on the
basis that per-seed McNemar found no significant accuracy difference in any of 5
seeds. That is a correct statement about accuracy and a misleading one about
safety.

**Evidence:** ResNet-50's uncertainty flags 73% of its own missed tumors at 5%
deferral. ViT's flags 36%. ViT calls 36% of its missed tumors "no tumor" at 90%+
confidence, ResNet-50 10%.

**Action:** documented as a real difference in `MODEL_CARD.md` and `README.md`,
with an explicit recommendation for ResNet-50 on safety grounds. Not an
overclaim that was cut; an underclaim that was corrected.

### 7. The `patient_id` column that could not prove its own claim

**Found:** `results/leakage_audit.md` established the split is grouped by
near-duplicate cluster. The evidence was a console log from a run in July. The
committed manifest's `patient_id` column was 100% null, so nobody cloning the
repo could verify it.

**Action:** bug fixed in `src/code.py`, manifest regenerated, split confirmed
byte-identical. 0 of 4,784 clusters straddle a split boundary, and that is now
checkable in three lines of pandas from a fresh clone.

---

## What was checked and left alone, with the reason

Not every hit is an overclaim. These were reviewed and kept.

| location | hit | why it stays |
|---|---|---|
| `MISSION.md:27` | "clinical grade" | A statement of the owner's goal, not a claim about the current tool. `MISSION.md` is also frozen for every session. |
| `MISSION.md:100` | "trustworthy signal" | Same. It is the definition of success, phrased as a future state. |
| `results/leakage_audit.md:113` | "98.9% accuracy" | Used only to repudiate the number as leakage-inflated. That is the correct context, and it is the only context in which it may appear. `results/**` is frozen. |
| `manuscript/manuscript.md` (several) | "generalization", "98.9%", "trustworthy" | Every instance is either explicitly negated ("do not establish generalization beyond this single dataset") or is repudiating the old number. The file is a historical snapshot and must not be edited; see `manuscript/README.md`. |
| `analysis/power_analysis.py`, `src/xai_and_stats.py`, `src/code.py:1468` | "trustworthy", "reliable", "robust" | Ordinary technical vocabulary about statistical estimators, not claims about model performance. "Reliable CI estimates" is a statement about bootstrap iterations. |
| `analysis/results/**/*.json` | bare accuracy figures | Machine-readable result files. The dataset is identified by file path and by the embedded `provenance` block. No prose claim is being made. |

**Finding worth stating plainly: the pre-existing repository was already largely
clean.** Earlier sessions did careful work here. The manuscript in particular
negates its own generalisation claims in the same sentence it makes them. The
overclaiming risk in this project was never sloppy adjectives. It was one
structural assumption, that BRISC was external data, which was wrong and which
everything else was built on top of.

---

## Standing checks

Things to re-run rather than trust.

**The cheapest external-validation smoke alarm, courtesy of session A.** If a
model scores *higher* on a supposedly external dataset than on its own internal
held-out test split, suspect contamination before celebrating. ViT seed 42 scored
0.9728 on full BRISC and 0.9613 internally. That single comparison would have
caught this on day one.

**Before adopting any future external dataset:**

1. Read the dataset paper's provenance section before downloading it. BRISC says
   what it is made of, in plain English, in its own *Scientific Data* article.
2. Hash every image in both sets and compare. `docs/check_brisc_overlap.py` is
   the template. It takes 15 minutes on CPU.
3. Compare the external score against the internal held-out score. Higher is a
   red flag, not a result.

**Before any release or push to `main`:** `bash docs/overclaim_grep.sh`, and
resolve every hit.

---

## Session C, the application text

**Status: reviewed against what session C has published so far.** See
`handoff/ISSUES.md` for the corrections sent.

Text a clinician reads under time pressure is where overclaiming does the most
direct harm, so it gets the strictest standard in the repository:

- No accuracy figure without its dataset name and n visible in the same view.
- The words "external", "externally validated", "independent dataset",
  "validated on 6,000 scans" must not appear. BRISC is 80% the training data.
- No "diagnosis", "detects", "confirms", "rules out". The tool classifies an
  image. It does not do any of those things.
- The tumor miss rate must be visible somewhere a user can find it, not only the
  accuracy.
- "No tumor" must never render as a clean bill of health. The model has four
  classes; a metastasis, a stroke or a bleed all land somewhere, and "no tumor"
  is one of the places they land.
- Confidence must not be presented as probability of correctness without the
  caveat that a third of ViT's missed tumors carried 90%+ confidence.
