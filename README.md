# Brain tumor MRI triage, for clinics without a radiologist

A research prototype that reads a brain MRI slice and flags whether it looks like
it needs urgent attention.

> ## Not for clinical use
>
> This is a research prototype. It has no regulatory clearance anywhere. No
> radiologist has ever used it on a real case. It has never been tested on a scan
> from a hospital, a named scanner, or a known patient population.
>
> **On the only data it has been tested on, it calls a real tumor "no tumor"
> about 1 time in 100.**
>
> Do not use it to make a decision about a person.

---

## Why this exists

In a lot of the world, a brain MRI is taken and then waits days or weeks for
someone qualified to read it. The scanner is there. The radiologist is not.

The idea is a second set of eyes that runs on a laptop, offline: flag the scans
that look abnormal, say how confident it is, show where it looked, and push the
uncertain ones to a human. Not to replace a doctor. To decide who gets looked at
first.

That is the goal. This repository is the work toward it, including the parts that
did not go well.

---

## Where the project actually stands

Honestly, and with the bad news first.

**It has one dataset.** Training and evaluation both come from the same public
Kaggle collection, itself a merge of three older public datasets. There is no
second source.

**The attempt at external validation failed.** We brought in BRISC 2025, a 6,000
image dataset published in 2025, to be the independent test. Then we hashed every
file. **4,787 of BRISC's 6,000 images are byte-for-byte the same files as images
in our training data.** Not similar images. The same files. 3,353 of them are in
the training split specifically.

Of the 4,793 tumor images in BRISC, the model had already seen 4,735.

So BRISC cannot tell us whether this model works on new patients. Nothing in this
repository can. That was the single most important question in the project and it
is still open. Details: [docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md).

**What is real:** a carefully built, leakage-audited training pipeline; genuine
uncertainty estimates that catch a majority of the model's own mistakes; a full
accounting of what is and is not known. The engineering is sound. The evidence
base is one dataset deep.

---

## Performance, honestly

Every number here is from the **internal held-out test split**: 1,112 images from
the same pool the model trained on, held out at the near-duplicate-cluster level.
It is not a real-world estimate. Nothing in this repository is.

Backbone: ResNet-50, 5 random seeds, MC-Dropout with 20 stochastic passes.

| what | number | 95% CI | on what |
|---|---|---|---|
| **Tumor called "no tumor"** | **1.00%** | 0.74 to 1.35 | 821 tumor images x 5 seeds |
| ...for gliomas specifically | 1.88% | 1.28 to 2.74 | 1,385 evaluations |
| ...for meningiomas | 1.13% | 0.68 to 1.85 | 1,330 evaluations |
| ...for pituitary tumors | 0.00% | 0.00 to 0.28 | 1,390 evaluations |
| No-tumor scan flagged as tumor | 1.72% | 1.17 to 2.52 | 291 no-tumor images x 5 seeds |
| Four-way accuracy | 0.964 | ≈0.948 to 0.977 | 1,112 images |

Intervals are Wilson score intervals except the accuracy row, which is the
bootstrap interval from the original run. **Read them as a floor on the
uncertainty, not a description of it** — see the systematic-misses point below.

Two things to sit with:

**Gliomas are missed most.** They are also the most aggressive of the three
families. The class this tool is worst at is the one where delay costs most.

**Some misses are confident misses.** Sending the most uncertain 20% of scans to
a human still leaves 3 missed tumors out of 821 for ResNet-50, and 17 for ViT.
The confidence score is useful. It is not a safety net.

**The misses are systematic.** All 88 missed-tumor events across all 10
checkpoints come from just 18 images, and 9 images account for 81% of them. One
image is missed by every model this project has ever trained, at 93% confidence.
That means every confidence interval here is narrower than the evidence really
justifies.

*An earlier version of this section also concluded that ensembling therefore
could not help. That went further than the evidence: on the uncontaminated BRISC
subset a 5-seed ensemble roughly halved the miss rate. It is measured below. The
correlated-error finding is real; the inference drawn from it was too strong.*

---

## The one number that is not from the training pool

After removing every BRISC image that overlaps the training data, **2,634 images
remain that the model genuinely has not seen.** This is the closest thing to an
external result this project has, and it is still not one: those images share
sources, scanners and preprocessing with the training data, and patient-level
overlap cannot be ruled out because neither dataset ships patient identifiers.

Shipped configuration: **ViT-B/16, 5-seed ensemble**, MC-Dropout T=20.

| what | number | 95% CI |
|---|---|---|
| **Tumor called "no tumor"** | **0.27%** | 0.07 to 0.55 |
| Sensitivity (tumor vs no tumor) | 99.73% | 99.45 to 99.93 |
| **Healthy scan flagged as tumor** | **9.33%** | 7.66 to 11.02 |
| Four-way accuracy | 96.20% | 95.52 to 96.92 |

**Read the third row.** The miss rate is the number this project watches hardest
and it holds up. Specificity is what degrades: from 98.3% internally to 90.7%
here. Roughly **1 healthy person in 11 gets a false alarm**, which in a rural
setting means a referral costing travel, money, time and fear. The alternative
ResNet-50 configuration is worse again, at nearly 1 in 5.

**The tool is safe in the direction it was built to be safe in, and expensive in
the other.**

Two more things a clinic would feel:

- **It defers 41% of scans to a human.** That is the price of the lowest miss
  rate. In a clinic with no radiologist, four scans in ten come straight back.
  Whether that is usable is a decision nobody has made yet.
- **Coronal and sagittal scans get far more false alarms than axial ones.** ViT
  specificity is 96% axial, 82% coronal, 93% sagittal. The training pool is
  mostly axial.

Full breakdown, including deferral behaviour and per-seed spread:
[MODEL_CARD.md](MODEL_CARD.md).

---

## What this is not

- Not a diagnosis. A doctor decides.
- Not autonomous. A human reads the scan, always.
- Not a full tumor detector. Three tumor families plus no-tumor. **A metastasis
  or any other tumor type can come back as "no tumor".**
- Not tested on stroke, bleeds, MS, or any non-tumor pathology. A brain
  haemorrhage would likely be labelled "no tumor", correctly and uselessly.
- Not tested on children.
- Not tested on any named scanner or sequence other than T1.
- Not fairness-audited. **The training data carries no age, sex or ethnicity, so
  no subgroup analysis is possible at all.** For a tool aimed at under-served
  populations that is a serious hole, not a formality.

The complete list: [LIMITATIONS.md](LIMITATIONS.md).

---

## How it works

1. A brain MRI slice goes in, resized to 224x224.
2. An input check is meant to reject images that are not brain MRI. **It has no
   published measurements yet**, so there is currently no evidence about what it
   catches or misses. The app refuses to start without it rather than run
   unguarded.
3. A ResNet-50, fine-tuned from ImageNet weights, runs 20 times with dropout
   left on. The 20 outputs are averaged.
4. Spread across those 20 passes is the uncertainty. High spread means the tool
   says "I am not sure, a human should look".
5. A Grad-CAM heatmap shows which part of the image drove the decision.

Two backbones were trained and compared, ViT-B/16 and ResNet-50. Their accuracy
is statistically indistinguishable. Their **safety behaviour is not**: ResNet-50's
uncertainty flags 73% of its own missed tumors at a 5% deferral rate, ViT's flags
36%. That difference is invisible if you compare on accuracy alone, and it is the
one that matters clinically.

---

## Reproducing this

```bash
git clone https://github.com/medsharma/Brain-Tumor-Algo-for-Clinics
cd Brain-Tumor-Algo-for-Clinics

conda env create -f reproducibility/environment.yml
conda activate mri-algo
# or: pip install -r reproducibility/requirements.txt
```

Rebuild the leakage-safe split:

```bash
python src/code.py --data-root ./data/brain_tumor \
    --manifest data/split_manifest.csv --force-manifest
```

Full training run, 2 backbones x 5 seeds. This ran on CPU and took a long time:

```bash
python src/code.py --data-root ./data/brain_tumor \
    --manifest data/split_manifest.csv --results-dir results \
    --epochs 30 --batch-size 32 --lr 1e-4 --weight-decay 0.01 \
    --label-smoothing 0.1 --workers 4 --mc-T 20 --early-stop-patience 5 \
    --seeds 42 123 7 2024 31 --model both
```

Reproduce the two checks that reframed this project:

```bash
python docs/check_brisc_overlap.py      # BRISC vs training-data overlap
python docs/internal_safety_metrics.py  # tumor miss rate and deferral behaviour
```

Full command list and environment notes:
[reproducibility/README.md](reproducibility/README.md).

**Trained checkpoints are not in this repository.** Each is over GitHub's 100 MB
limit. Until they are deposited in an archive, nobody outside the original
machine can verify any number here. Deposit instructions:
[reproducibility/CHECKPOINT_DEPOSIT.md](reproducibility/CHECKPOINT_DEPOSIT.md).
**No DOI exists**, because the deposit has not been made.

---

## Running the app

The app is a local web page served by a process on your own machine. It binds
`127.0.0.1` only and refuses to start on any network-visible address. Nothing is
uploaded anywhere and no internet connection is used.

```bash
python -m app.launch                 # opens a browser at the first free port from 8765
python -m app.launch --port 9000 --no-browser
```

**It refuses to start unless the real safety configuration, the real input
check, and the real explainer are all present.** While any of those is a stub it
will not run at all, except under `MRI_CLINIC_DEV_MODE=1`, and every screen then
carries a red DEVELOPMENT BUILD banner. That refusal is deliberate. Do not
disable it.

What it will and will not accept:

- One brain MRI slice at a time, JPEG or PNG.
- **No DICOM.** Export the slice from your viewer first, using the window your
  radiographer normally uses. Guessing a window level and width would change the
  image silently.
- **No study-level answer.** It judges one image. It does not combine slices.

Speed, measured on CPU (Windows 11, torch 2.12.1+cpu, 8 threads, ResNet-50,
single seed, MC-Dropout T=20): about **130 ms per image** once warm, plus about
0.5 s to load the model.

Tests: `python -m pytest app/tests` (178 tests).

---

## Repository layout

| path | what |
|---|---|
| `src/code.py` | training pipeline, models, MC-Dropout, split builder |
| `src/input_validation.py` | out-of-scope input rejection |
| `src/explain_runtime.py` | runtime heatmap generation |
| `app/` | the clinic-facing application |
| `analysis/` | validation, calibration, safety and explainability analysis |
| `docs/` | provenance, licensing, and the BRISC overlap investigation |
| `reproducibility/` | environment, run commands, data and checkpoint provenance |
| `results/` | training run outputs, leakage audit |
| `manuscript/` | historical snapshot only, see `manuscript/README.md` |
| `handoff/` | working notes between parallel development sessions |

---

## Data and licensing

**Training data:** Kaggle "Brain Tumor MRI Dataset" (Nickparvar, 2023), 7,200
images, CC0 1.0 Public Domain. A merge of Br35H, SARTAJ and the Figshare Cheng
collection. Redistributed here under CC0.

**Evaluation data:** BRISC 2025 (Fateh et al., *Scientific Data* 2026,
arXiv:2506.14318), CC BY 4.0. Not redistributed here. Download it yourself and
verify against the hashes in [docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md).

Both trace back to the same three source collections. That is the whole problem.

**Code: no license.** This repository is public and carries no license file,
which means all rights reserved by default. Nobody may legally reuse this code
until a license is chosen. That is a decision nobody has made yet, not an
oversight waiting to be typed in. See
[docs/OPEN_QUESTIONS.md](docs/OPEN_QUESTIONS.md) item 1, which recommends
Apache-2.0 and explains why.

---

## What would have to happen before this touches a patient

1. External validation on data with no lineage to Br35H, SARTAJ or Figshare.
2. A reader study with radiologists on real cases.
3. Prospective evaluation at a clinic like the target sites.
4. Demographic and subgroup analysis, which needs data that has demographics.
5. A regulatory pathway for a specific country. None chosen, none started.
6. A monitoring plan, because performance drifts and nobody would notice.

None of these are done. None are in progress.

---

## Documents

- [docs/FOR_CLINICIANS.md](docs/FOR_CLINICIANS.md) — plain words, no maths, for
  anyone deciding whether this should go near a patient
- [MODEL_CARD.md](MODEL_CARD.md) — read this before trusting any output
- [LIMITATIONS.md](LIMITATIONS.md) — everything this cannot do
- [docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md) — where the images came from
  and why BRISC is not external
- [MISSION.md](MISSION.md) — project mission and working rules

## Contact

Issues, including anything you think is a clinical safety problem:
<https://github.com/medsharma/Brain-Tumor-Algo-for-Clinics/issues>
