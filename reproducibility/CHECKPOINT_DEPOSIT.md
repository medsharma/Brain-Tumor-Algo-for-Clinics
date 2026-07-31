# Depositing the trained checkpoints

**Status: not done. No DOI exists yet.**

Until this deposit is made, **nobody outside the original machine can verify any
number in this project.** Every metric in `MODEL_CARD.md` traces back to ten
files that exist in exactly one place, on one laptop, unbacked-up. That is a
single point of failure for the entire evidence base, and it is the cheapest
serious problem in this repository to fix.

---

## Why not GitHub

Each ResNet-50 checkpoint is 220 MB and each ViT-B/16 checkpoint is 459 MB. Ten
files, **3.4 GB total.** GitHub rejects single files over 100 MB.

Git LFS was considered and rejected: the free tier gives 1 GB of storage and
1 GB/month of bandwidth, so a single clone by a single reviewer would exhaust the
monthly quota. It is the wrong tool for archival research artefacts anyway,
because LFS gives no DOI, no permanence guarantee, and disappears if the repo is
deleted.

**Zenodo is the right home.** Free, CERN-operated, gives a permanent DOI, accepts
depositions up to 50 GB, and issues a versioned DOI so a later checkpoint can be
added without breaking the citation to this one.

---

## What to deposit

All ten checkpoints from `results/20260703_155524/`:

```
resnet50/seed_42/best_resnet50_seed42.pth      220,460,701 bytes
resnet50/seed_123/best_resnet50_seed123.pth    220,461,139
resnet50/seed_7/best_resnet50_seed7.pth        220,460,263
resnet50/seed_2024/best_resnet50_seed2024.pth  220,461,641
resnet50/seed_31/best_resnet50_seed31.pth      220,460,701
vit/seed_42/best_vit_seed42.pth                459,097,175
vit/seed_123/best_vit_seed123.pth              459,097,433
vit/seed_7/best_vit_seed7.pth                  459,096,917
vit/seed_2024/best_vit_seed2024.pth            459,097,691
vit/seed_31/best_vit_seed31.pth                459,097,175
```

Plus, so the checkpoints are usable rather than just present:

- `results/master_summary.json`
- `results/20260703_155524/comparison_summary.json`
- every `results/20260703_155524/*/seed_*/config.json` and `summary.json`
- `data/split_manifest.csv`, because a checkpoint without its split is not
  reproducible
- `CHECKSUMS.txt`, generated below

sha256 for all ten is in [MODEL_CARD.md](../MODEL_CARD.md#checkpoint-hashes).
Deposit them and the model card table becomes independently checkable.

---

## Steps

### 1. Generate the checksum file

```bash
cd "C:\Users\medha\OneDrive\Documents\MRI ALGO\results\20260703_155524"
find . -name "*.pth" | sort | xargs sha256sum > CHECKSUMS.txt
cat CHECKSUMS.txt   # confirm against MODEL_CARD.md before uploading
```

If any hash disagrees with the model card, **stop.** A checkpoint has changed
since 2026-07-31 and the model card no longer describes the files you are about
to publish.

### 2. Create the deposition

Log in at <https://zenodo.org> with GitHub or ORCID. New upload.

Upload the ten `.pth` files, the JSON summaries, `data/split_manifest.csv` and
`CHECKSUMS.txt`. At 3.4 GB this will take a while; Zenodo's browser uploader is
fine but the API is more reliable for files this size:

```bash
# needs a Zenodo personal access token with deposit:write and deposit:actions
export ZENODO_TOKEN="..."

BUCKET=$(curl -s -X POST "https://zenodo.org/api/deposit/depositions?access_token=$ZENODO_TOKEN" \
    -H "Content-Type: application/json" -d '{}' | python -c "import sys,json;print(json.load(sys.stdin)['links']['bucket'])")

for f in $(find . -name "*.pth"); do
    curl --progress-bar -o /dev/null -X PUT \
        "$BUCKET/$(basename $f)?access_token=$ZENODO_TOKEN" --upload-file "$f"
done
```

### 3. Metadata

- **Type:** Software / Model (Zenodo's "Software" type, with "Model" in the
  keywords, is the closest fit).
- **Title:** `Trained checkpoints for a brain tumor MRI triage classifier
  (ViT-B/16 and ResNet-50, 5 seeds, run 20260703_155524)`
- **Authors:** as on the repository.
- **License:** must match whatever the code license ends up being. **This is
  currently undecided.** See `docs/OPEN_QUESTIONS.md`. Do not deposit under a
  license that contradicts the repository.
- **Related identifiers:** `isSupplementTo` the GitHub repository URL, and
  `isDerivedFrom` the Kaggle training dataset.
- **Version:** `20260703_155524`.

### 4. Description field

Paste this. It is the part that stops someone downloading these and drawing the
wrong conclusion:

> Trained model checkpoints for a 4-class brain tumor MRI classifier (glioma,
> meningioma, pituitary, no tumor). Two backbones, ViT-B/16 and ResNet-50, five
> seeds each (42, 123, 7, 2024, 31), fine-tuned from ImageNet weights with
> MC-Dropout (T=20) for uncertainty estimation.
>
> **These are research artefacts. They are not a medical device, have no
> regulatory clearance, and must not be used for clinical decisions.**
>
> On a held-out split of the training pool the model calls a real tumor "no
> tumor" about 1 time in 100 (ResNet-50: 1.00%, 95% CI 0.74 to 1.35). It has
> never been evaluated on data from an independent source. See MODEL_CARD.md and
> LIMITATIONS.md in the linked repository before any use.
>
> Training data: Kaggle "Brain Tumor MRI Dataset" (Nickparvar 2023, CC0), a merge
> of Br35H, SARTAJ and the Figshare Cheng collection. No scanner, site or
> demographic metadata exists for any image, so no fairness or subgroup analysis
> is possible.

### 5. Publish, then wire the DOI back in

Publishing is irreversible on Zenodo. Files cannot be changed afterwards, only
superseded by a new version.

Once it is published, replace every "No DOI exists" statement with the real DOI.
They are in:

- `MODEL_CARD.md`, "Version and provenance"
- `README.md`, "Reproducing this"
- `LIMITATIONS.md`, "Reproducibility"
- `reproducibility/README.md`, "Data and code availability"
- this file

```bash
grep -rn "No DOI exists" .
```

### 6. Verifying a download

```bash
sha256sum -c CHECKSUMS.txt
```

Then confirm the checkpoint actually loads and reproduces a published number:

```bash
python -c "
import torch
sd = torch.load('best_resnet50_seed42.pth', map_location='cpu')
print(type(sd), len(sd) if hasattr(sd,'__len__') else '')
"
```

---

## If Zenodo is not an option

- **Hugging Face Hub** takes model files, gives a persistent URL and versioning,
  and its model card format maps onto `MODEL_CARD.md` directly. No DOI, and the
  hosting guarantee is a company's rather than CERN's.
- **OSF** gives DOIs and takes large files.
- **Figshare** gives DOIs, 20 GB free.

Zenodo is still the default: DOI, permanence, and no commercial dependency.

---

## Open items

- Code license undecided, which blocks the deposit license field. See
  `docs/OPEN_QUESTIONS.md`.
- Nobody has confirmed the ten checkpoints are backed up anywhere at all. Until
  this deposit exists, a disk failure destroys the evidence base for this
  project.
