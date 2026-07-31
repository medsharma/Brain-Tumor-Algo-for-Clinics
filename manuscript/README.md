# Historical snapshot. Do not edit these files.

**The live manuscript lives in a different repository:
<https://github.com/medsharma/Brain-Tumor-ML-Publication>, frozen at commit
`b53c0b3`.**

The files in this directory (`manuscript.md`,
`supplementary_TRIPOD-AI_checklist.md`) were copied here when this clinical
repository was created. They are a **point-in-time snapshot of the paper draft,
kept for reference.** They are not the paper.

- Do not edit them here. Edits here will never reach the paper.
- Do not push anything to the publication repository. It is frozen and is
  deliberately not configured as a remote in this checkout.
- If the paper genuinely needs updating, that is a separate decision to be made
  with the human first. See `MISSION.md`, "Repositories".

---

## What has changed since this snapshot was taken

Two things in this repository now supersede statements in the snapshot, and in
the two `analysis/` documents the snapshot leans on.

### 1. "No external cohort exists" is superseded, but the conclusion is not

`analysis/EXTERNAL_VALIDATION_GAP.md` and `analysis/HANDOFF.md` both state that no
external or held-out cohort was available for this project, and the manuscript's
framing as an internal-validation study rests on that.

A candidate was found and evaluated on 2026-07-31: **BRISC 2025** (Fateh et al.,
*Scientific Data* 2026, arXiv:2506.14318), 6,000 expert-annotated T1 images. The
trained checkpoints were run on it, inference only.

**It failed the independence test.** 4,787 of its 6,000 images are byte-identical
files to images in `data/split_manifest.csv`; 3,353 are in the train split. The
BRISC paper states it was collated from Cheng/Figshare, SARTAJ and Br35H via the
same Kaggle merge used for training. Of its 4,793 tumor-bearing images, the model
has already seen 4,735.

So **the manuscript's framing as an internal-validation study remains correct.**
The reasoning behind it is now stronger: it rests on a measured negative result
rather than on having searched and found nothing.

Anyone who does eventually update the paper should replace "no external cohort is
available" with "one candidate external cohort was identified and rejected on
provenance grounds, with the overlap quantified", and cite
[`../docs/DATA_PROVENANCE.md`](../docs/DATA_PROVENANCE.md).

### 2. The reproducibility placeholders are resolved

The snapshot's "how to read this draft" note lists outstanding
`[PLACEHOLDER: ...]` markers including dataset-license confirmation and a
checkpoint archive URL. In **this** repository those are now resolved or
explicitly scoped:

- Training data license: **CC0 1.0**, redistribution permitted, caveats written
  out in [`../docs/DATA_PROVENANCE.md`](../docs/DATA_PROVENANCE.md).
- BRISC license: **CC BY 4.0**, attribution given.
- Checkpoint archive: not yet deposited. Instructions and a DOI slot in
  [`../reproducibility/CHECKPOINT_DEPOSIT.md`](../reproducibility/CHECKPOINT_DEPOSIT.md).
- Hardware, environment and run-command placeholders: resolved in
  [`../reproducibility/README.md`](../reproducibility/README.md).
- `environment.yml` was broken and is fixed. It pinned a torch version the conda
  channel does not carry.

The remaining markers in the snapshot (funding, conflicts of interest,
protocol/registration, ethics approval, Figure 1 graphic, literature comparison)
still need author input and are outside this repository's scope.

### 3. What the manuscript does not cover at all

The snapshot is a development-and-internal-validation paper. It says nothing
about the clinical questions this repository now documents: tumor miss rate as
the primary safety endpoint, per-class miss rates, deferral behaviour, confident
misses, out-of-scope input rejection, or the fact that no demographic data exists
so no fairness analysis is possible.

For those, read [`../MODEL_CARD.md`](../MODEL_CARD.md) and
[`../LIMITATIONS.md`](../LIMITATIONS.md). They are the current documents. The
manuscript is the historical one.
