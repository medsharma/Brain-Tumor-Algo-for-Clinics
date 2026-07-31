# Cross-session issues

**Append only.** Never edit or delete another session's entry. Prefix every entry
with your session letter and a UTC timestamp. Newest at the bottom.

---

## E — 2026-07-31T00:54Z — BRISC IS NOT AN INDEPENDENT EXTERNAL COHORT. READ THIS BEFORE YOU WRITE THE WORD "EXTERNAL" ANYWHERE.

**This is the loudest thing I will say all session. Everyone please read it.**

The BRISC 2025 paper says, in its own words, where its images came from. I pulled
the peer-reviewed version (Fateh et al., *Scientific Data*, 2026,
`s41597-026-06753-y`, open access at PMC12982668). It states that BRISC was
collated from:

- the **Cheng / Figshare** brain tumor dataset,
- **SARTAJ** ("Brain Tumor Classification (MRI)"),
- **Br35H** ("Brain Tumor Detection 2020"),

and that these were **aggregated through the Kaggle "Brain Tumor MRI Dataset"
(Nickparvar) collection.**

That is the same dataset this project trained on. `data/brain_tumor/` **is** the
Nickparvar merge of Br35H, SARTAJ and Figshare. See
`reproducibility/README.md` and `results/leakage_audit.md`.

So BRISC is not a second, independent cohort. It is a re-annotated,
de-duplicated, re-split **subset of the same image pool** the model was trained
on. Some BRISC images are plausibly the very images in our training split.

The BRISC authors also state plainly: *"While complete subject-level independence
cannot be guaranteed due to the source limitations ... multiple images from the
same subject may therefore be present within a single split."* Their
de-duplication was internal to BRISC. Nobody de-duplicated BRISC against **our**
training split, because nobody knew they had to.

### What this does and does not break

It does **not** invalidate anyone's code, and it does **not** mean anyone
contaminated anything. Nobody trained on BRISC. Contract rule 1 is intact.

It **does** change what the resulting number means. A BRISC score is not
"performance on data from another hospital." At best it is "performance on a
cleaner, expert-re-annotated cut of our own source pool, with different
preprocessing." That is still worth measuring. It is a much weaker claim.

The specific risk: if a meaningful share of BRISC is near-duplicate to our
**train** split, the BRISC number is partly a memorisation score, and it will
read as better than the model really is.

### What I am doing about it

Running a phash near-duplicate match of all 6,000 BRISC images against all 7,200
internal images, at the same Hamming threshold (5) that `src/code.py` uses to
define a near-duplicate cluster for the leakage-safe split. Script:
`docs/check_brisc_overlap.py`. CPU only, no GPU, no model, no labels fitted.
Result lands in `docs/results/brisc_overlap.json`. I will append the numbers here
the moment I have them.

### What I am asking each session to do

- **A** — please do not describe BRISC as "external validation" without a
  qualifier until my overlap number lands. If the train-overlap fraction is
  material, the headline BRISC metrics need a second version computed on the
  **non-overlapping subset only**, and that subset version is the one that
  belongs in the model card. I will do the subsetting in a column you can join
  on if you want it; tell me the format you prefer.
- **B** — your out-of-scope work is unaffected. But "false rejection rate on
  BRISC" inherits the same caveat.
- **C** — do not put "validated on an external dataset" in the UI. Not in a
  tooltip, not in an About box, not in a footer. I will send full UI text
  review separately.
- **D** — unaffected, but do not cite BRISC as evidence of cross-site
  generalisation in any figure caption.

I would rather be wrong about this loudly now than have it found by a reviewer,
or worse, after this thing is in a clinic.

---

## E — 2026-07-31T00:54Z — Advance notice: I will make the one permitted edit to `src/code.py`

Per `prompts/CONTRACTS.md`, session E may fix exactly one documented bug in
`src/code.py` and nothing else. Announcing it now, doing it late in my run.

**The bug** (documented in `results/leakage_audit.md`): in
`build_split_manifest()`, the phash branch computes the split from `df_phash` but
writes the original `df` to CSV. So `data/split_manifest.csv` has a `patient_id`
column that is 100% null, even though phash clustering genuinely ran and the
split is genuinely grouped. The split is correct. The CSV just cannot prove it.

**The fix**, one line, at `src/code.py:358`-ish, inside the phash branch:

```python
df["patient_id"] = df_phash["patient_id"]
```

**What it does not change:** nothing about split assignment, nothing about
training, nothing importable. It adds cluster IDs to a column that is currently
empty. No function signature changes. No behaviour changes for any caller.

**Verification I will run:** regenerate the manifest and confirm the
`(filepath, split)` assignment is identical to the current file, row for row. If
it is not identical, I stop and escalate here in capital letters, because that
would mean the trained checkpoints no longer match the manifest and every number
in this project is suspect.

If anyone objects, say so here before I do it. If you have `src/code.py` open,
you will want to pull after I push.
