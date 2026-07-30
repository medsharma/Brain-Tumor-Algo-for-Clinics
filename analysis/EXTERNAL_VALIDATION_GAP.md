# External Validation: Known Gap

**Status: no external or held-out dataset is available in this project. This is a
real limitation of the current work, not a formality — read before writing any
generalization claim in the manuscript.**

*(Note: the repo was reorganized after this doc was written — `code.py` and
`split_manifest.csv` now live at `src/code.py` and `data/split_manifest.csv`.
References below are preserved as originally written; the substance is
unaffected by the reorganization.)*

## What was checked

Before writing this doc, the repository and local filesystem were searched for
any second imaging source (different scanner, institution, or acquisition
protocol) that could serve as an external validation set:

- `data/` contains exactly one dataset: `data/brain_tumor/{glioma,meningioma,
  pituitary,notumor}/`, 1,800 images/class, 7,200 total.
- `results/leakage_audit.md` (produced by another session auditing
  `code.py::build_split_manifest()`) identifies this as almost certainly the
  public Kaggle "Brain Tumor MRI Dataset" (a combination of the Br35H, SARTAJ,
  and Figshare brain-tumor sources published as one slice-level Kaggle
  dataset), based on filename conventions (`Te-gl_1.jpg`, `Tr-me_995.jpg`,
  `Te-aug-me_1.jpg`, ...). This has **not** been confirmed against the
  original download/DOI — see `reproducibility/README.md`'s data-availability
  placeholder.
- No second directory, archive, or reference to another scanner/site/cohort
  was found anywhere in the repo (code, configs, docs) or on the local
  filesystem outside this project.
- `code.py::evaluate_ood()` / `run_ood_evaluation()` exist and are wired up
  (`--ood-dir` CLI flag) but are an **out-of-distribution entropy-separation
  check** (is the model's uncertainty higher on non-brain-MRI images?), not
  external validation (does the model generalize to brain MRI from a
  different scanner/protocol/population?). These answer different questions;
  do not conflate them in the manuscript. No `--ood-dir` has been run yet
  either, as of this writing.

## Why this matters for this dataset specifically

Because the likely source is a **combination of three public sub-sources**
(Br35H, SARTAJ, Figshare) merged into one Kaggle release, the "single dataset"
already has some heterogeneity in acquisition — but the current train/val/test
split does not stratify or track which sub-source each image came from, so:

- We cannot currently measure whether performance is uniform across the
  merged sub-sources, or whether the model is implicitly learning
  source-specific artifacts (scanner vendor, preprocessing, image size/crop
  conventions used when each sub-source was assembled) rather than
  tumor-relevant features.
- A held-out **test split from the same merged pool** (which is what
  `split_manifest.csv` currently provides, now leakage-audited at the
  near-duplicate-cluster level per `results/leakage_audit.md`) demonstrates
  the model doesn't memorize training images, but it does **not** demonstrate
  the model generalizes beyond this particular data collection's site mix,
  scanner mix, preprocessing pipeline, or patient population.

## What a reviewer will ask, and what we can honestly say today

| Question | Honest answer with current evidence |
|---|---|
| "Does the model generalize to scans from a different hospital/scanner?" | **Unknown.** Not tested. No external cohort available. |
| "Is the held-out test performance at least not inflated by leakage?" | Yes, with caveats — `results/leakage_audit.md` shows the split is grouped at the phash near-duplicate-cluster level (1,107 multi-image clusters kept intact), which is materially stronger than a naive per-file split, but this is still an in-distribution split of the same merged Kaggle pool. |
| "Could the reported accuracy be inflated by shared preprocessing/artifact signatures across the merged sub-sources?" | **Possible, untested.** No sub-source labels are tracked in `split_manifest.csv` to check this. |
| "Is the uncertainty-quantification (MC-Dropout) calibration expected to hold under distribution shift?" | **No basis to claim yes.** Calibration was only measured in-distribution (`analysis/results/calibration/`); calibration is well known to degrade under covariate shift, and this has not been tested. |

## What would be needed to close this gap

In order of practical feasibility, cheapest first:

1. **Sub-source stratification within the existing data.** If the three
   merged sources (Br35H / SARTAJ / Figshare) can be identified per-image
   (e.g., from filename patterns, image dimensions, or metadata), add a
   `source` column to `split_manifest.csv` and report per-source test
   accuracy. This is the cheapest possible step toward a generalization
   signal and requires no new data acquisition — just provenance work on the
   existing files. Not attempted in this session; flagged as a quick win for
   whoever next touches the data pipeline.
2. **A genuinely external public brain-tumor MRI dataset** with class labels
   compatible with this 4-class scheme (glioma / meningioma / pituitary /
   no-tumor), acquired independently of the Kaggle merge above — e.g. a
   distinct Figshare or TCIA (The Cancer Imaging Archive) brain tumor
   collection not already folded into this Kaggle release. This would need
   to be located, license-checked, and downloaded; `code.py`'s existing
   `evaluate_ood()` infrastructure is close to but not identical to what's
   needed (it does entropy-based OOD detection, not labeled-accuracy
   external validation) — a labeled external-eval function would need to be
   added, mirroring `run_single_seed()`'s test-evaluation block but pointed
   at the external manifest instead of the internal test split.
3. **A prospectively or separately collected institutional cohort**, ideally
   with documented scanner make/model and acquisition protocol metadata, if
   this work is intended for clinical deployment claims. This is a real data
   -acquisition effort (IRB, data-sharing agreements, etc.) well beyond the
   scope of a coding session and should be scoped as a follow-up study, not
   squeezed into this manuscript.

## Manuscript guidance

Until one of the above exists:

- Do **not** write "generalizes to clinical practice," "generalizes across
  scanners," or similar claims anywhere in the manuscript.
- State explicitly in Limitations that all reported metrics are on a
  held-out split of a single (albeit three-source-merged) public dataset,
  and that external validation on an independent cohort is future work.
- If reviewers require external validation before acceptance (likely for a
  Nature-family or top specialty clinical-imaging venue), option 1 above
  (per-sub-source stratified reporting) is the minimum additional analysis
  that could plausibly be completed without new data acquisition, and should
  be attempted before submission if time allows. It is not a substitute for
  a truly external cohort, and the manuscript should say so if used.
