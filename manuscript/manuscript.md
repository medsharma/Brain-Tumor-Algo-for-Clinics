# A Monte Carlo Dropout Framework for Uncertainty-Aware Brain Tumor MRI Classification: Development and Internal Validation of ViT-B/16 and ResNet-50 Models

*Reported per the TRIPOD+AI statement (Collins et al., BMJ 2024;385:e078378). Section numbers below map directly to the completed checklist in `manuscript/supplementary_TRIPOD-AI_checklist.md`; TRIPOD+AI item numbers are noted in brackets.*

> **How to read this draft.** All quantitative results are now real, drawn from the completed 5-seed × 2-backbone run (`results/master_summary.json`, `results/20260703_155524/comparison_summary.json`, `analysis/results/*.json`) — none of the numbers in §5 (Results) or the Abstract are placeholders anymore. The remaining `[PLACEHOLDER: ...]` markers are all non-quantitative and require author/institutional input that cannot be inferred from the data or code: funding, conflicts of interest, protocol/registration, ethics-approval basis, intended clinical user/care pathway, dataset-license confirmation, a checkpoint/data archive URL, a Figure 1 flow-diagram graphic, and a literature comparison in Discussion. See `analysis/HANDOFF.md` for the full status of what's been verified.

---

## 1. Title and abstract *[Items 1–2]*

### 1.1 Title *[Item 1]*

Title above identifies: prediction-model development study, internal validation, target population (brain MRI), and outcome (4-class tumor classification). Confirmed final: `analysis/EXTERNAL_VALIDATION_GAP.md` establishes that no external or held-out cohort is available for this project, so "internal validation" is the correct and final framing — not a placeholder pending resolution.

### 1.2 Abstract *[Item 2]*

Structured per the separate TRIPOD+AI-for-Abstracts checklist (Background/Objective, Methods, Results, Discussion). Drafted with placeholders so its shape is fixed before numbers land:

> **Background:** Reliable uncertainty quantification is a prerequisite for any clinical deployment of an automated MRI triage tool. **Objective:** To develop and internally validate two deep-learning classifiers (ViT-B/16, ResNet-50) for 4-class brain tumor MRI classification (glioma, meningioma, pituitary, no tumor) with calibrated, non-collapsing uncertainty estimates via Monte Carlo (MC) Dropout, and to compare the two architectures on discrimination, calibration, and selective-prediction performance. **Methods:** Retrospective single-dataset study; N = 7,200 images from 4 classes (1,800/class), split 70/15/15 (train/val/test, phash near-duplicate-cluster grouped) into 4,979/1,109/1,112 images. Both backbones were partially fine-tuned (ViT: last 2 encoder blocks + head; ResNet-50: layer4 + head) and evaluated with MC Dropout (T=20 stochastic forward passes) across 5 seeds (42, 123, 7, 2024, 31). Metrics: accuracy, macro-F1, macro-AUC (bootstrap 95% CI, n=1000), expected calibration error (ECE, 15-bin), Brier score, area under the risk-coverage curve (AURC), and per-seed McNemar's test comparing the two architectures. **Results:** Across 5 seeds, ViT-B/16 achieved mean test accuracy 0.962 (range 0.957–0.964), macro-F1 0.962, macro-AUC 0.994 (95% CI ≈0.950–0.973 accuracy, ≈0.990–0.997 AUC, pooled across seeds); ResNet-50 achieved mean accuracy 0.964 (range 0.960–0.967), macro-F1 0.964, macro-AUC 0.994 (95% CI ≈0.953–0.975 accuracy, ≈0.990–0.997 AUC). ECE was 0.070 (ViT) and 0.074 (ResNet-50); Brier 0.068 and 0.066; AURC 0.010 and 0.011, respectively. Per-seed McNemar's test found no significant difference between architectures in any of the 5 seeds (0/5, p<0.05), with the direction of effect favoring ResNet-50 in 4/5 seeds and ViT in 1/5 — not a consistent effect. **Conclusion:** Both architectures achieve comparable, well-discriminating (macro-AUC ≈0.99), moderately-calibrated (ECE 0.07–0.08) performance on this internal test split under MC-Dropout uncertainty quantification, with no statistically significant or directionally consistent advantage for either backbone; selective prediction via entropy-based deferral recovers accuracy ≥0.98 at 95% coverage for both. These results support MC Dropout as a non-collapsing uncertainty-quantification approach for this task, but do not establish generalization beyond this single (three-source-merged) public dataset — external validation remains an open gap (see Limitations).

---

## 2. Introduction

### 2.1 Background *[Item 3a–3c]*

**3a — Clinical context and rationale.** Brain tumor subtype on MRI (glioma / meningioma / pituitary adenoma / no tumor) informs urgency of referral and initial management planning. Automated triage classifiers have been proposed to support radiologist workflow, but clinical adoption of any such model is gated on whether its confidence estimates are trustworthy enough to support a defer-to-human decision — not merely on top-line accuracy.

**3a — Related work and method rationale (uncertainty quantification approach).** An earlier iteration of this project used a deterministic auxiliary "uncertainty head" — a single additional output regressed against a confidence target in the same forward pass as the classification head — as a computationally cheap alternative to sampling-based uncertainty quantification. That approach exhibited **confidence collapse**: the auxiliary head converged to a narrow, near-constant confidence output largely uninformative of whether the classification head's prediction was correct, a failure mode consistent with the broader literature on deterministic single-network uncertainty estimation, where an auxiliary head optimized jointly with (or downstream of) the primary task loss has no mechanism forcing its output to track epistemic uncertainty rather than collapsing to a task-loss-convenient shortcut. This is the direct motivation for the design decision reported in Section 3.6 below: this project uses **MC Dropout** (Gal & Ghahramani, 2016) instead — a sampling-based approach with an established theoretical grounding as approximate Bayesian inference, T=20 stochastic forward passes at eval time, dropout active — precisely because it does not depend on a separately-trained head that can degenerate independently of the classification objective. This choice, and the reasoning above, is treated here as the study's designed methodology, not as a finding to be reported in Results/Discussion; no deterministic-uncertainty-head results are reported anywhere in this manuscript.

**3a — Related work and method rationale (data splitting).** The same earlier iteration also reported 98.9% test accuracy under a naive per-file random split. A leakage audit of the current pipeline (`results/leakage_audit.md`) traced this to the source dataset's structure: the meningioma class contains 203 augmented near-duplicate images (`Te-aug-me_*`, `Tr-aug_*` — rotated/brightness-jittered copies of originals), and more broadly, perceptual-hash clustering across the full dataset found 1,107 clusters (of 4,784 total) containing more than one near-duplicate image, together covering 2,416/7,200 images (33.6%) — none of which a naive per-file random split accounts for. A prior split of that kind could freely place near-duplicates of the same underlying image on both sides of the train/test boundary, inflating test accuracy through memorization rather than generalization. This is the second, independent motivation (alongside confidence collapse, above) for this project's methodological pivot: the current pipeline's `build_split_manifest()` uses phash-cluster grouping (Hamming distance ≤ 5) so that every image in a near-duplicate cluster is confined to a single split — confirmed active for the current `data/split_manifest.csv` in the leakage audit referenced above, not a no-op fallback. As with the confidence-collapse motivation, this is reported here as designed methodology; no numbers from the prior leakage-affected split are reported anywhere in this manuscript.

**3b — Intended users and care pathway.** [PLACEHOLDER: confirm intended deployment context and user — e.g., decision-support at point of radiologist review vs. triage prioritization — this affects how Discussion §6.3 (Usability) should be written.]

**3c — Health disparities.** Confirmed: the source dataset carries no demographic (age, sex, race/ethnicity) or site metadata at the per-image level (§3.9 Fairness). No disparity or subgroup analysis was possible, and none is reported anywhere in this manuscript — stated here explicitly as a scope limitation, not omitted.

### 2.2 Objectives *[Item 4]*

This is a **model development and internal validation** study (not an external validation study — confirmed final, see §6.2 Limitations and `analysis/EXTERNAL_VALIDATION_GAP.md`). Objectives:

1. Develop 4-class brain tumor MRI classifiers using ViT-B/16 and ResNet-50 backbones with MC-Dropout-based uncertainty quantification.
2. Report calibration (ECE, Brier), discrimination (macro-AUC, accuracy, macro-F1) and selective-prediction (risk-coverage, AURC) performance with bootstrap 95% CIs across 5 random seeds.
3. Compare the two architectures' predictive performance via per-seed McNemar's test.
4. An entropy-based out-of-distribution (OOD) detection check was planned as an optional secondary robustness measure (`src/code.py`'s `--ood-dir` flag) but was **not run**: no out-of-distribution image set was available for this project. This objective is not pursued in this manuscript; it is noted here as scoped-out rather than omitted silently.

---

## 3. Methods

### 3.1 Data sources *[Item 5a–5b]*

**5a.** Images are organized under `data/brain_tumor/<class>/`, 4 classes (glioma, meningioma, pituitary, notumor), 1,800 images per class (7,200 total). Per `results/leakage_audit.md`, filenames carry `Te-`/`Tr-` (test/train) prefixes baked in from the original authors' split, plus an augmented-image marker for meningioma (`Te-aug-me_*`, `Tr-aug_*`, 203 files) — consistent with the public Kaggle "Brain Tumor MRI Dataset" (Nickparvar, M. *Brain Tumor MRI Dataset*. Kaggle, 2023, https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), itself a combination of the Br35H, SARTAJ, and Figshare brain-tumor sources. This project does **not** use the original authors' train/test split (the `Te-`/`Tr-` prefixes are not treated as a split assignment); §3.7 describes the split actually used. The image count matches the dataset exactly: Kaggle **Version 2** of this dataset comprises 7,200 images (Training + Testing folders), identical to the 7,200 (1,800/class) present here — so this is the Nickparvar dataset as distributed, not a locally resampled derivative. The meningioma `-aug-` files are author-supplied augmentations included in that official release. **[PLACEHOLDER: confirm the exact license terms from the Kaggle listing before submission; the Kaggle page does not surface a formal DOI, so cite as above (Nickparvar 2023, Version 2) unless a DOI is later confirmed.]**

**5b.** Original imaging acquisition date range / enrollment period is **unavailable** — this is a secondary, pre-aggregated public dataset with no acquisition-date metadata retained at the per-image level.

### 3.2 Participants *[Item 6a–6c]*

**6a.** Source is a pre-aggregated public dataset (see §3.1); original acquiring center(s), scanner vendor/field strength, and geographic site are **not recoverable** from the data or metadata available to this project. This is stated here explicitly, and carried into Limitations (§6.2) as a scope constraint, rather than left unaddressed.

**6b.** Inclusion: all images in the 4 class folders at `data/brain_tumor/`. No exclusion criteria were applied in `build_split_manifest()` beyond file-extension filtering (`.jpg/.jpeg/.png/.bmp/.tiff/.tif`).

**6c.** No interventions applicable (imaging classification task, not a treatment-effect study).

### 3.3 Outcome *[Item 8a–8c]*

**8a.** Outcome is the 4-class tumor label (glioma / meningioma / pituitary / no tumor), assigned as a single categorical ground-truth label per image at the time the source dataset was compiled; no separate outcome-assessment timing applies (label is a property of the image, not a longitudinal event).

**8b–8c.** Radiologic ground-truth labeling methodology and labeler qualifications are inherited from the source dataset (§3.1) and are **unknown** — not independently verifiable by this project, and no labeling protocol is published alongside the Kaggle release. Stated explicitly here rather than assumed.

### 3.4 Predictors *[Item 9a–9c]*

**9a.** The sole predictor is the MRI image itself (whole-image classification; no hand-crafted or tabular predictors, no predictor pre-selection).

**9b.** Preprocessing: resize + ImageNet-statistics normalization (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`); training-split augmentation includes `RandomHorizontalFlip` (left/right axial symmetry is anatomically valid for this view) — `RandomVerticalFlip` and saturation `ColorJitter` are deliberately excluded (no meaningful vertical symmetry; RGB-converted MRI carries no true color information to jitter). See `src/code.py::get_transforms()`.

**9c.** Not applicable (no human-judgment-based predictor measurement).

### 3.5 Sample size *[Item 10]*

Sample size (7,200 images) is fixed by the source dataset; no prospective power calculation was performed to target a specific CI width. Post-hoc achieved precision (test N=1,112; bootstrap 95% CI, n=1,000, averaged across the 5 seeds): accuracy CI half-width ≈1.1–1.2 percentage points for both models (ViT ≈[0.950, 0.973]; ResNet-50 ≈[0.953, 0.975]); macro-AUC CI half-width ≈0.3–0.4 points (ViT ≈[0.990, 0.997]; ResNet-50 ≈[0.990, 0.997]). `analysis/results/power_analysis.md` additionally shows the per-seed McNemar comparison is **underpowered** at the current test-set size: achieved power for the observed discordance rates ranged 0.000–0.285 across seeds, and reaching 80–90% power for a McNemar comparison at these effect sizes would require a test set of roughly 4,400–13,400 images depending on seed (vs. the current 1,112) — this is reported as a limitation on the model-comparison claim specifically, not on the discrimination/calibration metrics themselves.

### 3.6 Missing data *[Item 11]*

No missing images/labels within the 4 class folders (complete-case by construction — every file present was assigned exactly one class label by its containing folder). No imputation was applicable.

### 3.7 Analytical methods *[Item 12a–12g]*

**12a — Data partitioning.** 70/15/15 train/val/test split, generated by `build_split_manifest()` with a descending-priority leakage-safety cascade:
1. Patient-ID regex parsing from filename stem, grouped so all images from one parsed ID stay in one split.
2. If patient IDs cannot be parsed for every file (the case here — see note below), perceptual-hash (pHash, Hamming distance ≤ 5) clustering of near-duplicate images within each class, using the same grouped-split logic so near-duplicate slices don't leak across splits.
3. Per-file stratified fallback (with an explicit warning logged) only if `imagehash` is unavailable or clustering fails.

For the current `data/split_manifest.csv`: patient-ID regex parsing matched 0/7,200 filenames (source filenames carry no recoverable patient/scan identifier — see §3.1), so the pipeline fell through to step 2. `results/leakage_audit.md` independently re-ran the build with logging and confirms step 2 (phash-cluster grouping) is genuinely what executed, not a silent fallback to step 3: perceptual hashing produced 4,784 clusters from 7,200 images, of which 1,107 (23.1%) contain more than one near-duplicate image (2,416 images, 33.6% of the dataset, affected; largest cluster 28 images), and the log explicitly records `"Using phash-cluster-stratified split."` `build_split_manifest()`'s saved CSV does not retain the phash cluster IDs in its `patient_id` column (a cosmetic gap — the IDs exist only in the internal working copy used for grouping, not a leakage bug: `df["split"]` is correctly computed from the phash-grouped assignment, only `df["patient_id"]` in the saved CSV is uninformative). This is confirmed, not a placeholder.

Observed split sizes (from the existing `data/split_manifest.csv`, not a placeholder — already generated, deterministic):

| Split | glioma | meningioma | pituitary | notumor | Total |
|---|---|---|---|---|---|
| train | 1,254 | 1,268 | 1,246 | 1,211 | 4,979 |
| val   | 269   | 266   | 276   | 298   | 1,109 |
| test  | 277   | 266   | 278   | 291   | 1,112 |

**12b — Predictor handling.** Not applicable beyond §3.4 (single image predictor, standard normalization).

**12c — Model class, architecture, and optimization.**

- **ViT-B/16** (`BrainTumorViT`): torchvision ViT-B/16 backbone; only encoder blocks 10–11 and the classification head are trainable, all earlier blocks frozen.
- **ResNet-50** (`BrainTumorResNet50`): torchvision ResNet-50 (ImageNet1K_V2 weights); only `layer4` and the classification head are trainable, `conv1/bn1/layer1-3` frozen — including keeping frozen-stage BatchNorm layers in `.eval()` mode even during `model.train()`, to prevent running-statistic drift in frozen stages.
- Shared head design: `LayerNorm → Dropout(p=0.3) → Linear(→256) → GELU → Dropout(p=0.3) → Linear(256→4)`.
- Optimizer: AdamW, lr=1e-4, weight decay=0.01; `ReduceLROnPlateau` scheduling; label smoothing=0.1; early stopping (patience=5) on validation loss; up to 30 epochs.
- 5 independent seeds per architecture (42, 123, 7, 2024, 31), each with `setup_reproducibility()` fixing all RNGs and disabling non-deterministic cuDNN kernels.
- Internal validation strategy: held-out validation split for early stopping/model selection, held-out test split (never used for any training decision) for all reported performance metrics.

**12d — Performance by group.** No sub-group (e.g. multi-site) performance breakdown applicable — single aggregated dataset, no site metadata (see §3.9 Fairness).

**12e — Metrics.**
- Discrimination: accuracy, macro-F1, macro-AUC (one-vs-rest), each with bootstrap 95% CI (percentile method, n=1,000 resamples, `src/code.py::bootstrap_ci`).
- Calibration: expected calibration error (ECE, 15 equal-width confidence bins) and multiclass Brier score (`src/code.py::compute_calibration_metrics`).
- Selective prediction: risk-coverage curve and area under it (AURC), entropy-ranked rejection, plus accuracy-at-fixed-coverage (80/90/95%) (`src/code.py::compute_risk_coverage_curve`).
- Model comparison: per-seed McNemar's test (continuity-corrected, `src/code.py::mcnemar_test`) comparing ViT vs. ResNet-50 predictions on the identical test set, for each of the 5 seeds independently, plus a summary of how many seeds reach significance (p<0.05) and whether the direction of the effect is consistent across seeds.
- Secondary robustness: entropy-based OOD-detection AUROC (in-distribution test entropy vs. an external image set), if run (see §2.2 objective 4 placeholder).

**12f–12g.** Not applicable (no post-hoc recalibration or external-cohort re-derivation reported in this development-and-internal-validation study — revisit if Session B's HANDOFF.md establishes an external validation arm).

### 3.8 Class imbalance *[Item 13]*

Dataset is balanced by construction (1,800 images/class); no class-weighting, oversampling, or class-imbalance correction was applied or needed. `StratifiedShuffleSplit` (used in the per-file fallback path only) and the group-split logic both stratify by class regardless.

### 3.9 Fairness *[Item 14]*

No demographic, site, or scanner metadata is available in the source dataset (see §3.1–3.2), so no fairness/subgroup evaluation was performed. **This is stated here as a methodological fact, and carried into Limitations (§6.2) as a scope constraint on what this study can claim.**

### 3.10 Model output *[Item 15]*

Output is a 4-way softmax over {glioma, meningioma, pituitary, notumor}, MC-Dropout-averaged over T=20 stochastic passes (`predict_with_uncertainty`); the reported class is `argmax` of the mean probability vector, with predictive entropy of that mean vector reported as the per-prediction uncertainty score used for calibration and selective-prediction analysis. No fixed decision threshold beyond argmax is used (multiclass, not a binary risk-threshold task); coverage-based deferral thresholds (§3.7, 12e) are reported at fixed coverage levels (80/90/95%) rather than as a single deployment threshold recommendation, pending Discussion §6.3.

### 3.11 Training vs. evaluation setting *[Item 16]*

Single dataset, single acquisition context throughout — training, validation, and test splits are all drawn from the same source pool via the split described in §3.7. No distributional differences between training and evaluation settings are known or expected beyond ordinary sampling variation from the same split cascade. This is a study-design constraint, elaborated in §6.2.

### 3.12 Ethical approval *[Item 17]*

**[PLACEHOLDER: state approving body / exemption basis, or confirm that use of a de-identified public dataset does not require separate IRB approval at this institution — do not leave this blank in the submitted manuscript.]**

---

## 4. Open science and transparency *[Item 18a–18f]*

- **18a — Funding.** [PLACEHOLDER: funding source(s) and role of funder, or state "no external funding."]
- **18b — Conflicts of interest.** [PLACEHOLDER: disclose for every author, or state "none declared."]
- **18c — Protocol.** [PLACEHOLDER: link/DOI to a pre-registered analysis protocol, or state "no protocol was prepared in advance."]
- **18d — Registration.** [PLACEHOLDER: registry name/number, or state "not registered."]
- **18e — Data sharing.** See `reproducibility/README.md` "Data and code availability" — currently placeholder pending confirmation of dataset redistribution rights.
- **18f — Code sharing.** See `reproducibility/README.md` — code is in this repository; exact public URL/DOI is a placeholder pending release decision.

### 4.1 Patient and public involvement *[Item 19]*

No patient or public involvement in study design, conduct, or reporting (retrospective secondary-data analysis of a pre-existing public dataset) — this follows directly from the study design (§3.1–3.2: pre-aggregated public dataset, no prospective recruitment) and is confirmed accurate.

---

## 5. Results

*Every subsection below is currently a placeholder shape with no numbers filled in. Do not estimate, round from a smoke-test run, or otherwise approximate any value here — every number must come from `results/master_summary.json` (or `analysis/results/*.json`) once that file exists and reflects a completed, non-smoke-test, full 5-seed run. See `figures/README.md` for the exact schema and `reproducibility/README.md` for how to distinguish a real run from a smoke test via its `provenance.is_smoke_test_run` flag.*

### 5.1 Participant/image flow *[Item 20a–20c]*

Split counts are reported in §3.7 (real, from the existing `data/split_manifest.csv`). Flow: 7,200 images considered (4 class folders under `data/brain_tumor/`) → 0 excluded (no exclusion criteria applied beyond file-extension filtering, §3.2) → phash near-duplicate-cluster grouped 70/15/15 split → 4,979 train / 1,109 val / 1,112 test. **[PLACEHOLDER: render this as a Figure 1 flow diagram graphic — the counts above are final and ready to plot; no figure-generation script for this specific diagram exists yet in `figures/`, unlike the other TRIPOD+AI figures which are all wired up.]**

### 5.2 Model development *[Item 21]*

Per-seed training stopped early (patience=5 on validation loss) well before the 30-epoch budget in all but one run. Best epoch (by validation loss) and stopping epoch, from `results/20260703_155524/{vit,resnet50}/seed_<seed>/epoch_metrics.csv`:

| Model | Seed | Stopped at epoch | Best epoch | Best val-loss | Best val-acc |
|---|---|---|---|---|---|
| ViT-B/16 | 42 | 10 | 5 | 0.4054 | 0.9757 |
| ViT-B/16 | 123 | 13 | 8 | 0.4019 | 0.9739 |
| ViT-B/16 | 7 | 12 | 7 | 0.4114 | 0.9702 |
| ViT-B/16 | 2024 | 18 | 13 | 0.4038 | 0.9784 |
| ViT-B/16 | 31 | 10 | 5 | 0.4152 | 0.9693 |
| ResNet-50 | 42 | 29 | 24 | 0.3996 | 0.9748 |
| ResNet-50 | 123 | 13 | 8 | 0.4061 | 0.9720 |
| ResNet-50 | 7 | 14 | 9 | 0.4081 | 0.9748 |
| ResNet-50 | 2024 | 12 | 7 | 0.4024 | 0.9784 |
| ResNet-50 | 31 | 12 | 7 | 0.4227 | 0.9693 |

Best weights (lowest validation loss) were restored before test-set evaluation in every run, per `src/code.py`'s early-stopping implementation. ResNet-50 seed 42 is the outlier, training the full 29 epochs before triggering early stopping; all other runs stopped between epochs 10–18.

### 5.3 Model specification *[Item 22]*

Full architecture and hyperparameters are specified in §3.7 (already real, not placeholder — these are fixed pipeline settings, not experimental outcomes). Final trained weights: 10 checkpoints (2 architectures × 5 seeds), `results/20260703_155524/{vit,resnet50}/seed_<seed>/best_{vit,resnet50}_seed<seed>.pth`. **[PLACEHOLDER: public archive URL/DOI for these checkpoints — fill in once a release/hosting decision is made, see `reproducibility/README.md` "Data and code availability."]**

### 5.4 Model performance *[Item 23a–23b]*

Values below are the across-seed **mean** of the 5-seed distribution in `results/master_summary.json` (source of truth; matches `results/20260703_155524/comparison_summary.json`); accuracy/AUC 95% CI is the mean of the 5 per-seed bootstrap CI bounds (n=1,000 each). Per-seed values are in the supplementary table immediately below.

| Model | Accuracy (95% CI) | Macro-F1 | Macro-AUC (95% CI) | ECE | Brier | AURC | Acc@80% cov. | Acc@90% cov. | Acc@95% cov. | OOD-AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| ViT-B/16   | 0.9615 (0.950–0.973) | 0.9616 | 0.9938 (0.990–0.997) | 0.0699 | 0.0677 | 0.0102 | 0.9899 | 0.9866 | 0.9797 | not run |
| ResNet-50  | 0.9642 (0.953–0.975) | 0.9638 | 0.9939 (0.990–0.997) | 0.0738 | 0.0659 | 0.0105 | 0.9912 | 0.9878 | 0.9801 | not run |

OOD-AUROC is reported as "not run" rather than a number: no out-of-distribution image set was available for this project (§2.2, `code.py --ood-dir` not exercised).

**Supplementary — per-seed values** (seeds 42, 123, 7, 2024, 31):

| Model | Seed | Accuracy | Macro-F1 | Macro-AUC | ECE | Brier | AURC |
|---|---|---|---|---|---|---|---|
| ViT-B/16 | 42 | 0.9640 | 0.9639 | 0.9948 | 0.0812 | 0.0689 | 0.0079 |
| ViT-B/16 | 123 | 0.9640 | 0.9639 | 0.9953 | 0.0721 | 0.0663 | 0.0058 |
| ViT-B/16 | 7 | 0.9613 | 0.9610 | 0.9933 | 0.0673 | 0.0640 | 0.0093 |
| ViT-B/16 | 2024 | 0.9622 | 0.9621 | 0.9919 | 0.0647 | 0.0661 | 0.0174 |
| ViT-B/16 | 31 | 0.9568 | 0.9580 | 0.9936 | 0.0643 | 0.0733 | 0.0108 |
| ResNet-50 | 42 | 0.9649 | 0.9648 | 0.9909 | 0.0680 | 0.0652 | 0.0204 |
| ResNet-50 | 123 | 0.9604 | 0.9600 | 0.9947 | 0.0757 | 0.0697 | 0.0089 |
| ResNet-50 | 7 | 0.9649 | 0.9647 | 0.9971 | 0.0731 | 0.0594 | 0.0027 |
| ResNet-50 | 2024 | 0.9667 | 0.9665 | 0.9954 | 0.0750 | 0.0630 | 0.0072 |
| ResNet-50 | 31 | 0.9640 | 0.9628 | 0.9916 | 0.0774 | 0.0721 | 0.0134 |

**Model comparison (McNemar's test, per seed):** No significant difference between architectures in any seed (0/5, p<0.05). Direction of effect favored ResNet-50 in 4/5 seeds (42, 7, 2024, 31) and ViT-B/16 in 1/5 (123) — not a consistent direction, and not statistically significant regardless: p-values ranged 0.211–1.000, chi2 0.00–1.56 (continuity-corrected). Per `analysis/results/power_analysis.md`, this comparison is underpowered at the current test-set size (achieved power 0.000–0.285 across seeds) — the absence of significance should be read as "not detected at this sample size," not as strong evidence of equivalence.

| Seed | ViT acc | ResNet-50 acc | chi2 | p-value | Direction |
|---|---|---|---|---|---|
| 42 | 0.9640 | 0.9649 | 0.0000 | 1.0000 | ResNet-50 better |
| 123 | 0.9640 | 0.9586 | 0.6250 | 0.4292 | ViT better |
| 7 | 0.9613 | 0.9658 | 0.5517 | 0.4576 | ResNet-50 better |
| 2024 | 0.9622 | 0.9667 | 0.5517 | 0.4576 | ResNet-50 better |
| 31 | 0.9568 | 0.9649 | 1.5610 | 0.2115 | ResNet-50 better |

**Figures (all generated from the real completed run — see `figures/README.md`):**
- Calibration/reliability diagrams — `figures/output/calibration_reliability.png`, `calibration_temperature_scaling_bonus.png`
- Risk-coverage curves — `figures/output/risk_coverage.png`
- Confusion matrices — `figures/output/confusion_matrix.png`
- Per-class ROC curves — `figures/output/roc_curves.png`
- Grad-CAM panel (ResNet-50) — `figures/output/gradcam_panel.png`
- McNemar comparison table — `figures/output/mcnemar_table.png` / `.md`

### 5.5 Model updating *[Item 24]*

Not applicable — no post-deployment updating cycle in this development study.

---

## 6. Discussion

### 6.1 Interpretation *[Item 25]*

Both architectures discriminate well on this internal test split (macro-AUC ≈0.994 for both) and achieve comparable accuracy (~96.2–96.4% mean across 5 seeds), with no statistically significant or directionally consistent difference between them (§5.4 McNemar summary: 0/5 seeds significant, direction split 4/5 ResNet-50 vs. 1/5 ViT). Calibration is moderate rather than excellent for either model under MC-Dropout mean-probability averaging (ECE 0.070–0.074, Brier 0.066–0.068). As a complementary check, post-hoc temperature scaling (Guo et al., 2017) on the deterministic (dropout-off, single-pass) softmax substantially improves calibration for both architectures: fitted temperature T averaged 0.62 for both models (ViT range 0.57–0.66; ResNet-50 range 0.58–0.65 across seeds), reducing deterministic test ECE from a mean of 0.066→0.020 (ViT) and 0.072→0.016 (ResNet-50), with a smaller Brier improvement (ViT 0.067→0.065; ResNet-50 0.065→0.062) — full per-seed values in `analysis/results/calibration/calibration_comparison.md`. As noted in §3.7 12e, this deterministic-softmax calibration is complementary to, not a replacement for, the MC-Dropout mean-probability ECE/Brier reported in §5.4 above — the two reflect different (single-pass vs. T=20-averaged stochastic) predictive distributions and are not directly interchangeable numbers, but both point the same direction: the raw softmax is overconfident, and either MC-Dropout averaging or temperature scaling meaningfully reduces that overconfidence. Selective prediction is effective for both models: deferring the most-uncertain 5% of cases by entropy raises accuracy to ≈0.980 (§6.3, 27a).

**Fairness caveat, stated explicitly rather than left silent:** no subgroup or disparity analysis was possible for any result above (§3.9) — the source dataset carries no demographic or site metadata. Every accuracy, calibration, and selective-prediction number in §5.4 should be read as an aggregate over an unknown and unauditable mix of patient subpopulations, not as evidence of uniform performance across any subgroup.

**[PLACEHOLDER: add a comparison against prior published brain-tumor-MRI classification work, with citations — this requires a literature search/author judgment call on which prior work is most relevant and cannot be fabricated here. Note for that comparison: the retired deterministic auxiliary-head predecessor to this project reported 98.9% accuracy under a leakage-affected split (§2.1) and must not be cited as this project's own prior result or compared against as if it were a different published study.]**

The choice of MC Dropout as the uncertainty-quantification method (§2.1) should **not** be re-litigated here as a limitation "discovered" during this study — it was a designed methodological choice made in advance of this training run, for the reasons given in §2.1 (avoiding the confidence-collapse failure mode observed in a deterministic auxiliary-head approach used in an earlier iteration of this project, which was retired before any results in this manuscript were produced). If MC Dropout's own limitations are worth discussing here (compute cost of T forward passes at inference, sensitivity to dropout rate/placement), frame them as a comparison against the alternative UQ approaches considered — deep ensembles, deterministic evidential heads — not as a flaw discovered in this study's own method after the fact.

### 6.2 Limitations *[Item 26]*

- **Single-dataset origin.** All training, validation, and test data are drawn from one aggregated public source (§3.1–3.2); no independent site or scanner variation is represented, so performance estimates reflect in-distribution generalization only, not cross-institutional generalization.
- **No external institutional validation.** Confirmed, final: no second imaging source (different scanner, institution, or acquisition protocol) exists in this project — see `analysis/EXTERNAL_VALIDATION_GAP.md` for the full search record. All reported metrics are on a held-out split of a single (three-source-merged) public dataset; external validation on an independent cohort is future work, consistent with the "internal validation" framing used throughout the Title (§1.1) and Objectives (§2.2).
- **Retrospective design.** Labels and images were assembled retrospectively from a pre-existing public dataset (§3.1), not prospectively collected for this study; the usual retrospective-study caveats apply (label provenance not independently auditable, no control over acquisition protocol variation).
- **No clinical reader study.** This is a computational internal-validation study; no comparison against radiologist performance, no prospective clinical-workflow evaluation, and no evidence here that MC-Dropout uncertainty estimates, however well-calibrated on this test split, would change clinician behavior or improve patient outcomes if deployed. Claims should be scoped to "candidate decision-support component with calibrated internal-validation performance," not to any clinical-effectiveness claim.
- **No fairness/subgroup analysis** (§3.9) — the source dataset carries no demographic or site metadata, so this study cannot speak to whether performance or calibration is consistent across patient subgroups. This is a scope limitation on every result in §5.4, not just a caveat at the end.

### 6.3 Usability *[Item 27a–27c]*

**27a — Handling poor/missing input.** No OOD-AUROC metric is available in this manuscript (§5.4 — not run). The risk-coverage results do support a proposed coverage-based deferral rule: at 95% coverage (deferring the highest-entropy 5% of predictions to a human reader), observed accuracy on the retained 95% is ≈0.980 (ViT) / ≈0.980 (ResNet-50) versus ≈0.962/0.964 at full coverage — i.e. deferring the most-uncertain 1 in 20 cases recovers roughly 1.6–1.8 percentage points of accuracy on the rest. This is a candidate deferral threshold, not a validated clinical decision rule — it has not been tested against genuinely out-of-distribution or poor-quality input, only against in-distribution test-set entropy ranking (§3.11).

**27b — User expertise required.** [PLACEHOLDER: state intended user (radiologist-in-the-loop vs. autonomous triage) — ties back to §2.1 (3b) placeholder.]

**27c — Future work.** (1) External validation on an independent cohort from a different scanner/institution (closing the §6.2 gap) — cheapest first step is per-sub-source stratified reporting within the existing merged dataset, per `analysis/EXTERNAL_VALIDATION_GAP.md`. (2) A prospective reader study comparing clinician performance/behavior with vs. without model assistance, to test whether the calibrated uncertainty estimates reported here actually change clinical decisions. (3) Fairness/subgroup evaluation, contingent on acquiring a dataset with demographic or site metadata, since none exists here (§3.9).

---

## 7. Other information

Covered in §4 (Open science) per TRIPOD+AI's consolidation of funding/COI/protocol/registration/data-and-code-sharing into a single "Other information" checklist block; see `manuscript/supplementary_TRIPOD-AI_checklist.md` for the item-by-item cross-reference.
