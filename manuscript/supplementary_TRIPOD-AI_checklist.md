# Supplementary Material: TRIPOD+AI Reporting Checklist

Completed against the TRIPOD+AI statement (Collins GS, Moons KGM, Dhiman P, et al.
*TRIPOD+AI statement: updated guidance for reporting clinical prediction models that
use regression or machine learning methods.* BMJ 2024;385:e078378). This study is a
**model development and internal-validation study (D)** — items marked "E" (evaluation
of an already-developed model on new data) are not applicable unless Session B's
external-validation work changes the study design (see manuscript §6.2).

Status legend: **Done** = fully addressed with real content; **Structural placeholder**
= section exists with the correct shape but is waiting on a real number or a fact to be
confirmed; **N/A** = not applicable to this study design.

| # | Section / Topic | Applies to | Manuscript location | Status |
|---|---|---|---|---|
| 1 | Title | D;E | §1.1 | Structural placeholder — revise "internal validation" pending §6.2 external-validation confirmation |
| 2 | Abstract | D;E | §1.2 | Structural placeholder — shape fixed, numbers pending `results/master_summary.json` |
| 3a | Background — rationale | D;E | §2.1 | Done — includes MC-Dropout method rationale as related work/motivation |
| 3b | Background — intended users/pathway | D;E | §2.1 | Structural placeholder |
| 3c | Background — health disparities | D;E | §2.1 | Structural placeholder |
| 4 | Objectives | D;E | §2.2 | Done (objective 4 has a sub-placeholder re: whether OOD eval was run) |
| 5a | Data sources | D;E | §3.1 | Mostly done — dataset identity confirmed via filename-pattern audit (`results/leakage_audit.md`); exact version/license/DOI citation still pending |
| 5b | Data — enrollment/follow-up timeframe | D;E | §3.1 | Structural placeholder — likely "not available" for secondary public data |
| 6a | Participants — setting/centers/location | D;E | §3.2 | Structural placeholder |
| 6b | Participants — inclusion/exclusion | D;E | §3.2 | Done |
| 6c | Participants — interventions | D;E | §3.2 | N/A — stated explicitly |
| 7 | Data preparation | D;E | §3.4 (9b covers preprocessing; no separate cleaning step beyond folder-based labeling) | Done |
| 8a | Outcome — definition/timing | D;E | §3.3 | Done |
| 8b | Outcome — assessor blinding/qualifications | D;E | §3.3 | Structural placeholder |
| 8c | Outcome — masking | D;E | §3.3 | Structural placeholder |
| 9a | Predictors — selection | D | §3.4 | Done — N/A rationale stated (single image predictor) |
| 9b | Predictors — definition/timing/blinding | D;E | §3.4 | Done |
| 9c | Predictors — assessor quals | D;E | §3.4 | N/A — stated explicitly |
| 10 | Sample size | D;E | §3.5 | Structural placeholder — needs achieved-precision statement once results exist |
| 11 | Missing data | D;E | §3.6 | Done |
| 12a | Data partitioning | D | §3.7 (12a) | Done — split-safety branch (phash-cluster grouping) confirmed active via `results/leakage_audit.md`, not a placeholder |
| 12b | Variable handling | D | §3.7 (12b) | N/A — stated explicitly |
| 12c | Model type/building/tuning/internal validation | D | §3.7 (12c) | Done |
| 12d | Performance by group | D;E | §3.7 (12d) | N/A — stated explicitly (§3.9 cross-reference) |
| 12e | Performance measures | D;E | §3.7 (12e) | Done |
| 12f | Performance adjustment | E | §3.7 (12f) | N/A for D-only study |
| 12g | New-prediction calculation | E | §3.7 (12g) | N/A for D-only study |
| 13 | Class imbalance | D;E | §3.8 | Done |
| 14 | Fairness | D;E | §3.9 | Done — explicit "not possible" statement, carried into Limitations |
| 15 | Model output | D | §3.10 | Done |
| 16 | Training vs. evaluation data differences | D;E | §3.11 | Done |
| 17 | Ethical approval | D;E | §3.12 | Structural placeholder — must not be left blank at submission |
| 18a | Funding | D;E | §4 | Structural placeholder |
| 18b | Conflicts of interest | D;E | §4 | Structural placeholder |
| 18c | Protocol | D;E | §4 | Structural placeholder |
| 18d | Registration | D;E | §4 | Structural placeholder |
| 18e | Data sharing | D;E | §4; `reproducibility/README.md` | Structural placeholder |
| 18f | Code sharing | D;E | §4; `reproducibility/README.md` | Structural placeholder |
| 19 | Patient and public involvement | D;E | §4.1 | Structural placeholder (states "none," pending confirmation) |
| 20a | Participant/image flow | D;E | §5.1 | Structural placeholder — flow diagram pending |
| 20b | Descriptive statistics | D;E | §5.1 | Split counts done (real, from `data/split_manifest.csv`); demographic descriptives N/A (§3.9) |
| 20c | Train/test characteristic comparison | E | §5.1 | N/A for D-only study |
| 21 | Model development — participant/event counts per stage | D;E | §5.2 | Structural placeholder |
| 22 | Model specification | D | §5.3 | Architecture/hyperparameters done (real); checkpoint reference pending |
| 23a | Model performance | D;E | §5.4 | Structural placeholder — table shape fixed, values pending `results/master_summary.json` |
| 23b | Performance by group | D;E | §5.4 | N/A — no subgroup data (§3.9) |
| 24 | Model updating | E | §5.5 | N/A for D-only study |
| 25 | Interpretation | D;E | §6.1 | Structural placeholder |
| 26 | Limitations | D;E | §6.2 | Done — single-dataset origin, no external validation (pending HANDOFF.md check), retrospective design, no reader study, no fairness analysis |
| 27a | Usability — poor/missing input | D | §6.3 | Structural placeholder |
| 27b | Usability — user expertise | D | §6.3 | Structural placeholder |
| 27c | Usability — future research | D;E | §6.3 | Structural placeholder |

## Notes on items intentionally kept out of Results/Discussion

Per the project's pivot away from (a) a deterministic auxiliary "uncertainty head" that
exhibited confidence collapse, and (b) a naive per-file data split later found to permit
near-duplicate leakage (`results/leakage_audit.md`, ~33.6% of images in near-duplicate
clusters): both prior issues and their failure modes are reported **once each**, in §2.1
(Background, item 3a) as the stated rationale for this study's actual methodology (MC
Dropout; phash-cluster-grouped splitting). No item in the Results (§5) or Discussion
§6.1/§6.2 restates either as a limitation "discovered" during this study's own
evaluation — the method and split actually used throughout this manuscript are MC
Dropout and the leakage-safe split respectively, and no numbers from either prior,
retired approach are reported anywhere. This is a deliberate editorial choice, not an
oversight, and should be preserved through future revisions of this manuscript.
