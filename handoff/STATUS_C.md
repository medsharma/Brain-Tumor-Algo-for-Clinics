# Session C — the offline clinic application

Last updated: 2026-07-31T03:10Z

## Headline

**The app is built and works. It is not safe in front of a patient, and it
enforces that itself by refusing to start.**

Session A has found that BRISC 2025 is roughly 80% the training set
republished. So the project has no external validation result, and neither
does this app. What C can prove is narrower and worth stating precisely: the
shipped app computes **bit-identical** results to the research pipeline. Every
number session A eventually produces on genuinely unseen data is the app's
number too, exactly, at every threshold.

Tests: **219 passing.**

## The brief's first question, answered

**Does the shipped configuration hold the same tumour miss rate as the
research pipeline? Yes, identically.**

| measurement | result |
|---|---|
| BRISC images compared, all four classes | 200 |
| Max absolute difference in class probabilities | **0.0** |
| Max absolute difference in `p_tumor` | **0.0** |
| Max absolute difference in predictive entropy | **0.0** |
| Predicted label mismatches | **0** |

Compared `src.code.BrainTumorResNet50.predict_with_uncertainty` against
`app.core.model.run_mc_dropout`, same checkpoint, same seeds. Proving the
computations identical settles the question, rather than measuring the miss
rate twice and negotiating what "close" means.

Reproduce: `python -m app.tools.validate_deployed_config --parity-images 200`

## Speed, and what it cost

**133 ms per image**, ResNet-50, T=20, CPU only, 8 threads. 260 ms end to end
over HTTP including file read, heatmap and PNG encoding. Against a 30 s budget.

**Nothing was traded away.** Both backbones put their only dropout in the
classification head, so the naive implementation recomputes an identical trunk
20 times. Computing it once and running only the head T times is an exact
algebraic identity, verified at max difference 0.0 against both the naive path
and `src/code.py`'s own method.

| configuration | median |
|---|---|
| ResNet-50, 1 seed, T=20, naive | 2625 ms |
| ResNet-50, 1 seed, T=20, deployed | **133 ms** |
| ViT-B/16, 1 seed, T=20 | 217 ms |
| ResNet-50, 5 seeds, T=20 | 805 ms |

Consequences for A, raised as issue C-6:

- **Keep T=20.** Reducing T saves nothing (the trunk dominates) and is
  measurably worse for deferral stability.
- **The ensemble is affordable.** 805 ms is 2.7% of budget. Choose ensemble
  size and backbone on the miss rate, not on speed.
- **Do not quantise.** It would move the decision boundary and break the
  bit-identity result, for a saving on a 0.4% budget.

## The finding A should read before fixing a threshold

Raised as issue C-5, measured on 400 internal validation images. BRISC not
touched, nothing fitted.

Internal-val entropy is very tightly distributed: interquartile range 0.485 to
0.547 bits. Run-to-run noise at T=20 is 0.029 bits, about half that range.

How often two independent runs disagree on deferring the same scan:

| threshold, bits | T=5 | T=10 | T=20 |
|---|---|---|---|
| 0.4 | 4.5% | 1.3% | 0.5% |
| **0.5** | **36.6%** | **31.9%** | **26.8%** |
| 0.7 | 2.4% | 1.5% | 0.7% |

A threshold near 0.5 gives a tool that answers differently one time in four on
a rerun. Threshold *placement* dominates T by a wide margin: T=5 to T=20 moves
it 36.6% to 26.8%, while 0.5 to 0.7 moves it 26.8% to 0.7%.

## Cross-session state

| dependency | status |
|---|---|
| A, `deployment_config.json` | **not published.** App runs on the stub in dev mode only. |
| B, `src/input_validation.py` | **not published.** Stub in place; app refuses to start clinically. |
| D, `src/explain_runtime.py` | **published and integrated.** Real Grad-CAM rendering, D's `HEATMAP_CAVEAT` shown verbatim. |

**A note on how D was integrated, for whoever merges.** Nothing is on `main`
yet, so C could not rebase onto it. D's file was pulled into C's worktree with
`git show origin/session/D:src/explain_runtime.py` and added to the worktree's
local `info/exclude`, so it is present for testing but **not committed on
`session/C`**. D still owns that file and it arrives on `main` from D's branch.
C's green test run therefore depends on D's branch being merged too. C's
explainer tests skip cleanly if the file is absent, so `session/C` alone is
still green, just with those tests skipped.

Open questions for A, in `handoff/ISSUES.md`:

- **C-1, entropy units.** `src/code.py` uses `log2` (bits); Contract 1 says
  nats. 1.443x apart. Read wrong, the app under-defers, and that comes out as
  confident `NO TUMOR` calls. The app assumes bits (the over-deferring
  direction) and says so loudly. **Please confirm before deployment.**
- **C-2**, study-level threshold. **C-3**, calibration error field.
  **C-4**, real checkpoint hashes. **C-5**, threshold placement. **C-6**, the
  ensemble is affordable.

## One observation for D

On `brisc2025_test_00001_gl_ax_t1.jpg`, Grad-CAM puts its hottest region on
the frontal mass, which is reassuring, but also puts substantial weight
outside the skull entirely, bottom-left and right edge. One image is not a
finding, and this is D's territory. Flagging it in case it is useful input to
the clinical explainability work.

## Contamination self-audit

Required by CONTRACTS.md. Run as a test, not a one-off grep, in
`app/tests/test_contamination_audit.py`:

- The shipped app contains no backward pass, no optimiser, no `.fit()`.
- Exactly one `.train()` call exists, on `nn.Dropout` submodules only, with a
  test asserting no BatchNorm layer is ever left in training mode.
- The shipped app contains no reference to BRISC and no dataset path. Checked
  by parsing identifiers and string literals, so docstrings explaining *why*
  BRISC must not be touched do not trip it.
- Every threshold arrives from the config. The app **refuses to start** if
  `thresholds_fitted_on` mentions BRISC.
- `app/tools/` reads BRISC for measurement only. The miss-rate curve ships
  with a warning that reading a threshold off it would be circular.

## What C shipped

| Path | What it is |
|---|---|
| `app/core/` | Inference core. No UI dependency, importable, testable headlessly. |
| `app/server.py` | Local web shell. Binds `127.0.0.1` only, verified by parsing the source for wildcard literals. |
| `app/static/` | UI. Self-contained, no CDN, no web fonts, CSP-enforced. |
| `app/tests/` | 219 tests. |
| `app/tools/` | CPU benchmark and deployed-path validation. |
| `app/packaging/` | PyInstaller spec, build script that refuses to build on failing tests. |
| `app/README.md` | Install and run, written for a clinic worker. |
| `app/DEPLOYED_CONFIG_VALIDATION.md` | Everything measured, and what it does not prove. |
| `analysis/results/safety/deployment_config.SCHEMA.json` | The stub, per CONTRACTS.md. |

## What the app does not do

- **No DICOM.** Refused explicitly with an instruction to export as JPEG or
  PNG. Guessing a window level, width, rescale slope and intercept changes the
  image drastically and silently.
- **No study-level call.** Per-slice only, until A fits a study threshold.
- **No percentages** until A reports a calibration error. Words only.
- **No external validation claim** anywhere in the interface.

---

# Integration update, 2026-07-30T23:10Z. Both stubs replaced. 244 tests pass.

## The app is running on real configs for the first time

| dependency | before | now |
|---|---|---|
| A, `deployment_config.json` | not published, stub | **real**, vit ensemble5, 5 seeds |
| B, `rejector_config.json` | not published, stub | **real**, `precheck_only` |
| D, `src/explain_runtime.py` | pulled in, uncommitted | **merged**, committed |

Tests: **244 passing**, up from 219. The extra ones are D's explainer tests,
which used to skip, plus a new guard described below.

**Parity still holds, on the real config.** Research path against deployed path,
200 BRISC images, ResNet-50:

| measurement | result |
|---|---|
| max abs probability difference | **0.0** |
| max abs entropy difference | **0.0** |
| predicted label mismatches | **0** |

## The app stays on CPU. This is now load-bearing.

The environment has a working GPU. `src/code.py` picks CUDA automatically, and
the research pipeline now runs there.

**The app does not, and must not.** `app/core/model.py` loads with
`map_location="cpu"` and never moves a tensor to a device. Nothing under `app/`
references CUDA except a benchmark that only reports whether it exists.

That is deliberate. MC Dropout draws random masks, and CUDA draws them from a
different RNG stream than CPU, so GPU results are not bit-identical to CPU and
cannot be. Measured during integration: 0.17% of labels flip between devices.
The parity result above is the strongest claim this project has. Moving the app
to a GPU would destroy it, to speed up a path that already runs in 133 ms
against a 30 s budget, on laptops that have no GPU anyway.

## C-1 is resolved: the entropy unit question

`src/code.py` uses `torch.log2`, so its `entropy` is in **bits**. Session A's
prediction cache column is in **nats**. Both were right and they were different
numbers.

The published config now states it outright: `entropy_units: "nats"`, and
`entropy_defer_threshold: 0.0378` is in nats. The app reads the declared unit
instead of assuming. The `entropy_units_assumed` readiness warning does not
fire.

## A bug I introduced by finishing the other sessions

**Completing A, B and D silently removed "DEVELOPMENT BUILD - NOT FOR CLINICAL
USE" from the screen.**

`app/static/app.js` hides that banner on `status.state !== "clinical"`, and
`Readiness.state` returned `"clinical"` as soon as there were no blockers and no
warnings. Every readiness check asked a *wiring* question: is the config real,
is the validator installed, is the explainer installed, do the checkpoints
exist. All of them started passing the moment A, B and D published.

Nothing was asking the separate and more important question: **has this been
shown to work on data it did not train on, and has a clinician ever looked at
it.** The answer to both is still no.

Fixed with a `no_external_validation` finding in `readiness.check`. It is a
**warning, not a blocker**, so the app still starts for development. State is
`development`, the banner stays up, and the message says why:

> This tool has never been tested on data it did not train on. BRISC 2025 was
> the intended external test set and roughly 80% of it turned out to be the
> training data republished. [...] No clinician has reviewed a single output.
> Not for clinical use.

The check is **fail-safe**. It clears only on an exact attestation
(`external_validation.status == "independent_cohort"`) that nothing in this repo
writes. Absent, malformed, wrong-typed and near-miss values all keep the banner.
You cannot clear it by forgetting to fill in a field.

`app/tests/test_no_external_validation_guard.py`, 13 tests, pins this in both
directions so it cannot regress the next time a wiring check starts passing.

## What C should still not do

- **No study-level call.** A published a per-slice threshold only. C-2 stands.
- **No percentages shown to a user.** Words only.
- **No DICOM.**
- **Heatmap: ViT only.** Session D measured Grad-CAM on ResNet-50 landing on the
  tumour 8.4% of the time against a 1.7% chance baseline. The shipped backbone
  is ViT at 41%, which is the one worth drawing. Do not add any text implying a
  sensible-looking heatmap means the call is more likely right: D measured that
  it is very slightly the reverse.

## The number a clinic operator will feel

The shipped config defers **41.2%** of scans to a human. That is the cost of the
lowest miss rate, and in a clinic with no radiologist it means four scans in ten
come straight back. A human should decide whether that trade is right. Both
configurations are measured in `analysis/results/safety/ensemble_vs_single.csv`.
