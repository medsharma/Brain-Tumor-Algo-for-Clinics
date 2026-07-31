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
