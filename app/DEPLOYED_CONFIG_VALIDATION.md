# Deployed configuration: validation

What the shipped app actually does, measured, on the hardware it will run on.

Last updated: 2026-07-31.

---

## The headline

**Does the shipped configuration hold the same tumour miss rate as the
research pipeline? Yes. Not close to it. Identical.**

The app's inference path and `src/code.py`'s inference path produce **bit-identical
probabilities** on the same checkpoint and the same images.

| measurement | result |
|---|---|
| Images compared, stratified across all four BRISC classes | 200 |
| Maximum absolute difference in class probabilities | **0.0** |
| Maximum absolute difference in `p_tumor` | **0.0** |
| Maximum absolute difference in predictive entropy | **0.0** |
| Predicted label mismatches | **0** |

Because the two paths produce the same numbers, the miss rate is the same at
**every** threshold, not just the one that happens to ship. There is no pair of
numbers to compare and no argument to have about how close is close enough.

Reproduce with:

```
python -m app.tools.validate_deployed_config --parity-images 200
```

Raw output: `app/benchmarks/deployed_config_validation.json`.

### Why this is the right question to answer

The obvious approach is to measure the miss rate twice, once in the research
pipeline and once in the app, and check the two are similar. That leaves you
comparing two noisy estimates on a finite sample and negotiating what
"similar" means.

Proving the computations are identical is stronger and settles it. Whatever
session A measures, the app inherits exactly.

### What is compared

- **Research path**: `src.code.BrainTumorResNet50.predict_with_uncertainty`,
  T full forward passes through the entire network.
- **Deployed path**: `app.core.model.run_mc_dropout`, trunk computed once and
  the classification head run T times.

Both seeded identically before each image, so a match means bit-identical
rather than merely close.

---

## The one thing this does not prove

**It does not prove the app is safe. It proves the app faithfully reproduces
whatever the research pipeline does, including its mistakes.**

Session A has found that BRISC 2025 is not external data. About 80% of it
(4,791 of 6,000 images) is the training set republished under new names, at
perceptual-hash distance 0, confirmed by pixel comparison. See
`handoff/ISSUES.md`.

So at the time of writing **this project has no external validation result**,
and neither does this app. Bit-identity to the research pipeline is a
statement about engineering, not about clinical safety. When session A
publishes a miss rate on the genuinely-unseen clean subset, that number is the
app's number too, by the argument above. Until then there is no number.

The app reflects this. It refuses to start on a placeholder config, and it
does not claim external validation anywhere in its interface.

---

## Speed on CPU

Windows 11, CPU only, torch 2.12.1+cpu, 8 torch threads, no CUDA. Median per
image over 28 internal validation images, after warm-up.

| configuration | forward passes | median | against a 30 s budget |
|---|---|---|---|
| ResNet-50, 1 seed, T=20, **naive** | 20 full | 2625 ms | 8.8% |
| **ResNet-50, 1 seed, T=20, deployed** | 1 trunk + 20 head | **133 ms** | **0.4%** |
| ResNet-50, 1 seed, T=10 | 1 trunk + 10 head | 116 ms | 0.4% |
| ResNet-50, 1 seed, T=5 | 1 trunk + 5 head | 174 ms | 0.6% |
| ViT-B/16, 1 seed, T=20 | 1 trunk + 20 head | 217 ms | 0.7% |
| ResNet-50, **5 seeds**, T=20 | 5 trunk + 100 head | 805 ms | 2.7% |

Add about 35 ms for the Grad-CAM heatmap, measured by session D. End to end,
including file reading and PNG encoding, a real upload over HTTP measured
**260 ms**.

Model load at startup: about 0.5 s per checkpoint.

### What was traded away to get it

**Nothing.** That is unusual enough to be worth explaining.

Both backbones put their only dropout layers in the classification head.
Everything before the head is deterministic at eval time, so the naive
implementation recomputes an identical trunk twenty times. Computing it once
and running only the head T times is an exact algebraic identity.

The app does not take that on trust. `verify_mc_equivalence` runs both paths
under the same seed and compares:

- Against the naive path on synthetic and real input: **max difference 0.0**.
- Against `src/code.py`'s own method on 200 BRISC images: **max difference 0.0**.

If that check ever fails, the engine falls back to the slow path rather than
shipping a different sampling distribution than was measured.

### What this means for the configuration choice

The brief listed four speed options in order of preference. Measured, three of
them turn out to be unnecessary:

1. **Single seed instead of an ensemble** — not needed for speed. The 5-seed
   ensemble costs 805 ms, which is 2.7% of the budget. Session A should choose
   ensemble size on the miss rate alone.
2. **Reduce T** — pointless. The trunk dominates, so T=5 is not faster than
   T=20 in any useful sense, and T=5 is measurably worse for deferral
   stability (below). **Keep T=20.**
3. **ONNX Runtime or quantisation** — not needed, and actively unwanted.
   Quantisation moves the decision boundary, which would require re-verifying
   the miss rate and would break the bit-identity result above. There is no
   reason to accept that risk for a saving on a 0.4% budget.
4. **One backbone** — ResNet-50 is the default and is faster, but ViT at
   217 ms is also comfortably within budget. Again, choose on the miss rate.

---

## Deferral stability, and why threshold placement matters more than T

The brief asks that reducing T be checked against deferral decisions, not just
accuracy. Checked. The finding is that T is not the problem.

Measured on 400 internal validation images, ResNet-50 seed 42. For each image,
40 Monte Carlo passes were drawn once and cut into **disjoint** blocks of size
T. Two blocks of the same size are two honest independent runs of the tool on
the same scan.

Disjoint matters. Comparing a T=10 estimate against a T=20 estimate that
contains those same ten passes would make them look far more alike than two
real runs are.

### The entropy distribution is very narrow

| percentile | 5 | 25 | 50 | 75 | 95 |
|---|---|---|---|---|---|
| entropy, bits | 0.448 | 0.485 | 0.511 | 0.547 | 0.714 |

Half of all images sit inside a band 0.062 bits wide.

### Run-to-run noise is comparable to that band

| T | mean absolute change in entropy between two independent runs |
|---|---|
| 5 | 0.058 bits |
| 10 | 0.041 bits |
| 20 | 0.029 bits |

Noise falls as 1/sqrt(T), exactly as it should, which is a good sign the
measurement is sound. But even at T=20 the noise is about half the entire
interquartile range.

### So the threshold's position dominates

How often two independent runs disagree about deferring the same scan:

| threshold, bits | T=5 | T=10 | T=20 |
|---|---|---|---|
| 0.3 | 0.1% | 0.0% | 0.0% |
| 0.4 | 4.5% | 1.3% | 0.5% |
| **0.5** | **36.6%** | **31.9%** | **26.8%** |
| 0.6 | 11.7% | 5.8% | 3.3% |
| 0.7 | 2.4% | 1.5% | 0.7% |
| 0.8+ | under 1% | under 1% | under 1% |

**A deferral threshold near 0.5 bits produces an unstable tool.** About one
scan in four would get a different answer on a second run. A clinic worker who
reruns a scan and sees "UNCERTAIN" then "NO TUMOR" will stop trusting the
tool, and they would be right to.

Going from T=5 to T=20 moves that from 36.6% to 26.8%. Moving the threshold
from 0.5 to 0.7 moves it from 26.8% to 0.7%. Spending compute on T cannot fix
what threshold placement causes.

Raised to session A as `handoff/ISSUES.md` entry C-5.

Reproduce with `python -m app.tools.benchmark_cpu`. Raw output:
`app/benchmarks/cpu_benchmark.json`.

---

## Preprocessing parity

The failure this project was most exposed to: if the app prepares images even
slightly differently from `get_transforms("test")`, every measured number
describes a different program than the one a clinic runs, and nothing raises
an error.

`app/tests/test_preprocessing_parity.py` asserts the two produce **identical**
tensors, at exact equality rather than a tolerance. It covers:

- Real BRISC images across all four classes.
- Non-square images, where a squashing resize and an aspect-preserving resize
  diverge. BRISC images are all square, so this drift would be invisible on
  BRISC alone.
- 16-bit and already-RGB inputs.
- EXIF orientation, which the analysis pipeline ignores and the app must too.

It also includes a meta-test that deliberately introduces the two most likely
real drifts and confirms the comparison catches them. A test asserting
equality is worthless if it could never fail.

---

## Entropy units, unresolved

`src/code.py` computes predictive entropy with `log2`, so its numbers are in
**bits**, maximum 2.0 for four classes. Contract 1 describes the cached
entropy column as **nats**, maximum 1.3863. They differ by a factor of 1.443.

Contract 2's `entropy_defer_threshold` does not state its unit. Read a bits
threshold as nats and the app defers far fewer scans than intended, and the
difference comes out as confident `NO TUMOR` calls on scans a human was meant
to see.

The app resolves this explicitly rather than assuming: an optional
`entropy_units` key wins; otherwise a threshold above 1.3863 must be bits;
otherwise it falls back to bits and says so loudly at startup. Bits is the
larger number for the same scan, so the fallback is the one that defers more.

Raised to session A as entry C-1. **This should be settled before deployment,
not left to the fallback.**

---

## Startup gate

The app will not produce a clinical call unless all of these hold. Each is
enforced in code, not documented as an intention.

| check | what happens if it fails |
|---|---|
| Config is not the placeholder stub | refuses to start |
| Session B's input check is installed | refuses to start |
| Session D's explainer is installed | refuses to start |
| Preprocessing matches the training pipeline | refuses to start, even in development mode |
| Checkpoint file present and matching its recorded SHA-256 | refuses to start |
| `thresholds_fitted_on` does not mention BRISC | refuses to start |
| Config schema version supported | refuses to start |
| Tumour miss rate in the config above 5% | starts, with a warning |
| Entropy unit had to be assumed | starts, with a warning |

`MRI_CLINIC_DEV_MODE=1` downgrades the three stub checks to warnings and puts
a red banner on every screen. The preprocessing check is never downgraded.

---

## Current state

| dependency | status |
|---|---|
| Session A, `deployment_config.json` | **not yet published.** App runs on the stub in development mode only. |
| Session B, `src/input_validation.py` | **not yet published.** Stub in place; app refuses to start clinically. |
| Session D, `src/explain_runtime.py` | **published and integrated.** 189 tests pass with the real explainer. |

When A and B publish, re-run:

```
python -m pytest app/tests -q
python -m app.tools.validate_deployed_config --parity-images 200
python -m app.tools.benchmark_cpu
```

and update this file. The parity result is expected to be unchanged, since it
depends on the model and the code paths, not on the thresholds.
