# Cross-session issues

**Append only.** Never edit or delete another session's entry. Prefix every
entry with your session letter and a UTC timestamp.

---

## C-1 — 2026-07-31T01:20:44Z — Session C — Entropy units: bits or nats. For session A.

**Blocking risk to the deferral rule. Please answer in `handoff/STATUS_A.md`.**

There is a conflict between two frozen documents.

- `src/code.py`, in both `predict_with_uncertainty` methods, computes
  predictive entropy with `torch.log2`. Those numbers are in **bits**. Maximum
  for 4 classes is 2.0.
- `CONTRACTS.md` Contract 1 describes the `entropy` column as **nats**.
  Maximum for 4 classes is ln(4) = 1.3863.

They differ by a factor of about 1.443.

Contract 2's `entropy_defer_threshold` does not say which unit it is in. If A
writes a threshold in bits and the app reads it as nats, the app defers far
fewer scans than A intended. Those under-deferred scans come out as confident
`NO TUMOR` calls on images a human was supposed to look at. That is the exact
failure mode this project cares about most.

**What C has done meanwhile.** `app/core/config.py` resolves the unit in this
order:

1. An optional `entropy_units` key in the config, if A adds one. This is
   additive, so it does not change the frozen schema and needs no contract
   change.
2. Inference. A threshold above 1.3863 cannot be nats, so it must be bits.
3. Fallback to bits, logged loudly as `ASSUMED`, and surfaced in the app's
   startup warnings. Bits are the larger number for the same scan, so this is
   the assumption that defers **more** scans to a human. When nobody has said
   which was meant, over-deferring is the error worth making.

**What C is asking for.** Either add `"entropy_units": "bits"` (or `"nats"`)
to `deployment_config.json`, or state it in `STATUS_A.md`. Case 3 above should
never be what ships.

---

## C-2 — 2026-07-31T01:20:44Z — Session C — Study-level threshold for a folder of slices. For session A.

**Not blocking. C has shipped the honest behaviour and will not invent a number.**

A real MRI study is many slices. The model judges one. `C_application.md`
notes that taking the maximum tumour probability across slices is a reasonable
default but moves the operating point, and says to raise it to A rather than
derive it here.

Why it moves: with N slices you take N draws and keep the largest. The more
slices, the more chances one crosses the threshold, so a per-slice threshold
applied to a study raises the false-alarm rate by an amount that depends on N.
Nothing errors. The tool just starts crying wolf and the clinic stops trusting
it.

**What C has shipped.** `TriageEngine.analyze_folder()` runs every slice and
returns per-slice results. It returns `study_level_call = None` and tells the
operator, in words, that combining slices needs a threshold nobody has
measured.

**What C is asking for.** If A wants a study-level call, fit a separate
threshold on the internal validation split, grouped by study, and add
`"series_tumor_threshold"` to `deployment_config.json`. Additive key, no
contract change. The app already reads it and switches the behaviour on when
it is present. If A decides a study-level call is not supportable, say so and
C will keep the per-slice-only behaviour permanently.

---

## C-3 — 2026-07-31T01:20:44Z — Session C — Request: calibration error in the config. For session A.

**Not blocking. Affects what the UI is allowed to display.**

`C_application.md` says never to show more precision than the calibration
justifies, and that displaying "94.7%" from a model with an ECE around 0.07 is
lying with decimal places.

Contract 2's `expected_performance` block has no field for calibration error.

**What C has shipped.** The app reads an optional
`expected_performance.ece`. Its display policy:

| measured ECE | what the screen shows |
|---|---|
| absent | words only, no percentage at all |
| <= 0.02 | percentage to the nearest 1 point |
| <= 0.05 | nearest 5 points |
| <= 0.15 | nearest 10 points, plus a stated range |
| > 0.15 | words only, no percentage |

While the field is absent, the app shows "High / Moderate / Low confidence"
and no number. That is the honest default when nobody has measured how well
the probabilities track reality.

**What C is asking for.** Add `"ece"` to `expected_performance`, measured on
BRISC. Additive key, no contract change.

---

## C-4 — 2026-07-31T01:20:44Z — Session C — Checkpoint SHA-256 values, please. For session A.

Contract 2's `checkpoints[].sha256` matters more than it looks. The app
verifies it before loading and refuses to run on a mismatch, because every
safety number in the config was measured on one exact file. An all-zeros hash
is treated as "unknown" and skips the check, which is the stub's behaviour and
should not be what ships.

Please put the real hashes in. `app/core/hashing.py:sha256_file` computes them
the same way if it helps.

---

## C-5 — 2026-07-31T02:05Z — Session C — Where you put the deferral threshold matters far more than T. For session A.

**Read this before fixing `entropy_defer_threshold`. It changes what a good
threshold looks like.**

Measured on 400 internal validation images, ResNet-50 seed 42. BRISC not
touched. Nothing fitted.

### The measurement

For each image, 40 Monte Carlo passes were drawn once and cut into **disjoint**
blocks of size T. Two blocks of the same size are two honest independent runs
of the tool on the same scan. The question asked was: do they agree on whether
to defer.

Disjoint matters. Comparing a T=10 estimate against a T=20 estimate that
contains those same ten passes makes them look far more alike than two real
runs would, and understates the instability.

### Result 1: the entropy distribution is extremely tight

| percentile | 5 | 25 | 50 | 75 | 95 |
|---|---|---|---|---|---|
| entropy, bits | 0.448 | 0.485 | 0.511 | 0.547 | 0.714 |

Half of all internal val images sit between 0.485 and 0.547 bits. That is an
interquartile range of 0.062 bits.

### Result 2: run-to-run noise is comparable to that range

| T | mean absolute change in entropy between two independent runs |
|---|---|
| 5 | 0.058 bits |
| 10 | 0.041 bits |
| 20 | 0.029 bits |

Noise scales as 1/sqrt(T), exactly as it should, which is a good sign the
measurement is sound. But at T=20 the noise, 0.029 bits, is about half the
entire interquartile range of the distribution.

### Result 3: so threshold placement dominates everything

How often two independent runs of the same tool disagree about deferring the
same scan:

| threshold, bits | T=5 | T=10 | T=20 |
|---|---|---|---|
| 0.3 | 0.1% | 0.0% | 0.0% |
| 0.4 | 4.5% | 1.3% | 0.5% |
| **0.5** | **36.6%** | **31.9%** | **26.8%** |
| 0.6 | 11.7% | 5.8% | 3.3% |
| 0.7 | 2.4% | 1.5% | 0.7% |
| 0.8 and above | under 1% | under 1% | under 1% |

**A threshold near 0.5 bits gives an unstable tool.** Roughly one scan in four
would get a different answer if you ran it twice, at any T. Not because
anything is broken, but because the threshold would sit exactly on the peak of
a very narrow distribution.

A clinic worker who reruns a scan and gets "UNCERTAIN" then "NO TUMOR" stops
trusting the tool, and they would be right to.

### What C recommends

1. **Do not choose the threshold on accuracy alone.** Check where it lands
   relative to the entropy distribution. Anything in roughly 0.47 to 0.58 bits
   is a coin flip zone on internal val.
2. **Prefer a threshold at or above about 0.7 bits**, or below about 0.4, where
   two runs agree more than 99% of the time. If the accuracy-optimal threshold
   falls in the unstable band, that trade is worth making explicit rather than
   taking silently.
3. **T is not the lever.** Going from T=5 to T=20 improves agreement at 0.5
   bits only from 36.6% to 26.8%. Threshold placement moves it from 26.8% to
   under 1%. Spending compute on T to fix this would not work.
4. If no stable threshold gives an acceptable miss rate, that is a real finding
   and belongs in the report, not in a config file.

Raw numbers: `app/benchmarks/cpu_benchmark.json`. Reproduce with
`python -m app.tools.benchmark_cpu`.

---

## C-6 — 2026-07-31T02:05Z — Session C — The 5-seed ensemble is affordable. Choose on accuracy, not speed. For session A.

`C_application.md` assumed MC Dropout at T=20 across 5 seeds is 100 forward
passes and probably far too slow on a laptop CPU. Measured, it is not.

Both backbones put their only dropout layers in the classification head, so
everything before the head is deterministic at eval time. Computing the trunk
once and running only the head T times is an exact algebraic identity, not an
approximation. Verified against the naive implementation at **max absolute
difference 0.0**, and separately verified against
`src.code.BrainTumorResNet50.predict_with_uncertainty` on 200 BRISC images at
**max absolute difference 0.0** with zero predicted-label mismatches.

Measured on Windows 11, CPU only, torch 2.12.1+cpu, 8 threads, median per image:

| configuration | median |
|---|---|
| ResNet-50, 1 seed, T=20, naive full passes | 2625 ms |
| ResNet-50, 1 seed, T=20, trunk cached | **133 ms** |
| ResNet-50, 1 seed, T=5 | 174 ms |
| ViT-B/16, 1 seed, T=20 | 217 ms |
| ResNet-50, **5 seeds**, T=20 | **805 ms** |

Two consequences for A:

- **Reducing T saves nothing.** The trunk dominates; the head passes are close
  to free. T=5 is not faster than T=20 in any useful sense, and T=5 is
  measurably worse for deferral stability. Keep T=20.
- **The ensemble costs 805 ms, not 30 seconds.** If 5 seeds lowers the tumour
  miss rate, cost is not a reason to drop it. Please choose the backbone and
  ensemble size on the miss rate. Session C will ship whatever A picks; none
  of these options is near the latency budget.
