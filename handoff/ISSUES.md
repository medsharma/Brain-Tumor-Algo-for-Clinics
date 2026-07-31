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
