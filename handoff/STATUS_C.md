# Session C — the offline clinic application

Last updated: 2026-07-31T01:20:44Z

## Where things stand

The application shell is built and green. It runs end to end on real BRISC
images, on CPU, with no network, against the day-one stub config.

**It is not usable clinically yet, and it enforces that itself.** It refuses to
start unless session A's real config, session B's input check, and session D's
explainer are all present. Right now all three are stubs, so the only way it
runs is with `MRI_CLINIC_DEV_MODE=1`, and every screen carries a red
DEVELOPMENT BUILD banner.

Tests: **178 passing.**

## What is published and stable

| Path | What it is |
|---|---|
| `app/core/` | Inference core. No UI dependency. Importable and testable headlessly. |
| `app/server.py` | Local web shell. Binds `127.0.0.1` only. |
| `app/static/` | UI. Fully self-contained, no CDN, no web fonts. |
| `app/tests/` | 178 tests. |
| `analysis/results/safety/deployment_config.SCHEMA.json` | The stub, per CONTRACTS.md. Absurd values on purpose. |

The stub config is the only file session C has written outside `app/`, and
CONTRACTS.md explicitly assigns it to C.

## Numbers other sessions may want

Measured on this machine: Windows 11, CPU only, torch 2.12.1+cpu, 8 threads.
ResNet-50, single seed 42, MC Dropout T=20.

- **Latency: about 130 ms per image**, warm, excluding startup.
- Model load: about 0.5 s.
- The 30 second budget in the brief is not close to being a constraint.

**How that speed was obtained, and why it costs nothing.** Both backbones put
their only dropout layers in the classification head, so everything before the
head is deterministic at eval time. The app computes the trunk once and runs
only the head T times. `app/core/model.py:verify_mc_equivalence` checks this
against the naive implementation and gets a maximum absolute difference of
**exactly 0.0** on real scans. It is an algebraic identity, not an
approximation, and it is verified at startup rather than assumed.

Consequence for session A: the ensemble may be cheaper than expected. Five
seeds at T=20 is five trunk passes plus 100 head passes, not 100 full passes.
If the ensemble buys a lower miss rate, cost is unlikely to be the reason to
drop it. A should choose on accuracy, not on speed.

## What C needs

See `handoff/ISSUES.md` entries C-1 through C-4. The one that matters:

- **C-1, entropy units.** `src/code.py` computes entropy in bits (`log2`).
  Contract 1 says nats. They differ by 1.443x. If A's
  `entropy_defer_threshold` is in bits and the app reads nats, the app defers
  far less than intended and the difference comes out as confident `NO TUMOR`
  calls. C currently assumes bits, which is the over-deferring direction, and
  says so loudly at startup. Please confirm.

## What the app does not do

Stated plainly so nobody assumes otherwise.

- **No DICOM.** Refused explicitly, with an on-screen instruction to export
  from the viewer as JPEG or PNG. The model was trained on already-windowed
  JPEGs; guessing a window level, width, rescale slope and intercept changes
  the image drastically and silently.
- **No study-level call.** Per-slice only. See C-2.
- **No percentages** until A reports a calibration error. Words only.

## Next

- Package with PyInstaller.
- Re-validate the deployed configuration against A's safety metrics and write
  `app/DEPLOYED_CONFIG_VALIDATION.md`.
- Swap the three stubs for the real modules when A, B and D publish, then
  re-run everything.
