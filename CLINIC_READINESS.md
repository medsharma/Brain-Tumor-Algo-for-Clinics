# Can this go into a clinic yet?

**No. Not as a tool anybody acts on.**

**Yes, as a shadow pilot**, where a human reads every scan anyway and the tool's
answer changes nothing. That is worth doing and it is the next real step.

Last updated: 2026-08-01. Companion to [LIMITATIONS.md](LIMITATIONS.md) and
[MODEL_CARD.md](MODEL_CARD.md).

---

## The short version

The software is ready. The evidence is not.

You can install this on a clinic laptop today and it will run, offline, reliably,
and it will read a scan in under a second. None of that is the hard part.

The hard part is that **nobody has ever checked whether it works on scans from a
hospital it did not learn from, and no doctor has ever looked at one of its
answers.** Until both of those are true, a clinic acting on this tool is acting
on a number nobody has earned.

---

## What is ready

| | state | evidence |
|---|---|---|
| Runs offline on a clinic laptop | ready | `app/tests/test_offline.py`, no network path in the app |
| One-click install, no admin rights | ready | `BrainMRITriageSetup.exe`, installs per-user |
| Reinstall over a running copy | ready as of 2026-08-01 | `app/tests/test_installer_upgrade.py` |
| Uninstall from Add or remove programs | ready as of 2026-08-01 | per-user entry; **leaves the audit log in place on purpose** |
| Reads DICOM from the scanner | ready as of 2026-08-01, unmeasured route | `app/core/dicom_loading.py`, see point 6 |
| A whole study's slices in one go | ready as of 2026-08-01 | judged one by one, never merged into a study-level call |
| Result sheet can be filed against a patient | ready as of 2026-08-01 | operator-typed case reference, export only, never stored |
| Refuses to start when misconfigured | ready | `app/core/readiness.py`, blocks on stub config, wrong preprocessing, missing weights |
| Says "development build" on every screen | ready | stays up until somebody records external validation |
| Rejects out-of-scope images | ready | session B's check, fails closed |
| Shows where it looked | ready | session D's heatmap, with its own caveat |
| Audit line per scan, no image kept | ready | `app/core/audit.py`, SHA-256 only |
| Result sheet carries a date | ready as of 2026-08-01 | previously it did not, which made it useless as a record |
| The shipped program matches its paperwork | **checked as of 2026-08-01** | `app/tools/verify_shipped_package.py` |
| Speed | ready | 0.6 to 0.8 s per scan on a laptop CPU |

Software checks: 323 tests pass. The built package reads real brain MRI end to
end, and reads DICOM. Run all three before anything leaves the building:

```
python -m pytest app/tests -q
python app/tools/smoke_test_package.py
python app/tools/verify_shipped_package.py
```

---

## What it does, measured on the package itself

All 2,634 clean-subset images, read by the built Windows executable over its own
interface, on 2026-08-01. Not rescored from a cache. This is the program a clinic
would install.

| | measured |
|---|---|
| Real tumours shown as "no tumour" | **0.27%** (4 of 1,476), 95% CI 0.11 to 0.69 |
| Tumour vs no-tumour sensitivity | 99.73% |
| Specificity | 94.65% |
| Four-way accuracy | 96.16% |
| Sent to a human | 5.28% of all scans, 2.17% of tumours |
| Healthy scans shown as tumour | 3.71% |
| Refused as unreadable | 0.15% (4 of 2,634) |
| Speed | 0.7 s per scan |

Two things to notice in that table, both about meningiomas.

**All four tumours it sent home were meningiomas.** Glioma and pituitary miss
rates were zero here. So were all four scans it refused to read. The failures are
not spread evenly, they sit in one class, and 500 meningiomas is not enough to
say how bad that is.

**Deferral catches almost nothing.** The miss rate before deferring is 0.27%.
After deferring, 0.28%. The four tumours it sent home were sent home
confidently, and no confidence rule in this tool would have caught them.

Reproduce with `python app/tools/verify_shipped_package.py`, which is also the
release gate: it fails when the package and its documentation disagree.

**Every one of these numbers comes from data that shares its sources and its
preparation with the training data.** They are an upper bound. See section 1
below.

---

## What is not ready

In order of how much it matters to a patient.

### 1. It has never been tested on scans from a hospital it did not learn from

This is the whole ballgame and it is still open.

BRISC 2025 was brought in to close it. It did not. 80% of BRISC is the training
data republished under new file names, and the overlap sits almost entirely in
the tumour classes. What survived removing the duplicates is 2,634 images, and
even those come from the same sources and the same preparation pipeline as the
training data. Neither dataset ships patient identifiers, so other slices of the
same patients may still be in training.

**Every number this project has ever produced comes from one data pool.** The
clean subset is an upper bound on real-world performance, not an estimate of it.

*What clears it:* a few hundred scans from a scanner and a hospital this project
has never touched, with labels from that hospital, run once, reported whatever
it says.

### 2. No clinician has ever reviewed a single output

Not one. Nobody clinically qualified has looked at what this tool says, or at
the heatmaps it produces, or at the images it gets wrong. It is not known
whether the images it fails on are even labelled correctly.

*What clears it:* a radiologist reads a set of cases blind, then reviews the
tool's answers on the same cases, and says in writing whether it helps or gets
in the way.

### 3. It sends real tumours home, and confidence does not catch all of them

On the cleanest data available it sends roughly 2 to 3 tumours in 1,000 home.
Some of those are sent home **confidently**: ViT calls 36% of its missed tumours
"no tumour" with 90%+ confidence. Deferring the most uncertain fifth of all cases
still leaves misses on the table.

The misses are not random. Across every checkpoint ever trained, 18 distinct
images out of 821 account for all of them, and nine images account for 81%. One
image is missed by every model this project has produced, at 93% confidence.

Two things follow, and both matter:

- **Ensembling does not fix it.** The errors are correlated across seeds and
  across architectures, so averaging does not cancel them.
- **Every confidence interval in this project is narrower than the evidence
  justifies.** The effective sample size behind the miss rate is about a dozen
  hard cases, not 821 independent ones.

*What clears it:* nothing, entirely. This is the residual risk of the tool. It is
managed by the instruction printed on every result: **if the patient has
symptoms, refer them anyway.** That instruction is the actual safety control.
The model is not.

### 4. The tool's own paperwork described a different program

Found and fixed on 2026-08-01. Worth stating because of what it says about
process, not just about this one file.

`deployment_config.json` shipped performance figures left over from a previous
operating point. All 2,634 clean-subset images were then run through the built
package itself. What it actually does, against what it said it does:

| | config said | package does |
|---|---|---|
| sends to a human | 41.19% | **5.28%** |
| clears a healthy scan | 98.62% | **94.65%** |
| miss rate after deferral | 0.09% | **0.28%** |
| tumour miss rate | 0.27% | 0.27% |
| four-way accuracy | 96.20% | 96.16% |
| sensitivity | 99.12% | 99.73% |

Nothing was wrong with the model. The last two rows match, and the miss rate is
exactly as published. The numbers written on the box were describing the version
before the last threshold change. A clinic planning staffing on "4 in 10 need a
human read" would have planned for a tool that does not exist.

**The row that matters most is the third one.** Deferral no longer improves the
miss rate. It sends 5.3% of scans to a human, catches 2.2% of tumours, and the
miss rate before and after deferring is the same 0.27%. That is a defensible
trade, made deliberately in commit `e7dc1ba` on the evidence that the wider net
caught no extra tumours and cost 28 points of usability. But it means **the
deferral rule is not a safety net any more.** The control that catches a missed
tumour is the instruction to refer a symptomatic patient regardless. There is no
second one.

*What clears it:* `app/tools/verify_shipped_package.py`, which runs the built
package over the published clean subset and fails when measured and documented
behaviour disagree, plus `app/tests/test_config_matches_the_program.py`, which
fails the moment a threshold moves without the figures being remeasured. Fixed
and both in place as of 2026-08-01.

### 5. At a real clinic prevalence, most tumour flags will be false alarms

This is arithmetic, not a flaw, and it has to be planned for.

The PPV table in `analysis/results/safety/OPERATING_POINT.md` projects from
internal validation, where specificity was 100%. Measured on the clean subset
through the shipped package it is 94.65%. That changes the picture completely:

| | doc says (internal val) | measured, unseen data |
|---|---|---|
| specificity | 100% | **94.65%** |
| PPV at 2% prevalence | 100% | about 28% |
| PPV at 5% prevalence | 100% | about 50% |
| PPV at 10% prevalence | 100% | about 67% |
| referrals per 100 scans at 5% prevalence | 4.9 | about 10 |

So at a plausible rural prevalence, **about half the people this tool flags will
not have a tumour**, and the clinic will refer roughly twice as many people as
the current documentation implies. NPV stays above 99.9%, which is the reassuring
half of the same arithmetic.

That is not automatically a bad trade. Missing a tumour is worse than a wasted
journey. But the journey is not free: travel, a lost day of work, money out of
pocket, a specialist slot another patient needed, and weeks of fear. And referral
capacity has a hard ceiling. If the tool refers more people than the regional
hospital can see, the genuinely urgent cases wait longer and the tool has made
things worse.

*What clears it:* agreeing the referral budget with the receiving hospital
**before** the pilot starts, and setting the threshold to fit it.

### 6. It reads DICOM now, and that conversion is unmeasured

Fixed on 2026-08-01. Clinics produce DICOM, and the app used to refuse it and
tell somebody to open a viewer, choose a window, and export a JPEG. That did not
remove the decision, it moved it to whoever was standing at the laptop, who in a
clinic with no radiologist is the person least equipped to make it.

The app now converts DICOM itself: rescale slope and intercept applied, then the
VOI LUT or window stored in the file, which is the setting a radiographer or the
scanner already chose. Only when the file carries no window does it fall back to
a percentile stretch, and then it says so on screen in those words. MONOCHROME1
is inverted. Multi-frame files are refused rather than silently sliced.

**What is still open: nobody has measured how much the conversion moves the
answer.** Every published figure for this tool was measured on already-windowed
images exported from a viewer. A DICOM converted here is a different picture. So
every DICOM result carries a note saying exactly that, on screen and on the
printed sheet, and the audit log records which route the image came in by.

*What clears it:* a run of the same studies both ways, DICOM straight in against
viewer-exported JPEG, comparing the calls. Shadow mode is where that happens for
free.

### 7. It is not approved as a medical device anywhere

No clearance, no registration, no notified body, no submission. In most countries
software that triages patients is a regulated medical device, and the rules
depend on the country you launch in.

*What clears it:* naming the launch country, finding out its classification for
triage software, and deciding whether the pilot fits under a research or clinical
evaluation route.

### 8. Populations and sequences nobody has measured

- T1 only. Other sequences: unmeasured.
- Three planes, and specificity is worst on coronal.
- Adults. Children: unmeasured.
- Three tumour families. Metastases, lymphoma, abscess, rarer tumours: not in the
  label set at all. The tool cannot say "this is something else". It can only
  pick one of four answers.
- Post-operative brains, implants, motion artefact: unmeasured.

A metastasis shown to this tool comes back as one of four answers, and the tool
has no way to say it does not know.

### 9. There is no login, and the audit log is sensitive

Anyone at the laptop can use it. The audit log holds one line per scan with a
SHA-256 of the image. That hash is not a picture, but it is a stable identifier,
so the log can show the same scan was read twice.

*What clears it:* deciding who may use the laptop, who may read the log, how long
logs are kept, and answering the data protection question for the launch country.

---

## The only deployment that is defensible today: shadow mode

Definition: **the tool's answer does not change what happens to any patient.**

- Every scan is read by whoever reads it now, on the normal pathway.
- The tool is run on the same scan, and its answer is recorded.
- Nobody sees the tool's answer before the human decision is made. Not the nurse,
  not the clinical officer, not the patient.
- At the end, the two are compared.

What to record for each scan: the tool's call, its confidence, whether it
deferred, the human's eventual finding, and the outcome if it is known.

**Half of that is already collected.** The app writes one line per scan to
`%LOCALAPPDATA%\BrainMRITriage\audit\`, holding the call, the confidence, the
probabilities, the model and config version, and a SHA-256 of the image. No
image, and no filename unless somebody ticks the box. The missing half is the
human's finding, which has to be recorded alongside and joined on the image
hash. Treat that log as sensitive: the hash is not a picture, but it is a stable
identifier.

Target: at least 300 scans, including at least 30 with tumours, from the actual
clinic on the actual scanner. Fewer than that and the miss rate cannot be
distinguished from zero either way.

What it costs: nobody's care changes, so the clinical risk is zero. The cost is
staff time to export slices and record answers.

What it buys: the first honest external number this project has ever had, and it
comes from the exact place the tool would be deployed.

**Do not skip to advisory mode because shadow mode looked good on 20 scans.**

---

## The gate: what must be true before a clinic acts on a result

Every one of these, not most of them.

1. [ ] At least 300 scans from a hospital outside the training sources, scored
       once, reported whatever they say.
2. [ ] Tumour miss rate on those scans with a confidence interval, and the upper
       bound of that interval is one the clinic has explicitly accepted.
3. [ ] A radiologist has reviewed a sample of outputs, including every miss, and
       has written down whether the tool helps.
4. [ ] The referral rate at the clinic's real prevalence has been agreed with the
       hospital that receives the referrals, in writing.
5. [ ] The route from scanner to this tool is defined, with a fixed windowing
       rule, and somebody has checked the answer is stable across it.
6. [ ] The regulatory position in the launch country is written down, with a
       named route.
7. [ ] Who may use the laptop, who may read the audit log, and how long logs are
       kept are all decided.
8. [ ] `python app/tools/verify_shipped_package.py` passes on the exact package
       being installed.
9. [ ] Every clinic user has been told, out loud and in writing: **a "no tumour"
       result is not a clean bill of health, and a patient with symptoms gets
       referred regardless of what this says.**

Until 1 to 3 are done, the "development build" banner stays up. It is not
decoration and it is not pessimism. It is currently accurate.

---

## What to do next, in order

1. **Get outside scans.** One hospital, one scanner, a few hundred images with
   local labels. Everything else is blocked behind this, and it is cheap.
2. **Run shadow mode in one clinic.** Nothing changes for any patient.
3. **Put the results in front of a radiologist.** Including the misses.
4. **Settle the referral budget with the receiving hospital**, then set the
   threshold to fit it rather than the other way round.
5. **Decide the DICOM question.** Either support it properly or measure how much
   the window choice moves the answer.
6. **Answer the regulatory question** for one named country.

Steps 1 to 3 are the only ones that can turn this from a prototype into
something a clinic should trust. None of them is a code change.
