# Brain MRI Triage

A second pair of eyes on a brain MRI scan, on a laptop, with no internet.

You load one scan. In a few seconds you get one of four answers, how sure the
tool is, and a picture showing where it looked.

**This is not a diagnosis.** A doctor or specialist makes the call. The tool
helps you decide who to send on first.

---

## Read this first

The tool knows three kinds of tumour: glioma, meningioma, and pituitary. Plus
"no tumour".

It does **not** know about cancers that spread to the brain from elsewhere
(metastases), and it does not know rarer tumour types.

So a "NO TUMOR" answer means *none of the three types it was taught*. It does
not mean the brain is clear. If the patient has symptoms, refer them anyway.
The tool does not overrule what you can see in front of you.

---

## What the four answers mean

| What you see | What to do |
|---|---|
| **TUMOR — refer urgently** | Refer this patient urgently for a specialist read. |
| **NO TUMOR** | None of the three types this tool knows. If the patient has symptoms, refer anyway. |
| **UNCERTAIN — needs human read** | The tool is not confident enough. Send the scan to a human reader. |
| **CANNOT READ THIS IMAGE — not a supported brain MRI** | The tool cannot judge this file. Check it is a brain MRI slice saved as JPEG or PNG. |

Under each answer you get **High**, **Moderate**, or **Low confidence** in
plain words.

If you also see a percentage, treat it as rough. The tool only shows a number
when its accuracy has been measured, and it deliberately rounds. A tool that
says "94.7%" when it is typically wrong by 7 points is inventing precision it
does not have.

---

## Installing it

You need a laptop with **8 GB of memory or more**. You do not need a graphics
card. You do not need a fast machine.

### Windows

1. Install **Python 3.10 or newer** from [python.org](https://www.python.org/downloads/).
   During the install, tick the box that says **"Add Python to PATH"**. This
   matters; without it nothing else works.
2. Copy this whole folder onto the laptop.
3. Open the `app` folder and double-click **`install_and_run.bat`**.

The first time, it spends a few minutes downloading what it needs. **This is
the only time the tool ever uses the internet.** After that it works with the
network switched off, forever.

### macOS or Linux

1. Make sure Python 3.10 or newer is installed. Check with `python3 --version`.
2. Open a terminal in this folder and run:

   ```
   bash app/install_and_run.sh
   ```

### A version with no Python at all

If you want something a clinic can run without installing Python, build a
standalone version on a machine that does have Python:

```
app\packaging\build_windows.bat
```

That produces `dist\BrainMRITriage\`. Copy that folder to the clinic laptop.
It runs by double-clicking `BrainMRITriage.exe`.

---

## Two files you must add before it will work

The app will **refuse to start** without both of these. That refusal is on
purpose, and it is explained further down.

**1. The model file.** This is the trained model, about 220 MB. It is not
included in this folder because it is too large for the code repository. Copy
it from:

```
results\20260703_155524\resnet50\seed_42\best_resnet50_seed42.pth
```

**2. The settings file**, named `deployment_config.json`. This holds the
safety thresholds: how sure the tool has to be before it says "tumour", and
how unsure it has to be before it sends a scan to a human.

Put it at:

```
analysis\results\safety\deployment_config.json
```

Or put it next to the program and the app will find it.

If you only have `deployment_config.SCHEMA.json`, that is a **placeholder with
made-up numbers**, and the app will not run on it.

---

## Using it

1. Start the app. A browser window opens by itself.
2. Click the big box, or drag a scan onto it.
3. Wait a few seconds.
4. Read the answer.

**Save this result** writes a single file you can print or put in the
patient's record. It contains the answer, both pictures, and the disclaimer.

By default the saved file does **not** include the original file name,
because file names often contain patient names. There is a tick box if you
want it included, for example when the result goes into that patient's own
file.

---

## The picture showing where it looked

Next to the original scan you get the same scan with colour over it. The
colour shows which parts of the image pushed the tool towards its answer.

Use it as a sanity check. If the colour sits over something that looks
abnormal, that is reassuring. **If the colour is nowhere near anything
abnormal, distrust the answer**, even a confident one. That is the single most
useful thing this picture does.

It is not proof. A tool can look at the right place and still be wrong.

The slider changes how strong the colour is. The button hides it entirely.

---

## What it cannot do

Stated plainly, because a vague limit is worse than a clear one.

**DICOM files are not supported.** Most scanners produce DICOM. This tool does
not read it.

The reason is not laziness. DICOM stores raw numbers from the scanner, not a
picture. Turning those into an image needs a window level and width, plus a
rescale slope and intercept. Get them wrong and the scan looks completely
different, with no error and no warning, and the tool would confidently give
you a wrong answer. The model was trained on images that had already been
converted with a radiographer's settings.

**What to do instead:** open the study in your normal viewer, export the slice
you want as JPEG or PNG using the window your radiographer normally uses, then
load that file.

**One slice at a time, not a whole study.** A real MRI is many slices. This
tool judges one image. It will not combine slices into a single answer,
because doing that properly needs a separate safety threshold that has not
been measured yet. Guessing one would make the tool raise false alarms more
often, and nobody would notice.

**T1 scans of the brain.** That is what it has been tested on. Other sequences
and other body parts are outside what it knows.

**Never used on its own.** A human stays in the loop.

---

## Why it sometimes refuses to start

If you see **REFUSING TO START**, nothing is wrong with the laptop.

The app checks four things before it will give a clinical answer:

1. The settings file holds real measured thresholds, not placeholders.
2. The out-of-scope check is installed, so it can reject things that are not
   brain MRI.
3. The heatmap code is installed, so the picture shows something real.
4. The way it prepares images matches the way the model was tested.

If any of those is missing, it stops and says so, rather than giving an answer
it cannot stand behind.

A made-up threshold in a tool like this is worse than no tool at all. It looks
authoritative and it is meaningless.

Show the message to whoever installed the app.

---

## Where your data goes

Nowhere. That is the point.

- Scans are held in memory while being read, then dropped. They are never
  saved to disk by this app.
- The app listens on `127.0.0.1`, which means this laptop only. Another
  computer on the same network cannot reach it, even on shared clinic wifi.
- There is no cloud, no analytics, no crash reporting, no automatic updates.
- The app makes no internet connection at any point after installation.

### The record it keeps

Every result is written to a log file on this laptop, so a clinic can answer
"what did the tool say about this scan, and which version said it". A future
regulatory review will want the same.

The log records: the time, a fingerprint of the image, the model and settings
version, the answer, and how long it took.

The log **never** records the file name, and never records the image itself.
File names are scrambled with a key unique to this laptop, so repeat scans of
the same file line up in the log, but the names cannot be recovered from it.

Find the log at:

- Windows: `%LOCALAPPDATA%\BrainMRITriage\audit\`
- macOS: `~/Library/Application Support/BrainMRITriage/audit/`
- Linux: `~/.local/share/BrainMRITriage/audit/`

Back this folder up the way you back up any other clinical record.

---

## Checking it really is offline

Worth doing once when you install it, in front of whoever needs convincing.

1. Turn off wifi and unplug any network cable.
2. Start the app.
3. Load a scan.

It behaves exactly the same. Nothing times out, nothing hangs.

For a stricter check, the test suite blocks every way the program could open a
network connection and then runs the whole pipeline, so it proves nothing is
even *attempted*:

```
python -m pytest app/tests/test_offline.py -v
```

---

## For whoever maintains this

```
python -m pytest app/tests -q          # the whole suite
python -m app --no-browser             # run without opening a browser
python -m app --port 9000              # pick the port
```

Development mode, which permits the placeholder settings file. **Never set
this on a clinic laptop:**

```
set MRI_CLINIC_DEV_MODE=1
```

Environment variables:

| Variable | What it does |
|---|---|
| `MRI_CLINIC_CONFIG` | Path to the settings file |
| `MRI_CLINIC_DATA_DIR` | Where the log and install key live |
| `MRI_CLINIC_DEV_MODE` | `1` permits the placeholder config. Development only. |

The code is arranged so all the decisions live in `app/core/`, which has no
web dependency and can be used from a script:

```python
from app.core.engine import TriageEngine

engine = TriageEngine()
result = engine.analyze_path("scan.jpg")
print(result.call, result.confidence)
```

`app/server.py` is a thin shell over that. Measured performance and the
validation of the shipped configuration are in
[`DEPLOYED_CONFIG_VALIDATION.md`](DEPLOYED_CONFIG_VALIDATION.md).
