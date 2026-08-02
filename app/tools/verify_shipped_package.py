#!/usr/bin/env python
"""Run the BUILT package on the published clean subset and check it agrees.

Why this exists
---------------
`smoke_test_package.py` proves the package starts and can read a scan. That is a
plumbing test: sixteen images, and the only thing it can catch is a package that
rejects everything.

This asks the harder question. **Does the program a clinic actually runs produce
the numbers this project publishes about it?**

Nothing guarantees that it does. The figures in `deployment_config.json` come
from `analysis/`, computed by a separate script, from cached probabilities, on a
developer's checkout. The clinic runs a PyInstaller bundle with its own copy of
the config, its own preprocessing path, its own thresholds and its own ensemble
loading code. Every one of those is a place where the shipped program can drift
away from the evidence written about it, quietly, while every unit test passes.

That failure is invisible and it is the dangerous kind. A wrong threshold in the
bundle does not crash. It reads scans all day and sends a different fraction of
tumours home than the number printed on the box.

So: real images, through the real executable, over HTTP, exactly as the clinic
uses it. Then compare against what the config claims.

What it measures, in the order the mission cares about
-----------------------------------------------------
1. **Tumour miss rate.** A real tumour displayed as "no tumour". The number that
   sends a sick patient home. Everything else is secondary.
2. **Tumour vs no-tumour accuracy.** The call a clinic acts on.
3. **Deferral.** How often it says "I am not sure, ask a human", and whether the
   misses are concentrated in what it defers, which is the only reason deferral
   is a safety feature rather than an inconvenience.
4. **Four-way accuracy.** Tumour type. Useful, secondary, a clinic refers either
   way.

The sample
----------
Images are drawn from the **clean subset only**: the BRISC 2025 images whose
nearest internal *training* image is more than perceptual-hash Hamming distance
5 away, as defined in `analysis/results/brisc/brisc_overlap_per_image.csv`. The
other 56% of BRISC is the training set republished under new file names, and
scoring the model on those measures memory, not skill.

The draw is deterministic (sorted, fixed stride), so two runs of this tool on the
same package compare like with like.

**This is a consistency check on the artifact, not new evidence about the model.**
The sample is a few hundred images out of 2,634 and it cannot establish a miss
rate of a fraction of a percent. It can establish that the shipped program is not
behaving like a different program from the one that was measured. For what this
model's performance actually is, and what it is not, read `LIMITATIONS.md`.

Usage
-----
    python app/tools/verify_shipped_package.py
    python app/tools/verify_shipped_package.py --per-class 120 --json report.json

Exit code 0 means the shipped package agrees with its own paperwork.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]

OVERLAP_CSV = REPO_ROOT / "analysis" / "results" / "brisc" / "brisc_overlap_per_image.csv"
CONFIG_JSON = REPO_ROOT / "analysis" / "results" / "safety" / "deployment_config.json"

#: Where BRISC 2025 is unpacked. Overridable, because it is not in the repo and
#: never will be: it is 6,000 images and it is somebody else's dataset.
BRISC_ROOT = Path(os.environ.get(
    "MRI_TRIAGE_BRISC_ROOT",
    r"C:\Users\medha\Downloads\archive (1)\brisc2025"))

TUMOUR_CLASSES = ("glioma", "meningioma", "pituitary")
ALL_CLASSES = TUMOUR_CLASSES + ("notumor",)


# ------------------------------------------------------------------ statistics

def wilson(successes: int, n: int, z: float = 1.959963985) -> Tuple[float, float]:
    """95% Wilson interval. Correct at zero successes, where normal-approx is not.

    A miss rate of 0 out of 200 is not proof of a miss rate of 0, and an interval
    that says [0, 0] would be a lie told by arithmetic.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


# ------------------------------------------------------------------ the sample

def clean_subset(csv_path: Path) -> Dict[str, List[str]]:
    """The published clean subset, grouped by true class."""
    by_class: Dict[str, List[str]] = {name: [] for name in ALL_CLASSES}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("clean_vs_train") != "True":
                continue
            name = row.get("class_name", "")
            if name in by_class:
                by_class[name].append(row["image_path"])
    for name in by_class:
        by_class[name].sort()
    return by_class


def evenly_spaced(items: Sequence[str], count: int) -> List[str]:
    """A deterministic spread across the whole list, not the first N.

    The files are named by scanner and plane, so the first N of a sorted list is
    one plane and one acquisition. That is a different question from the one
    being asked.
    """
    if count >= len(items):
        return list(items)
    stride = len(items) / count
    return [items[int(i * stride)] for i in range(count)]


# ------------------------------------------------------------------ the server

def start_package(exe: Path, port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [str(exe), "--no-browser", "--port", str(port)],
        cwd=str(exe.parent),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def wait_for(port: int, process: subprocess.Popen, timeout: int = 300) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise SystemExit(
                f"the app exited during startup with code {process.returncode}")
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/status", timeout=5) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(3)
    raise SystemExit(f"the app did not answer on port {port} within {timeout}s")


def analyse(port: int, image: Path) -> dict:
    boundary = uuid.uuid4().hex
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{image.name}"\r\n'.encode(),
        b"Content-Type: image/jpeg\r\n\r\n",
        image.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/analyze", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


# ------------------------------------------------------------------ the report

def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _relative_to_dataset(path: Path, root: Path) -> str:
    """Dataset-relative path, falling back to the file name."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dist", type=Path, default=REPO_ROOT / "dist" / "BrainMRITriage")
    parser.add_argument("--port", type=int, default=8901)
    parser.add_argument("--per-class", type=int, default=80,
                        help="images per class, drawn evenly from the clean subset")
    parser.add_argument("--brisc-root", type=Path, default=BRISC_ROOT)
    parser.add_argument("--json", type=Path, default=None,
                        help="write the full result to this file")
    parser.add_argument("--from-json", type=Path, default=None,
                        help="re-derive the report from a previous run's records "
                             "instead of reading the images again")
    parser.add_argument("--update-config", action="store_true",
                        help="write the measured figures into deployment_config.json, "
                             "keeping the old ones in its history")
    args = parser.parse_args()

    published = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
    expected = dict(published.get("expected_performance") or {})
    expected["_tumor_threshold"] = published.get("tumor_threshold")

    # Re-deriving from a saved run. The images were read once; reading 2,634 of
    # them again to recompute a percentage helps nobody.
    if args.from_json:
        saved = json.loads(args.from_json.read_text(encoding="utf-8"))
        records = saved.get("records") or []
        if not records:
            raise SystemExit(f"{args.from_json} holds no records")
        print(f"  from        {args.from_json} ({len(records)} images, not re-read)")
        report, failures = summarise(records, expected)
        print_report(report, expected, failures)
        if args.update_config:
            failures = update_config(published, report, failures, len(records))
        sys.exit(1 if failures else 0)

    exe = args.dist / "BrainMRITriage.exe"
    if not exe.is_file():
        raise SystemExit(f"{exe} not found. Build the package first.")
    if not OVERLAP_CSV.is_file():
        raise SystemExit(f"{OVERLAP_CSV} not found. Run the BRISC overlap audit first.")
    if not args.brisc_root.is_dir():
        raise SystemExit(
            f"BRISC 2025 is not at {args.brisc_root}.\n"
            f"Set MRI_TRIAGE_BRISC_ROOT or pass --brisc-root.")

    # Sensitivity and specificity are only defined against a cutoff, and the
    # cutoff lives outside expected_performance. It was carried into `expected`
    # above so the measured numbers use the threshold this package is running.
    by_class = clean_subset(OVERLAP_CSV)
    sample: List[Tuple[str, Path]] = []
    for name in ALL_CLASSES:
        for relative in evenly_spaced(by_class[name], args.per_class):
            sample.append((name, args.brisc_root / relative.replace("/", os.sep)))

    missing = [p for _, p in sample if not p.is_file()]
    if missing:
        raise SystemExit(
            f"{len(missing)} sampled images are not on this machine, "
            f"starting with {missing[0]}")

    print(f"  package     {exe}")
    print(f"  sample      {len(sample)} images from the clean subset "
          f"(n={sum(len(v) for v in by_class.values())})")
    print(f"  published   miss rate {percent(expected.get('tumor_miss_rate', float('nan')))}, "
          f"defer rate {percent(expected.get('defer_rate', float('nan')))}")
    print()

    process = start_package(exe, args.port)
    records: List[dict] = []
    try:
        status = wait_for(args.port, process)
        print(f"  state       {status.get('state')}")

        started = time.time()
        for index, (truth, path) in enumerate(sample, 1):
            result = analyse(args.port, path)
            records.append({
                # Relative to the dataset root, not absolute. This file is
                # checked in as the evidence behind the config's published
                # figures, and a public clinical repository is no place for a
                # map of somebody's hard drive.
                "path": _relative_to_dataset(path, args.brisc_root),
                "true_class": truth,
                "call_key": result.get("call_key"),
                "tumor_type": result.get("tumor_type"),
                # Enough to reproduce every published figure from the raw calls,
                # including the ones defined differently from what the screen
                # shows. p_tumor drives the referral decision; the four-way
                # argmax is what the miss rate in the config is defined on.
                "p_tumor": result.get("p_tumor"),
                "class_probabilities": result.get("class_probabilities"),
                "entropy": result.get("entropy"),
                "mutual_information": result.get("mutual_information"),
                "latency_ms": result.get("latency_ms"),
                "validator_is_stub": bool(result.get("validator_is_stub")),
                "explainer_is_stub": bool(result.get("explainer_is_stub")),
            })
            if index % 25 == 0 or index == len(sample):
                rate = (time.time() - started) / index
                print(f"  read        {index}/{len(sample)}  "
                      f"({rate:.1f}s each, {rate * (len(sample) - index) / 60:.0f} min left)")
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()

    report, failures = summarise(records, expected)
    print_report(report, expected, failures)

    if args.update_config:
        failures = update_config(published, report, failures, len(sample))

    if args.json:
        args.json.write_text(json.dumps(
            {"expected": expected, "observed": report, "failures": failures,
             "records": records}, indent=2), encoding="utf-8")
        print(f"\n  written to {args.json}")

    sys.exit(1 if failures else 0)


#: How many genuine brain MRI the package may refuse before that is a defect
#: rather than caution. Refusing is fail-safe -- the scan goes to a human -- so
#: a small rate is acceptable. A large one means the clinic cannot use the tool.
UNREADABLE_BUDGET = 0.01

#: How far the observed deferral rate may sit from the documented one before the
#: paperwork is describing a different program. Deferral is the safety net, so a
#: clinic planning who reads the deferred scans needs this number to be true.
DEFER_TOLERANCE = 0.10


def argmax_class(record: dict) -> Optional[str]:
    """The four-way winner, which is what the config's miss rate is defined on.

    Not the same question as what the screen says. The screen applies a
    ``p_tumor`` threshold and a deferral rule on top.
    """
    probabilities = record.get("class_probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        return None
    return max(probabilities.items(), key=lambda kv: kv[1])[0]


def summarise(records: List[dict], expected: dict) -> Tuple[dict, List[str]]:
    """Reproduce every published figure from the shipped package's own calls."""
    failures: List[str] = []

    tumours = [r for r in records if r["true_class"] in TUMOUR_CLASSES]
    healthy = [r for r in records if r["true_class"] == "notumor"]

    unreadable = [r for r in records if r["call_key"] == "cannot_read"]
    deferred = [r for r in records if r["call_key"] == "uncertain"]

    # THE number, as a clinic experiences it: a real tumour that the screen
    # showed as no-tumour, so the patient went home. A deferred tumour is not
    # missed; it was sent to a human, which is the system working.
    sent_home = [r for r in tumours if r["call_key"] == "no_tumor"]
    sent_home_rate = len(sent_home) / len(tumours) if tumours else 0.0
    sent_low, sent_high = wilson(len(sent_home), len(tumours))

    # The same event under the config's own definition, so the two can be
    # compared without arguing about what "miss" means: four-way argmax landing
    # on notumor for a tumour image, deferral ignored.
    argmax_scored = [r for r in records if argmax_class(r) is not None]
    argmax_missed = [r for r in tumours if argmax_class(r) == "notumor"]
    argmax_miss_rate = len(argmax_missed) / len(tumours) if tumours else 0.0
    argmax_low, argmax_high = wilson(len(argmax_missed), len(tumours))

    not_deferred = [r for r in argmax_scored if r["call_key"] != "uncertain"]
    tumours_not_deferred = [r for r in not_deferred if r["true_class"] in TUMOUR_CLASSES]
    missed_after_defer = [r for r in tumours_not_deferred if argmax_class(r) == "notumor"]

    four_way_correct = [
        r for r in argmax_scored if argmax_class(r) == r["true_class"]]

    # Sensitivity and specificity as the config defines them: the p_tumor
    # threshold alone, deferral ignored, so the number describes the classifier
    # rather than the workflow around it.
    threshold = float(expected.get("_tumor_threshold", 0) or 0)
    sensitivity = specificity = None
    if threshold > 0:
        scored_t = [r for r in tumours if isinstance(r.get("p_tumor"), (int, float))]
        scored_h = [r for r in healthy if isinstance(r.get("p_tumor"), (int, float))]
        if scored_t:
            sensitivity = sum(r["p_tumor"] >= threshold for r in scored_t) / len(scored_t)
        if scored_h:
            specificity = sum(r["p_tumor"] < threshold for r in scored_h) / len(scored_h)

    # A healthy scan called a tumour. Not dangerous, but a clinic that refers
    # everybody has not been helped, and it is how a tool loses its welcome.
    false_alarms = [r for r in healthy if r["call_key"] == "tumor"]

    decided = [r for r in records if r["call_key"] in ("tumor", "no_tumor")]
    correct_binary = [
        r for r in decided
        if (r["call_key"] == "tumor") == (r["true_class"] in TUMOUR_CLASSES)
    ]

    report = {
        "n": len(records),
        "n_tumour": len(tumours),
        "n_healthy": len(healthy),
        "tumours_sent_home": len(sent_home),
        "tumour_sent_home_rate": sent_home_rate,
        "tumour_sent_home_rate_ci": [sent_low, sent_high],
        "tumour_miss_rate_argmax": argmax_miss_rate,
        "tumour_miss_rate_argmax_ci": [argmax_low, argmax_high],
        "miss_rate_after_defer": (
            len(missed_after_defer) / len(tumours_not_deferred)
            if tumours_not_deferred else 0.0),
        "four_way_accuracy": (
            len(four_way_correct) / len(argmax_scored) if argmax_scored else 0.0),
        "binary_sensitivity": sensitivity,
        "binary_specificity": specificity,
        "false_alarm_rate": len(false_alarms) / len(healthy) if healthy else 0.0,
        "defer_rate": len(deferred) / len(records) if records else 0.0,
        "defer_rate_tumour": (
            sum(r["call_key"] == "uncertain" for r in tumours) / len(tumours)
            if tumours else 0.0),
        "defer_rate_healthy": (
            sum(r["call_key"] == "uncertain" for r in healthy) / len(healthy)
            if healthy else 0.0),
        "unreadable_rate": len(unreadable) / len(records) if records else 0.0,
        "binary_accuracy_on_decided": (
            len(correct_binary) / len(decided) if decided else 0.0),
        "sent_home_files": [Path(r["path"]).name for r in sent_home],
        "unreadable_files": [Path(r["path"]).name for r in unreadable],
    }

    # ---- pass or fail -------------------------------------------------------
    if any(r["validator_is_stub"] for r in records):
        failures.append(
            "the input check is a STUB in this package: nothing will be "
            "rejected as out of scope, so a photograph of a knee gets a tumour "
            "verdict")
    if any(r["explainer_is_stub"] for r in records):
        failures.append(
            "the heatmap is a STUB in this package: the overlay shown to a "
            "clinician is a test pattern, not where the model looked")

    if report["unreadable_rate"] > UNREADABLE_BUDGET:
        failures.append(
            f"{len(unreadable)} of {len(records)} "
            f"({percent(report['unreadable_rate'])}) ordinary T1 brain MRI were "
            f"refused as unreadable, over the {percent(UNREADABLE_BUDGET)} "
            f"budget. Refusing is safe but a clinic cannot use a tool that "
            f"turns away its own scans")

    published_miss = expected.get("tumor_miss_rate")
    if isinstance(published_miss, (int, float)) and tumours:
        # Fail only when the evidence says the shipped package is worse, not
        # when it merely looks worse on a small sample. The lower bound of the
        # observed interval sitting above the published point estimate is that
        # evidence.
        if argmax_low > published_miss:
            failures.append(
                f"this package misses {percent(argmax_miss_rate)} of tumours "
                f"(95% CI {percent(argmax_low)} to {percent(argmax_high)}), which "
                f"is worse than the {percent(published_miss)} it is documented as "
                f"missing. The shipped program is not the program that was "
                f"measured")

    published_defer = expected.get("defer_rate")
    if isinstance(published_defer, (int, float)):
        if abs(report["defer_rate"] - published_defer) > DEFER_TOLERANCE:
            failures.append(
                f"this package sends {percent(report['defer_rate'])} of scans to "
                f"a human. Its own config says {percent(published_defer)}. "
                f"Deferral is the safety net and the clinic staffs for it, so "
                f"the documented figure has to be the real one")

    published_specificity = expected.get("binary_specificity")
    if (specificity is not None and isinstance(published_specificity, (int, float))
            and published_specificity - specificity > 0.05):
        failures.append(
            f"this package clears {percent(specificity)} of healthy scans; its "
            f"config claims {percent(published_specificity)}. Every point of "
            f"that gap is a patient sent on a journey they did not need")

    return report, failures


#: Failures that writing the measured numbers into the config actually resolves.
#: Anything else is the package behaving differently from how it should, and no
#: amount of editing the paperwork fixes that.
_DOCUMENTATION_FAILURES = ("sends", "clears")


def update_config(published: dict, report: dict, failures: List[str],
                  n_sampled: int) -> List[str]:
    """Write the measured figures into the config, keeping what they replaced.

    Deliberately narrow. It rewrites the claims a package makes about itself and
    it never touches a threshold, because a threshold is a safety decision made
    on the internal validation split and this tool has only seen the outcome.

    It also refuses to run when the package is failing for any reason other than
    its paperwork. Editing the documentation to match a package that misses more
    tumours than it should would turn a loud failure into a quiet one, which is
    the exact bug this whole file exists to prevent.
    """
    real_problems = [
        f for f in failures
        if not any(f.startswith(f"this package {kind}") for kind in _DOCUMENTATION_FAILURES)
    ]
    if real_problems:
        print("\n  NOT updating the config: this package is failing for reasons "
              "the paperwork cannot fix.")
        return failures

    if n_sampled < 500:
        print(f"\n  NOT updating the config: {n_sampled} images is too small a "
              f"sample to publish as this package's performance. Rerun over the "
              f"whole clean subset (--per-class 1200).")
        return failures

    old = dict(published.get("expected_performance") or {})
    threshold = published.get("tumor_threshold")
    defer_cutoff = published.get("entropy_defer_threshold")

    updated = dict(old)
    updated.update({
        "n": report["n"],
        "tumor_miss_rate": report["tumour_miss_rate_argmax"],
        "tumor_miss_rate_ci": report["tumour_miss_rate_argmax_ci"],
        "tumour_sent_home_rate": report["tumour_sent_home_rate"],
        "miss_rate_after_defer": report["miss_rate_after_defer"],
        "binary_sensitivity": report["binary_sensitivity"],
        "binary_specificity": report["binary_specificity"],
        "four_way_accuracy": report["four_way_accuracy"],
        "defer_rate": report["defer_rate"],
        "defer_rate_tumour": report["defer_rate_tumour"],
        "defer_rate_healthy": report["defer_rate_healthy"],
        "unreadable_rate": report["unreadable_rate"],
        "measured_at_tumor_threshold": threshold,
        "measured_at_defer_cutoff": defer_cutoff,
        "measured_by": (
            "app/tools/verify_shipped_package.py, running the built Windows "
            "package over the whole clean subset through its own HTTP interface"),
    })

    history = list(published.get("expected_performance_history") or [])
    history.insert(0, {
        "replaced_utc": _utc_now(),
        "why": (
            "The figures below were measured under an earlier operating point "
            "and were not updated when the thresholds changed. They described a "
            "program that was no longer running."),
        "previous": old,
    })

    published["expected_performance"] = updated
    published["expected_performance_history"] = history
    CONFIG_JSON.write_text(json.dumps(published, indent=2) + "\n", encoding="utf-8")

    print(f"\n  {CONFIG_JSON} updated. The figures it publishes are now the ones "
          f"this package produced.")
    print("  The old block is kept in expected_performance_history.")
    return [f for f in failures if f in real_problems]


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _documented(expected: dict, key: str) -> str:
    value = expected.get(key)
    return percent(value) if isinstance(value, (int, float)) else "not documented"


def print_report(report: dict, expected: dict, failures: List[str]) -> None:
    print()
    print("  " + "-" * 68)
    print("                                   measured      documented")
    print(f"  Tumours SENT HOME              {percent(report['tumour_sent_home_rate']):>9}"
          f"      {'-':>10}   ({report['tumours_sent_home']} of {report['n_tumour']})")
    print(f"    95% interval                 "
          f"{percent(report['tumour_sent_home_rate_ci'][0])} to "
          f"{percent(report['tumour_sent_home_rate_ci'][1])}")
    print(f"  Tumour miss rate (argmax)      "
          f"{percent(report['tumour_miss_rate_argmax']):>9}      "
          f"{_documented(expected, 'tumor_miss_rate'):>10}")
    print(f"  Miss rate after deferral       "
          f"{percent(report['miss_rate_after_defer']):>9}      "
          f"{_documented(expected, 'miss_rate_after_defer'):>10}")
    print(f"  Sensitivity (p_tumor)          "
          + (f"{percent(report['binary_sensitivity']):>9}"
             if report["binary_sensitivity"] is not None else f"{'-':>9}")
          + f"      {_documented(expected, 'binary_sensitivity'):>10}")
    print(f"  Specificity (p_tumor)          "
          + (f"{percent(report['binary_specificity']):>9}"
             if report["binary_specificity"] is not None else f"{'-':>9}")
          + f"      {_documented(expected, 'binary_specificity'):>10}")
    print(f"  Four-way accuracy              "
          f"{percent(report['four_way_accuracy']):>9}      "
          f"{_documented(expected, 'four_way_accuracy'):>10}")
    print(f"  Sent to a human                {percent(report['defer_rate']):>9}      "
          f"{_documented(expected, 'defer_rate'):>10}")
    print(f"    of tumours                   {percent(report['defer_rate_tumour']):>9}")
    print(f"    of healthy scans             {percent(report['defer_rate_healthy']):>9}")
    print(f"  Healthy shown as TUMOUR        {percent(report['false_alarm_rate']):>9}"
          f"      {'-':>10}   (of {report['n_healthy']})")
    print(f"  Refused as unreadable          {percent(report['unreadable_rate']):>9}")
    print("  " + "-" * 68)

    for name in report["sent_home_files"]:
        print(f"    sent home: {name}")
    for name in report["unreadable_files"]:
        print(f"    refused:   {name}")

    if failures:
        print("\n  THIS PACKAGE MUST NOT GO TO A CLINIC:")
        for failure in failures:
            print(f"    - {failure}")
        return

    print("\n  The shipped package agrees with its own paperwork.")
    print("  That is a check on the build, not permission to use it on a "
          "patient.\n  Read LIMITATIONS.md and CLINIC_READINESS.md.")


if __name__ == "__main__":
    main()
