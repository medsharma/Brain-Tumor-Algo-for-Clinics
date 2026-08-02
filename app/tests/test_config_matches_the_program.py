"""The numbers printed on the box must describe the program inside it.

This is the failure these tests exist for, and it shipped.

`deployment_config.json` carries an `expected_performance` block. It is the
source for what the app tells people about itself, what the download page
publishes, and what a clinic would use to plan. On 2026-07-31 the deferral
cutoff was refitted, correctly, in commit e7dc1ba. The performance block was not
touched, and nothing anywhere noticed.

The result: the config said the tool sends 41.2% of scans to a human. The program
in the box sends about 4%. It said 98.6% of healthy scans are cleared. The
program clears about 90%. Both figures were true of the previous operating point
and of nothing that was actually running.

Nothing crashed. Every test passed. A clinic reading that file would have staffed
for a tool that does not exist, and a reviewer checking the claims against the
behaviour would have found them wrong.

So: the performance block must record the thresholds it was measured at, and
those must still be the thresholds in force. Change a threshold without
remeasuring and this fails, loudly, here, rather than quietly in a clinic.

The measuring itself is `app/tools/verify_shipped_package.py`, which runs the
built package over the published clean subset. These tests only hold the two
halves together.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "analysis" / "results" / "safety" / "deployment_config.json"

#: Floating point read back from JSON, compared against JSON. Equal to within
#: the width of a rounding error, not the width of a decision.
TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def config() -> dict:
    if not CONFIG_PATH.is_file():
        pytest.skip("no real deployment config on this machine")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def performance(config: dict) -> dict:
    return config.get("expected_performance") or {}


def test_the_performance_block_records_what_it_was_measured_at(performance):
    """Numbers with no operating point attached cannot be checked by anybody."""
    for field in ("measured_at_tumor_threshold", "measured_at_defer_cutoff"):
        assert field in performance, (
            f"expected_performance has no {field}. Without it nobody can tell "
            f"whether these figures describe the current settings or an older "
            f"set, which is exactly how the 41% deferral claim survived a "
            f"threshold change."
        )


def test_the_published_numbers_describe_the_thresholds_in_force(config, performance):
    """The guard. Change a threshold, remeasure, or this fails."""
    pairs = (
        ("tumor_threshold", "measured_at_tumor_threshold"),
        ("entropy_defer_threshold", "measured_at_defer_cutoff"),
    )
    for live_key, measured_key in pairs:
        live = config.get(live_key)
        measured = performance.get(measured_key)
        assert isinstance(live, (int, float)) and isinstance(measured, (int, float))
        assert abs(live - measured) < TOLERANCE, (
            f"{live_key} is {live} but expected_performance was measured at "
            f"{measured}. Every figure in that block describes a different "
            f"program from the one this config runs. Rerun "
            f"app/tools/verify_shipped_package.py and update the block."
        )


def test_the_defer_signal_and_the_notes_do_not_contradict_each_other(config):
    """The field is named for entropy and may hold something else entirely.

    ``entropy_defer_threshold`` held a mutual-information cutoff while the notes
    still told the reader it was compared against predictive entropy. Both
    cannot be true, and the app follows ``defer_signal``.
    """
    signal = str(config.get("defer_signal", "entropy")).lower()
    notes = " ".join(str(n) for n in config.get("notes") or []).lower()

    if signal == "mutual_information":
        assert "compared against the predictive entropy" not in notes, (
            "defer_signal is mutual_information, but the notes still say the "
            "cutoff is compared against predictive entropy. The app compares "
            "mutual information. One of the two is lying to whoever reads this "
            "file next."
        )
        assert "mutual information" in notes, (
            "the notes never mention mutual information, which is the quantity "
            "the cutoff is actually applied to"
        )


def test_the_performance_block_says_who_measured_it(performance):
    """Provenance, so the next person can rerun it rather than trust it."""
    assert str(performance.get("measured_by", "")).strip(), (
        "expected_performance does not say what produced it"
    )


def test_the_deferral_figure_is_present_and_sane(performance):
    """A clinic staffs to this number. It is not decoration."""
    defer_rate = performance.get("defer_rate")
    assert isinstance(defer_rate, (int, float)), "no deferral rate published"
    assert 0.0 <= defer_rate <= 1.0


# ----------------------------------------------------------------- the clock

def test_a_clock_before_the_build_date_is_reported(monkeypatch):
    """A clinic laptop is offline, so nothing corrects its clock.

    When the battery dies it comes back years in the past and stays there. Every
    result sheet is then dated wrongly, with no warning, and a wrongly dated
    report can be filed against the wrong visit.
    """
    from app.core import readiness

    finding = readiness._clock_finding("2026-07-31T02:50:49.187118+00:00")
    assert finding is None, "a normal clock must not raise this"

    from datetime import datetime, timedelta, timezone

    future = (datetime.now(timezone.utc) + timedelta(days=400)).isoformat()
    finding = readiness._clock_finding(future)
    assert finding is not None
    assert finding.code == "clock_is_wrong"
    assert "wrong date" in finding.message


def test_an_unreadable_build_date_does_not_cry_wolf():
    """An unparseable date is not evidence of a broken clock."""
    from app.core import readiness

    assert readiness._clock_finding("") is None
    assert readiness._clock_finding("not a date") is None
