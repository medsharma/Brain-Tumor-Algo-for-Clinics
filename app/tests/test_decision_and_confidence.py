"""Decision rules, and not lying with decimal places."""

from __future__ import annotations

import math

import pytest

from app.core import decision
from app.core.decision import Call, Confidence, decide, display_step_percent, format_probability

CLASS_NAMES = ("glioma", "meningioma", "pituitary", "notumor")
MAX_ENTROPY_BITS = 2.0


def _decide(p_tumor, entropy, tumor_threshold=0.5, defer=0.8, ece=None):
    remaining = (1.0 - p_tumor)
    probs = (p_tumor * 0.7, p_tumor * 0.2, p_tumor * 0.1, remaining)
    return decide(
        mean_probs=probs,
        p_tumor=p_tumor,
        entropy=entropy,
        class_names=CLASS_NAMES,
        tumor_threshold=tumor_threshold,
        entropy_defer_threshold=defer,
        max_entropy=MAX_ENTROPY_BITS,
        ece=ece,
    )


# ------------------------------------------------------------------ ordering

def test_deferral_is_checked_before_the_tumour_threshold():
    """A shaky scan goes to a human rather than being forced into a yes or no."""
    result = _decide(p_tumor=0.99, entropy=1.5, defer=0.8)
    assert result.call is Call.UNCERTAIN


def test_a_confident_high_probability_is_a_tumour_call():
    result = _decide(p_tumor=0.95, entropy=0.2)
    assert result.call is Call.TUMOR


def test_a_confident_low_probability_is_a_no_tumour_call():
    result = _decide(p_tumor=0.05, entropy=0.2)
    assert result.call is Call.NO_TUMOR


def test_the_threshold_boundary_goes_to_tumour():
    """At exactly the threshold, err towards referral, not towards home."""
    assert _decide(p_tumor=0.5, entropy=0.1, tumor_threshold=0.5).call is Call.TUMOR


def test_a_zero_deferral_threshold_disables_deferral():
    """Session A may choose not to defer at all. That must not defer everything."""
    result = _decide(p_tumor=0.9, entropy=1.9, defer=0.0)
    assert result.call is Call.TUMOR


# ---------------------------------------------------------------- confidence

def test_confidence_words_are_reachable():
    high = _decide(p_tumor=0.99, entropy=0.10, defer=0.8)
    moderate = _decide(p_tumor=0.65, entropy=0.55, defer=0.8)
    low = _decide(p_tumor=0.52, entropy=0.70, defer=0.8)

    assert high.confidence is Confidence.HIGH
    assert moderate.confidence is Confidence.MODERATE
    assert low.confidence is Confidence.LOW


def test_a_scan_near_the_threshold_is_never_high_confidence():
    """A margin of two percentage points is not high confidence, however calm
    the Monte Carlo passes were."""
    result = _decide(p_tumor=0.52, entropy=0.01, defer=0.8)
    assert result.confidence is not Confidence.HIGH


def test_a_disagreeing_model_is_never_high_confidence():
    """A comfortable margin does not rescue a scan the model was unsure about."""
    result = _decide(p_tumor=0.99, entropy=0.75, defer=0.8)
    assert result.confidence is not Confidence.HIGH


def test_a_deferred_scan_reports_low_confidence():
    assert _decide(p_tumor=0.9, entropy=1.5, defer=0.8).confidence is Confidence.LOW


# ---------------------------------------------------------------- precision

def test_no_percentage_is_shown_when_calibration_is_unknown():
    """The honest position before session A measures it."""
    assert display_step_percent(None) is None
    assert format_probability(0.947, None) is None
    assert _decide(p_tumor=0.95, entropy=0.1, ece=None).probability_text is None


def test_a_poorly_calibrated_model_gets_coarse_numbers():
    """At an ECE around 0.07, "94.7%" is invented precision."""
    text = format_probability(0.947, ece=0.07)
    assert text is not None
    assert "94.7" not in text
    assert "95%" in text or "90%" in text
    assert "realistically" in text


def test_a_very_poorly_calibrated_model_gets_no_number_at_all():
    assert display_step_percent(0.30) is None
    assert format_probability(0.947, ece=0.30) is None


def test_a_well_calibrated_model_may_show_single_points():
    assert display_step_percent(0.01) == 1
    assert "95%" in format_probability(0.9502, ece=0.01)


@pytest.mark.parametrize("ece,expected_step", [(0.0, 1), (0.02, 1), (0.05, 5), (0.07, 10), (0.15, 10)])
def test_precision_steps_widen_as_calibration_worsens(ece, expected_step):
    assert display_step_percent(ece) == expected_step


def test_the_stated_band_reflects_the_calibration_error():
    text = format_probability(0.80, ece=0.07)
    assert "73%" in text and "87%" in text


def test_percentages_never_leave_the_zero_to_hundred_range():
    assert "110%" not in (format_probability(0.99, ece=0.15) or "")
    assert "-" not in (format_probability(0.01, ece=0.15) or "")


# ---------------------------------------------------------------- secondary

def test_the_tumour_type_is_only_offered_on_a_tumour_call():
    assert _decide(p_tumor=0.95, entropy=0.1).tumor_type is not None
    assert _decide(p_tumor=0.05, entropy=0.1).tumor_type is None


def test_the_tumour_type_is_the_highest_scoring_tumour_class():
    result = decide(
        mean_probs=(0.10, 0.70, 0.10, 0.10),
        p_tumor=0.90,
        entropy=0.2,
        class_names=CLASS_NAMES,
        tumor_threshold=0.5,
        entropy_defer_threshold=0.8,
        max_entropy=MAX_ENTROPY_BITS,
        ece=None,
    )
    assert result.tumor_type == "Meningioma"


# --------------------------------------------------------------- disclaimer

def test_the_disclaimer_names_every_limit_the_brief_requires():
    text = decision.DISCLAIMER_FULL.lower()
    assert "not a diagnosis" in text
    assert "qualified human" in text
    assert "glioma" in text and "meningioma" in text and "pituitary" in text
    assert "no-tumour" in text
    assert "metastases" in text
    assert "rarer" in text


def test_the_disclaimer_warns_that_no_tumour_is_not_a_clean_bill():
    assert "may still contain a tumour it was never taught to see" in decision.DISCLAIMER_FULL


def test_cannot_read_produces_no_numbers():
    result = decision.cannot_read("Not a brain MRI.")
    assert result.call is Call.CANNOT_READ
    assert result.p_tumor is None
    assert result.probability_text is None
    assert result.confidence is Confidence.NONE


def test_entropy_maxima_are_right_for_four_classes():
    from app.core.config import MAX_ENTROPY_BITS as bits, MAX_ENTROPY_NATS as nats

    assert bits == pytest.approx(2.0)
    assert nats == pytest.approx(math.log(4))
