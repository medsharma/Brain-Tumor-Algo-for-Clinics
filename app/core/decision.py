"""Turning model output into the one call a clinic worker acts on.

Four outcomes, and only four. The wording is fixed here so the UI, the export
and the audit log can never drift apart.

On confidence wording: the model's probabilities are only as trustworthy as
its calibration error. If the expected calibration error on external data is
around 0.07, then a screen reading "94.7% confident" is lying with decimal
places. This module refuses to print more precision than the measured
calibration supports, and prints no number at all while the calibration is
unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# --------------------------------------------------------------------------
# The disclaimer. One source of truth, shown on every result and every export.
# --------------------------------------------------------------------------

DISCLAIMER_SHORT = (
    "Not a diagnosis. A qualified human makes the call."
)

DISCLAIMER_FULL = (
    "This tool is not a diagnosis and does not replace a doctor. "
    "A qualified human makes the call.\n"
    "It recognises three tumour families only: glioma, meningioma and "
    "pituitary, plus no-tumour. "
    "It does not cover metastases, and it does not cover rarer tumour types. "
    "A scan it calls NO TUMOR may still contain a tumour it was never taught "
    "to see.\n"
    "It judges one image at a time, not a whole study, and it has been "
    "validated on T1 brain MRI only."
)


class Call(str, Enum):
    """The four outcomes. String values are what the operator reads."""

    TUMOR = "TUMOR — refer urgently"
    NO_TUMOR = "NO TUMOR"
    UNCERTAIN = "UNCERTAIN — needs human read"
    CANNOT_READ = "CANNOT READ THIS IMAGE — not a supported brain MRI"

    @property
    def key(self) -> str:
        """Stable machine identifier for the audit log and the UI stylesheet."""
        return self.name.lower()


#: What the operator should do next. Shown under the call.
NEXT_STEP: dict[Call, str] = {
    Call.TUMOR: "Refer this patient urgently for a specialist read.",
    Call.NO_TUMOR: (
        "No tumour of the three types this tool knows. "
        "If the patient has symptoms, refer anyway. This tool does not "
        "overrule what you can see in front of you."
    ),
    Call.UNCERTAIN: (
        "Send this scan to a human reader. The tool is not confident enough "
        "to give an answer you should act on."
    ),
    Call.CANNOT_READ: (
        "The tool cannot judge this image. Check it is a brain MRI slice in "
        "JPEG or PNG form. If it is, send it to a human reader."
    ),
}


class Confidence(str, Enum):
    HIGH = "High confidence"
    MODERATE = "Moderate confidence"
    LOW = "Low confidence"
    NONE = "No confidence estimate"

    @property
    def key(self) -> str:
        return self.name.lower()


# --------------------------------------------------------------------------
# Precision policy
# --------------------------------------------------------------------------

def display_step_percent(ece: float | None) -> int | None:
    """How coarsely a probability may be shown, given calibration error.

    ``None`` means show no number at all: either calibration was never
    measured, or it is so poor that any percentage would mislead.

    A model with an expected calibration error of 0.07 is typically wrong by
    about 7 percentage points, so quoting single percentage points invents
    precision that does not exist.
    """
    if ece is None:
        return None
    if ece <= 0.02:
        return 1
    if ece <= 0.05:
        return 5
    if ece <= 0.15:
        return 10
    return None


def format_probability(probability: float, ece: float | None) -> str | None:
    """A percentage rounded to what the calibration justifies, plus its band.

    Returns ``None`` when no number should be shown.
    """
    step = display_step_percent(ece)
    if step is None:
        return None

    percent = 100.0 * probability
    rounded = int(round(percent / step) * step)
    rounded = max(0, min(100, rounded))

    if ece is None:
        return f"about {rounded}%"

    margin = int(round(100.0 * ece))
    if margin <= 0:
        return f"about {rounded}%"
    low = max(0, rounded - margin)
    high = min(100, rounded + margin)
    return f"about {rounded}% (realistically {low}% to {high}%)"


# --------------------------------------------------------------------------
# Confidence wording
# --------------------------------------------------------------------------

#: Fractions of the deferral threshold. Below the first, the model is well
#: inside the region session A judged reliable.
_ENTROPY_HIGH_FRACTION = 0.50
_ENTROPY_MODERATE_FRACTION = 0.80

#: Distance of p_tumor from the operating threshold.
_MARGIN_HIGH = 0.25
_MARGIN_MODERATE = 0.10


def confidence_level(
    p_tumor: float,
    entropy: float,
    tumor_threshold: float,
    entropy_defer_threshold: float,
    max_entropy: float,
    uncertainty: float | None = None,
) -> Confidence:
    """Plain-words confidence, from uncertainty and distance to the threshold.

    ``uncertainty`` must be measured on the SAME signal the deferral cutoff was
    fitted on, because the two are divided by each other. Passing four-way
    entropy while ``entropy_defer_threshold`` holds a mutual-information cutoff
    compares two different quantities on two different scales: a confident scan
    with entropy 0.03 against an MI cutoff of 0.0116 scores a ratio of 2.6 and
    is reported as low confidence, which is how every answer ended up saying
    "Low confidence" after the deferral signal changed. Defaults to ``entropy``
    only for callers that have nothing else.

    This is a display heuristic, not a calibrated quantity, and it is
    described that way in the UI. It combines two things that can each make a
    call shaky: the Monte Carlo passes disagreeing with each other, and the
    tumour probability sitting close to the line where the call flips.

    The worse of the two wins. A comfortable margin does not rescue a scan the
    model was internally unsure about, and vice versa.
    """
    value = entropy if uncertainty is None else uncertainty
    reference = entropy_defer_threshold if entropy_defer_threshold > 0 else max_entropy
    entropy_ratio = value / reference if reference > 0 else 1.0

    # Margin has to be measured against the room available on each side of the
    # threshold, not as a raw distance.
    #
    # The raw version was `abs(p_tumor - tumor_threshold)` compared against a
    # fixed 0.25 for High and 0.10 for Moderate. That silently breaks whenever
    # the threshold is near an end of the scale. With the threshold at 0.970,
    # the largest margin any tumour call could possibly have was 0.030, so
    # **every tumour call was pinned to "Low confidence" by arithmetic**, no
    # matter how certain the model was. Driving the real app end to end, a
    # correct glioma call at p_tumor = 0.9984 came back "Low confidence".
    #
    # Normalising by the distance to the nearer end of the scale makes the
    # number mean the same thing at any threshold: 1.0 is as far from the
    # decision boundary as it is possible to get.
    if p_tumor >= tumor_threshold:
        room = 1.0 - tumor_threshold
    else:
        room = tumor_threshold
    margin = abs(p_tumor - tumor_threshold) / room if room > 0 else 0.0

    if entropy_ratio <= _ENTROPY_HIGH_FRACTION and margin >= _MARGIN_HIGH:
        return Confidence.HIGH
    if entropy_ratio <= _ENTROPY_MODERATE_FRACTION and margin >= _MARGIN_MODERATE:
        return Confidence.MODERATE
    return Confidence.LOW


# --------------------------------------------------------------------------
# The decision itself
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    call: Call
    confidence: Confidence
    reason: str
    next_step: str
    probability_text: str | None
    p_tumor: float | None
    entropy: float | None
    tumor_type: str | None
    tumor_type_probability: float | None


#: Human-readable class names for display. The model's internal names are
#: terse; these are what a clinic worker reads.
DISPLAY_CLASS_NAMES: dict[str, str] = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitary tumour",
    "notumor": "No tumour",
}


def decide(
    *,
    mean_probs: tuple[float, ...],
    p_tumor: float,
    entropy: float,
    class_names: tuple[str, ...],
    tumor_threshold: float,
    entropy_defer_threshold: float,
    max_entropy: float,
    ece: float | None,
    mutual_information: float | None = None,
    defer_signal: str = "entropy",
) -> Decision:
    """Apply the operating point to one image's Monte Carlo output.

    Order matters. Deferral is checked before the tumour threshold, so a scan
    the model was unsure about goes to a human rather than being forced into a
    yes or no by an arbitrary cut.

    ``defer_signal`` selects what "unsure" is measured on, and the choice is not
    cosmetic. Four-way predictive entropy mixes two different things: not
    knowing whether there is a tumour, and not knowing which of three families
    it belongs to. Only the first changes what a clinic does. Measured on the
    uncontaminated BRISC subset, deferring on four-way entropy at a cutoff
    fitted to defer 10% of internal validation actually deferred 26% of scans,
    including 43% of healthy ones, because the model is broadly unsure which
    class an unfamiliar healthy brain belongs to.

    Mutual information isolates the epistemic part, "I have not seen anything
    like this", from that class ambiguity. At the same fitted budget it defers
    14%, and 15% of healthy scans. It is the signal that survives the move to
    unseen data, so it is what the shipped config selects.
    """
    if defer_signal == "mutual_information":
        _uncertainty = mutual_information if mutual_information is not None else entropy
    else:
        _uncertainty = entropy

    confidence = confidence_level(
        p_tumor=p_tumor,
        entropy=entropy,
        tumor_threshold=tumor_threshold,
        entropy_defer_threshold=entropy_defer_threshold,
        max_entropy=max_entropy,
        uncertainty=_uncertainty,
    )

    tumor_indices = [i for i, name in enumerate(class_names) if name != "notumor"]
    best_tumor_index = max(tumor_indices, key=lambda i: mean_probs[i])
    tumor_type = DISPLAY_CLASS_NAMES.get(
        class_names[best_tumor_index], class_names[best_tumor_index]
    )
    tumor_type_probability = float(mean_probs[best_tumor_index])

    uncertainty = _uncertainty
    if entropy_defer_threshold > 0 and uncertainty > entropy_defer_threshold:
        return Decision(
            call=Call.UNCERTAIN,
            confidence=Confidence.LOW,
            reason=(
                f"The model's repeated readings of this scan disagreed with each "
                f"other more than the safe limit allows "
                f"({uncertainty:.3f} against a limit of {entropy_defer_threshold:.3f})."
            ),
            next_step=NEXT_STEP[Call.UNCERTAIN],
            probability_text=None,
            p_tumor=p_tumor,
            entropy=entropy,
            tumor_type=tumor_type,
            tumor_type_probability=tumor_type_probability,
        )

    if p_tumor >= tumor_threshold:
        return Decision(
            call=Call.TUMOR,
            confidence=confidence,
            reason="The scan looks like one of the three tumour types this tool knows.",
            next_step=NEXT_STEP[Call.TUMOR],
            probability_text=format_probability(p_tumor, ece),
            p_tumor=p_tumor,
            entropy=entropy,
            tumor_type=tumor_type,
            tumor_type_probability=tumor_type_probability,
        )

    return Decision(
        call=Call.NO_TUMOR,
        confidence=confidence,
        reason="The scan does not look like any of the three tumour types this tool knows.",
        next_step=NEXT_STEP[Call.NO_TUMOR],
        probability_text=format_probability(1.0 - p_tumor, ece),
        p_tumor=p_tumor,
        entropy=entropy,
        tumor_type=None,
        tumor_type_probability=None,
    )


def cannot_read(reason: str) -> Decision:
    """The fourth call. Comes from session B's input validation, or a bad file."""
    return Decision(
        call=Call.CANNOT_READ,
        confidence=Confidence.NONE,
        reason=reason,
        next_step=NEXT_STEP[Call.CANNOT_READ],
        probability_text=None,
        p_tumor=None,
        entropy=None,
        tumor_type=None,
        tumor_type_probability=None,
    )
