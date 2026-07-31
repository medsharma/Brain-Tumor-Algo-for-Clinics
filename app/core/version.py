"""Version identity, written into every result and every audit line."""

from __future__ import annotations

APP_NAME = "Brain MRI Triage"

#: Bump on any change that could move a call. The audit log records it, so a
#: clinic can tell which build produced which answer.
APP_VERSION = "1.0.0-pilot"

#: What the tool is built to read. Deliberately narrow. This is a statement
#: about inputs, not about performance, and it is worded that way on purpose.
INTENDED_SCOPE = (
    "T1 brain MRI slices, one image at a time, saved as JPEG or PNG. "
    "Three tumour families plus no-tumour."
)


def validation_statement(expected_performance: dict, is_stub: bool) -> str:
    """One honest line about what has actually been measured.

    The app used to print "Validated on: ..." next to the intended scope.
    That was an overclaim. Session A found that BRISC 2025 is roughly 80% the
    training set republished, so at the time of writing the project has no
    external validation result at all. Saying otherwise on a clinical screen
    is exactly the kind of thing that gets people hurt.

    This reads the config rather than hard-coding a claim, so the line stops
    being a lie the moment session A publishes real numbers, and stays honest
    if they never do.
    """
    if is_stub:
        return (
            "NOT VALIDATED. This is a development build running on placeholder "
            "numbers. Nothing it says means anything clinically."
        )

    dataset = str(expected_performance.get("dataset", "")).strip()
    n = expected_performance.get("n")
    miss_rate = expected_performance.get("tumor_miss_rate")

    if not dataset or not isinstance(n, int) or n <= 0:
        return (
            "No performance figures have been recorded for this build. Treat "
            "every answer as unverified."
        )

    if isinstance(miss_rate, (int, float)):
        return (
            f"Measured on {dataset}, {n:,} images. Of real tumours, "
            f"{miss_rate:.1%} were called no-tumour. Ask whoever installed "
            f"this tool whether that dataset was genuinely unseen by the model."
        )

    return f"Measured on {dataset}, {n:,} images."
