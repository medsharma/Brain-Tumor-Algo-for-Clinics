"""Version identity, written into every result and every audit line."""

from __future__ import annotations

APP_NAME = "Brain MRI Triage"

#: Bump on any change that could move a call. The audit log records it, so a
#: clinic can tell which build produced which answer.
APP_VERSION = "1.0.0-pilot"

#: What the tool is built to read. Deliberately narrow. This is a statement
#: about inputs, not about performance, and it is worded that way on purpose.
#:
#: It does not repeat the class list. That is in the disclaimer directly above
#: it on screen, and saying it twice in four lines teaches people to skim both.
INTENDED_SCOPE = "T1 brain MRI slices, one at a time. DICOM, JPEG or PNG."


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
        # A rate per thousand, because "0.3%" of a number nobody has in mind is
        # not a quantity anyone can feel. Three patients is.
        #
        # And it states the limit itself instead of ending on "ask whoever
        # installed this tool whether that dataset was genuinely unseen", which
        # was the old wording. That question was passed to the one person least
        # able to answer it, when the answer is known and is this: the test data
        # comes from the same sources as the training data.
        per_thousand = miss_rate * 1000
        return (
            f"Tested on {n:,} images it never trained on, from the same sources "
            f"as the data it learned from. It missed {per_thousand:.0f} tumours "
            f"in 1,000."
        )

    return f"Tested on {dataset}, {n:,} images. No miss rate recorded."
