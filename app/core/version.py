"""Version identity, written into every result and every audit line."""

from __future__ import annotations

APP_NAME = "Brain MRI Triage"

#: Bump on any change that could move a call. The audit log records it, so a
#: clinic can tell which build produced which answer.
APP_VERSION = "1.0.0-pilot"

#: What this build has been shown to do. Deliberately narrow.
VALIDATED_SCOPE = (
    "T1 brain MRI slices, one image at a time, saved as JPEG or PNG. "
    "Three tumour families plus no-tumour."
)
