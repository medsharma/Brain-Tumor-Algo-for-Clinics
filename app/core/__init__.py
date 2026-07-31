"""Inference core for the offline clinic app.

No UI dependency. Import this from a script, a test, or a notebook and it
behaves identically to the running app.

    from app.core.engine import TriageEngine
    engine = TriageEngine()
    result = engine.analyze_path("scan.jpg")
    print(result.call, result.confidence)
"""

from __future__ import annotations

from .version import APP_NAME, APP_VERSION

__all__ = ["APP_NAME", "APP_VERSION"]
