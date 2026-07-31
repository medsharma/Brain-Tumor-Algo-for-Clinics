"""The page and its script must agree, and the page must say what it must say.

A mistyped element id does not raise anything. The script quietly does nothing
and a section of the result page stays blank. On a triage tool that could mean
the disclaimer or the confidence wording silently not appearing, which is the
kind of failure nobody notices until it matters.

These checks are static, so they run in a second and never flake.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "static"
HTML = (STATIC / "index.html").read_text(encoding="utf-8")
JS = (STATIC / "app.js").read_text(encoding="utf-8")
CSS = (STATIC / "app.css").read_text(encoding="utf-8")

HTML_IDS = set(re.findall(r'\bid="([^"]+)"', HTML))
JS_IDS = set(re.findall(r'\bel\(\s*"([^"]+)"\s*\)', JS))


def test_every_id_the_script_touches_exists_in_the_page():
    missing = sorted(JS_IDS - HTML_IDS)
    assert not missing, (
        f"app.js looks up ids that index.html does not define: {missing}. "
        f"These fail silently and leave part of the result blank."
    )


def test_every_call_key_has_a_style():
    """All four calls must be visually distinct, not just textually."""
    for key in ("tumor", "no_tumor", "uncertain", "cannot_read"):
        assert f".call-box.{key}" in CSS, f"no colour defined for the {key} call"


def test_the_page_promises_the_things_the_brief_requires():
    lowered = HTML.lower()
    assert "not a diagnosis" in lowered
    assert "dicom" in lowered
    assert "one brain mri slice at a time" in lowered
    assert "nothing leaves this laptop" in lowered


def test_the_page_says_dicom_is_unsupported_rather_than_staying_vague():
    assert "DICOM files are not supported" in HTML
    assert "JPEG or PNG" in HTML


def test_the_page_warns_that_it_judges_one_slice_not_a_study():
    assert "does not combine slices" in HTML


def test_the_disclaimer_block_is_present_and_marked_up_as_a_note():
    assert 'class="disclaimer"' in HTML
    assert 'role="note"' in HTML
    assert 'id="disclaimer-text"' in HTML


def test_the_disclaimer_appears_in_the_result_and_the_footer():
    """Two places, so it is visible whether or not the operator scrolls."""
    assert JS.count("disclaimer") >= 3
    assert 'id="footer-disclaimer"' in HTML


def test_the_heatmap_caveat_has_a_home_on_the_page():
    """Session D publishes the wording; the page must have somewhere to put it."""
    assert 'id="heatmap-caveat"' in HTML
    assert "heatmap-caveat" in JS


def test_the_tumour_type_is_marked_secondary():
    assert "Secondary information" in HTML
    assert "A clinic refers either way" in HTML


def test_the_heatmap_has_both_a_slider_and_a_toggle():
    assert 'id="overlay-slider"' in HTML
    assert 'id="overlay-toggle"' in HTML
    assert 'type="range"' in HTML


def test_the_original_and_the_overlay_are_shown_side_by_side():
    assert 'id="img-original"' in HTML
    assert 'id="img-overlay"' in HTML
    assert ".image-pair" in CSS


def test_the_filename_checkbox_defaults_to_off():
    """Off by default, because file names routinely carry patient names."""
    checkbox = re.search(r'<input type="checkbox" id="include-filename"[^>]*>', HTML)
    assert checkbox is not None
    assert "checked" not in checkbox.group(0)


def test_no_inline_style_or_script_in_the_page():
    """The Content-Security-Policy forbids both. An inline block would break
    the page only on the clinic laptop, not on a developer's machine."""
    assert "<style" not in HTML.lower()
    assert not re.search(r"<script(?![^>]*\bsrc=)", HTML, re.I)
    assert not re.search(r'\bstyle="', HTML)


def test_no_external_resource_in_any_static_file():
    pattern = re.compile(r"https?://|//cdn\.|googleapis|jsdelivr|unpkg", re.I)
    for name, text in (("index.html", HTML), ("app.js", JS), ("app.css", CSS)):
        assert not pattern.search(text), f"{name} references an outside resource"


def test_the_font_stack_uses_only_fonts_a_machine_already_has():
    """A web font would silently fail to load on an offline laptop."""
    assert "@font-face" not in CSS
    assert "system-ui" in CSS


@pytest.mark.parametrize(
    "handler",
    ["save-report", "new-scan", "error-retry", "overlay-toggle", "overlay-slider"],
)
def test_every_control_is_wired_to_something(handler):
    assert handler in JS, f"the {handler} control has no listener"
