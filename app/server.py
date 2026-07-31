"""Local web shell over ``app.core``. Bound to 127.0.0.1 and nothing else.

Why a local web app rather than a desktop window
------------------------------------------------
A clinic laptop is whatever the clinic already owns, which in practice means a
mix of Windows versions and the occasional old Mac. A local web app runs the
same on all of them, renders a side-by-side heatmap comparison without fighting
a widget toolkit, and needs no GUI libraries in the installer, which keeps the
bundle small enough to move on a USB stick. The real cost is the one that
matters, and it is honest to name it: there is a server process, and when a
server process dies a non-technical user sees a blank browser tab and has no
idea what happened. That is mitigated here rather than ignored: the launcher
opens the browser itself, prints a plain-language message if the port is
already taken, and keeps the console window open on a crash so there is
something to read. A PySide6 window would remove that failure mode but add a
much heavier build and a worse image comparison view. The trade lands on the
web app.

Binding
-------
``127.0.0.1`` only, never ``0.0.0.0``. A clinic laptop on a shared clinic
network must not serve patient scans to every other machine on that network.
This is enforced in :func:`run`, and ``app/tests/test_binding.py`` asserts it.

No patient data leaves the machine. Uploaded images are held in memory, never
written to disk, and never sent anywhere. There is no telemetry and no crash
reporting.
"""

from __future__ import annotations

import base64
import io
import ipaddress
import logging
import socket
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .core import decision, explain_adapter, version as version_module
from .core.engine import TriageEngine, TriageResult
from .core.readiness import NotReadyError

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

#: Upload ceiling. Far above one MRI slice, far below anything that could
#: exhaust memory on an 8 GB laptop.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024

#: The only address the app is allowed to bind. Loopback means this machine
#: and nothing else on the network can reach it.
LOOPBACK_HOST = "127.0.0.1"

_engine: TriageEngine | None = None


class BindingRefused(RuntimeError):
    """Someone asked the app to listen on a network-visible address."""


def assert_loopback(host: str) -> None:
    """Refuse any bind address that is not loopback.

    ``0.0.0.0`` means every interface, which on a clinic network means every
    other machine in the building can fetch patient scans from this laptop.
    """
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise BindingRefused(
            f"Refusing to listen on {host!r}. The app binds loopback only."
        ) from exc

    if not address.is_loopback:
        raise BindingRefused(
            f"Refusing to listen on {host}. That address is reachable from the "
            f"network, which would serve patient scans to every other machine "
            f"on it. The app binds {LOOPBACK_HOST} only."
        )


def get_engine() -> TriageEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="The app is still starting up.")
    return _engine


def set_engine(engine: TriageEngine | None) -> None:
    """Install the engine. Called by :func:`run` and by tests."""
    global _engine
    _engine = engine


# --------------------------------------------------------------------------
# Image encoding
# --------------------------------------------------------------------------

def _png_data_uri(array: np.ndarray | None, upscale_to: int = 448) -> str | None:
    """Encode an RGB array as a self-contained data URI.

    Data URIs rather than served URLs, so a patient image never has a fetchable
    address and never touches disk. Upscaled with nearest-neighbour so the
    operator sees the exact pixels the model saw, just larger, with no
    interpolation inventing detail that was not there.
    """
    if array is None:
        return None
    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    if upscale_to and image.width < upscale_to:
        image = image.resize((upscale_to, upscale_to), Image.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _result_payload(result: TriageResult, include_images: bool = True) -> dict[str, Any]:
    payload = result.to_dict(include_display_name=True)
    if include_images:
        payload["original_png"] = _png_data_uri(result.original_rgb)
        payload["overlay_png"] = _png_data_uri(result.overlay_rgb)
        payload["has_heatmap"] = result.overlay_rgb is not None
    return payload


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title=f"{version_module.APP_NAME} (offline)",
        version=version_module.APP_VERSION,
        docs_url=None,      # no interactive docs; they pull in a CDN bundle
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def local_only(request, call_next):  # type: ignore[no-untyped-def]
        """Second line of defence behind the loopback bind.

        If the app is ever run behind something that forwards traffic, this
        still refuses requests that did not originate on this machine.
        """
        client = request.client
        if client is not None:
            try:
                if not ipaddress.ip_address(client.host).is_loopback:
                    return JSONResponse(
                        {"detail": "This app serves the local machine only."},
                        status_code=403,
                    )
            except ValueError:
                return JSONResponse({"detail": "Unrecognised client address."}, status_code=403)

        response = await call_next(request)
        # No caching of anything: a browser cache holding patient scans on a
        # shared clinic laptop is a leak with no upside.
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        # Everything the page needs is inline or same-origin. This header makes
        # an accidental CDN reference fail loudly instead of silently working
        # on a developer machine that happens to be online.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'; font-src 'self'; "
            "form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        engine = get_engine()
        readiness = engine.readiness
        return {
            "app_name": version_module.APP_NAME,
            "app_version": version_module.APP_VERSION,
            "intended_scope": version_module.INTENDED_SCOPE,
            "validation_statement": version_module.validation_statement(
                dict(engine.config.expected_performance), engine.config.is_stub
            ),
            "state": readiness.state,
            "dev_mode": readiness.dev_mode,
            "warnings": [
                {"code": finding.code, "message": finding.message}
                for finding in readiness.warnings
            ],
            "config": {
                "is_stub": engine.config.is_stub,
                "schema_version": engine.config.schema_version,
                "created_utc": engine.config.created_utc,
                "backbone": engine.config.chosen_backbone,
                "seeds": list(engine.config.chosen_seeds),
                "mc_T": engine.config.mc_T,
                "tumor_threshold": engine.config.tumor_threshold,
                "entropy_defer_threshold": engine.config.entropy_defer_threshold,
                "entropy_units": engine.config.entropy_units,
                "thresholds_fitted_on": engine.config.thresholds_fitted_on,
                "expected_performance": dict(engine.config.expected_performance),
            },
            "supports_dicom": False,
            "supports_series": engine.config.series_tumor_threshold is not None,
            "disclaimer": decision.DISCLAIMER_FULL,
            "disclaimer_short": decision.DISCLAIMER_SHORT,
            "threads": engine.threads,
        }

    @app.post("/api/analyze")
    async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:
        engine = get_engine()
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="That file is too large.")
        result = engine.analyze_bytes(
            data, file.filename or "unnamed", with_heatmap=True
        )
        return _result_payload(result)

    @app.post("/api/export")
    async def export(payload: dict[str, Any]) -> Response:
        """Build a self-contained HTML report the clinic can print or file.

        The disclaimer is written into the report by the server, not copied
        from the browser, so it cannot be edited out of an export by anyone
        fiddling with the page.
        """
        html = _build_report(payload)
        return Response(
            content=html,
            media_type="text/html; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="mri-triage-report.html"'},
        )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _build_report(payload: dict[str, Any]) -> str:
    """A printable, offline, self-contained result page.

    Whether the original filename appears is the operator's choice, because a
    report that goes into a patient's own file legitimately needs it, while
    one that goes anywhere else does not. The default is to leave it out.
    """
    include_name = bool(payload.get("include_filename"))
    name = _escape(payload.get("display_name")) if include_name else "(filename withheld)"

    images = ""
    for label, key in (("Original", "original_png"), ("Where the model looked", "overlay_png")):
        uri = payload.get(key)
        if isinstance(uri, str) and uri.startswith("data:image/png;base64,"):
            images += (
                f'<figure><img src="{_escape(uri)}" alt="{_escape(label)}">'
                f"<figcaption>{_escape(label)}</figcaption></figure>"
            )

    # Session D's wording, shown verbatim. Falls back to D's own published
    # text if the exported payload predates the field.
    caveat = payload.get("heatmap_caveat") or explain_adapter.FALLBACK_CAVEAT
    if images:
        images += f"<p class='caveat'>{_escape(caveat)}</p>"

    notes = "".join(f"<li>{_escape(note)}</li>" for note in payload.get("notes") or [])
    notes_block = f"<ul class='notes'>{notes}</ul>" if notes else ""

    tumor_type = payload.get("tumor_type")
    type_block = (
        f"<p class='secondary'><span>Most likely type, secondary information only:</span> "
        f"{_escape(tumor_type)}</p>"
        if tumor_type
        else ""
    )

    probability = payload.get("probability_text")
    probability_block = f"<p class='prob'>{_escape(probability)}</p>" if probability else ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Brain MRI triage result</title>
<style>
 body {{ font-family: "Segoe UI", system-ui, sans-serif; margin: 2rem; color: #111; max-width: 50rem; }}
 .call {{ font-size: 2rem; font-weight: 700; padding: 1rem; border-radius: .5rem;
          background: #f0f0f0; border-left: .5rem solid #666; }}
 .call.tumor {{ background: #fdecea; border-color: #b3261e; }}
 .call.no_tumor {{ background: #e8f5e9; border-color: #2e7d32; }}
 .call.uncertain {{ background: #fff4e5; border-color: #e08600; }}
 .call.cannot_read {{ background: #eceff1; border-color: #546e7a; }}
 .conf {{ font-size: 1.15rem; margin-top: .5rem; }}
 .secondary span {{ color: #555; font-size: .9rem; }}
 figure {{ display: inline-block; margin: 0 1rem 1rem 0; }}
 img {{ width: 20rem; max-width: 100%; image-rendering: pixelated; border: 1px solid #ccc; }}
 figcaption {{ font-size: .85rem; color: #555; text-align: center; }}
 .disclaimer {{ margin-top: 2rem; padding: 1rem; border: 2px solid #b3261e;
                border-radius: .5rem; background: #fff8f8; white-space: pre-wrap; }}
 .meta {{ margin-top: 1.5rem; font-size: .8rem; color: #555; }}
 .meta td {{ padding: .15rem .75rem .15rem 0; vertical-align: top; }}
 .notes li {{ color: #7a4600; }}
 .caveat {{ font-size: .9rem; color: #444; max-width: 42rem; }}
</style></head><body>
<h1>Brain MRI triage result</h1>
<div class="call {_escape(payload.get('call_key'))}">{_escape(payload.get('call'))}</div>
<p class="conf"><strong>{_escape(payload.get('confidence'))}</strong></p>
{probability_block}
<p>{_escape(payload.get('reason'))}</p>
<p><strong>What to do next:</strong> {_escape(payload.get('next_step'))}</p>
{type_block}
{notes_block}
<h2>Images</h2>
{images or "<p>No images available for this result.</p>"}
<div class="disclaimer"><strong>Read this before acting on the result above.</strong>
{_escape(payload.get('disclaimer') or decision.DISCLAIMER_FULL)}</div>
<table class="meta">
 <tr><td>Image file</td><td>{name}</td></tr>
 <tr><td>Image fingerprint (SHA-256)</td><td>{_escape(payload.get('image_sha256'))}</td></tr>
 <tr><td>App version</td><td>{_escape(payload.get('app_version'))}</td></tr>
 <tr><td>Model</td><td>{_escape(payload.get('model_backbone'))},
     seed(s) {_escape(payload.get('model_seeds'))}</td></tr>
 <tr><td>Config version</td><td>{_escape(payload.get('config_version'))}</td></tr>
 <tr><td>Repeated readings</td><td>{_escape(payload.get('mc_passes'))}</td></tr>
 <tr><td>DICOM</td><td>Not supported by this version. Images must be JPEG or PNG.</td></tr>
</table>
</body></html>
"""


# --------------------------------------------------------------------------
# Launch
# --------------------------------------------------------------------------

def find_free_port(preferred: int = 8765, attempts: int = 20) -> int:
    """First free loopback port at or after ``preferred``."""
    for offset in range(attempts):
        candidate = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((LOOPBACK_HOST, candidate))
                return candidate
            except OSError:
                continue
    raise RuntimeError(
        f"Could not find a free port between {preferred} and {preferred + attempts - 1}."
    )


def run(
    host: str = LOOPBACK_HOST,
    port: int | None = None,
    *,
    open_browser: bool = True,
    engine: TriageEngine | None = None,
) -> None:
    """Start the local server. Refuses any non-loopback host."""
    import uvicorn

    assert_loopback(host)

    if engine is None:
        engine = TriageEngine()
    set_engine(engine)

    ok, note = engine.verify_fast_path()
    if not ok:
        log.warning("Fast Monte Carlo path disagreed with the reference path: %s", note)

    port = port or find_free_port()
    url = f"http://{host}:{port}/"

    if open_browser:
        import threading
        import webbrowser

        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    print(f"\n  {version_module.APP_NAME} is running.")
    print(f"  Open this address in your browser: {url}")
    print("  This address works on this laptop only. Nothing is sent over the internet.")
    print("  To stop the app, close this window.\n")

    uvicorn.run(create_app(), host=host, port=port, log_level="warning", access_log=False)


__all__ = ["create_app", "run", "assert_loopback", "set_engine", "get_engine", "BindingRefused"]
