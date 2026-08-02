"""Reading the file a scanner actually produces.

Every clinic this tool is for produces DICOM. Until now the app refused it and
told the operator to open the study in a viewer, pick a window, and export a
JPEG. That instruction is the single biggest thing standing between this tool
and a clinic being able to use it, and it is worse than an inconvenience: **the
window the operator picks changes the picture, and a changed picture can change
the call.** In a clinic with no radiologist, the person doing that is the person
least equipped to choose.

So this converts DICOM, and the whole design is about not making that choice
silently.

How the picture is made
-----------------------
1. **Rescale.** ``RescaleSlope`` and ``RescaleIntercept`` turn stored integers
   into the scanner's real values. Skipping this is the classic error and it
   changes nothing visibly, which is what makes it dangerous.
2. **Window, from the file wherever possible.** If the file carries a VOI LUT or
   a ``WindowCenter``/``WindowWidth``, that is the window a radiographer or the
   scanner already chose, and it is exactly the number the old instructions
   asked a human to reproduce by hand. Using it removes a guess rather than
   adding one.
3. **Only if the file has no window**, fall back to a 1st-to-99th percentile
   stretch over the pixels, and say so in words that reach the screen. A
   fallback that happens quietly is the failure this whole module is trying to
   avoid.
4. **MONOCHROME1 is inverted**, because in that photometric interpretation low
   values are white. Getting this wrong produces a photographic negative of a
   brain, which the model has never seen and will happily judge anyway.

What this does not do
---------------------
**It does not make DICOM a measured input.** Every number this project
publishes was measured on already-windowed 8-bit images from public datasets.
A DICOM converted here is a different picture from the JPEG the same study would
have produced in a viewer, by an amount nobody has measured. So every result
from this path carries a note saying so, on screen and in the exported sheet,
and the audit log records the file extension it came from.

**It does not pick a slice for you.** A multi-frame DICOM is refused, exactly as
a multi-frame TIFF is.

**It does not guess at compressed pixel data it cannot decode.** JPEG2000 and
JPEG-LS need codec libraries that are not bundled. If one turns up, the operator
is told to export from their viewer, which is the old route and still works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image

#: Percentile stretch used only when the file carries no window of its own.
FALLBACK_PERCENTILES = (1.0, 99.0)


@dataclass(frozen=True)
class Converted:
    """A DICOM turned into a picture, plus how that was done."""

    image: Image.Image

    #: One sentence for the operator, shown on the result and in the export.
    note: str

    #: Short machine-readable description of where the window came from:
    #: "voi_lut", "window_tag", or "percentile_fallback".
    window_source: str


class DicomError(ValueError):
    """The DICOM could not be turned into a picture. Message is for a clinic."""

    def __init__(self, message: str, *, kind: str = "dicom_unreadable") -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind


def available() -> bool:
    """Is DICOM support present in this build?"""
    try:
        import pydicom  # noqa: F401
    except ImportError:
        return False
    return True


UNAVAILABLE_MESSAGE = (
    "This looks like a DICOM file, and DICOM support is missing from this "
    "build.\n"
    "\n"
    "What to do: export the slice from your viewer as JPEG or PNG, using the "
    "window your radiographer normally uses, then load that file."
)


def _first_number(value: Any) -> Optional[float]:
    """DICOM numbers arrive as scalars, strings, or lists of presets."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) or (
            isinstance(value, Sequence) and not isinstance(value, (str, bytes))):
        for item in value:
            number = _first_number(item)
            if number is not None:
                return number
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _apply_voi(pixels: np.ndarray, dataset: Any) -> tuple[np.ndarray, str, str]:
    """Window the pixels, preferring anything the file itself specifies."""
    try:
        from pydicom.pixels import apply_voi_lut  # pydicom 3
    except ImportError:  # pragma: no cover - older pydicom
        try:
            from pydicom.pixel_data_handlers.util import apply_voi_lut
        except ImportError:
            apply_voi_lut = None  # type: ignore[assignment]

    has_lut = bool(getattr(dataset, "VOILUTSequence", None))
    centre = _first_number(getattr(dataset, "WindowCenter", None))
    width = _first_number(getattr(dataset, "WindowWidth", None))

    if apply_voi_lut is not None and (has_lut or (centre is not None and width)):
        try:
            windowed = apply_voi_lut(pixels, dataset)
        except Exception:  # noqa: BLE001 - a bad LUT must not lose the scan
            windowed = None
        if windowed is not None:
            if has_lut:
                return np.asarray(windowed), "voi_lut", (
                    "Converted from DICOM using the brightness table stored in "
                    "the file.")
            return np.asarray(windowed), "window_tag", (
                f"Converted from DICOM using the window stored in the file "
                f"(centre {centre:g}, width {width:g}).")

    low, high = np.percentile(pixels, FALLBACK_PERCENTILES)
    if high <= low:
        low, high = float(np.min(pixels)), float(np.max(pixels))
    clipped = np.clip(pixels, low, high)
    return clipped, "percentile_fallback", (
        "This DICOM carries no window setting, so the picture was made by "
        "stretching the middle 98% of its values. That is a guess. Check the "
        "image below looks like the scan you expect before trusting the "
        "answer.")


def to_image(data: bytes) -> Converted:
    """Turn DICOM bytes into an 8-bit greyscale picture, or refuse with a reason."""
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - guarded by available()
        raise DicomError(UNAVAILABLE_MESSAGE, kind="dicom_unsupported") from exc

    import io

    try:
        dataset = pydicom.dcmread(io.BytesIO(data), force=True)
    except Exception as exc:  # noqa: BLE001 - any parse failure is the same to a clinic
        raise DicomError(
            f"That DICOM file could not be read. It may be damaged or "
            f"incomplete. ({exc})") from exc

    if "PixelData" not in dataset:
        raise DicomError(
            "That DICOM file holds no image. It may be a report, a structured "
            "document, or the header of a study rather than a slice.",
            kind="dicom_no_pixels")

    frames = int(getattr(dataset, "NumberOfFrames", 1) or 1)
    if frames > 1:
        raise DicomError(
            f"That DICOM holds {frames} frames. This tool judges one slice at a "
            f"time and will not silently pick one for you. Export the slice you "
            f"want, or load the single-slice file.",
            kind="dicom_multiframe")

    try:
        pixels = dataset.pixel_array
    except Exception as exc:  # noqa: BLE001 - missing codecs land here
        raise DicomError(
            "That DICOM stores its image in a compressed form this build "
            "cannot decode.\n"
            "\n"
            "What to do: open the study in your viewer and export the slice as "
            "JPEG or PNG, using the window your radiographer normally uses.\n"
            f"\n(Technical detail: {exc})",
            kind="dicom_compressed") from exc

    if pixels.ndim > 2:
        # Colour DICOM, or a frame axis that survived the check above. A brain
        # MRI slice is greyscale; anything else is not what this reads.
        if pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
            pixels = np.asarray(Image.fromarray(pixels.astype(np.uint8)).convert("L"))
        else:
            raise DicomError(
                f"That DICOM's image has an unexpected shape {pixels.shape}. "
                f"This tool reads a single greyscale slice.",
                kind="dicom_shape")

    pixels = pixels.astype(np.float64)

    # Stored integers are not the scanner's values until this is applied.
    slope = _first_number(getattr(dataset, "RescaleSlope", None))
    intercept = _first_number(getattr(dataset, "RescaleIntercept", None))
    if slope is not None or intercept is not None:
        pixels = pixels * (slope if slope is not None else 1.0) + (
            intercept if intercept is not None else 0.0)

    windowed, window_source, note = _apply_voi(pixels, dataset)
    windowed = np.asarray(windowed, dtype=np.float64)

    low = float(np.min(windowed))
    high = float(np.max(windowed))
    if high <= low:
        raise DicomError(
            "That DICOM's image is a single flat value, with no picture in it.",
            kind="dicom_blank")

    eight_bit = ((windowed - low) / (high - low) * 255.0).round()

    # MONOCHROME1 means low values are white. Left alone it produces a
    # photographic negative of a brain, which the model has never seen and will
    # judge anyway.
    if str(getattr(dataset, "PhotometricInterpretation", "")).strip() == "MONOCHROME1":
        eight_bit = 255.0 - eight_bit
        note += " Brightness was inverted, as the file specifies."

    image = Image.fromarray(eight_bit.clip(0, 255).astype(np.uint8), mode="L")

    return Converted(image=image, note=note, window_source=window_source)


#: Appended to every DICOM-derived result. Not optional and not softened: the
#: published miss rate was measured on exported JPEG, and this is a different
#: picture from the one a viewer would have produced.
UNMEASURED_PATH_NOTE = (
    "This scan was converted from DICOM by the app. Every performance figure "
    "published for this tool was measured on images exported from a viewer, "
    "not on this conversion. Treat the answer with more caution than usual, "
    "and check the picture below looks right."
)
