"""Reading image files, and refusing the ones we should not read.

The model was trained on 8-bit JPEG and PNG brain MRI slices. Anything else
either needs conversion decisions we are not qualified to make silently, or is
not an image at all.

DICOM is read, as of 2026-08-01, because every clinic this tool is for produces
it and the previous answer -- open a viewer, choose a window, export a JPEG --
put the most consequential decision in the conversion into the hands of whoever
was standing at the laptop. The conversion is in ``dicom_loading`` and it
explains itself on screen every time. If DICOM support is missing from a build,
the old refusal comes back with the old instructions, which still work.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from . import dicom_loading, hashing

#: Pillow's decompression-bomb ceiling. A brain MRI slice is well under a
#: megapixel; anything near this is not a scan.
MAX_PIXELS = 64_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

#: Refuse before decoding. 64 MB is far above any single MRI slice.
MAX_FILE_BYTES = 64 * 1024 * 1024

SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
DICOM_EXTENSIONS = frozenset({".dcm", ".dicom", ".ima"})

_DICOM_MAGIC = b"DICM"
_DICOM_MAGIC_OFFSET = 128


class ImageLoadError(ValueError):
    """The file cannot be read as a supported brain MRI image.

    ``message`` is written for a clinic worker, not a developer. It surfaces
    directly in the UI.
    """

    def __init__(self, message: str, *, kind: str = "unreadable") -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind


@dataclass(frozen=True)
class LoadedImage:
    """A decoded image plus the identifiers the audit trail needs.

    ``display_name`` stays in memory for the operator's benefit and is never
    written to disk. ``filename_hash`` is what gets logged.
    """

    image: Image.Image
    sha256: str
    display_name: str
    filename_hash: str
    extension: str
    width: int
    height: int
    mode: str
    file_bytes: int

    #: "image" for a file that was already a picture, "dicom" for one this app
    #: converted. Everything published about this tool's accuracy was measured
    #: on the first kind.
    source_format: str = "image"

    #: What the operator needs to know about how the picture was made. Empty
    #: for an ordinary JPEG, where there is nothing to say.
    conversion_notes: tuple[str, ...] = ()


def looks_like_dicom(data: bytes, filename: str = "") -> bool:
    """True for DICOM by magic bytes or by extension.

    The magic check is the reliable one. Extension is a fallback for the
    preamble-less files some scanners emit.
    """
    if Path(filename).suffix.lower() in DICOM_EXTENSIONS:
        return True
    if len(data) > _DICOM_MAGIC_OFFSET + 4:
        if data[_DICOM_MAGIC_OFFSET:_DICOM_MAGIC_OFFSET + 4] == _DICOM_MAGIC:
            return True
    return data[:4] == _DICOM_MAGIC


#: Kept for builds without DICOM support, and as the wording every other refusal
#: in this file is measured against. See ``dicom_loading.UNAVAILABLE_MESSAGE``.
DICOM_REFUSAL = dicom_loading.UNAVAILABLE_MESSAGE


def load_image_bytes(data: bytes, filename: str, audit_dir: str | Path) -> LoadedImage:
    """Decode raw file bytes into a ``LoadedImage``, or refuse with a reason.

    Args:
        data: The file contents.
        filename: Original name. Used for the extension and the salted hash.
            Never stored in the clear.
        audit_dir: Where the per-install filename-hash key lives.
    """
    if not data:
        raise ImageLoadError("That file is empty.", kind="empty")

    if len(data) > MAX_FILE_BYTES:
        raise ImageLoadError(
            f"That file is {len(data) / 1e6:.0f} MB, which is far larger than a "
            f"single MRI slice. Check you picked the right file.",
            kind="too_large",
        )

    extension = hashing.safe_extension(filename)

    if looks_like_dicom(data, filename):
        return _load_dicom(data, filename, extension, audit_dir)

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except UnidentifiedImageError as exc:
        raise ImageLoadError(
            "That file is not an image the app can read. Supported types are "
            "JPEG, PNG, BMP and TIFF.",
            kind="not_an_image",
        ) from exc
    except Image.DecompressionBombError as exc:
        raise ImageLoadError(
            "That image is unreasonably large and was not opened.",
            kind="too_large",
        ) from exc
    except OSError as exc:
        raise ImageLoadError(
            f"That image file is damaged or incomplete and could not be opened. ({exc})",
            kind="corrupt",
        ) from exc

    if image.width < 32 or image.height < 32:
        raise ImageLoadError(
            f"That image is only {image.width} by {image.height} pixels. That is "
            f"too small to be a usable MRI slice.",
            kind="too_small",
        )

    # Multi-frame TIFF or animated GIF: we would silently judge frame zero.
    n_frames = getattr(image, "n_frames", 1)
    if n_frames > 1:
        raise ImageLoadError(
            f"That file holds {n_frames} frames. The app judges one slice at a "
            f"time and will not silently pick one for you. Export the slice you "
            f"want as a single-frame JPEG or PNG.",
            kind="multiframe",
        )

    return LoadedImage(
        image=image,
        sha256=hashing.sha256_bytes(data),
        display_name=Path(filename).name,
        filename_hash=hashing.hash_filename(filename, audit_dir),
        extension=extension,
        width=image.width,
        height=image.height,
        mode=image.mode,
        file_bytes=len(data),
    )


def _load_dicom(data: bytes, filename: str, extension: str,
                audit_dir: str | Path) -> LoadedImage:
    """Convert a DICOM, or refuse it the way this app always used to.

    A build without DICOM support gives the old refusal and the old
    instructions, which still work. Anything else would leave a clinic reading
    "no module named pydicom" and guessing.
    """
    if not dicom_loading.available():
        raise ImageLoadError(dicom_loading.UNAVAILABLE_MESSAGE, kind="dicom")

    try:
        converted = dicom_loading.to_image(data)
    except dicom_loading.DicomError as exc:
        raise ImageLoadError(exc.message, kind=exc.kind) from exc

    image = converted.image
    if image.width < 32 or image.height < 32:
        raise ImageLoadError(
            f"That DICOM's image is only {image.width} by {image.height} "
            f"pixels. That is too small to be a usable MRI slice.",
            kind="too_small",
        )

    return LoadedImage(
        image=image,
        sha256=hashing.sha256_bytes(data),
        display_name=Path(filename).name,
        filename_hash=hashing.hash_filename(filename, audit_dir),
        extension=extension or ".dcm",
        width=image.width,
        height=image.height,
        mode=image.mode,
        file_bytes=len(data),
        source_format="dicom",
        # The unmeasured-path warning comes first. It is the one that changes
        # how much weight to put on the answer; how the window was chosen is
        # the detail behind it.
        conversion_notes=(dicom_loading.UNMEASURED_PATH_NOTE, converted.note),
    )


def load_image_path(path: str | Path, audit_dir: str | Path) -> LoadedImage:
    """Same as :func:`load_image_bytes`, reading from disk."""
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"No file at {path.name}.", kind="missing")
    return load_image_bytes(path.read_bytes(), path.name, audit_dir)


def is_supported_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS
