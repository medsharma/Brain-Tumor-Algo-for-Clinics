"""File handling, and the things a clinic will actually hand it.

DICOM is the big one. Every scanner in a clinic produces it, and the model was
trained on already-windowed JPEG. Turning one into the other needs the rescale
slope and intercept plus a window level and width, and getting those wrong
produces a picture that looks completely different with nothing raising an
error.

The app used to refuse DICOM and tell the operator to export a JPEG from a
viewer. That did not remove the decision, it moved it to whoever was standing at
the laptop, which in a clinic with no radiologist is the person least equipped
to make it. So the app converts DICOM itself, prefers the window stored in the
file, and says on screen how the picture was made.

These tests hold that: the conversion is right, the fallbacks are loud, and a
file the app cannot convert is refused in words rather than a traceback.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.core import dicom_loading, image_loading, paths
from app.core.image_loading import ImageLoadError


@pytest.fixture
def audit_dir():
    return paths.audit_dir()


def _grey_png(tmp_path, name="scan.png", size=(256, 256)):
    array = (np.random.default_rng(1).random(size[::-1]) * 255).astype("uint8")
    path = tmp_path / name
    Image.fromarray(array).save(path)
    return path


# ------------------------------------------------------------------- DICOM

pydicom = pytest.importorskip("pydicom", reason="DICOM support is optional")


def _dicom_bytes(rows=224, cols=224, *, centre=None, width=None,
                 photometric="MONOCHROME2", slope=1.0, intercept=0.0, frames=1,
                 flat=False):
    """A synthetic single-slice MR DICOM. Not a real scan, a real container."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    y, x = np.ogrid[:rows, :cols]
    radius = (((x - cols / 2) / (cols * 0.36)) ** 2
              + ((y - rows / 2) / (rows * 0.42)) ** 2)
    # A gradient, not a flat disc. Two grey levels survive any linear transform
    # once the result is normalised, so a flat phantom cannot tell a correct
    # conversion from one that ignored the rescale entirely.
    inside = np.clip(1.0 - radius, 0.0, 1.0)
    array = np.full((rows, cols), 500, dtype=np.uint16) if flat else (
        (inside * 900 + 60).astype(np.uint16))

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "MR"
    ds.Rows, ds.Columns = rows, cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope, ds.RescaleIntercept = slope, intercept
    if centre is not None:
        ds.WindowCenter = centre
    if width is not None:
        ds.WindowWidth = width
    if frames > 1:
        ds.NumberOfFrames = frames
    ds.PixelData = array.tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def _write(tmp_path, data, name="slice.dcm"):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_a_dicom_is_converted_rather_than_refused(tmp_path, audit_dir):
    loaded = image_loading.load_image_path(
        _write(tmp_path, _dicom_bytes(centre=500, width=900)), audit_dir)

    assert loaded.source_format == "dicom"
    assert loaded.mode == "L", "the model reads greyscale"
    assert (loaded.width, loaded.height) == (224, 224)


def test_the_window_in_the_file_is_used_and_named(tmp_path, audit_dir):
    """The file's own window is the radiographer's choice. Prefer it, say so."""
    loaded = image_loading.load_image_path(
        _write(tmp_path, _dicom_bytes(centre=500, width=900)), audit_dir)

    notes = " ".join(loaded.conversion_notes)
    assert "window stored in the file" in notes
    assert "500" in notes and "900" in notes


def test_a_file_with_no_window_says_the_picture_was_guessed(tmp_path, audit_dir):
    """A silent fallback is the failure this whole path is trying to avoid."""
    loaded = image_loading.load_image_path(
        _write(tmp_path, _dicom_bytes()), audit_dir)

    notes = " ".join(loaded.conversion_notes)
    assert "no window setting" in notes
    assert "guess" in notes
    assert "check the image" in notes.lower()


def test_every_dicom_result_says_the_figures_were_not_measured_on_it(
        tmp_path, audit_dir):
    """The published miss rate came from exported JPEG. This is another picture."""
    loaded = image_loading.load_image_path(
        _write(tmp_path, _dicom_bytes(centre=500, width=900)), audit_dir)

    assert dicom_loading.UNMEASURED_PATH_NOTE in loaded.conversion_notes
    assert loaded.conversion_notes[0] == dicom_loading.UNMEASURED_PATH_NOTE, (
        "the warning that changes how much to trust the answer goes first")


def test_monochrome1_is_inverted(tmp_path, audit_dir):
    """Left alone this produces a photographic negative of a brain."""
    normal = image_loading.load_image_path(
        _write(tmp_path, _dicom_bytes(centre=500, width=900), "a.dcm"), audit_dir)
    inverted = image_loading.load_image_path(
        _write(tmp_path, _dicom_bytes(centre=500, width=900,
                                      photometric="MONOCHROME1"), "b.dcm"), audit_dir)

    a = np.asarray(normal.image, dtype=int)
    b = np.asarray(inverted.image, dtype=int)
    assert abs((a + b) - 255).max() <= 1, "MONOCHROME1 was not inverted"
    assert "inverted" in " ".join(inverted.conversion_notes)


def test_rescale_slope_and_intercept_change_the_picture(tmp_path, audit_dir):
    """Skipping these is the classic DICOM error and it looks like nothing."""
    plain = image_loading.load_image_path(
        _write(tmp_path, _dicom_bytes(centre=500, width=900), "a.dcm"), audit_dir)
    rescaled = image_loading.load_image_path(
        _write(tmp_path, _dicom_bytes(centre=500, width=900,
                                      slope=2.0, intercept=-200), "b.dcm"), audit_dir)

    assert not np.array_equal(np.asarray(plain.image), np.asarray(rescaled.image))


def test_a_multiframe_dicom_is_refused_rather_than_silently_sliced(
        tmp_path, audit_dir):
    with pytest.raises(ImageLoadError) as excinfo:
        image_loading.load_image_path(
            _write(tmp_path, _dicom_bytes(frames=4)), audit_dir)

    assert excinfo.value.kind == "dicom_multiframe"
    assert "4 frames" in excinfo.value.message


def test_a_flat_dicom_with_no_picture_in_it_is_refused(tmp_path, audit_dir):
    with pytest.raises(ImageLoadError) as excinfo:
        image_loading.load_image_path(
            _write(tmp_path, _dicom_bytes(flat=True)), audit_dir)

    assert excinfo.value.kind == "dicom_blank"


def test_something_claiming_to_be_dicom_is_refused_in_words(tmp_path, audit_dir):
    """A traceback in the result box helps nobody at a clinic laptop."""
    path = tmp_path / "study.dcm"
    path.write_bytes(b"not really dicom at all")

    with pytest.raises(ImageLoadError) as excinfo:
        image_loading.load_image_path(path, audit_dir)

    message = excinfo.value.message
    assert excinfo.value.kind.startswith("dicom")
    assert "Traceback" not in message
    assert message[0].isupper()


def test_dicom_is_detected_without_the_preamble(tmp_path, audit_dir):
    """Some scanners emit DICOM with no 128-byte preamble."""
    assert image_loading.looks_like_dicom(b"DICM" + b"\x02\x00" * 100, "scan.bin")
    assert image_loading.looks_like_dicom(
        b"\x00" * 128 + b"DICM" + b"\x02\x00" * 100, "scan.bin")


def test_a_build_without_dicom_support_gives_the_old_instructions(
        tmp_path, audit_dir, monkeypatch):
    """Degrade to the route that always worked, not to a missing-module error."""
    monkeypatch.setattr(dicom_loading, "available", lambda: False)

    with pytest.raises(ImageLoadError) as excinfo:
        image_loading.load_image_path(
            _write(tmp_path, _dicom_bytes(centre=500, width=900)), audit_dir)

    message = excinfo.value.message
    assert excinfo.value.kind == "dicom"
    assert "export" in message.lower()
    assert "JPEG or PNG" in message
    assert "pydicom" not in message.lower()


def test_a_converted_dicom_reaches_the_operator_as_a_real_call(engine, tmp_path):
    """End to end: it produces a call, and the caveats travel with it."""
    result = engine.analyze_path(
        _write(tmp_path, _dicom_bytes(centre=500, width=900)))

    assert result.call_key in ("tumor", "no_tumor", "uncertain", "cannot_read")
    assert result.source_format == "dicom"
    assert any(dicom_loading.UNMEASURED_PATH_NOTE == note for note in result.notes)


# ------------------------------------------------------------ other refusals

def test_an_empty_file_is_refused(tmp_path, audit_dir):
    path = tmp_path / "empty.png"
    path.write_bytes(b"")
    with pytest.raises(ImageLoadError) as excinfo:
        image_loading.load_image_path(path, audit_dir)
    assert excinfo.value.kind == "empty"


def test_a_truncated_image_is_refused(tmp_path, audit_dir):
    source = _grey_png(tmp_path)
    data = source.read_bytes()
    broken = tmp_path / "broken.png"
    broken.write_bytes(data[: len(data) // 2])

    with pytest.raises(ImageLoadError):
        image_loading.load_image_path(broken, audit_dir)


def test_a_tiny_image_is_refused(tmp_path, audit_dir):
    path = _grey_png(tmp_path, "tiny.png", size=(16, 16))
    with pytest.raises(ImageLoadError) as excinfo:
        image_loading.load_image_path(path, audit_dir)
    assert excinfo.value.kind == "too_small"


def test_a_multiframe_file_is_refused_rather_than_silently_using_frame_zero(
    tmp_path, audit_dir
):
    """A multi-page TIFF is a study, and the app must not pick a slice for you."""
    frames = [
        Image.fromarray((np.random.default_rng(i).random((128, 128)) * 255).astype("uint8"))
        for i in range(4)
    ]
    path = tmp_path / "stack.tif"
    frames[0].save(path, save_all=True, append_images=frames[1:])

    with pytest.raises(ImageLoadError) as excinfo:
        image_loading.load_image_path(path, audit_dir)
    assert excinfo.value.kind == "multiframe"
    assert "4 frames" in excinfo.value.message


def test_a_plain_text_file_is_refused(tmp_path, audit_dir):
    path = tmp_path / "notes.txt"
    path.write_text("nothing to see", encoding="utf-8")
    with pytest.raises(ImageLoadError) as excinfo:
        image_loading.load_image_path(path, audit_dir)
    assert excinfo.value.kind == "not_an_image"


# ------------------------------------------------------------------ success

def test_a_normal_greyscale_png_loads(tmp_path, audit_dir):
    path = _grey_png(tmp_path)
    loaded = image_loading.load_image_path(path, audit_dir)

    assert loaded.width == 256 and loaded.height == 256
    assert len(loaded.sha256) == 64
    assert loaded.display_name == "scan.png"
    assert loaded.extension == ".png"


def test_a_real_brisc_jpeg_loads(brisc_glioma, audit_dir):
    loaded = image_loading.load_image_path(brisc_glioma[0], audit_dir)
    assert loaded.width > 0 and loaded.height > 0
    assert loaded.mode in {"L", "RGB"}


def test_the_content_hash_tracks_content_not_name(tmp_path, audit_dir):
    import shutil

    first = _grey_png(tmp_path, "a.png")
    second = tmp_path / "b.png"
    shutil.copy(first, second)

    assert (
        image_loading.load_image_path(first, audit_dir).sha256
        == image_loading.load_image_path(second, audit_dir).sha256
    )


def test_a_long_dotted_filename_does_not_leak_a_suffix():
    """``patient.john.smith`` must not turn into an extension in the log."""
    from app.core import hashing

    assert hashing.safe_extension("patient.john.smith") == ""
    assert hashing.safe_extension("scan.jpg") == ".jpg"
    assert hashing.safe_extension("scan.jpeg") == ".jpeg"
