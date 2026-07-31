"""File handling, and the things a clinic will actually hand it.

DICOM is the big one. Real scanners produce DICOM, the model was trained on
JPEG, and converting between them involves window level and width plus rescale
slope and intercept. Get those wrong and the image looks completely different
with nothing raising an error. So this version refuses DICOM out loud rather
than converting it badly in silence.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.core import image_loading, paths
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

def _fake_dicom(tmp_path, name="scan.dcm", with_preamble=True):
    path = tmp_path / name
    if with_preamble:
        path.write_bytes(b"\x00" * 128 + b"DICM" + b"\x02\x00" * 100)
    else:
        path.write_bytes(b"DICM" + b"\x02\x00" * 100)
    return path


def test_dicom_is_detected_by_its_magic_bytes(tmp_path, audit_dir):
    path = _fake_dicom(tmp_path)
    with pytest.raises(ImageLoadError) as excinfo:
        image_loading.load_image_path(path, audit_dir)
    assert excinfo.value.kind == "dicom"


def test_dicom_without_a_preamble_is_still_detected(tmp_path, audit_dir):
    """Some scanners emit DICOM with no 128-byte preamble."""
    path = _fake_dicom(tmp_path, name="scan.bin", with_preamble=False)
    with pytest.raises(ImageLoadError) as excinfo:
        image_loading.load_image_path(path, audit_dir)
    assert excinfo.value.kind == "dicom"


def test_a_dcm_extension_alone_is_enough(tmp_path, audit_dir):
    """A mislabelled file still gets the DICOM message, which is more useful."""
    path = tmp_path / "study.dcm"
    path.write_bytes(b"not really dicom either")
    with pytest.raises(ImageLoadError) as excinfo:
        image_loading.load_image_path(path, audit_dir)
    assert excinfo.value.kind == "dicom"


def test_the_dicom_message_tells_the_operator_what_to_do(tmp_path, audit_dir):
    """Not "unsupported file". An actual instruction."""
    path = _fake_dicom(tmp_path)
    with pytest.raises(ImageLoadError) as excinfo:
        image_loading.load_image_path(path, audit_dir)

    message = excinfo.value.message
    assert "does not read DICOM" in message
    assert "window" in message.lower()
    assert "export" in message.lower()
    assert "JPEG or PNG" in message


def test_dicom_refusal_reaches_the_operator_as_the_fourth_call(engine, tmp_path):
    path = _fake_dicom(tmp_path)
    result = engine.analyze_path(path)
    assert result.call_key == "cannot_read"
    assert "DICOM" in result.reason


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
