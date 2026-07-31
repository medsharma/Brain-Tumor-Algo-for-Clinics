"""Shared fixtures.

Tests never write to the operator's real data directory, and never load the
real deployment config unless a test explicitly asks for it.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BRISC_ROOT = Path(r"C:\Users\medha\Downloads\archive (1)\brisc2025\classification_task")

CHECKPOINT_ROOT = Path(
    r"C:\Users\medha\OneDrive\Documents\MRI ALGO\results\20260703_155524"
)


@pytest.fixture(scope="session", autouse=True)
def _isolated_data_dir(tmp_path_factory):
    """Keep the audit log and install key out of the real user directory."""
    target = tmp_path_factory.mktemp("clinic-data")
    os.environ["MRI_CLINIC_DATA_DIR"] = str(target)
    from app.core import hashing

    hashing.reset_install_key_cache()
    yield target


@pytest.fixture(scope="session")
def brisc_available() -> bool:
    return BRISC_ROOT.is_dir()


def _brisc_samples(class_folder: str, count: int = 3) -> list[Path]:
    folder = BRISC_ROOT / "test" / class_folder
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.jpg"))[:count]


@pytest.fixture(scope="session")
def brisc_glioma() -> list[Path]:
    files = _brisc_samples("glioma")
    if not files:
        pytest.skip("BRISC is not present on this machine")
    return files


@pytest.fixture(scope="session")
def brisc_no_tumor() -> list[Path]:
    files = _brisc_samples("no_tumor")
    if not files:
        pytest.skip("BRISC is not present on this machine")
    return files


@pytest.fixture(scope="session")
def brisc_mixed() -> list[Path]:
    """A few images from every class, for parity checks."""
    files: list[Path] = []
    for folder in ("glioma", "meningioma", "pituitary", "no_tumor"):
        files.extend(_brisc_samples(folder, 2))
    if not files:
        pytest.skip("BRISC is not present on this machine")
    return files


@pytest.fixture(scope="session")
def stub_config():
    """The day-one stub, loaded with the guard explicitly overridden."""
    from app.core.config import load_config

    return load_config(REPO_ROOT / "analysis/results/safety/deployment_config.SCHEMA.json",
                       allow_stub=True)


@pytest.fixture(scope="session")
def test_config(stub_config):
    """A workable config for end-to-end tests.

    Temperature 1.0 and threshold 0.5 replace the stub's absurd values so the
    model behaves normally. These are test scaffolding, not fitted operating
    points. Session A owns the real ones, and nothing here is ever written to
    a file that could be mistaken for a deployment config.
    """
    return dataclasses.replace(
        stub_config,
        temperature=1.0,
        tumor_threshold=0.5,
        entropy_defer_threshold=0.8,
    )


@pytest.fixture(scope="session")
def checkpoints_available(stub_config) -> bool:
    return all(Path(c.path).is_file() for c in stub_config.checkpoints)


@pytest.fixture(scope="session")
def engine(test_config, checkpoints_available):
    """A loaded engine. Session-scoped: loading a checkpoint is not cheap."""
    if not checkpoints_available:
        pytest.skip("Model checkpoints are not present on this machine")

    from app.core.engine import TriageEngine

    return TriageEngine(cfg=test_config, enforce_readiness=False)


@pytest.fixture
def synthetic_image(tmp_path) -> Path:
    """A small greyscale square. Used where real data is not required."""
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(0)
    array = (rng.random((256, 256)) * 255).astype("uint8")
    path = tmp_path / "synthetic.png"
    Image.fromarray(array, mode="L").save(path)
    return path
