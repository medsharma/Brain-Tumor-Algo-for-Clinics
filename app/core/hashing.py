"""Hashing, with patient identifiers in mind.

Two different jobs, and they need different treatment.

*Image content* is hashed with plain SHA-256. That hash is the audit trail's
join key: it lets someone later prove which exact pixels produced which call.
It leaks nothing, because you cannot reconstruct a scan from its hash.

*Filenames* are a different matter. Clinics name files things like
``SMITH_JOHN_1962-04-11_axial.jpg``. A plain hash of that is technically
reversible: anyone with a list of candidate names can hash them all and look
for matches. So filenames get an HMAC under a random per-install key that
never leaves the laptop. Deterministic within one clinic, so repeat scans of
the same file line up in the log. Useless to anyone who gets hold of the log
alone.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import stat
from pathlib import Path

_CHUNK = 1024 * 1024

#: Truncated to 16 hex characters. Collision risk is negligible at clinic
#: volumes and it keeps the log readable.
FILENAME_HASH_LENGTH = 16

_INSTALL_KEY_FILENAME = "install_key.bin"
_install_key_cache: bytes | None = None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_or_create_install_key(key_dir: Path) -> bytes:
    """Read the per-install HMAC key, generating it on first run.

    Stored beside the audit log, owner-readable only where the filesystem
    supports it. If this file is lost, old filename hashes stop matching new
    ones. Nothing else breaks, and no clinical result depends on it.
    """
    key_path = key_dir / _INSTALL_KEY_FILENAME
    if key_path.is_file():
        key = key_path.read_bytes()
        if len(key) >= 32:
            return key

    key_dir.mkdir(parents=True, exist_ok=True)
    key = secrets.token_bytes(32)
    key_path.write_bytes(key)
    try:
        os.chmod(key_path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        # Windows filesystems may refuse. The key is still local-only.
        pass
    return key


def get_install_key(key_dir: str | Path) -> bytes:
    global _install_key_cache
    if _install_key_cache is None:
        _install_key_cache = _load_or_create_install_key(Path(key_dir))
    return _install_key_cache


def reset_install_key_cache() -> None:
    """Test hook. Forces the next call to re-read from disk."""
    global _install_key_cache
    _install_key_cache = None


def hash_filename(filename: str, key_dir: str | Path) -> str:
    """One-way, salted, install-local hash of a filename.

    Use this anywhere a filename would otherwise be written to disk or into an
    export. Never log the raw name.
    """
    key = get_install_key(key_dir)
    digest = hmac.new(key, filename.encode("utf-8", errors="replace"), hashlib.sha256)
    return digest.hexdigest()[:FILENAME_HASH_LENGTH]


#: The only suffixes allowed into the audit log. An allowlist rather than a
#: length rule, because ``patient.john.smith`` has a short final segment and
#: would otherwise be logged as the extension ``.smith``.
LOGGABLE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
    ".dcm", ".dicom", ".ima",   # refused, but worth recording that it was tried
    ".txt", ".pdf", ".zip",     # common operator mistakes
})


def safe_extension(filename: str) -> str:
    """The file extension, if it is one we recognise. Otherwise nothing.

    Kept in the log because it is diagnostically useful and carries no
    identity. An allowlist is used rather than a length limit: a name like
    ``patient.john.smith`` would pass a length check and put a surname in the
    audit trail.
    """
    suffix = Path(filename).suffix.lower()
    return suffix if suffix in LOGGABLE_EXTENSIONS else ""
