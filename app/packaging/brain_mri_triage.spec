# PyInstaller spec for the offline brain MRI triage app.
#
# Produces a folder that runs on a machine with no Python installed.
#
# Build with:
#     pip install pyinstaller
#     pyinstaller app/packaging/brain_mri_triage.spec --noconfirm
#
# Output lands in dist/BrainMRITriage/.
#
# Two things this spec deliberately does.
#
# 1. It bundles no model weights. A checkpoint is over 200 MB and there are
#    ten of them. They are gitignored, they are not the app's to redistribute,
#    and which one ships is session A's decision recorded in the deployment
#    config. The installer copies them separately. See app/README.md.
#
# 2. It excludes matplotlib, scipy, sklearn, and the rest of the research
#    stack. The app does not import them. Leaving them in would roughly double
#    the bundle for no reason, and a clinic often installs from a USB stick.

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

SPEC_DIR = Path(SPECPATH).resolve()
APP_DIR = SPEC_DIR.parent
REPO_ROOT = APP_DIR.parent

block_cipher = None

hidden_imports = (
    collect_submodules("uvicorn")
    + collect_submodules("fastapi")
    + [
        "app.core.engine",
        "app.core.config",
        "app.core.model",
        "app.core.preprocess",
        "app.core.decision",
        "app.core.audit",
        "app.core.readiness",
        "app.core.validation_adapter",
        "app.core.explain_adapter",
        "multipart",
        "python_multipart",
    ]
)

# The UI, and the config the app reads at startup.
datas = [
    (str(APP_DIR / "static"), "app/static"),
    (str(REPO_ROOT / "analysis" / "results" / "safety"), "analysis/results/safety"),
]

excludes = [
    "matplotlib",
    "scipy",
    "sklearn",
    "pandas",          # only app/tools/ needs it, and tools are not shipped
    "IPython",
    "notebook",
    "jupyter",
    "tkinter",
    "torch.distributions",
    "torchvision.datasets",
    "torchvision.io",
    "test",
    "unittest",
    "pytest",
]

a = Analysis(
    [str(APP_DIR / "packaging" / "entrypoint.py")],
    pathex=[str(REPO_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BrainMRITriage",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX-packed binaries trip antivirus on clinic laptops
    console=True,       # the console is where a startup refusal is readable
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="BrainMRITriage",
)
