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
        # Session B's input check and session D's heatmap. The app reaches
        # these with importlib.import_module("src.input_validation") and
        # ("src.explain_runtime"), so PyInstaller's static analysis cannot see
        # them and will not collect them unless they are named here.
        #
        # Leaving them out does not produce a broken build. It produces a build
        # that compiles, launches, and then REFUSES TO START, because
        # readiness.check finds both adapters in their stub state and blocks.
        # That is the safety machinery doing its job, and it is also a very
        # confusing way to discover a packaging mistake. `src/` has no
        # __init__.py and is a PEP 420 namespace package, which is exactly the
        # shape PyInstaller is least likely to pick up on its own.
        "src",
        "src.input_validation",
        "src.explain_runtime",
        # explain_runtime imports cv2 inside a function, deliberately, to keep
        # it off the ResNet path's cost. A lazy import is invisible to static
        # analysis, and the shipped backbone is ViT, whose attention-rollout
        # path is exactly the one that needs it.
        "cv2",
        # input_validation imports these inside functions too.
        "scipy.ndimage",
        "skimage.filters",
        "skimage.morphology",
        # DICOM. image_loading imports it lazily through dicom_loading so a
        # build without it degrades to the old refusal instead of failing to
        # start, which also means static analysis cannot see it.
        "pydicom",
        "pydicom.pixels",
        "pydicom.encaps",
        "pydicom.uid",
    ]
)

# The UI, and the config the app reads at startup.
datas = [
    (str(APP_DIR / "static"), "app/static"),
    (str(REPO_ROOT / "analysis" / "results" / "safety"), "analysis/results/safety"),
]

# Session B's fitted rejector config has to travel with the app, and it has to
# land where `src/input_validation.py` looks for it.
#
# That module resolves its own location: `Path(__file__).parent.parent /
# "analysis/results/ood/rejector_config.json"`. Frozen, `__file__` is inside
# `_internal`, so the file must be at `_internal/analysis/results/ood/`. Copying
# it next to the .exe does not work, and fails silently: `load_config` falls
# back to the UNFITTED hand-set thresholds, which reject about 8% of real brain
# MRI including the `fg_solidity` rule that fitting removes entirely. The app
# looks fine and quietly refuses one real scan in twelve.
#
# Only the two small files. `analysis/results/ood/` also holds cached scores and
# feature matrices worth hundreds of megabytes that the app never reads.
_ood = REPO_ROOT / "analysis" / "results" / "ood"
for _name in ("rejector_config.json", "rejector_stats.npz"):
    if (_ood / _name).is_file():
        datas.append((str(_ood / _name), "analysis/results/ood"))

# Excluding the research stack roughly halves the bundle, which matters when a
# clinic installs from a USB stick. Nothing here is imported by the app.
#
# Do NOT add "unittest" or "test" to this list. It looks like free savings and
# is not: torch.utils._config_module imports unittest at module scope, so the
# frozen build dies with ModuleNotFoundError before it reaches any of the
# app's own code. Learned the hard way; the build ran fine and the executable
# did not start.
#
# Likewise leave torch's own submodules alone. torch.utils.data pulls in
# torch.distributed, which pulls in more than is obvious from the import
# graph, and pruning it saves little for real breakage risk.
#
# scipy and scikit-image were on this list when session B's input check was not
# yet wired in, under the comment "nothing here is imported by the app". That
# stopped being true. `src/input_validation.py` uses `scipy.ndimage` and
# `skimage.filters.threshold_otsu` to find the brain-shaped region, and imports
# them inside the function, so nothing static catches it.
#
# The failure mode was nasty. The app started, looked healthy, and rejected
# **100% of real brain MRI** with "the file could not be read as an image
# (No module named 'scipy')". The validator catches any exception and fails
# closed, which is the right instinct and made a missing dependency look like a
# data problem. Do not re-add these without checking what imports them.
#
# Build this in a CPU-only environment. The app never touches a GPU: it loads
# with map_location="cpu" and no tensor is ever moved to a device. Building it
# where CUDA torch is installed bundles the CUDA libraries anyway, and they are
# enormous: cublasLt64 alone is 456 MB, torch_cuda.dll 401 MB, cufft 272 MB.
# That was 2.3 GB of a 4.8 GB folder, downloaded by clinics, for code that
# cannot execute. See app/packaging/build_windows.bat, which builds in a
# dedicated CPU virtualenv.
excludes = [
    "matplotlib",
    "sklearn",
    "pandas",          # only app/tools/ needs it, and tools are not shipped
    # pandas is excluded but pyarrow was still arriving through it and costing
    # 80 MB. imageio and its bundled ffmpeg come in through scikit-image; the
    # app only uses threshold_otsu and convex_hull_image and never decodes a
    # video. Together these were another ~165 MB of a clinic's download.
    "pyarrow",
    "imageio",
    "imageio_ffmpeg",
    "IPython",
    "notebook",
    "jupyter",
    "tkinter",
    "pytest",
    "_pytest",
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
