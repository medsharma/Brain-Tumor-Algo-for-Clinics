@echo off
REM Build the standalone Windows app. Run this once, on a build machine with
REM Python installed. The result runs on clinic laptops that have no Python.
REM
REM Usage, from the repository root:
REM     app\packaging\build_windows.bat

setlocal
cd /d "%~dp0..\.."

echo.
echo === Build environment ===
REM Built in a dedicated CPU-only virtualenv, and that is not optional.
REM
REM The app never touches a GPU: it loads with map_location="cpu" and never
REM moves a tensor to a device. But PyInstaller bundles whatever torch it finds,
REM so building on a machine with CUDA torch installed ships the CUDA libraries
REM anyway. cublasLt64 is 456 MB, torch_cuda.dll is 401 MB, cufft is 272 MB.
REM That was 2.3 GB of a 4.8 GB download, for code that cannot execute.
REM
REM Building here instead takes the folder from 4.8 GB to 2.4 GB.
if not exist .buildenv-cpu (
    echo Creating .buildenv-cpu
    python -m venv .buildenv-cpu || goto :failed
)
REM Every step below runs through %PY%, never through plain `python`.
REM
REM That is not tidiness. This script used to create the CPU virtualenv, install
REM CPU torch into it, and then run PyInstaller with whatever `python` happened
REM to be on PATH. On a machine with CUDA torch installed system-wide, that
REM bundles the CUDA build regardless of what is sitting in .buildenv-cpu, and
REM the warning above becomes a comment describing a thing the script does not
REM do. It shipped a 4.8 GB folder that way, with 2.4 GB of GPU libraries the
REM app cannot reach, on a download aimed at clinics with poor connections.
set PY=.buildenv-cpu\Scripts\python.exe
%PY% -m pip install --upgrade pip || goto :failed
%PY% -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || goto :failed
%PY% -m pip install numpy pillow fastapi uvicorn python-multipart scipy scikit-image opencv-python-headless pydicom pyinstaller || goto :failed

REM Build-time only. None of this is bundled: matplotlib, sklearn and pandas are
REM all in the `excludes` list in brain_mri_triage.spec, and pytest is too.
REM
REM   pytest, httpx2  - the test suite. httpx2 is what starlette's TestClient
REM                     imports; without it the three files covering the HTTP
REM                     layer fail to collect, which reads as "the tests are
REM                     broken" rather than "a dependency is missing".
REM   matplotlib,     - prepare_clinic_install.py imports src\code.py to check
REM   pandas,           the stripped checkpoints still predict identically, and
REM   scikit-learn      code.py pulls in the whole research stack at import.
%PY% -m pip install pytest httpx2 matplotlib pandas scikit-learn || goto :failed

echo.
echo === Checking this really is the CPU build ===
REM Cheap, and it catches the whole class of mistake above before spending
REM twenty minutes building the wrong thing.
%PY% -c "import torch, sys; v = torch.__version__; print('  torch', v); sys.exit(0 if 'cu' not in v.split('+')[-1] else 1)" || goto :wrong_torch

echo.
echo === Running the test suite ===
echo A build that fails its tests must not reach a clinic.
%PY% -m pytest app\tests -q || goto :tests_failed

echo.
echo === Building ===
%PY% -m PyInstaller app\packaging\brain_mri_triage.spec --noconfirm || goto :failed

echo.
echo === Adding the model files ===
REM Copies the checkpoints in, strips the training leftovers out of them,
REM verifies the stripped copies still give bit-identical answers, and rewrites
REM the settings file to use relative paths.
%PY% app\tools\prepare_clinic_install.py || goto :failed

echo.
echo === Smoke testing the BUILT package ===
echo The source tests cannot see packaging bugs. This runs the real .exe on
echo real scans. Two bugs got through here before this step existed: a build
echo that refused to start, and a build that rejected every brain MRI it saw.
%PY% app\tools\smoke_test_package.py || goto :smoke_failed

echo.
echo === Packing it into one downloadable file ===
%PY% app\tools\pack_release.py || goto :failed

echo.
echo === Done ===
echo The app is in:  dist\BrainMRITriage\
echo The zip is in:  release\BrainMRITriage-windows.zip
echo.
echo Copy either one to the clinic laptop. From the zip: extract it and
echo double-click BrainMRITriage.exe. No Python needed on that machine.
echo About 2.4 GB. A 4 GB USB stick is enough.
goto :eof

:wrong_torch
echo.
echo BUILD STOPPED. .buildenv-cpu has a CUDA build of torch in it.
echo.
echo Building from it would bundle about 2.4 GB of GPU libraries that this app
echo can never use, on a download meant for clinics with slow connections.
echo.
echo Delete the .buildenv-cpu folder and run this script again. It will rebuild
echo the environment from the CPU-only wheel index.
exit /b 1

:smoke_failed
echo.
echo BUILD STOPPED. The package was built but it does not work.
echo Read the failures above. Do NOT copy dist\BrainMRITriage to a clinic.
exit /b 1

:tests_failed
echo.
echo BUILD STOPPED. The tests did not pass, so nothing was built.
exit /b 1

:failed
echo.
echo BUILD FAILED.
exit /b 1
