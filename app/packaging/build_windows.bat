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
set PY=.buildenv-cpu\Scripts\python.exe
%PY% -m pip install --upgrade pip || goto :failed
%PY% -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || goto :failed
%PY% -m pip install numpy pillow fastapi uvicorn python-multipart scipy scikit-image opencv-python-headless pyinstaller || goto :failed

echo.
echo === Running the test suite ===
echo A build that fails its tests must not reach a clinic.
python -m pytest app\tests -q || goto :tests_failed

echo.
echo === Building ===
python -m PyInstaller app\packaging\brain_mri_triage.spec --noconfirm || goto :failed

echo.
echo === Adding the model files ===
REM Copies the checkpoints in, strips the training leftovers out of them,
REM verifies the stripped copies still give bit-identical answers, and rewrites
REM the settings file to use relative paths.
python app\tools\prepare_clinic_install.py || goto :failed

echo.
echo === Smoke testing the BUILT package ===
echo The source tests cannot see packaging bugs. This runs the real .exe on
echo real scans. Two bugs got through here before this step existed: a build
echo that refused to start, and a build that rejected every brain MRI it saw.
python app\tools\smoke_test_package.py || goto :smoke_failed

echo.
echo === Packing it into one downloadable file ===
python app\tools\pack_release.py || goto :failed

echo.
echo === Done ===
echo The app is in:  dist\BrainMRITriage\
echo The zip is in:  release\BrainMRITriage-windows.zip
echo.
echo Copy either one to the clinic laptop. From the zip: extract it and
echo double-click BrainMRITriage.exe. No Python needed on that machine.
echo About 2.4 GB. A 4 GB USB stick is enough.
goto :eof

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
