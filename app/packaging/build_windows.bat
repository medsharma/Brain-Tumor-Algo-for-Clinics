@echo off
REM Build the standalone Windows app. Run this once, on a build machine with
REM Python installed. The result runs on clinic laptops that have no Python.
REM
REM Usage, from the repository root:
REM     app\packaging\build_windows.bat

setlocal
cd /d "%~dp0..\.."

echo.
echo === Installing build dependencies ===
python -m pip install --upgrade pip || goto :failed
python -m pip install -r app\requirements.txt || goto :failed
python -m pip install pyinstaller || goto :failed

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
echo === Done ===
echo The app is in:  dist\BrainMRITriage\
echo Copy that whole folder to the clinic laptop and run BrainMRITriage.exe
echo It needs about 5 GB. Use an 8 GB or larger USB stick.
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
