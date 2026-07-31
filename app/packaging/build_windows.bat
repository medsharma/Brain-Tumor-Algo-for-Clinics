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
echo === Done ===
echo The app is in:  dist\BrainMRITriage\
echo.
echo Before giving this to a clinic you must still:
echo   1. Copy the model file into dist\BrainMRITriage\models\
echo   2. Copy session A's deployment_config.json next to BrainMRITriage.exe
echo      and point its checkpoint path at that models folder.
echo.
echo Without both, the app will refuse to start. That is deliberate.
goto :eof

:tests_failed
echo.
echo BUILD STOPPED. The tests did not pass, so nothing was built.
exit /b 1

:failed
echo.
echo BUILD FAILED.
exit /b 1
