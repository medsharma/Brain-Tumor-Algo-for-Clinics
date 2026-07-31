@echo off
REM ===========================================================================
REM  Brain MRI Triage - install and run
REM
REM  Double-click this file. It installs what the app needs the first time,
REM  then starts the app and opens it in your browser. After the first run it
REM  just starts the app.
REM
REM  It needs Python 3.10 or newer installed. If you do not have it, see
REM  app\README.md.
REM
REM  It needs the internet ONCE, for the first install only. After that the
REM  app never uses the internet again.
REM ===========================================================================

setlocal
cd /d "%~dp0.."

set VENV=%CD%\.venv

echo.
echo   Brain MRI Triage
echo.

python --version >nul 2>&1
if errorlevel 1 (
  echo   Python is not installed, or Windows cannot find it.
  echo.
  echo   Install Python 3.10 or newer from python.org, and tick
  echo   "Add Python to PATH" during the install. Then run this file again.
  echo.
  pause
  exit /b 1
)

if not exist "%VENV%\Scripts\python.exe" (
  echo   First run. Setting up. This takes a few minutes and needs the internet.
  echo   You only have to do this once.
  echo.
  python -m venv "%VENV%" || goto :failed
  "%VENV%\Scripts\python.exe" -m pip install --upgrade pip --quiet || goto :failed
  "%VENV%\Scripts\python.exe" -m pip install -r app\requirements.txt || goto :failed
  echo.
  echo   Setup finished. The app will not need the internet again.
  echo.
)

"%VENV%\Scripts\python.exe" -m app
if errorlevel 1 (
  echo.
  echo   The app stopped. The message above says why.
  echo.
  pause
)
goto :eof

:failed
echo.
echo   Setup failed. Show the messages above to whoever installed the app.
echo.
pause
exit /b 1
