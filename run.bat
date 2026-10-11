@echo off
rem MedHarness desktop launcher - Windows.
rem First run: creates a private .venv, installs the 5 requirements, starts the app.
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (set "PY=py") else (set "PY=python")

%PY% --version >nul 2>nul
if not %errorlevel%==0 (
  echo [MedHarness] Python not found. Install Python 3.10+ from https://python.org
  echo             and tick "Add Python to PATH" during install.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [MedHarness] First run - creating a private Python environment (.venv)...
  %PY% -m venv .venv
  if not exist ".venv\Scripts\python.exe" (
    echo [MedHarness] Could not create a venv. Reinstall Python 3.10+ and tick "Add to PATH".
    pause
    exit /b 1
  )
)

".venv\Scripts\python.exe" -c "import fastapi, uvicorn, httpx, pydantic, h2" >nul 2>nul
if not %errorlevel%==0 (
  echo [MedHarness] Installing requirements - one time only, about 30 seconds...
  ".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
)

echo [MedHarness] Starting the desktop app at http://127.0.0.1:8787 ...
".venv\Scripts\python.exe" -m app.main
pause
