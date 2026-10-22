@echo off
cd /d "%~dp0"
if not exist .venv (
  echo Premiere installation, patiente une minute...
  py -3 -m venv .venv 2>nul || python -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip >nul
  python -m pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
start "" pythonw phaselab.py
