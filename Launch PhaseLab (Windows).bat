@echo off
cd /d "%~dp0"
if not exist .venv (
  echo First install, please wait a minute...
  py -3 -m venv .venv 2>nul || python -m venv .venv
)
call .venv\Scripts\activate.bat
python -c "import numpy, scipy, soundfile, PySide6, pyqtgraph, sounddevice" 2>nul || (
  echo Installing components...
  python -m pip install --upgrade pip >nul
  python -m pip install -r requirements.txt
)
start "" pythonw phaselab.py
