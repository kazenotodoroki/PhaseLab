#!/bin/bash
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "First install, please wait a minute..."
  python3 -m venv .venv
fi
source .venv/bin/activate
python -c "import numpy, scipy, soundfile, PySide6, pyqtgraph, sounddevice" 2>/dev/null || {
  echo "Installing components..."
  python -m pip install --upgrade pip >/dev/null
  python -m pip install -r requirements.txt
}
python phaselab.py
