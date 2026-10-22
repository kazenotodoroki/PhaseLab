#!/bin/bash
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Première installation, patiente une minute..."
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip >/dev/null
  python -m pip install -r requirements.txt
else
  source .venv/bin/activate
fi
python phaselab.py
