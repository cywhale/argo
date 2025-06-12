#!/bin/bash

# for linux version
pyinstaller odbargo-cli.py \
  --name odbargo-cli \
  --onefile \
  --noconfirm \
  --clean \
  --log-level=WARN \
  --hidden-import websockets

# for windows version
pyinstaller odbargo-cli.py --name odbargo-cli.exe --onefile --noconfirm --clean --log-level=WARN --hidden-import websockets --icon=icon.ico --version-file version.txt

# an alternative version that use nuitka
python -m nuitka --onefile --output-filename=odbargo-cli --include-module=websockets odbargo-cli.py

