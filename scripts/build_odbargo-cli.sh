#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
APP_NAME="odbargo-cli"

mkdir -p "$DIST_DIR/mac_cli" "$DIST_DIR/linux_cli" "$DIST_DIR/win_cli"

case "$(uname -s)" in
  Darwin)
    pyinstaller "$ROOT_DIR/cli.spec"
    mv -f "$DIST_DIR/$APP_NAME" "$DIST_DIR/mac_cli/$APP_NAME"
    ;;
  Linux)
    pyinstaller "$ROOT_DIR/cli_entry.py" \
      --name "$APP_NAME" \
      --onefile \
      --noconfirm \
      --clean \
      --log-level=WARN \
      --hidden-import websockets
    mv -f "$DIST_DIR/$APP_NAME" "$DIST_DIR/linux_cli/$APP_NAME"
    ;;
  MINGW*|MSYS*|CYGWIN*)
    pyinstaller "$ROOT_DIR/cli_entry.py" \
      --name "$APP_NAME.exe" \
      --onefile \
      --noconfirm \
      --clean \
      --log-level=WARN \
      --hidden-import websockets \
      --icon "$ROOT_DIR/icon.ico" \
      --version-file "$ROOT_DIR/version.txt"
    mv -f "$DIST_DIR/$APP_NAME.exe" "$DIST_DIR/win_cli/$APP_NAME.exe"
    ;;
  *)
    echo "Unsupported OS for build_odbargo-cli.sh"
    exit 1
    ;;
esac

# Optional alternative build (Nuitka)
# python -m nuitka --onefile --output-filename="$APP_NAME" --include-module=websockets "$ROOT_DIR/cli_entry.py"
