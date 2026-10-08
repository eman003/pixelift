#!/bin/bash
# Development helper: add a launcher entry for this source checkout / venv
# to ~/.local/share/applications (use --remove to undo).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_ID="io.github.pixelift.Pixelift"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
DESKTOP="$DATA/applications/$APP_ID.desktop"
ICON="$DATA/icons/hicolor/scalable/apps/$APP_ID.svg"
if [ "${1:-}" = "--remove" ]; then
    rm -f "$DESKTOP" "$ICON"; echo "Removed launcher entry."; exit 0
fi
EXE="$ROOT/.venv/bin/pixelift"
[ -x "$EXE" ] || EXE="$(command -v pixelift)"
install -Dm644 "$ROOT/pixelift/data/icons/hicolor/scalable/apps/$APP_ID.svg" "$ICON"
mkdir -p "$(dirname "$DESKTOP")"
sed "s|^Exec=pixelift|Exec=$EXE|" "$ROOT/packaging/$APP_ID.desktop" > "$DESKTOP"
update-desktop-database -q "$DATA/applications" 2>/dev/null || true
echo "Installed $DESKTOP (Exec=$EXE)"
