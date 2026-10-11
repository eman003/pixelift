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
[ -x "$EXE" ] || EXE="$(command -v pixelift || true)"
if [ -z "$EXE" ]; then
    echo "pixelift not found: create .venv in $ROOT (see README) or put pixelift on PATH." >&2
    exit 1
fi
case "$EXE" in
    *[\"\`\$\\]*)
        echo "Cannot write a launcher for a path containing \", \`, \$ or \\: $EXE" >&2
        exit 1
        ;;
esac
install -Dm644 "$ROOT/pixelift/data/icons/hicolor/scalable/apps/$APP_ID.svg" "$ICON"
mkdir -p "$(dirname "$DESKTOP")"
# Quote the path (it may contain spaces); copy every other line unchanged.
while IFS= read -r line; do
    case "$line" in
        Exec=pixelift*) line="Exec=\"$EXE\"${line#Exec=pixelift}" ;;
    esac
    printf '%s\n' "$line"
done < "$ROOT/packaging/$APP_ID.desktop" > "$DESKTOP"
update-desktop-database -q "$DATA/applications" 2>/dev/null || true
echo "Installed $DESKTOP (Exec=$EXE)"
