#!/bin/bash
# Build build/Pixelift-<version>-x86_64.AppImage.
# The AppImage bundles the app, CPU PyTorch, numpy and Pillow, and uses the
# host's Python 3.12 + GTK4/libadwaita (Ubuntu 24.04 and derivatives).
source "$(dirname "$0")/common.sh"
build_vendor

APPDIR="$BUILD/AppDir"
rm -rf "$APPDIR"
stage_app "$APPDIR/usr/lib/pixelift"
stage_desktop "$APPDIR/usr"
cp "$APPDIR/usr/share/applications/$APP_ID.desktop" "$APPDIR/$APP_ID.desktop"
cp "$APPDIR/usr/share/icons/hicolor/scalable/apps/$APP_ID.svg" "$APPDIR/$APP_ID.svg"
ln -sf "$APP_ID.svg" "$APPDIR/.DirIcon"
cat > "$APPDIR/AppRun" <<'APPRUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
export PIXELIFT_APPDIR="$HERE/usr/lib/pixelift"
exec "$HERE/usr/bin/pixelift" "$@"
APPRUN
chmod 755 "$APPDIR/AppRun"

# A tagged release, verified before it is ever run.
TOOL_VERSION="1.9.1"
TOOL_SHA256="ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0"
TOOL="$BUILD/tools/appimagetool-$TOOL_VERSION-x86_64.AppImage"
if [ ! -x "$TOOL" ]; then
    mkdir -p "$(dirname "$TOOL")"
    curl -fsSL -o "$TOOL.part" \
        "https://github.com/AppImage/appimagetool/releases/download/$TOOL_VERSION/appimagetool-x86_64.AppImage"
    if ! echo "$TOOL_SHA256  $TOOL.part" | sha256sum --check --status; then
        rm -f "$TOOL.part"
        echo "appimagetool $TOOL_VERSION failed checksum verification" >&2
        exit 1
    fi
    chmod +x "$TOOL.part"
    mv "$TOOL.part" "$TOOL"
fi
OUT="$BUILD/Pixelift-${VERSION}-x86_64.AppImage"
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$TOOL" --comp zstd -n "$APPDIR" "$OUT"
echo "Built $OUT"
