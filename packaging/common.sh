#!/bin/bash
# Shared build helpers: stage the app + bundled Python dependencies into a dir.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build"
APP_ID="io.github.pixelift.Pixelift"
VERSION="$(cd "$ROOT" && python3 -c 'import pixelift; print(pixelift.__version__)')"
PYVER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
VENDOR="$BUILD/vendor-$PYVER"

build_vendor() {
    # CPU PyTorch + numpy + Pillow, for the system Python ABI. Cached between builds.
    if [ -f "$VENDOR/.complete" ]; then return; fi
    rm -rf "$VENDOR"
    python3 -m pip install --no-cache-dir --target "$VENDOR" \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple torch numpy pillow
    # Trim files not needed at runtime.
    rm -rf "$VENDOR"/torch/include "$VENDOR"/torch/share/cmake "$VENDOR"/bin \
           "$VENDOR"/torch/test "$VENDOR"/caffe2
    find "$VENDOR" -name "__pycache__" -type d -prune -exec rm -rf {} +
    touch "$VENDOR/.complete"
}

# stage_app <libdir>  -> <libdir>/{app,vendor}
stage_app() {
    local lib="$1"
    mkdir -p "$lib/app"
    (cd "$ROOT" && tar --exclude='__pycache__' -cf - pixelift) | tar -xf - -C "$lib/app"
    cp -a "$VENDOR" "$lib/vendor"
    rm -f "$lib/vendor/.complete"
    install -m 755 "$ROOT/packaging/install-gpu-runtime.sh" "$lib/install-gpu-runtime.sh"
    chmod -R u+rwX,go+rX,go-w "$lib"
}

# stage_desktop <prefix>  -> launcher, desktop entry, icon, metainfo under <prefix>
stage_desktop() {
    local prefix="$1"
    install -Dm 755 /dev/null "$prefix/bin/pixelift"
    sed "s/@PYVER@/$PYVER/" "$ROOT/packaging/launcher.sh" > "$prefix/bin/pixelift"
    install -Dm 644 "$ROOT/packaging/$APP_ID.desktop" \
        "$prefix/share/applications/$APP_ID.desktop"
    install -Dm 644 "$ROOT/pixelift/data/icons/hicolor/scalable/apps/$APP_ID.svg" \
        "$prefix/share/icons/hicolor/scalable/apps/$APP_ID.svg"
    install -Dm 644 "$ROOT/packaging/$APP_ID.metainfo.xml" \
        "$prefix/share/metainfo/$APP_ID.metainfo.xml"
}
