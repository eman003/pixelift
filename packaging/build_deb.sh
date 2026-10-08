#!/bin/bash
# Build build/pixelift_<version>_amd64.deb for Ubuntu 24.04+ (Python 3.12).
# The package bundles the CPU build of PyTorch; see install-gpu-runtime.sh for GPUs.
source "$(dirname "$0")/common.sh"
build_vendor

PKG="$BUILD/deb/pixelift"
rm -rf "$PKG"
stage_app "$PKG/usr/lib/pixelift"
stage_desktop "$PKG/usr"
install -Dm 644 "$ROOT/LICENSE" "$PKG/usr/share/doc/pixelift/copyright"
install -Dm 644 "$ROOT/README.md" "$PKG/usr/share/doc/pixelift/README.md"

PYNEXT="3.$(( ${PYVER#3.} + 1 ))"
SIZE_KB="$(du -sk "$PKG/usr" | cut -f1)"
mkdir -p "$PKG/DEBIAN"
cat > "$PKG/DEBIAN/control" <<CONTROL
Package: pixelift
Version: $VERSION
Section: graphics
Priority: optional
Architecture: amd64
Depends: python3 (>= $PYVER), python3 (<< $PYNEXT), python3-gi (>= 3.42), gir1.2-gtk-4.0, gir1.2-adw-1 (>= 1.5)
Recommends: python3-pip
Installed-Size: $SIZE_KB
Maintainer: Pixelift contributors <noreply@example.com>
Homepage: https://github.com/xinntao/Real-ESRGAN
Description: Local AI image upscaler and photo restorer (Real-ESRGAN, GTK4)
 Upscale images 2x or 4x with Real-ESRGAN super-resolution, and restore old
 photographs: dust, scratches, fading, colour casts and faces, with optional
 colorization of black-and-white photos. All processing runs locally; images
 are never uploaded. Includes a desktop app and the pixelift command-line
 tool. AI models are downloaded on first use.
CONTROL
cat > "$PKG/DEBIAN/postinst" <<'POSTINST'
#!/bin/sh
set -e
command -v update-desktop-database >/dev/null && update-desktop-database -q /usr/share/applications || true
command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q -t -f /usr/share/icons/hicolor || true
POSTINST
cp "$PKG/DEBIAN/postinst" "$PKG/DEBIAN/postrm"
chmod 755 "$PKG/DEBIAN/postinst" "$PKG/DEBIAN/postrm"

find "$PKG" -type d -exec chmod 755 {} +
OUT="$BUILD/pixelift_${VERSION}_amd64.deb"
dpkg-deb --root-owner-group -Zzstd --build "$PKG" "$OUT"
echo "Built $OUT"
