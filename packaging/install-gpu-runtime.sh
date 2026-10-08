#!/bin/sh
# Install a GPU-enabled PyTorch for the packaged app (.deb / AppImage), which
# ships the CPU build to stay small. Nothing is changed system-wide.
#
#   install-gpu-runtime.sh cu126     # NVIDIA, CUDA 12.6 (default; driver >= 560)
#   install-gpu-runtime.sh cu130     # NVIDIA, CUDA 13.0 (newer drivers)
#   install-gpu-runtime.sh rocm7.2   # AMD Radeon (ROCm-supported GPUs)
#   install-gpu-runtime.sh xpu       # Intel Arc / Core Ultra
#   install-gpu-runtime.sh remove    # go back to the bundled CPU build
set -eu
VARIANT="${1:-cu126}"
TARGET="${XDG_DATA_HOME:-$HOME/.local/share}/pixelift/runtime"
if [ "$VARIANT" = "remove" ]; then
    rm -rf "$TARGET"; echo "Removed $TARGET"; exit 0
fi
if ! python3 -m pip --version >/dev/null 2>&1; then
    echo "pip is required: sudo apt install python3-pip" >&2; exit 1
fi
echo "Downloading PyTorch ($VARIANT) — this is a large download (2–4 GB)…"
rm -rf "$TARGET.new"
python3 -m pip install --no-cache-dir --target "$TARGET.new" torch \
    --index-url "https://download.pytorch.org/whl/$VARIANT"
rm -rf "$TARGET"
mv "$TARGET.new" "$TARGET"
echo "Installed to $TARGET. Restart Pixelift; check the device with:"
echo "  pixelift --list-devices"
