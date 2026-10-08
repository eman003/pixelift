#!/bin/sh
# Pixelift launcher (installed as /usr/bin/pixelift and used by AppRun).
# Uses the system Python + PyGObject/GTK4 and the bundled PyTorch/Pillow/numpy.
APPDIR="${PIXELIFT_APPDIR:-/usr/lib/pixelift}"
PYTHON="${PIXELIFT_PYTHON:-/usr/bin/python3}"
WANT_PY="@PYVER@"

if ! "$PYTHON" -c "import sys; sys.exit(0 if '%d.%d' % sys.version_info[:2] == '$WANT_PY' else 1)" 2>/dev/null; then
    msg="Pixelift needs Python $WANT_PY (Ubuntu 24.04). Install it from source on this system instead (see README)."
    echo "$msg" >&2
    command -v zenity >/dev/null 2>&1 && zenity --error --text="$msg" 2>/dev/null
    exit 1
fi

# Optional GPU-enabled PyTorch installed by install-gpu-runtime.sh takes precedence.
GPU_RUNTIME="${XDG_DATA_HOME:-$HOME/.local/share}/pixelift/runtime"
PYTHONPATH="$APPDIR/app"
[ -d "$GPU_RUNTIME/torch" ] && PYTHONPATH="$PYTHONPATH:$GPU_RUNTIME"
PYTHONPATH="$PYTHONPATH:$APPDIR/vendor"
export PYTHONPATH
exec "$PYTHON" -s -m pixelift "$@"
