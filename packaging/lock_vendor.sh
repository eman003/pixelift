#!/bin/bash
# Regenerate packaging/requirements-cp312.txt: resolve the newest CPU PyTorch,
# numpy and Pillow (plus dependencies) for CPython 3.12 / x86_64 and record
# each wheel's SHA-256. Nothing is installed; review the diff before committing.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/packaging/requirements-cp312.txt"
REPORT="$(mktemp)"
trap 'rm -f "$REPORT"' EXIT
python3 -m pip install --dry-run --ignore-installed --quiet --report "$REPORT" \
    --python-version 3.12 --implementation cp --abi cp312 \
    --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 \
    --platform manylinux2014_x86_64 --only-binary=:all: \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch numpy 'pillow>=11.3'
python3 - "$REPORT" > "$OUT" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("""# Hash-locked runtime dependencies bundled into the .deb, AppImage and Flatpak
# (CPython 3.12, x86_64; CPU PyTorch). Installed with --require-hashes, so a
# changed or substituted wheel on either index fails the build.
#
# Regenerate (then review the diff) with packaging/lock_vendor.sh.""")
for i in sorted(r["install"], key=lambda i: i["metadata"]["name"].lower()):
    m = i["metadata"]
    h = i["download_info"]["archive_info"]["hashes"]["sha256"]
    print(f"{m['name'].lower()}=={m['version']} \\\n    --hash=sha256:{h}")
PY
echo "Wrote $OUT"
