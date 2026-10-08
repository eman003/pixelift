#!/usr/bin/env python3
"""Download (and checksum-verify) AI models without starting the GUI.

Usage:
    python scripts/download_models.py                 # recommended model
    python scripts/download_models.py --all
    python scripts/download_models.py realesr-general-x4v3 realesrgan-x2plus
    python scripts/download_models.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pixelift.cli import run_cli
from pixelift.models import all_specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="*", help="model ids (default: recommended)")
    parser.add_argument("--all", action="store_true", help="download every model")
    parser.add_argument("--list", action="store_true", help="list models and exit")
    args = parser.parse_args()
    if args.list:
        return run_cli(["--list-models"])
    ids = [s.id for s in all_specs()] if args.all else (args.models or ["recommended"])
    cli_args: list[str] = []
    for model_id in ids:
        cli_args += ["--download-model", model_id]
    return run_cli(cli_args)


if __name__ == "__main__":
    raise SystemExit(main())
