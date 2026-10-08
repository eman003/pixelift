"""Logging setup: detailed logs go to a rotating file, never to dialogs."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from pixelift.storage import paths

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(threadName)s] %(message)s"


def setup_logging(verbose: bool = False, console: bool = False) -> Path | None:
    """Configure root logging. Returns the log file path (None if unwritable)."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    log_path: Path | None = paths.log_file()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(file_handler)
    except OSError:
        log_path = None

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setLevel(logging.DEBUG if verbose else logging.WARNING)
        stream.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        root.addHandler(stream)

    logging.captureWarnings(True)
    return log_path
