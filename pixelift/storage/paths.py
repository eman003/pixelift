"""XDG base-directory locations used by the application."""

from __future__ import annotations

import os
from pathlib import Path

from pixelift import APP_SLUG


def _xdg(var: str, fallback: str) -> Path:
    value = os.environ.get(var)
    base = Path(value) if value and os.path.isabs(value) else Path.home() / fallback
    return base / APP_SLUG


def data_dir() -> Path:
    """~/.local/share/pixelift"""
    return _xdg("XDG_DATA_HOME", ".local/share")


def config_dir() -> Path:
    """~/.config/pixelift"""
    return _xdg("XDG_CONFIG_HOME", ".config")


def state_dir() -> Path:
    """~/.local/state/pixelift (logs)"""
    return _xdg("XDG_STATE_HOME", ".local/state")


def cache_dir() -> Path:
    """~/.cache/pixelift"""
    return _xdg("XDG_CACHE_HOME", ".cache")


def models_dir() -> Path:
    override = os.environ.get("PIXELIFT_MODELS_DIR")
    return Path(override) if override else data_dir() / "models"


def log_file() -> Path:
    return state_dir() / "app.log"
