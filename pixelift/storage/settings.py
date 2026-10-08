"""Persistent user settings stored as JSON in ~/.config/pixelift/."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pixelift.core.image_processor import ProcessingOptions
from pixelift.storage import paths
from pixelift.utils.image_utils import DEFAULT_TEMPLATE, OUTPUT_FORMATS

log = logging.getLogger(__name__)

TILE_SIZES = (0, 256, 512, 1024)  # 0 = automatic
THEMES = ("system", "light", "dark")
EXISTING = ("skip", "overwrite", "rename")


@dataclass
class Settings:
    # Processing
    model: str = "realesrgan"
    scale: int = 4
    device: str = "auto"  # auto | cpu | cuda | cuda:N | xpu | xpu:N
    tile_size: int = 0
    gpu_memory_limit_mb: int = 0  # 0 = no limit
    # Output
    output_dir: str = ""  # "" = <original directory>/upscaled
    output_format: str = "png"
    quality: int = 92
    filename_template: str = DEFAULT_TEMPLATE
    existing: str = "skip"
    preserve_metadata: bool = True
    # Performance
    concurrent_jobs: int = 0  # 0 = automatic
    cpu_threads: int = 0  # 0 = all cores
    gpu_enabled: bool = True
    # Appearance / state
    theme: str = "system"
    first_run_complete: bool = False
    last_open_dir: str = ""

    def normalise(self) -> Settings:
        """Clamp invalid values (e.g. from a hand-edited file) to defaults."""
        default = Settings()
        if self.scale not in (2, 4):
            self.scale = default.scale
        if self.tile_size not in TILE_SIZES:
            self.tile_size = 0
        if self.output_format not in OUTPUT_FORMATS:
            self.output_format = default.output_format
        self.quality = min(100, max(1, int(self.quality)))
        if self.theme not in THEMES:
            self.theme = default.theme
        if self.existing not in EXISTING:
            self.existing = default.existing
        self.concurrent_jobs = min(8, max(0, int(self.concurrent_jobs)))
        self.cpu_threads = max(0, int(self.cpu_threads))
        self.gpu_memory_limit_mb = max(0, int(self.gpu_memory_limit_mb))
        return self

    def processing_options(self) -> ProcessingOptions:
        return ProcessingOptions(
            scale=self.scale,
            model=self.model,
            output_format=self.output_format,
            quality=self.quality,
            output_dir=Path(self.output_dir).expanduser() if self.output_dir else None,
            filename_template=self.filename_template or DEFAULT_TEMPLATE,
            existing=self.existing,  # type: ignore[arg-type]
            preserve_metadata=self.preserve_metadata,
        )


def settings_path() -> Path:
    return paths.config_dir() / "settings.json"


def load_settings(path: Path | None = None) -> Settings:
    path = path or settings_path()
    try:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Settings()
    except (OSError, ValueError):
        log.warning("Settings file %s is unreadable; using defaults", path, exc_info=True)
        return Settings()
    known = {f.name: f for f in dataclasses.fields(Settings)}
    values: dict[str, Any] = {}
    for key, value in raw.items():
        field = known.get(key)
        default = getattr(Settings, key, None) if field else None
        if field and (default is None or isinstance(value, type(default))):
            values[key] = value
    return Settings(**values).normalise()


def save_settings(settings: Settings, path: Path | None = None) -> None:
    path = path or settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".settings-", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(dataclasses.asdict(settings), fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        log.exception("Could not save settings to %s", path)
