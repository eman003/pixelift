"""Persistent user settings stored as JSON in ~/.config/pixelift/."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pixelift.core import camera_looks as cl
from pixelift.core import lighting
from pixelift.core.image_processor import ProcessingOptions
from pixelift.core.restoration import settings as rs
from pixelift.storage import paths
from pixelift.utils.image_utils import DEFAULT_TEMPLATE, OUTPUT_FORMATS

log = logging.getLogger(__name__)

TILE_SIZES = (0, 256, 512, 1024)  # 0 = automatic
THEMES = ("system", "light", "dark")
EXISTING = ("skip", "overwrite", "rename")
MODES = ("upscale", "restore")
_STANDARD = rs.LEVEL_STAGES[rs.STANDARD]


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
    # Lighting (applied before upscaling); the lighting_<name> values are the
    # Custom profile's adjustments, -100..100.
    lighting_profile: str = lighting.ORIGINAL
    lighting_intensity: int = 100
    lighting_exposure: int = 0
    lighting_brightness: int = 0
    lighting_contrast: int = 0
    lighting_highlights: int = 0
    lighting_shadows: int = 0
    lighting_temperature: int = 0
    lighting_tint: int = 0
    lighting_saturation: int = 0
    # Camera look (applied after the lighting). The look_<name> values are the
    # Custom look's controls (cl.CUSTOM_NAMES); saved looks are named sets of
    # them, selected as "user:<name>".
    camera_look: str = cl.ORIGINAL
    camera_look_intensity: int = cl.DEFAULT_INTENSITY
    camera_look_grain: str = cl.GRAIN_AUTO
    camera_look_favorites: list[str] = field(default_factory=list)
    camera_look_saved: dict[str, dict[str, int]] = field(default_factory=dict)
    look_exposure: int = 0
    look_contrast: int = 0
    look_highlights: int = 0
    look_shadows: int = 0
    look_temperature: int = 0
    look_tint: int = 0
    look_saturation: int = 0
    look_vibrance: int = 0
    look_red: int = 0
    look_orange: int = 0
    look_yellow: int = 0
    look_green: int = 0
    look_aqua: int = 0
    look_blue: int = 0
    look_purple: int = 0
    look_magenta: int = 0
    look_sharpness: int = 0
    # Photo restoration (mode "restore"). The restore_<stage> values are the
    # Custom level's stages; restore_<color> the manual colour correction.
    mode: str = "upscale"  # upscale | restore
    restore_preset: str = rs.PRESET_RESTORE
    restore_level: str = rs.STANDARD
    restore_dust: int = _STANDARD.dust
    restore_scratches: int = _STANDARD.scratches
    restore_noise: int = _STANDARD.noise
    restore_fading: int = _STANDARD.fading
    restore_sharpness: int = _STANDARD.sharpness
    restore_face: str = _STANDARD.face
    restore_auto_color: bool = _STANDARD.auto_color
    restore_detail: bool = _STANDARD.detail
    restore_fidelity: int = rs.DEFAULT_FIDELITY
    restore_temperature: int = 0
    restore_tint: int = 0
    restore_exposure: int = 0
    restore_contrast: int = 0
    restore_saturation: int = 0
    restore_colorize_strength: int = rs.DEFAULT_COLORIZE_STRENGTH
    restore_colorize_vivid: int = 0
    restore_preserve_tones: bool = True
    restore_modern: str = rs.DEFAULT_MODERN
    restore_scale: int = 2  # used by the "+ Upscale" presets
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
        if self.lighting_profile not in {p.id for p in lighting.all_profiles()}:
            self.lighting_profile = default.lighting_profile
        self.lighting_intensity = min(100, max(0, int(self.lighting_intensity)))
        for name in lighting.ADJUSTMENT_NAMES:
            key = f"lighting_{name}"
            setattr(self, key, min(100, max(-100, int(getattr(self, key)))))
        self._normalise_restoration(default)
        self._normalise_camera_look(default)
        return self

    def _normalise_camera_look(self, default: Settings) -> None:
        self.camera_look_saved = {
            str(name)[:60]: cl.clean_custom_values(values)
            for name, values in (
                self.camera_look_saved.items() if isinstance(self.camera_look_saved, dict) else ()
            )
            if str(name).strip() and isinstance(values, dict)
        }
        if not self._look_exists(self.camera_look):
            self.camera_look = default.camera_look
        favorites = (
            self.camera_look_favorites if isinstance(self.camera_look_favorites, list) else []
        )
        self.camera_look_favorites = list(
            dict.fromkeys(
                f
                for f in favorites
                if isinstance(f, str) and f != cl.ORIGINAL and self._look_exists(f)
            )
        )
        self.camera_look_intensity = min(100, max(0, int(self.camera_look_intensity)))
        if self.camera_look_grain not in cl.GRAIN_CHOICES:
            self.camera_look_grain = default.camera_look_grain
        for name, value in cl.clean_custom_values(self.look_values()).items():
            setattr(self, f"look_{name}", value)

    def _look_exists(self, look_id: object) -> bool:
        if not isinstance(look_id, str):
            return False
        if look_id.startswith(cl.USER_PREFIX):
            return look_id[len(cl.USER_PREFIX) :] in self.camera_look_saved
        return cl.has_look(look_id)

    def look_values(self) -> dict[str, int]:
        """The Custom look's control values."""
        return {name: getattr(self, f"look_{name}") for name in cl.CUSTOM_NAMES}

    def camera_look_settings(self) -> cl.CameraLookSettings:
        look = self.camera_look
        if look.startswith(cl.USER_PREFIX):
            values = self.camera_look_saved.get(look[len(cl.USER_PREFIX) :], {})
        else:
            values = self.look_values()
        return cl.CameraLookSettings(
            look, self.camera_look_intensity, self.camera_look_grain, cl.custom_recipe(values)
        )

    def _normalise_restoration(self, default: Settings) -> None:
        if self.mode not in MODES:
            self.mode = default.mode
        if self.restore_preset not in rs.PRESETS:
            self.restore_preset = default.restore_preset
        if self.restore_level not in rs.LEVELS:
            self.restore_level = default.restore_level
        if self.restore_face not in rs.FACE_MODES:
            self.restore_face = default.restore_face
        if self.restore_modern not in rs.MODERN_MODES:
            self.restore_modern = default.restore_modern
        if self.restore_scale not in (2, 4):
            self.restore_scale = default.restore_scale
        for name in (*rs.STAGE_SLIDERS, "fidelity", "colorize_strength", "colorize_vivid"):
            key = f"restore_{name}"
            setattr(self, key, min(100, max(0, int(getattr(self, key)))))
        for name in rs.COLOR_SLIDERS:
            key = f"restore_{name}"
            setattr(self, key, min(100, max(-100, int(getattr(self, key)))))

    def restoration_stages(self) -> rs.Stages:
        """The Custom level's stages."""
        return rs.Stages(
            *(getattr(self, f"restore_{name}") for name in rs.STAGE_SLIDERS),
            face=self.restore_face,
            auto_color=self.restore_auto_color,
            detail=self.restore_detail,
        )

    def set_restoration_stages(self, stages: rs.Stages) -> None:
        for name in rs.STAGE_SLIDERS:
            setattr(self, f"restore_{name}", getattr(stages, name))
        self.restore_face = stages.face
        self.restore_auto_color = stages.auto_color
        self.restore_detail = stages.detail

    def restoration(self) -> rs.RestorationSettings:
        colorize, upscale = rs.preset_flags(self.restore_preset)
        return rs.RestorationSettings(
            level=self.restore_level,
            custom=self.restoration_stages(),
            fidelity=self.restore_fidelity,
            color=lighting.Adjustments(
                **{name: getattr(self, f"restore_{name}") for name in rs.COLOR_SLIDERS}
            ),
            colorize=colorize,
            colorize_strength=self.restore_colorize_strength,
            colorize_vivid=self.restore_colorize_vivid,
            preserve_tones=self.restore_preserve_tones,
            modern=self.restore_modern,
            scale=self.restore_scale if upscale else 1,
        )

    def lighting(self) -> lighting.LightingSettings:
        custom = lighting.Adjustments(
            **{name: getattr(self, f"lighting_{name}") for name in lighting.ADJUSTMENT_NAMES}
        )
        return lighting.LightingSettings(self.lighting_profile, self.lighting_intensity, custom)

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
            lighting=self.lighting(),
            camera_look=self.camera_look_settings(),
            restoration=self.restoration() if self.mode == "restore" else None,
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
