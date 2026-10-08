"""Presets: one-click recipes made of existing settings.

A preset is only a set of values for settings Pixelift already has — the
lighting profile and intensity, the camera look, its intensity and grain, and
(for Old Photo) the restoration level. Choosing one writes those values; the
lighting, camera look and restoration code do the work, so improvements there
improve the presets too. Presets never change the scale, model or file format.

A preset is a starting point: every value stays editable, and the preset then
shows as modified (``Natural · Modified``) until it is chosen again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pixelift.core import camera_looks as cl
from pixelift.core import lighting
from pixelift.core.restoration import settings as rs

UPSCALE, RESTORE = "upscale", "restore"
ORIGINAL = "original"


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    description: str  # a few plain words
    modes: tuple[str, ...]
    values: dict[str, Any] = field(default_factory=dict)  # settings field -> value


def _creative(
    light: str, light_intensity: int, look: str, look_intensity: int, grain: str
) -> dict[str, Any]:
    return {
        "lighting_profile": light,
        "lighting_intensity": light_intensity,
        "camera_look": look,
        "camera_look_intensity": look_intensity,
        "camera_look_grain": grain,
    }


BOTH = (UPSCALE, RESTORE)
_PRESETS: tuple[Preset, ...] = (
    Preset(
        ORIGINAL,
        "Original",
        "Your photo as it is",
        BOTH,
        _creative(lighting.ORIGINAL, 100, cl.ORIGINAL, cl.DEFAULT_INTENSITY, cl.GRAIN_AUTO),
    ),
    Preset(
        "old-photo",
        "Old Photo",
        "Repairs dust, scratches and fading",
        (RESTORE,),
        # The level only: whether to colorize or upscale stays the user's choice.
        {
            "restore_level": rs.STANDARD,
            **_creative(lighting.ORIGINAL, 100, cl.ORIGINAL, cl.DEFAULT_INTENSITY, cl.GRAIN_AUTO),
        },
    ),
    Preset(
        "natural",
        "Natural",
        "Balanced light and clean colour",
        BOTH,
        _creative("natural-daylight", 70, "fujifilm-provia", 35, "off"),
    ),
    Preset(
        "portrait",
        "Portrait",
        "Natural skin, gentle contrast",
        (UPSCALE,),
        _creative("natural-daylight", 50, "nikon-portrait", 60, "off"),
    ),
    Preset(
        "landscape",
        "Landscape",
        "Rich detail and true colour",
        (UPSCALE,),
        _creative("natural-daylight", 60, "nikon-landscape", 55, "off"),
    ),
    Preset(
        "cinematic",
        "Cinematic",
        "Moody contrast, muted colour",
        (UPSCALE,),
        _creative("cinematic", 60, "cinematic-film", 55, "low"),
    ),
    Preset(
        "film",
        "Film",
        "Soft, warm film colour",
        (UPSCALE,),
        _creative("golden-hour", 30, "portra", 60, cl.GRAIN_AUTO),
    ),
    Preset(
        "vintage",
        "Vintage",
        "Warm, faded and nostalgic",
        BOTH,
        _creative("golden-hour", 40, "classic-film", 45, "low"),
    ),
    Preset(
        "monochrome",
        "Monochrome",
        "Clean black and white",
        BOTH,
        _creative(lighting.ORIGINAL, 100, "black-white", 85, "low"),
    ),
)
_BY_ID = {p.id: p for p in _PRESETS}


def all_presets() -> tuple[Preset, ...]:
    return _PRESETS


def presets_for(mode: str) -> list[Preset]:
    """The presets offered in a mode (Upscale or Restore Photos), in display order."""
    return [p for p in _PRESETS if mode in p.modes]


def has_preset(preset_id: str) -> bool:
    return preset_id in _BY_ID


def get_preset(preset_id: str) -> Preset:
    return _BY_ID[preset_id]


def apply(settings: Any, preset_id: str) -> None:
    """Write a preset's values into ``settings`` and remember the choice."""
    preset = _BY_ID[preset_id]
    for name, value in preset.values.items():
        setattr(settings, name, value)
    settings.preset = preset_id


def matches(settings: Any, preset_id: str) -> bool:
    """Whether ``settings`` still hold every value of the preset."""
    preset = _BY_ID[preset_id]
    return all(getattr(settings, name) == value for name, value in preset.values.items())


def status(settings: Any) -> tuple[str, bool]:
    """(name to show, modified) for the current preset.

    No preset chosen: "Original" when nothing is applied, otherwise "Custom".
    """
    preset_id = settings.preset
    if preset_id in _BY_ID:
        return _BY_ID[preset_id].name, not matches(settings, preset_id)
    if matches(settings, ORIGINAL):
        return _BY_ID[ORIGINAL].name, False
    return "Custom", False


def label(settings: Any) -> str:
    name, modified = status(settings)
    return f"{name} · Modified" if modified else name
