"""What the user picked for photo restoration (no GUI, no PyTorch).

Mirrors the lighting profiles: a *level* (Light / Standard / Heavy) is a named
set of :class:`Stages`; *Custom* uses the user's own stage values. The other
choices (face mode, fidelity, colorization, colour correction, Modern Finish,
upscaling) apply on top of any level.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field

from pixelift.core.lighting import Adjustments

LIGHT, STANDARD, HEAVY, CUSTOM = "light", "standard", "heavy", "custom"
LEVELS = (LIGHT, STANDARD, HEAVY, CUSTOM)
LEVEL_LABELS = {
    LIGHT: "Light",
    STANDARD: "Standard",
    HEAVY: "Heavy",
    CUSTOM: "Custom",
}
LEVEL_DESCRIPTIONS = {
    LIGHT: "For photographs in good condition: colour, contrast, light denoise and sharpening",
    STANDARD: "Recommended: dust, scratches, noise, colour and faces",
    HEAVY: "For badly damaged photographs: stronger repair and AI detail reconstruction",
    CUSTOM: "Choose each restoration stage yourself",
}

FACE_OFF, FACE_NATURAL, FACE_STRONG = "off", "natural", "strong"
FACE_MODES = (FACE_OFF, FACE_NATURAL, FACE_STRONG)
FACE_LABELS = {FACE_OFF: "Off", FACE_NATURAL: "Natural", FACE_STRONG: "Strong"}

MODERN_OFF = "off"
MODERN_MODES = (MODERN_OFF, "natural", "clean", "vivid", "professional")
MODERN_LABELS = {
    MODERN_OFF: "Off",
    "natural": "Natural",
    "clean": "Clean",
    "vivid": "Vivid",
    "professional": "Professional",
}

# Processing presets: what the one-click choices in the UI switch on.
PRESET_RESTORE, PRESET_COLORIZE, PRESET_UPSCALE, PRESET_FULL = (
    "restore",
    "colorize",
    "upscale",
    "full",
)
PRESETS = (PRESET_RESTORE, PRESET_COLORIZE, PRESET_UPSCALE, PRESET_FULL)
PRESET_LABELS = {
    PRESET_RESTORE: "Restore",
    PRESET_COLORIZE: "Restore + Colorize",
    PRESET_UPSCALE: "Restore + Upscale",
    PRESET_FULL: "Full Restoration",
}


def preset_flags(preset: str) -> tuple[bool, bool]:
    """(colorize, upscale) for a processing preset."""
    return (
        preset in (PRESET_COLORIZE, PRESET_FULL),
        preset in (PRESET_UPSCALE, PRESET_FULL),
    )


@dataclass(frozen=True)
class Stages:
    """Strength of each restoration stage, 0..100 (0 = stage off)."""

    dust: int = 0
    scratches: int = 0
    noise: int = 0
    fading: int = 0
    sharpness: int = 0
    face: str = FACE_OFF
    auto_color: bool = False  # correct colour casts / yellowing automatically
    detail: bool = False  # AI detail reconstruction (Real-ESRGAN) without upscaling

    def clamped(self) -> Stages:
        def pct(v: object) -> int:
            return min(100, max(0, int(v)))  # type: ignore[call-overload]

        return Stages(
            pct(self.dust),
            pct(self.scratches),
            pct(self.noise),
            pct(self.fading),
            pct(self.sharpness),
            self.face if self.face in FACE_MODES else FACE_OFF,
            bool(self.auto_color),
            bool(self.detail),
        )


STAGE_SLIDERS = ("dust", "scratches", "noise", "fading", "sharpness")

LEVEL_STAGES: dict[str, Stages] = {
    LIGHT: Stages(noise=20, fading=35, sharpness=25, auto_color=True),
    STANDARD: Stages(
        dust=50,
        scratches=40,
        noise=40,
        fading=50,
        sharpness=35,
        face=FACE_NATURAL,
        auto_color=True,
    ),
    HEAVY: Stages(
        dust=75,
        scratches=70,
        noise=60,
        fading=70,
        sharpness=40,
        face=FACE_NATURAL,
        auto_color=True,
        detail=True,
    ),
}

# The manual colour-correction sliders reuse the lighting engine's adjustments.
COLOR_SLIDERS = ("temperature", "tint", "exposure", "contrast", "saturation")

DEFAULT_FIDELITY = 50
DEFAULT_COLORIZE_STRENGTH = 80
DEFAULT_MODERN = "natural"


@dataclass(frozen=True)
class RestorationSettings:
    level: str = STANDARD
    custom: Stages = LEVEL_STAGES[STANDARD]  # used when level == CUSTOM
    # 0 = keep the original pixels, 100 = trust the AI fully.
    fidelity: int = DEFAULT_FIDELITY
    color: Adjustments = field(default_factory=Adjustments)  # manual colour correction
    colorize: bool = False  # only ever applied to black-and-white photos
    colorize_strength: int = DEFAULT_COLORIZE_STRENGTH  # 0 = grey, 100 = full
    colorize_vivid: int = 0  # 0 = natural (historical) colours, 100 = vivid
    preserve_tones: bool = True  # keep the original brightness, only add colour
    modern: str = DEFAULT_MODERN
    scale: int = 1  # 1 = no upscaling; 2 or 4 = AI upscaling after restoration

    def stages(self) -> Stages:
        if self.level == CUSTOM:
            return self.custom.clamped()
        return LEVEL_STAGES.get(self.level, LEVEL_STAGES[STANDARD])

    def validate(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"unknown restoration level: {self.level}")
        if self.modern not in MODERN_MODES:
            raise ValueError(f"unknown Modern Finish: {self.modern}")
        if self.scale not in (1, 2, 4):
            raise ValueError("restoration scale must be 1, 2 or 4")
        if self.stages().face not in FACE_MODES:
            raise ValueError("unknown face restoration mode")

    def is_identity(self) -> bool:
        """True when nothing at all would change the pixels."""
        s = self.stages()
        return (
            not any(getattr(s, name) for name in STAGE_SLIDERS)
            and s.face == FACE_OFF
            and not s.auto_color
            and not s.detail
            and self.color.is_neutral
            and not (self.colorize and self.colorize_strength > 0)
            and self.modern == MODERN_OFF
            and self.scale == 1
        )

    def ai_scale(self) -> int:
        """The factor the AI upscaler runs at (0: not used).

        Detail reconstruction without upscaling runs it at 2× and resizes back.
        """
        if self.scale > 1:
            return self.scale
        return 2 if self.stages().detail else 0

    def tag(self, colorized: bool = True) -> str:
        """Filename tag: "" for the default Standard restoration.

        The level is named when it is not Standard; any other non-default
        choice adds a short hash, so results made with different settings
        never share a file name (like the lighting tag). The colorization
        settings only count for photos that are colorized (``colorized``).
        """
        parts = [] if self.level == STANDARD else [self.level]
        signature = self._signature(colorized)
        default = RestorationSettings(level=self.level, colorize=self.colorize)
        if signature != default._signature(colorized):
            parts.append(hashlib.sha1(signature.encode()).hexdigest()[:6])
        return "-".join(parts)

    def _signature(self, colorized: bool = True) -> str:
        values: list[object] = [self.fidelity, self.modern, *dataclasses.astuple(self.color)]
        if self.level == CUSTOM:
            values += dataclasses.astuple(self.custom.clamped())
        if self.colorize and colorized:
            values += [self.colorize_strength, self.colorize_vivid, self.preserve_tones]
        return ",".join(f"{v:g}" if isinstance(v, float) else str(v) for v in values)
