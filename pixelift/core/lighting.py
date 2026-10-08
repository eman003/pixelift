"""Lighting profiles: tonal and colour adjustments applied before upscaling.

Everything here is NumPy (no GUI, no PyTorch) and is shared by the
processing pipeline, the CLI and the live preview.

A profile is just a named set of ``Adjustments``; the intensity scales every
adjustment linearly towards zero, so 0 % is always the untouched image.
Adding a profile is one ``register_profile(...)`` call — the UI, settings and
CLI list whatever is registered.

All curves keep 0 and 1 fixed and are monotonic, so adjustments never
introduce clipping or banding of their own; exposure and white-balance gains
roll highlights off smoothly instead of clipping them.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass

import numpy as np
from PIL import Image

ORIGINAL = "original"
CUSTOM = "custom"

# Rows processed at once: bounds the float32 working memory on huge images.
_CHUNK_PIXELS = 1 << 20
_GAMMA = 2.2
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass(frozen=True)
class Adjustments:
    """Each value is in -100..100; 0 is neutral.

    exposure     ±2 EV in linear light, highlights roll off instead of clipping
    brightness   midtone gamma (black and white points stay fixed)
    contrast     S-curve around mid-grey
    highlights   recover (−) or boost (+) the bright tones
    shadows      lift (+) or deepen (−) the dark tones
    temperature  cooler (−) / warmer (+)
    tint         green (−) / magenta (+)
    saturation   less (−) / more (+) colour; gentler on already-saturated colours
    """

    exposure: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    temperature: float = 0.0
    tint: float = 0.0
    saturation: float = 0.0

    @property
    def is_neutral(self) -> bool:
        return all(v == 0 for v in dataclasses.astuple(self))

    @property
    def changes_colour(self) -> bool:
        """Whether a grey image stops being grey (saturation keeps grey grey)."""
        return self.temperature != 0 or self.tint != 0

    def scaled(self, factor: float) -> Adjustments:
        if factor == 1:
            return self
        return Adjustments(*(v * factor for v in dataclasses.astuple(self)))

    def clamped(self) -> Adjustments:
        return Adjustments(*(min(100.0, max(-100.0, float(v))) for v in dataclasses.astuple(self)))


ADJUSTMENT_NAMES: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(Adjustments))


@dataclass(frozen=True)
class LightingProfile:
    id: str
    name: str
    description: str
    adjustments: Adjustments = Adjustments()


_PROFILES: dict[str, LightingProfile] = {}


def register_profile(profile: LightingProfile) -> None:
    """Add (or replace) a profile. Custom stays last in ``all_profiles()``."""
    _PROFILES[profile.id] = profile
    if CUSTOM in _PROFILES and profile.id != CUSTOM:
        _PROFILES[CUSTOM] = _PROFILES.pop(CUSTOM)


def all_profiles() -> list[LightingProfile]:
    return list(_PROFILES.values())


def get_profile(profile_id: str) -> LightingProfile:
    return _PROFILES[profile_id]


for _profile in (
    LightingProfile(ORIGINAL, "Original", "No lighting adjustment"),
    LightingProfile(
        "natural-daylight",
        "Natural Daylight",
        "Balanced exposure and neutral colours",
        Adjustments(exposure=5, contrast=10, highlights=-20, shadows=20, saturation=5),
    ),
    LightingProfile(
        "bright-clean",
        "Bright & Clean",
        "A brighter image with lifted shadows",
        Adjustments(
            exposure=20, brightness=15, contrast=5, highlights=-10, shadows=35, temperature=-3
        ),
    ),
    LightingProfile(
        "golden-hour",
        "Golden Hour",
        "Warmer tones and softer highlights",
        Adjustments(
            brightness=5,
            contrast=-5,
            highlights=-25,
            shadows=10,
            temperature=45,
            tint=8,
            saturation=10,
        ),
    ),
    LightingProfile(
        "studio",
        "Studio",
        "Clean, bright lighting with controlled shadows",
        Adjustments(
            exposure=10, brightness=10, contrast=15, highlights=-20, shadows=10, temperature=-2
        ),
    ),
    LightingProfile(
        "cinematic",
        "Cinematic",
        "Deeper shadows, controlled highlights and stronger contrast",
        Adjustments(
            exposure=-5,
            contrast=35,
            highlights=-30,
            shadows=-25,
            temperature=-8,
            tint=-4,
            saturation=-10,
        ),
    ),
    LightingProfile(
        "low-light-recovery",
        "Low Light Recovery",
        "Brighten dark areas while protecting highlights",
        Adjustments(
            exposure=30, brightness=25, contrast=-5, highlights=-35, shadows=60, saturation=10
        ),
    ),
    LightingProfile(
        "cool-daylight",
        "Cool Daylight",
        "Cooler temperature with crisp contrast",
        Adjustments(contrast=15, highlights=-10, temperature=-35, tint=-5, saturation=5),
    ),
    LightingProfile(
        "vivid",
        "Vivid",
        "Stronger colours and contrast",
        Adjustments(contrast=20, highlights=-10, shadows=5, saturation=45),
    ),
    LightingProfile(
        "high-contrast",
        "High Contrast",
        "Dramatic highlights and shadows",
        Adjustments(contrast=60, highlights=25, shadows=-35, saturation=5),
    ),
    LightingProfile(CUSTOM, "Custom", "Set each adjustment yourself"),
):
    register_profile(_profile)


@dataclass(frozen=True)
class LightingSettings:
    """What the user picked: a profile, its intensity and the Custom values."""

    profile: str = ORIGINAL
    intensity: int = 100  # percent, 0..100; ignored by Custom
    custom: Adjustments = Adjustments()

    def adjustments(self) -> Adjustments:
        """The effective adjustments (intensity applied)."""
        if self.profile == CUSTOM:
            return self.custom.clamped()
        profile = _PROFILES.get(self.profile)
        if profile is None:
            return Adjustments()
        return profile.adjustments.scaled(min(100, max(0, self.intensity)) / 100)

    @property
    def active(self) -> bool:
        return not self.adjustments().is_neutral

    def tag(self) -> str:
        """Short filename-safe label for the effective lighting; "" when neutral.

        Different effective adjustments give different tags, so an output made
        with other lighting is never mistaken for this one's.
        """
        adjustments = self.adjustments()
        if adjustments.is_neutral:
            return ""
        if self.profile == CUSTOM:
            values = ",".join(f"{v:g}" for v in dataclasses.astuple(adjustments))
            return f"{CUSTOM}-{hashlib.sha1(values.encode()).hexdigest()[:6]}"
        intensity = min(100, max(0, self.intensity))
        return self.profile if intensity == 100 else f"{self.profile}-{intensity}"


# --- processing -------------------------------------------------------------
def apply_lighting(
    rgb: np.ndarray, adjustments: Adjustments, out: np.ndarray | None = None
) -> np.ndarray:
    """Apply ``adjustments`` to an (H, W, 3) ``uint8`` array.

    Returns ``rgb`` itself when the adjustments are neutral. Pass ``out=rgb``
    to work in place (no full-size copy); either way the float working memory
    is bounded by processing a band of rows at a time.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("expected an (H, W, 3) uint8 array")
    if adjustments.is_neutral:
        if out is not None and out is not rgb:
            out[...] = rgb
            return out
        return rgb
    if out is None:
        out = np.empty_like(rgb)
    height, width = rgb.shape[:2]
    rows = max(1, _CHUNK_PIXELS // max(width, 1))
    plan = _Plan(adjustments.clamped())
    for y0 in range(0, height, rows):
        y1 = min(height, y0 + rows)
        band = plan.run(plan.decode(rgb[y0:y1]))
        np.multiply(band, 255, out=band)
        np.add(band, 0.5, out=band)
        np.clip(band, 0, 255, out=band)
        out[y0:y1] = band.astype(np.uint8)
    return out


def apply_to_pil(img: Image.Image, adjustments: Adjustments) -> Image.Image:
    """Apply to a Pillow image, keeping its alpha channel (used by the preview)."""
    if adjustments.is_neutral:
        return img
    alpha = img.getchannel("A") if img.mode in ("RGBA", "LA") else None
    rgb = np.array(img.convert("RGB"))
    result = Image.fromarray(apply_lighting(rgb, adjustments, out=rgb), "RGB")
    if alpha is not None:
        result.putalpha(alpha)
    return result


class _Plan:
    """Adjustments converted to curve parameters, applied to float bands in 0..1."""

    def __init__(self, adj: Adjustments) -> None:
        # White balance as per-channel gains in linear light, normalised so a
        # neutral grey keeps its luminance.
        t, m = adj.temperature / 100, adj.tint / 100
        gains = np.array(
            [1 + 0.25 * t + 0.1 * m, 1 - 0.2 * m, 1 - 0.25 * t + 0.1 * m], dtype=np.float32
        )
        gains = gains / float(gains @ _LUMA)
        gains *= np.float32(2.0 ** (2.0 * adj.exposure / 100))
        # The first stage is per channel, so it is a 256-entry lookup per channel.
        levels = np.linspace(0, 1, 256, dtype=np.float32)
        self.lut = np.repeat(levels[:, None], 3, axis=1)
        if t or m or adj.exposure:
            self.lut = _linear_gains(self.lut, gains)
        self.shadows = adj.shadows / 100
        self.highlights = adj.highlights / 100
        self.contrast = 0.5 * adj.contrast / 100
        self.gamma = 2.0 ** (-0.7 * adj.brightness / 100)
        self.saturation = adj.saturation / 100

    def decode(self, band: np.ndarray) -> np.ndarray:
        """uint8 band -> float32 in 0..1 with white balance and exposure applied."""
        x = np.empty(band.shape, dtype=np.float32)
        for c in range(3):
            np.take(self.lut[:, c], band[..., c], out=x[..., c])
        return x

    def run(self, x: np.ndarray) -> np.ndarray:
        if self.shadows or self.highlights:
            x = self._tone(x)
        if self.contrast:
            u = x - 0.5
            # 0.5 + u(1+c) - 4c u³: fixed ends, monotonic for |c| <= 0.5.
            x = 0.5 + u * (1 + self.contrast) - (4 * self.contrast) * u * u * u
        if self.gamma != 1:
            np.clip(x, 0, 1, out=x)
            np.power(x, np.float32(self.gamma), out=x)
        if self.saturation:
            x = self._saturate(x)
        return x

    def _tone(self, x: np.ndarray) -> np.ndarray:
        np.clip(x, 0, 1, out=x)
        lum = _luma(x)
        inv = 1 - lum
        # Both curves keep 0 and 1 fixed and stay monotonic for |s|, |h| <= 1.
        target = lum + self.shadows * lum * inv * inv + self.highlights * lum * lum * inv
        np.clip(target, 0, 1, out=target)
        eps = np.float32(1e-6)
        # Darken by scaling towards black, brighten by scaling towards white:
        # either way hue is kept and no channel leaves 0..1. As x*a + b with
        # per-pixel a, b to avoid full-size temporaries.
        darker = target < lum
        a = np.where(darker, target / np.maximum(lum, eps), (1 - target) / np.maximum(inv, eps))
        b = np.where(darker, np.float32(0), 1 - a)
        x *= a[..., None]
        x += b[..., None]
        return x

    def _saturate(self, x: np.ndarray) -> np.ndarray:
        np.clip(x, 0, 1, out=x)
        lum = _luma(x)[..., None]
        x -= lum  # chroma
        if self.saturation > 0:
            # Vibrance-style: boost colours less once they are already
            # saturated (gentle on skin), and never push a channel out of
            # range, which would shift the hue.
            r, g, b = x[..., 0], x[..., 1], x[..., 2]
            hi = np.maximum(np.maximum(r, g), b)[..., None]
            lo = np.minimum(np.minimum(r, g), b)[..., None]
            factor = 1 + self.saturation * (1 - 0.5 * (hi - lo))
            eps = np.float32(1e-6)
            room = np.minimum((1 - lum) / np.maximum(hi, eps), lum / np.maximum(-lo, eps))
            # Soft minimum of the boost and the room left before a channel
            # clips: colours approach the gamut edge smoothly instead of piling
            # up on it.
            boost = factor - 1
            headroom = np.maximum(room - 1, 0)
            factor = 1 + boost * headroom / (boost + headroom + eps)
            x *= factor
        else:
            x *= np.float32(1 + self.saturation)
        x += lum
        return x


def _luma(x: np.ndarray) -> np.ndarray:
    # Explicit sum: much faster than a reduction over a length-3 axis.
    lum = x[..., 0] * _LUMA[0]
    lum += x[..., 1] * _LUMA[1]
    lum += x[..., 2] * _LUMA[2]
    return lum


def _linear_gains(x: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Per-channel gains applied in (approximately) linear light."""
    x = np.power(np.clip(x, 0, 1), np.float32(_GAMMA))
    for c in range(3):
        g = float(gains[c])
        if g <= 1:
            x[..., c] *= g
        else:
            # Extended Reinhard with white point g: maps [0, 1] onto [0, 1]
            # (no clipping) and is the identity at g = 1.
            v = x[..., c] * g
            x[..., c] = v * (1 + v / (g * g)) / (1 + v)
    return np.power(x, np.float32(1 / _GAMMA))
