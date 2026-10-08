"""Faded-image recovery, colour-cast correction and the Modern Finish looks.

Per-channel levels (what fading and yellowing actually change) are applied as
lookup tables; every other tonal and colour adjustment reuses the lighting
engine (:func:`pixelift.core.lighting.apply_lighting`), so restoration and the
lighting profiles share one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pixelift.core.lighting import Adjustments, apply_lighting
from pixelift.core.restoration.analysis import Levels, measure_levels

# The narrowest tonal range a channel is stretched from: avoids blowing up
# genuinely low-contrast scenes (fog, snow) and noise.
MIN_SPAN = 0.35
MAX_GAMMA_SHIFT = 0.35  # midtone colour-balance limit (gamma 0.74 .. 1.35)


@dataclass(frozen=True)
class ModernFinish:
    adjustments: Adjustments
    clarity: float  # local-contrast amount 0..1


MODERN_FINISHES: dict[str, ModernFinish] = {
    "natural": ModernFinish(
        Adjustments(contrast=6, highlights=-10, shadows=12, saturation=4), 0.10
    ),
    "clean": ModernFinish(
        Adjustments(exposure=4, brightness=3, contrast=8, highlights=-15, shadows=18, saturation=3),
        0.14,
    ),
    "vivid": ModernFinish(
        Adjustments(contrast=14, highlights=-12, shadows=12, saturation=20), 0.18
    ),
    "professional": ModernFinish(
        Adjustments(contrast=10, highlights=-20, shadows=16, saturation=6), 0.22
    ),
}


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * min(1.0, max(0.0, t))


def level_luts(
    levels: Levels, fading: int, auto_color: bool, monochrome: bool
) -> np.ndarray | None:
    """(256, 3) uint8 lookup tables, or None when nothing would change."""
    t = fading / 100
    color_fix = auto_color and not monochrome
    if t <= 0 and not color_fix:
        return None
    # The same stretch for every channel changes contrast only; per-channel
    # differences (what corrects a cast) are applied as far as the photo
    # shows signs of ageing, so a good photo's real colours are kept.
    low = np.full(3, levels.low.min())
    high = np.full(3, levels.high.max())
    if color_fix:
        evidence = ageing_evidence(levels)
        low = low + evidence * (levels.low - low)
        high = high + evidence * (levels.high - high)
    # How far towards the measured black/white points (always some when
    # correcting colour: unequal channel ranges *are* the cast).
    amount = _lerp(0.6, 1.0, t) if t > 0 else 0.6
    low = low * amount
    high = 1 - (1 - high) * amount
    span = np.maximum(high - low, MIN_SPAN)
    centre = (low + high) / 2
    low = np.where(high - low < MIN_SPAN, centre - span / 2, low)

    x = np.linspace(0, 1, 256, dtype=np.float64)[:, None]
    out = np.clip((x - low) / span, 0, 1)
    if color_fix:
        # Neutralise the remaining midtone cast (yellowing, magenta shift):
        # move each channel's midtone median towards their common grey.
        mid = np.clip(
            (levels.mid * (levels.high - levels.low) + levels.low - low) / span, 0.05, 0.95
        )
        target = float(np.mean(mid))
        gamma = np.log(target) / np.log(mid)
        gamma = np.clip(gamma, 1 - MAX_GAMMA_SHIFT, 1 + MAX_GAMMA_SHIFT)
        # Partial, and only as far as the photo shows signs of ageing (lifted
        # blacks): a warm, red or green scene in a good photo is not a cast.
        gamma = gamma ** (_lerp(0.3, 0.5, t) * ageing_evidence(levels))
        out = out**gamma
    return np.round(out * 255).astype(np.uint8)


def ageing_evidence(levels: Levels) -> float:
    """0 for a photo in good condition, 1 for a clearly faded or yellowed print.

    Signs: lifted blacks (faded dyes) or tinted whites (yellowed paper, a
    dye layer gone). A strongly coloured scene in a good photo has neither.
    """
    lifted = (float(levels.low.max()) - 0.03) / 0.10
    tinted = (float(levels.high.max() - levels.high.min()) - 0.10) / 0.15
    return min(1.0, max(0.0, lifted, tinted))


def recovery_adjustments(
    levels: Levels, fading: int, monochrome: bool, auto_color: bool
) -> Adjustments:
    """Gentle tone shaping after the levels stretch (via the lighting engine)."""
    t = fading / 100
    if t <= 0:
        return Adjustments()
    shadows = 14 * t if levels.shadows_clipped > 0.02 else 6 * t
    highlights = -16 * t if levels.highlights_clipped > 0.02 else -6 * t
    saturation = 0.0 if monochrome or not auto_color else 18 * t  # faded dyes
    return Adjustments(
        contrast=6 * t, shadows=shadows, highlights=highlights, saturation=saturation
    )


def restore_tones(rgb: np.ndarray, fading: int, auto_color: bool, monochrome: bool) -> np.ndarray:
    """Recover faded contrast and correct colour casts. Returns a new array or ``rgb``."""
    if fading <= 0 and not (auto_color and not monochrome):
        return rgb
    levels = measure_levels(rgb)
    luts = level_luts(levels, fading, auto_color, monochrome)
    out = rgb
    if luts is not None:
        out = np.empty_like(rgb)
        for c in range(3):
            np.take(luts[:, c], rgb[..., c], out=out[..., c])
    adjust = recovery_adjustments(levels, fading, monochrome, auto_color)
    if not adjust.is_neutral:
        out = apply_lighting(out, adjust, out=out if out is not rgb else None)
    return out


def adjust(rgb: np.ndarray, adjustments: Adjustments) -> np.ndarray:
    """Lighting-engine adjustment that never modifies ``rgb`` itself."""
    if adjustments.is_neutral:
        return rgb
    return apply_lighting(rgb, adjustments)
