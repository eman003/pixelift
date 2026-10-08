"""Camera looks: camera- and film-inspired renderings, applied after the lighting.

Everything here is NumPy and Pillow (no GUI, no PyTorch) and is shared by the
processing pipeline, the restoration pipeline, the CLI and the live preview.

The looks are Pixelift's own interpretations of the *style* commonly
associated with a camera maker's picture profiles or a film stock. They are
not manufacturer presets, do not reproduce any official colour science, and
imply no affiliation with or endorsement by the brands named.

A look is a :class:`LookRecipe`: the lighting engine's :class:`Adjustments`
(white balance, exposure, tone, contrast, saturation — shared, not
reimplemented) plus what makes a rendering feel photographic rather than like
a filter:

* a highlight shoulder (roll-off) and a matte toe (fade),
* vibrance and eight-band hue / saturation / luminance colour separation,
* skin-tone protection, black-and-white channel mixing and split toning,
* clarity (midtone micro-contrast), sharpening and film grain.

Processing runs in two stages so it fits the existing pipelines:

``apply_look`` (the *grade*) works at the input resolution, right after the
lighting and before the AI upscaler — cheap, and the model then builds on the
final tones, exactly like the lighting. ``finish_look`` (sharpening and grain)
works on the final, upscaled image: the AI model would otherwise smooth the
grain away and exaggerate sharpening halos.

The intensity scales every parameter linearly towards zero, so 0 % (and the
Original look) is always the untouched image. Adding a look is one
``register_look(...)`` call — the UI, settings and CLI list whatever is
registered.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter

# The luminance weights, row-band size and luma helper are shared with lighting.
from pixelift.core.lighting import (
    _CHUNK_PIXELS,
    _LUMA,
    Adjustments,
    TonePlan,
    _luma,
    apply_lighting,
    gamut_safe,
)

ORIGINAL = "original"
CUSTOM = "custom"
USER_PREFIX = "user:"  # saved Custom looks: "user:<name>"
DEFAULT_INTENSITY = 50

# Grain choice: the look's own amount ("auto") or a fixed level.
GRAIN_AUTO = "auto"
GRAIN_LEVELS: dict[str, float] = {"off": 0.0, "low": 30.0, "medium": 55.0, "high": 85.0}
GRAIN_CHOICES: tuple[str, ...] = (GRAIN_AUTO, *GRAIN_LEVELS)

# The colour bands of the hue / saturation / luminance controls and their
# centres on the HSV colour wheel (degrees); neighbours blend smoothly.
BANDS: tuple[str, ...] = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")
BAND_HUES = np.array([0, 30, 60, 120, 180, 240, 270, 300], dtype=np.float32)

_GRAIN_SEED = 0x5EED_1F7


# --- recipes -------------------------------------------------------------------
@dataclass(frozen=True)
class Band:
    """Adjustment of one colour band, each -100..100 (hue in degrees, ±30).

    A positive hue moves the colour towards the next band (red → orange →
    yellow → green …).
    """

    hue: float = 0.0
    saturation: float = 0.0
    luminance: float = 0.0

    def scaled(self, factor: float) -> Band:
        return Band(self.hue * factor, self.saturation * factor, self.luminance * factor)


NEUTRAL_BANDS: tuple[Band, ...] = (Band(),) * len(BANDS)


def bands(**values: tuple[float, float, float]) -> tuple[Band, ...]:
    """``bands(green=(-6, 20, -4))`` -> all eight bands, unnamed ones neutral."""
    unknown = set(values) - set(BANDS)
    if unknown:
        raise ValueError(f"unknown colour band(s): {', '.join(sorted(unknown))}")
    return tuple(Band(*values[name]) if name in values else Band() for name in BANDS)


@dataclass(frozen=True)
class Toning:
    """A colour cast added to the shadows or highlights (split toning)."""

    hue: float = 0.0  # degrees on the colour wheel: 35 warm amber, 200 teal…
    amount: float = 0.0  # 0..100


@dataclass(frozen=True)
class LookRecipe:
    """Everything a look does. 0 everywhere is the untouched image.

    tone            the lighting engine's adjustments (exposure, contrast,
                    highlights, shadows, temperature, tint, saturation…)
    vibrance        -100..100, boosts muted colours more than saturated ones
    bands           hue / saturation / luminance of the eight colour bands
    rolloff         0..100, a gentle highlight shoulder: tones approach white
                    smoothly and very bright colours desaturate like film
    fade            0..100, lifts the black point (matte, film-like toe)
    shadow_tint     split toning of the shadows
    highlight_tint  split toning of the highlights
    monochrome      0..100, conversion to black and white with ``mix``
    mix             black-and-white channel weights (normalised)
    clarity         -100..100, midtone micro-contrast (negative softens)
    sharpen         0..100, fine sharpening (applied to the final image)
    grain           0..100, film grain (applied to the final image)
    skin            0..100, how strongly skin tones keep their colour against
                    the look's colour changes; not scaled by the intensity
    """

    tone: Adjustments = field(default_factory=Adjustments)
    vibrance: float = 0.0
    bands: tuple[Band, ...] = NEUTRAL_BANDS
    rolloff: float = 0.0
    fade: float = 0.0
    shadow_tint: Toning = Toning()
    highlight_tint: Toning = Toning()
    monochrome: float = 0.0
    mix: tuple[float, float, float] = (0.2126, 0.7152, 0.0722)
    clarity: float = 0.0
    sharpen: float = 0.0
    grain: float = 0.0
    skin: float = 0.0

    @property
    def grade_neutral(self) -> bool:
        """Whether :func:`apply_look` leaves the image unchanged."""
        return (
            self.tone.is_neutral
            and self.vibrance == 0
            and all(b == Band() for b in self.bands)
            and self.rolloff == 0
            and self.fade == 0
            and self.shadow_tint.amount == 0
            and self.highlight_tint.amount == 0
            and self.monochrome == 0
            and self.clarity == 0
        )

    @property
    def finish_neutral(self) -> bool:
        """Whether :func:`finish_look` leaves the image unchanged."""
        return self.sharpen == 0 and self.grain == 0

    @property
    def is_neutral(self) -> bool:
        return self.grade_neutral and self.finish_neutral

    @property
    def changes_colour(self) -> bool:
        """Whether a grey image stops being grey."""
        return (
            self.tone.changes_colour
            or self.shadow_tint.amount != 0
            or self.highlight_tint.amount != 0
        )

    def scaled(self, factor: float) -> LookRecipe:
        """The look at ``factor`` (0..1) of its strength."""
        if factor == 1:
            return self
        f = factor
        return dataclasses.replace(
            self,
            tone=self.tone.scaled(f),
            vibrance=self.vibrance * f,
            bands=tuple(b.scaled(f) for b in self.bands),
            rolloff=self.rolloff * f,
            fade=self.fade * f,
            shadow_tint=Toning(self.shadow_tint.hue, self.shadow_tint.amount * f),
            highlight_tint=Toning(self.highlight_tint.hue, self.highlight_tint.amount * f),
            monochrome=self.monochrome * f,
            clarity=self.clarity * f,
            sharpen=self.sharpen * f,
            grain=self.grain * f,
        )

    def clamped(self) -> LookRecipe:
        def pct(v: float, low: float = -100.0) -> float:
            return min(100.0, max(low, float(v)))

        return dataclasses.replace(
            self,
            tone=self.tone.clamped(),
            vibrance=pct(self.vibrance),
            bands=tuple(
                Band(min(30.0, max(-30.0, b.hue)), pct(b.saturation), pct(b.luminance))
                for b in (tuple(self.bands) + NEUTRAL_BANDS)[: len(BANDS)]
            ),
            rolloff=pct(self.rolloff, 0),
            fade=pct(self.fade, 0),
            shadow_tint=Toning(self.shadow_tint.hue % 360, pct(self.shadow_tint.amount, 0)),
            highlight_tint=Toning(
                self.highlight_tint.hue % 360, pct(self.highlight_tint.amount, 0)
            ),
            monochrome=pct(self.monochrome, 0),
            clarity=pct(self.clarity),
            sharpen=pct(self.sharpen, 0),
            grain=pct(self.grain, 0),
            skin=pct(self.skin, 0),
        )

    def signature(self) -> str:
        """Short stable hash of the recipe (for file names)."""
        return hashlib.sha1(repr(self).encode()).hexdigest()[:6]


# --- the Custom look --------------------------------------------------------------
# The Custom look's controls, in UI order. Colour sliders set their band's
# saturation; sharpness is 0..100, everything else -100..100. (Grain is the
# separate grain choice, shared with every look.)
CUSTOM_TONE = ("exposure", "contrast", "highlights", "shadows", "temperature", "tint")
CUSTOM_NAMES: tuple[str, ...] = (*CUSTOM_TONE, "saturation", "vibrance", *BANDS, "sharpness")


def custom_range(name: str) -> tuple[int, int]:
    return (0, 100) if name == "sharpness" else (-100, 100)


def clean_custom_values(values: object) -> dict[str, int]:
    """Custom-look values with every control present, whole and in range
    (missing or unreadable values are 0)."""
    values = values if isinstance(values, Mapping) else {}
    clean = {}
    for name in CUSTOM_NAMES:
        low, high = custom_range(name)
        try:
            clean[name] = min(high, max(low, round(float(values.get(name, 0)))))
        except (TypeError, ValueError, OverflowError):
            clean[name] = 0
    return clean


def custom_recipe(values: Mapping[str, float]) -> LookRecipe:
    """A look from the Custom controls' values (missing names are 0)."""
    clean = clean_custom_values(values)

    def get(name: str) -> float:
        return float(clean[name])

    tone = Adjustments(**{name: get(name) for name in CUSTOM_TONE}, saturation=get("saturation"))
    return LookRecipe(
        tone=tone,
        vibrance=get("vibrance"),
        bands=tuple(Band(saturation=get(name)) for name in BANDS),
        sharpen=get("sharpness"),
        # Natural defaults for hand-made looks: smooth highlights, kind to skin.
        rolloff=15.0 if tone.contrast > 0 or tone.exposure > 0 else 0.0,
        skin=40.0,
    )


# --- registry ----------------------------------------------------------------------
@dataclass(frozen=True)
class Category:
    id: str
    name: str
    brand: bool = False  # a camera maker: labelled "<name>-inspired"


CATEGORIES: tuple[Category, ...] = (
    Category("sony", "Sony", brand=True),
    Category("canon", "Canon", brand=True),
    Category("nikon", "Nikon", brand=True),
    Category("fujifilm", "Fujifilm", brand=True),
    Category("leica", "Leica", brand=True),
    Category("hasselblad", "Hasselblad", brand=True),
    Category("film", "Film"),
    Category("monochrome", "Monochrome"),
)
_CATEGORY_BY_ID = {c.id: c for c in CATEGORIES}

DISCLAIMER = (
    "Camera-inspired looks are Pixelift's own interpretations of popular styles. They "
    "are not official manufacturer presets or colour science, and brand names are "
    "used only to describe the style, with no affiliation or endorsement."
)


@dataclass(frozen=True)
class CameraLook:
    id: str
    name: str
    category: str  # a Category id ("" for Original / Custom)
    description: str
    recipe: LookRecipe = LookRecipe()

    @property
    def monochrome(self) -> bool:
        return self.recipe.monochrome > 0

    @property
    def category_label(self) -> str:
        """ "Fujifilm-inspired", "Film look", "Black & white"…"""
        category = _CATEGORY_BY_ID.get(self.category)
        if category is None:
            return "Your look" if self.id == CUSTOM else ""
        if category.brand:
            return f"{category.name}-inspired"
        return "Black & white" if category.id == "monochrome" else f"{category.name} look"

    def in_category(self, category: str) -> bool:
        """For the category filter: Monochrome also lists brand B&W looks."""
        return self.category == category or (category == "monochrome" and self.monochrome)


_LOOKS: dict[str, CameraLook] = {}


def register_look(look: CameraLook) -> None:
    """Add (or replace) a look. Custom stays last in ``all_looks()``."""
    if look.category and look.category not in _CATEGORY_BY_ID:
        raise ValueError(f"unknown camera look category: {look.category}")
    _LOOKS[look.id] = look
    if CUSTOM in _LOOKS and look.id != CUSTOM:
        _LOOKS[CUSTOM] = _LOOKS.pop(CUSTOM)


def all_looks() -> list[CameraLook]:
    return list(_LOOKS.values())


def get_look(look_id: str) -> CameraLook:
    return _LOOKS[look_id]


def has_look(look_id: str) -> bool:
    return look_id in _LOOKS


A = Adjustments
T = Toning
_BW_CLASSIC = (0.30, 0.59, 0.11)
_BW_RED_FILTER = (0.42, 0.48, 0.10)  # glowing skin, darker skies

for _look in (
    CameraLook(ORIGINAL, "Original", "", "No camera look"),
    # --- Sony-inspired: clean, modern, crisp micro-contrast, neutral to cool.
    CameraLook(
        "sony-natural",
        "Sony Natural",
        "sony",
        "Clean, neutral rendering with crisp micro-contrast",
        LookRecipe(
            tone=A(contrast=8, highlights=-10, shadows=6, temperature=-4, saturation=-4),
            vibrance=6,
            bands=bands(green=(3, 0, 0), blue=(0, 6, -4)),
            rolloff=15,
            clarity=14,
            sharpen=22,
            skin=35,
        ),
    ),
    CameraLook(
        "sony-vivid",
        "Sony Vivid",
        "sony",
        "Punchy, controlled colour with a cool, modern edge",
        LookRecipe(
            tone=A(contrast=22, highlights=-12, shadows=-4, temperature=-3, saturation=16),
            vibrance=12,
            bands=bands(red=(0, 8, -4), green=(2, 14, -4), aqua=(0, 10, -2), blue=(-3, 14, -8)),
            rolloff=15,
            clarity=18,
            sharpen=28,
            skin=40,
        ),
    ),
    CameraLook(
        "sony-portrait",
        "Sony Portrait",
        "sony",
        "Clean, neutral skin with gentle contrast and crisp detail",
        LookRecipe(
            tone=A(contrast=6, highlights=-16, shadows=8, temperature=-1, tint=4, saturation=-4),
            bands=bands(red=(3, -8, 4), orange=(0, -6, 6), blue=(0, 6, -4)),
            rolloff=30,
            sharpen=16,
            skin=75,
        ),
    ),
    CameraLook(
        "sony-cinematic",
        "Sony Cinematic",
        "sony",
        "Video-style: soft highlights, muted colour, subtle teal shadows",
        LookRecipe(
            tone=A(contrast=10, highlights=-25, shadows=8, temperature=2, saturation=-10),
            bands=bands(orange=(0, 4, 4), green=(-8, -15, 0), blue=(-6, -5, -6)),
            rolloff=45,
            fade=12,
            shadow_tint=T(190, 15),
            highlight_tint=T(35, 10),
            clarity=6,
            sharpen=10,
            skin=55,
        ),
    ),
    # --- Canon-inspired: pleasant skin, warm reds, smooth highlights, warm.
    CameraLook(
        "canon-natural",
        "Canon Natural",
        "canon",
        "Faithful, slightly warm colour with smooth highlights",
        LookRecipe(
            tone=A(contrast=4, highlights=-10, shadows=6, temperature=6, saturation=-2),
            bands=bands(red=(2, 4, 0)),
            rolloff=25,
            sharpen=10,
            skin=45,
        ),
    ),
    CameraLook(
        "canon-standard",
        "Canon Standard",
        "canon",
        "Warm, rich reds and natural contrast",
        LookRecipe(
            tone=A(contrast=14, highlights=-8, shadows=2, temperature=8, saturation=10),
            vibrance=8,
            bands=bands(
                red=(-2, 12, -2),
                orange=(0, 4, 2),
                yellow=(-4, 6, 0),
                green=(-4, 8, 0),
                blue=(0, 8, -4),
            ),
            rolloff=25,
            clarity=8,
            sharpen=20,
            skin=45,
        ),
    ),
    CameraLook(
        "canon-portrait",
        "Canon Portrait",
        "canon",
        "Warm, luminous skin and soft transitions",
        LookRecipe(
            tone=A(contrast=2, highlights=-18, shadows=12, temperature=12, tint=4, saturation=-2),
            bands=bands(red=(4, 0, 6), orange=(2, -4, 8)),
            rolloff=35,
            clarity=-14,
            sharpen=4,
            skin=80,
        ),
    ),
    CameraLook(
        "canon-landscape",
        "Canon Landscape",
        "canon",
        "Lush greens, deep skies and crisp detail",
        LookRecipe(
            tone=A(contrast=18, highlights=-15, shadows=4, temperature=4, saturation=14),
            vibrance=16,
            bands=bands(green=(-6, 22, -2), aqua=(0, 10, -4), blue=(-4, 20, -8)),
            rolloff=20,
            clarity=22,
            sharpen=28,
            skin=25,
        ),
    ),
    # --- Nikon-inspired: neutral, natural greens and blues, balanced.
    CameraLook(
        "nikon-neutral",
        "Nikon Neutral",
        "nikon",
        "Flat, true-to-life rendering that keeps every tone",
        LookRecipe(
            tone=A(contrast=-4, highlights=-12, shadows=10, saturation=-8),
            rolloff=20,
            sharpen=10,
            skin=35,
        ),
    ),
    CameraLook(
        "nikon-standard",
        "Nikon Standard",
        "nikon",
        "Balanced contrast with natural greens and blues",
        LookRecipe(
            tone=A(contrast=12, highlights=-8, temperature=-1, saturation=8),
            vibrance=6,
            bands=bands(yellow=(-4, 4, 0), green=(4, 8, -2), blue=(0, 10, -4)),
            rolloff=15,
            clarity=10,
            sharpen=22,
            skin=40,
        ),
    ),
    CameraLook(
        "nikon-portrait",
        "Nikon Portrait",
        "nikon",
        "Natural, slightly warm skin with softened contrast",
        LookRecipe(
            tone=A(contrast=0, highlights=-14, shadows=12, temperature=6, saturation=-2),
            bands=bands(red=(2, -6, 4), orange=(0, -8, 6), green=(2, 6, 0), blue=(0, 4, -2)),
            rolloff=25,
            clarity=-12,
            sharpen=4,
            skin=80,
        ),
    ),
    CameraLook(
        "nikon-landscape",
        "Nikon Landscape",
        "nikon",
        "Clear blues, fresh greens and strong detail",
        LookRecipe(
            tone=A(contrast=16, highlights=-14, shadows=4, temperature=-3, saturation=12),
            vibrance=14,
            bands=bands(green=(4, 18, -4), aqua=(0, 12, -4), blue=(0, 22, -10)),
            rolloff=15,
            clarity=20,
            sharpen=28,
            skin=25,
        ),
    ),
    # --- Fujifilm-inspired: rich but controlled colour, pleasant greens,
    # softer highlights, film-like contrast.
    CameraLook(
        "fujifilm-classic",
        "Fujifilm Classic",
        "fujifilm",
        "Muted, documentary colour with hard shadows and soft highlights",
        LookRecipe(
            tone=A(contrast=12, highlights=-18, shadows=-6, temperature=2, saturation=-18),
            bands=bands(
                red=(-2, -6, -4), yellow=(-4, -14, 0), green=(-6, -14, -4), blue=(-6, -8, -10)
            ),
            rolloff=35,
            fade=6,
            shadow_tint=T(200, 10),
            highlight_tint=T(40, 6),
            clarity=10,
            grain=25,
            skin=55,
        ),
    ),
    CameraLook(
        "fujifilm-provia",
        "Fujifilm Provia",
        "fujifilm",
        "The all-rounder: natural, lively colour and pleasant greens",
        LookRecipe(
            tone=A(contrast=10, highlights=-12, shadows=2, saturation=6),
            vibrance=8,
            bands=bands(green=(-4, 8, 0), blue=(-2, 6, -4)),
            rolloff=25,
            clarity=6,
            sharpen=12,
            skin=50,
        ),
    ),
    CameraLook(
        "fujifilm-velvia",
        "Fujifilm Velvia",
        "fujifilm",
        "Saturated slide-film colour for landscapes",
        LookRecipe(
            tone=A(contrast=22, highlights=-12, shadows=-10, temperature=2, tint=3, saturation=20),
            vibrance=18,
            bands=bands(
                red=(-2, 14, -4),
                green=(-8, 24, -6),
                blue=(-4, 20, -10),
                purple=(0, 10, 0),
                magenta=(0, 10, 0),
            ),
            rolloff=20,
            clarity=14,
            sharpen=18,
            skin=60,
        ),
    ),
    CameraLook(
        "fujifilm-astia",
        "Fujifilm Astia",
        "fujifilm",
        "Soft slide film: gentle contrast, flattering skin, clear skies",
        LookRecipe(
            tone=A(contrast=4, highlights=-16, shadows=10, temperature=3, saturation=4),
            vibrance=4,
            bands=bands(red=(2, -4, 4), orange=(0, -6, 6), green=(-4, 8, 0), blue=(0, 10, -2)),
            rolloff=35,
            clarity=-4,
            sharpen=8,
            skin=75,
        ),
    ),
    CameraLook(
        "fujifilm-classic-negative",
        "Fujifilm Classic Negative",
        "fujifilm",
        "Snapshot negative film: hard tones, teal shadows, shifted greens",
        LookRecipe(
            tone=A(contrast=18, highlights=-22, shadows=-8, temperature=-2, saturation=-10),
            bands=bands(
                red=(4, 4, -6),
                orange=(-2, 0, 0),
                yellow=(6, -10, 0),
                green=(10, -10, -6),
                blue=(-8, -4, -10),
            ),
            rolloff=40,
            fade=8,
            shadow_tint=T(190, 14),
            highlight_tint=T(45, 8),
            clarity=8,
            grain=25,
            skin=50,
        ),
    ),
    # --- Leica-inspired: natural colour, strong tonal separation, subtle
    # contrast, clean highlights; documentary character.
    CameraLook(
        "leica-natural",
        "Leica Natural",
        "leica",
        "Natural colour with deep tonal separation and clean highlights",
        LookRecipe(
            tone=A(contrast=8, highlights=-14, shadows=4, temperature=2, saturation=-6),
            bands=bands(red=(0, 4, -4), blue=(0, 0, -6)),
            rolloff=30,
            clarity=16,
            sharpen=12,
            skin=65,
        ),
    ),
    CameraLook(
        "leica-monochrome",
        "Leica Monochrome",
        "leica",
        "Rich black and white with glowing skin and deep skies",
        LookRecipe(
            tone=A(contrast=14, highlights=-16, shadows=-4),
            rolloff=30,
            monochrome=100,
            mix=_BW_RED_FILTER,
            clarity=22,
            sharpen=14,
            grain=20,
        ),
    ),
    CameraLook(
        "leica-filmic",
        "Leica Filmic",
        "leica",
        "Editorial film character: soft blacks, warm highlights",
        LookRecipe(
            tone=A(contrast=10, highlights=-20, shadows=4, temperature=4, saturation=-12),
            bands=bands(green=(-6, -10, 0)),
            rolloff=40,
            fade=10,
            shadow_tint=T(210, 8),
            highlight_tint=T(40, 8),
            clarity=12,
            grain=20,
            skin=55,
        ),
    ),
    # --- Hasselblad-inspired: smooth gradations, accurate natural colour.
    CameraLook(
        "hasselblad-natural",
        "Hasselblad Natural",
        "hasselblad",
        "Smooth gradations and accurate, natural colour",
        LookRecipe(
            tone=A(contrast=4, highlights=-16, shadows=8, saturation=2),
            vibrance=4,
            bands=bands(red=(0, 2, 0), orange=(0, -2, 4)),
            rolloff=35,
            clarity=6,
            sharpen=10,
            skin=70,
        ),
    ),
    CameraLook(
        "hasselblad-filmic",
        "Hasselblad Filmic",
        "hasselblad",
        "Medium-format film feel: gentle shoulder and soft colour",
        LookRecipe(
            tone=A(contrast=10, highlights=-22, shadows=6, temperature=3, saturation=-6),
            rolloff=45,
            fade=8,
            shadow_tint=T(200, 10),
            highlight_tint=T(35, 8),
            clarity=8,
            grain=20,
            skin=60,
        ),
    ),
    # --- Generic film looks.
    CameraLook(
        "kodak",
        "Kodak-inspired",
        "film",
        "Warm, golden consumer-film colour",
        LookRecipe(
            tone=A(contrast=16, highlights=-14, shadows=-4, temperature=10, tint=2, saturation=12),
            vibrance=6,
            bands=bands(red=(2, 12, -4), yellow=(-2, 14, 2), green=(-6, 4, -4), blue=(-6, 10, -10)),
            rolloff=30,
            fade=4,
            highlight_tint=T(45, 8),
            grain=35,
            skin=50,
        ),
    ),
    CameraLook(
        "portra",
        "Portra-inspired",
        "film",
        "Soft portrait negative: warm skin, pastel colour, gentle contrast",
        LookRecipe(
            tone=A(contrast=-2, highlights=-22, shadows=12, temperature=8, saturation=-6),
            bands=bands(red=(4, -8, 4), orange=(2, -6, 8), green=(-6, -8, 4), blue=(-4, -10, 4)),
            rolloff=45,
            fade=8,
            shadow_tint=T(190, 6),
            highlight_tint=T(40, 6),
            grain=35,
            skin=75,
        ),
    ),
    CameraLook(
        "ektar",
        "Ektar-inspired",
        "film",
        "Vivid, fine-grained colour negative",
        LookRecipe(
            tone=A(contrast=18, highlights=-12, shadows=-4, temperature=3, saturation=18),
            vibrance=14,
            bands=bands(
                red=(0, 16, -4),
                orange=(0, 8, 0),
                yellow=(-2, 10, 0),
                green=(-4, 14, -4),
                blue=(-4, 22, -10),
            ),
            rolloff=25,
            clarity=8,
            grain=20,
            skin=60,
        ),
    ),
    CameraLook(
        "classic-film",
        "Classic Film",
        "film",
        "Faded vintage print with warm highlights and visible grain",
        LookRecipe(
            tone=A(contrast=12, highlights=-20, shadows=-2, temperature=6, saturation=-10),
            bands=bands(green=(-6, -10, 0)),
            rolloff=40,
            fade=20,
            shadow_tint=T(220, 10),
            highlight_tint=T(40, 12),
            grain=55,
            skin=55,
        ),
    ),
    CameraLook(
        "modern-film",
        "Modern Film",
        "film",
        "Clean film emulation: soft shoulder, restrained colour",
        LookRecipe(
            tone=A(contrast=8, highlights=-18, shadows=8, temperature=3, saturation=-4),
            bands=bands(orange=(0, -4, 4), green=(-4, -6, 0), blue=(-6, -4, -4)),
            rolloff=35,
            fade=8,
            shadow_tint=T(195, 8),
            clarity=6,
            grain=25,
            skin=60,
        ),
    ),
    CameraLook(
        "cinematic-film",
        "Cinematic",
        "film",
        "Teal-and-orange film grade with deep, soft shadows",
        LookRecipe(
            tone=A(contrast=20, highlights=-24, shadows=-6, temperature=-2, saturation=-10),
            bands=bands(orange=(0, 6, 4), green=(-10, -20, -4), blue=(-8, 0, -8)),
            rolloff=45,
            fade=6,
            shadow_tint=T(190, 22),
            highlight_tint=T(35, 16),
            clarity=8,
            skin=60,
        ),
    ),
    CameraLook(
        "black-white",
        "Black & White",
        "monochrome",
        "Classic black-and-white film with full tonal range",
        LookRecipe(
            tone=A(contrast=18, highlights=-10, shadows=-6),
            rolloff=25,
            monochrome=100,
            mix=_BW_CLASSIC,
            clarity=14,
            grain=25,
        ),
    ),
    CameraLook(CUSTOM, "Custom", "", "Set each adjustment yourself"),
):
    register_look(_look)
del A, T


# --- what the user picked ---------------------------------------------------------
@dataclass(frozen=True)
class CameraLookSettings:
    """The selected look, its intensity, the grain choice and the Custom values.

    ``custom`` is the recipe of the Custom look — or of the selected saved
    look (``user:<name>``), which then behaves like a built-in look.
    """

    look: str = ORIGINAL
    intensity: int = DEFAULT_INTENSITY  # percent, 0..100; ignored by Custom
    grain: str = GRAIN_AUTO
    custom: LookRecipe = field(default_factory=LookRecipe)

    @property
    def user_look(self) -> bool:
        return self.look.startswith(USER_PREFIX)

    def _strength(self) -> float:
        return 1.0 if self.look == CUSTOM else min(100, max(0, self.intensity)) / 100

    def recipe(self) -> LookRecipe:
        """The effective recipe (intensity and grain choice applied)."""
        if self.look == CUSTOM or self.user_look:
            base = self.custom.clamped()
        elif self.look in _LOOKS:
            base = _LOOKS[self.look].recipe
        else:
            return LookRecipe()
        if base.is_neutral and self.look == ORIGINAL:
            return base  # Original never gains grain
        if self.grain in GRAIN_LEVELS:
            base = dataclasses.replace(base, grain=GRAIN_LEVELS[self.grain])
        return base.scaled(self._strength())

    @property
    def active(self) -> bool:
        return not self.recipe().is_neutral

    def tag(self) -> str:
        """Short filename-safe label for the effective look; "" when neutral.

        Different effective looks give different tags, so an output made with
        another look is never mistaken for this one's.
        """
        recipe = self.recipe()
        if recipe.is_neutral:
            return ""
        if self.look == CUSTOM:
            return f"custom-look-{recipe.signature()}"
        if self.user_look:
            return f"look-{recipe.signature()}"
        tag = self.look
        intensity = min(100, max(0, self.intensity))
        if intensity != 100:
            tag += f"-{intensity}"
        if self.grain != GRAIN_AUTO:
            tag += f"-grain-{self.grain}"
        return tag

    def name(self) -> str:
        if self.user_look:
            return self.look[len(USER_PREFIX) :]
        return _LOOKS[self.look].name if self.look in _LOOKS else "Original"


# --- processing -----------------------------------------------------------------
def apply_look(
    rgb: np.ndarray,
    recipe: LookRecipe,
    out: np.ndarray | None = None,
    frame: tuple[int, int] | None = None,
    lut_size: int | None = None,
) -> np.ndarray:
    """Apply the grade (colour, tone and clarity) to an (H, W, 3) ``uint8`` array.

    Returns ``rgb`` itself when the grade is neutral. Pass ``out=rgb`` to
    work in place; either way the float working memory is bounded by
    processing a band of rows at a time. ``frame`` is the size of the whole
    picture in ``rgb``'s pixels when ``rgb`` is only a crop of it (so the
    clarity radius matches). ``lut_size`` trades accuracy for speed (the
    live preview uses ``PREVIEW_LUT_SIZE``).
    """
    _check(rgb)
    if recipe.grade_neutral:
        return _unchanged(rgb, out)
    recipe = recipe.clamped()
    if out is None:
        out = np.empty_like(rgb)
    lut = grade_lut(recipe, lut_size or LUT_SIZE)
    height, width = rgb.shape[:2]
    rows = max(1, _CHUNK_PIXELS // max(width, 1))
    for y0 in range(0, height, rows):
        y1 = min(height, y0 + rows)
        band = Image.fromarray(np.ascontiguousarray(rgb[y0:y1]), "RGB").filter(lut)
        out[y0:y1] = np.asarray(band)
    if recipe.clarity:
        span = max(frame or (width, height))
        sigma = min(40.0, max(2.0, span / 150))
        amount = 0.6 * recipe.clarity / 100

        def clarity(y8: np.ndarray, a: int, b: int, _y0: int) -> np.ndarray:
            y = y8[a:b].astype(np.float32) / 255
            detail = y - _blur(y8, sigma)[a:b]
            return amount * 4 * y * (1 - y) * detail

        _luma_pass(out, out, int(3 * sigma) + 2, clarity)
    return out


def finish_look(
    rgb: np.ndarray,
    recipe: LookRecipe,
    out: np.ndarray | None = None,
    frame: tuple[int, int] | None = None,
    output: tuple[int, int] | None = None,
) -> np.ndarray:
    """Sharpening and grain on the final (upscaled) ``uint8`` RGB image.

    Grain is deterministic (the same image gets the same grain) and its size
    follows the picture. ``frame`` is the whole picture's size in ``rgb``'s
    pixels (default ``rgb``'s size; set it for crops). For a preview smaller
    than the export, ``output`` is the export's size: grain and sharpening
    are then rendered as the export would look scaled down to the preview,
    rather than at preview pixels (which would look coarser and stronger).
    """
    _check(rgb)
    if recipe.finish_neutral:
        return _unchanged(rgb, out)
    recipe = recipe.clamped()
    if out is None:
        out = np.empty_like(rgb)
    height, width = rgb.shape[:2]
    span = max(frame or (width, height))
    ratio = span / max(output) if output else 1.0  # preview pixels per export pixel
    sharpen = 0.8 * recipe.sharpen / 100 * min(1.0, ratio)
    grain = 0.05 * recipe.grain / 100
    # Grain cells are 1/3000 of the picture (at least one export pixel).
    grain_size = max(1.0, span / ratio / 3000) * ratio
    if grain_size < 1:
        # Finer than a preview pixel: the export's grain averages out when
        # scaled down, by the cell size (the standard deviation of a mean).
        grain *= grain_size
        grain_size = 1.0
    margin = np.float32(0.03)
    floor = np.float32(1.5 / 255)  # leave noise-sized wiggles alone

    def finish(y8: np.ndarray, a: int, b: int, y0: int) -> np.ndarray:
        y = y8[a:b].astype(np.float32) / 255
        delta = np.zeros_like(y)
        if sharpen:
            detail = y - _blur(y8, 1.0)[a:b]
            detail = np.sign(detail) * np.maximum(np.abs(detail) - floor, 0)
            sharp = y + sharpen * detail
            # Limit overshoot to the local range: no halos around edges.
            low8, high8 = _local_range(y8[max(0, a - 1) : b + 1])
            skip = a - max(0, a - 1)
            low = low8[skip : skip + b - a].astype(np.float32) / 255 - margin
            high = high8[skip : skip + b - a].astype(np.float32) / 255 + margin
            delta += np.minimum(np.maximum(sharp, low), high) - y
        if grain:
            # Strongest in the midtones, as in film; blacks and whites stay clean.
            weight = 0.35 + 2.6 * y * (1 - y)
            delta += grain * weight * _grain_field(y0, y0 + (b - a), width, grain_size)
        return delta

    _luma_pass(rgb, out, 4 if sharpen else 0, finish)
    return out


def apply_to_pil(
    img: Image.Image,
    recipe: LookRecipe,
    lighting: Adjustments | None = None,
    frame: tuple[int, int] | None = None,
    quick: bool = False,
    output: tuple[int, int] | None = None,
    lut_size: int | None = None,
) -> Image.Image:
    """Lighting, grade and finish on a Pillow image, keeping alpha (the preview).

    Matches the export, which applies the finish after upscaling (pass the
    export's size as ``output``; see :func:`finish_look`). ``quick`` uses the
    smaller lookup table (for live slider updates); ``lut_size`` sets it.
    """
    lighting = lighting or Adjustments()
    if recipe.is_neutral and lighting.is_neutral:
        return img
    if img.mode in ("P", "PA") or (img.mode != "RGBA" and "transparency" in img.info):
        img = img.convert("RGBA")  # palette or colour-key transparency
    alpha = img.getchannel("A") if img.mode in ("RGBA", "LA") else None
    rgb = np.array(img.convert("RGB"))
    apply_lighting(rgb, lighting, out=rgb)
    size = lut_size or (PREVIEW_LUT_SIZE if quick else None)
    apply_look(rgb, recipe, out=rgb, frame=frame, lut_size=size)
    finish_look(rgb, recipe, out=rgb, frame=frame, output=output)
    result = Image.fromarray(rgb, "RGB")
    if alpha is not None:
        result.putalpha(alpha)
    return result


def sample_image(width: int = 160, height: int = 120) -> Image.Image:
    """A small synthetic scene (sky, foliage, skin, neutrals) for look thumbnails
    when no photo is available."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    u, v = x / max(width - 1, 1), y / max(height - 1, 1)
    sky = np.stack([0.45 + 0.25 * v, 0.62 + 0.2 * v, 0.92 - 0.05 * v], axis=-1)
    hills = v > 0.55 + 0.08 * np.sin(u * 7)
    foliage = np.stack([0.22 + 0.1 * u, 0.42 + 0.1 * (1 - v), 0.16 + 0 * u], axis=-1)
    img = np.where(hills[..., None], foliage, sky)
    face = ((u - 0.68) / 0.13) ** 2 + ((v - 0.42) / 0.2) ** 2 < 1
    skin = np.stack([0.86 - 0.2 * v, 0.64 - 0.18 * v, 0.52 - 0.16 * v], axis=-1)
    img = np.where(face[..., None], skin, img)
    ramp = v > 0.9
    img = np.where(ramp[..., None], np.repeat(u[..., None], 3, axis=-1), img)
    red = (u < 0.18) & (v > 0.6) & (v < 0.85)
    img = np.where(red[..., None], np.array([0.78, 0.16, 0.14], dtype=np.float32), img)
    return Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")


# --- implementation ---------------------------------------------------------------
LUT_SIZE = 65  # Pillow's maximum; grid points 4 levels apart (3.3 MB)
PREVIEW_LUT_SIZE = 33  # the usual grading-LUT size: ~6x quicker to build (0.4 MB)
THUMBNAIL_LUT_SIZE = 17  # look cards: tiny images, built in a few ms, never cached


def grade_lut(recipe: LookRecipe, size: int = LUT_SIZE) -> ImageFilter.Color3DLUT:
    """The grade's lookup table; recent export and preview tables are cached.

    Separate small caches, so rendering the look cards (uncached) never
    evicts the tables the live preview and the export are using.
    """
    if size >= LUT_SIZE:
        return _full_lut(recipe, size)
    if size >= PREVIEW_LUT_SIZE:
        return _preview_lut(recipe, size)
    return _build_lut(recipe, size)


def _build_lut(recipe: LookRecipe, size: int) -> ImageFilter.Color3DLUT:
    """The per-pixel grade baked into a 3D lookup table.

    Every per-pixel step of the grade depends only on the pixel's colour, so
    it is evaluated once on a grid (``_GradePlan``, the reference
    implementation) and applied with Pillow's trilinear 3D LUT in C — tens of
    times faster than evaluating it per pixel, which keeps slider drags live.
    """
    levels = np.linspace(0, 255, size).round().astype(np.uint8)
    b, g, r = np.meshgrid(levels, levels, levels, indexing="ij")  # red varies fastest
    grid = np.stack([r, g, b], axis=-1).reshape(-1, 1, 3)
    table = np.clip(_GradePlan(recipe).run(grid), 0, 1).reshape(-1, 3)
    return ImageFilter.Color3DLUT(size, table.astype(np.float32))


_full_lut = functools.lru_cache(maxsize=4)(_build_lut)
_preview_lut = functools.lru_cache(maxsize=8)(_build_lut)


def _check(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("expected an (H, W, 3) uint8 array")


def _unchanged(rgb: np.ndarray, out: np.ndarray | None) -> np.ndarray:
    if out is not None and out is not rgb:
        out[...] = rgb
        return out
    return rgb


def _hue_wheel(degrees: float) -> np.ndarray:
    """Zero-luminance chroma direction of a hue (max |channel| = 1)."""
    h = (degrees % 360) / 60
    k = (np.array([5.0, 3.0, 1.0]) + h) % 6
    rgb = 1 - np.clip(np.minimum(k, 4 - k), 0, 1)  # HSV with s = v = 1
    chroma = rgb - float(rgb @ _LUMA)
    return (chroma / np.abs(chroma).max()).astype(np.float32)


class _GradePlan:
    """The per-pixel part of a recipe on float RGB (the LUT's reference)."""

    def __init__(self, recipe: LookRecipe) -> None:
        self.recipe = recipe
        tone = recipe.tone
        # Saturation runs after the tonal stages and the reference copy taken
        # for skin protection, so protected skin keeps the look's white
        # balance and tones but not its colour boost.
        self.tone = TonePlan(dataclasses.replace(tone, saturation=0))
        self.saturation = TonePlan(Adjustments(saturation=tone.saturation))
        self.tone_active = not dataclasses.replace(tone, saturation=0).is_neutral
        self.rolloff = 0.9 * recipe.rolloff / 100
        self.fade = 0.12 * recipe.fade / 100
        self.vibrance = 0.8 * recipe.vibrance / 100
        self.hue_shift = np.array([b.hue for b in recipe.bands], dtype=np.float32)
        self.band_sat = np.array([b.saturation for b in recipe.bands], dtype=np.float32) / 100
        self.band_lum = np.array([b.luminance for b in recipe.bands], dtype=np.float32) / 100
        self.bands_active = any(b != Band() for b in recipe.bands)
        self.skin = recipe.skin / 100 if recipe.monochrome < 100 else 0.0
        self.monochrome = recipe.monochrome / 100
        mix = np.array(recipe.mix, dtype=np.float32).clip(0)
        self.mix = mix / max(float(mix.sum()), 1e-6)
        self.tints = [
            (_hue_wheel(t.hue), 0.12 * t.amount / 100, highlights)
            for t, highlights in ((recipe.shadow_tint, False), (recipe.highlight_tint, True))
            if t.amount
        ]

    def run(self, rgb: np.ndarray) -> np.ndarray:
        """uint8 (..., 3) -> float32 in about 0..1."""
        x = self.tone.decode(rgb)
        if self.tone_active:
            x = self.tone.run(x)
        np.clip(x, 0, 1, out=x)
        if self.rolloff:
            x = self._rolloff(x)
        reference = x.copy() if self.skin else None
        if self.saturation.saturation:
            x = self.saturation.run(x)
        if self.bands_active or self.vibrance:
            x = self._colour(x)
        if reference is not None:
            x = self._protect_skin(x, reference)
        if self.monochrome:
            np.clip(x, 0, 1, out=x)
            mono = x @ self.mix
            x += self.monochrome * (mono[..., None] - x)
        if self.tints:
            np.clip(x, 0, 1, out=x)
            lum = _luma(x)
            for direction, amount, highlights in self.tints:
                # Peaks in the shadows (or highlights); black and white stay neutral.
                weight = lum * lum * (1 - lum) if highlights else lum * (1 - lum) * (1 - lum)
                x += (amount * 27 / 4 * weight)[..., None] * direction
        if self.fade:
            np.clip(x, 0, 1, out=x)
            x *= np.float32(1 - self.fade)
            x += np.float32(self.fade)
        return x

    def _rolloff(self, x: np.ndarray) -> np.ndarray:
        """Highlight shoulder: above mid-grey, tones ease into white.

        y = k + (1-k)(u + r u²(1-u)) with u the position above the knee k:
        fixed ends, monotonic for r <= 1, and the slope at white is 1 - r.
        Brightening is done towards white, so near-white colours desaturate
        gently instead of clipping into neon.
        """
        knee = np.float32(0.5)
        lum = _luma(x)
        u = np.clip((lum - knee) / (1 - knee), 0, 1)
        target = lum + (1 - knee) * self.rolloff * u * u * (1 - u)
        a = (1 - target) / np.maximum(1 - lum, np.float32(1e-6))
        x *= a[..., None]
        x += (1 - a)[..., None]
        return x

    def _colour(self, x: np.ndarray) -> np.ndarray:
        """Per-band hue, saturation and luminance, and vibrance."""
        np.clip(x, 0, 1, out=x)
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        hi = np.maximum(np.maximum(r, g), b)
        lo = np.minimum(np.minimum(r, g), b)
        chroma = hi - lo
        factor = np.ones_like(chroma)
        if self.bands_active:
            hue = _hsv_hue(x, hi, chroma)
            # Near-greys carry no reliable hue: leave them (and their noise) alone.
            weight = np.clip(chroma / np.float32(0.1), 0, 1)

            def at(values: np.ndarray) -> np.ndarray:
                return np.interp(hue, BAND_HUES, values, period=360).astype(np.float32) * weight

            if self.hue_shift.any():
                x = _rotate_hue(x, np.radians(at(self.hue_shift)))
            if self.band_lum.any():
                lum_change = 0.5 * at(self.band_lum)
                darker = lum_change < 0
                scale = np.where(darker, 1 + lum_change, 1 - lum_change)
                x *= scale[..., None]
                x += np.where(darker, 0, lum_change)[..., None]
            if self.band_sat.any():
                factor *= 1 + at(self.band_sat)
        if self.vibrance:
            # Muted colours gain the most; saturated ones (and skin) little.
            factor *= 1 + self.vibrance * (1 - 0.8 * np.clip(chroma / np.float32(0.6), 0, 1))
        if self.band_sat.any() or self.vibrance:
            np.clip(x, 0, 1, out=x)
            lum = _luma(x)[..., None]
            x -= lum
            hi = x.max(axis=-1, keepdims=True)
            lo = x.min(axis=-1, keepdims=True)
            x *= gamut_safe(factor[..., None], lum, hi, lo)
            x += lum
        return x

    def _protect_skin(self, x: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Skin keeps its colour from before the look's colour changes.

        The look's tones (and white balance) still apply; only the chroma is
        pulled back towards the reference, as far as the pixel looks like skin.
        """
        r, g, b = reference[..., 0], reference[..., 1], reference[..., 2]
        hi = np.maximum(np.maximum(r, g), b)
        lo = np.minimum(np.minimum(r, g), b)
        chroma = hi - lo
        hue = _hsv_hue(reference, hi, chroma)
        sat = chroma / np.maximum(hi, np.float32(1e-6))
        # Skin: red-orange hues (centred on 20°) of moderate saturation, not too dark.
        # (Hue distance wraps around: pinkish skin sits just below 360°.)
        mask = np.clip(1 - np.abs((hue + 160) % 360 - 180) / 34, 0, 1)
        mask *= np.clip((sat - 0.08) / 0.1, 0, 1) * np.clip((0.75 - sat) / 0.15, 0, 1)
        mask *= np.clip((hi - 0.15) / 0.15, 0, 1)
        mask *= np.float32(self.skin)
        lum = _luma(x)
        ref_lum = _luma(reference)
        # x + m * ((ref - ref_lum) - (x - lum)), keeping x's luminance.
        reference -= ref_lum[..., None]
        reference += lum[..., None]
        reference -= x
        reference *= mask[..., None]
        x += reference
        return x


def _hsv_hue(x: np.ndarray, hi: np.ndarray, chroma: np.ndarray) -> np.ndarray:
    """HSV hue in degrees (0..360) of float RGB."""
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    c = np.maximum(chroma, np.float32(1e-6))
    hue = np.where(
        hi == r, np.mod((g - b) / c, 6), np.where(hi == g, (b - r) / c + 2, (r - g) / c + 4)
    )
    return hue * np.float32(60)


def _rotate_hue(x: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotate colours about the grey axis by ``angle`` (radians, per pixel).

    A positive angle turns red towards green (the HSV hue direction).
    Luminance is kept.
    """
    lum = _luma(x)
    mean = x.mean(axis=-1, keepdims=True)
    d = x - mean
    cos, sin = np.cos(angle)[..., None], np.sin(angle)[..., None]
    cross = np.stack(
        [d[..., 2] - d[..., 1], d[..., 0] - d[..., 2], d[..., 1] - d[..., 0]], axis=-1
    ) / np.float32(np.sqrt(3))
    x = mean + d * cos + cross * sin
    x += (lum - _luma(x))[..., None]
    return x


def _luma8(rgb: np.ndarray) -> np.ndarray:
    """Rec. 709 luminance of uint8 RGB as uint8."""
    y = rgb[..., 0].astype(np.uint32) * 54
    y += rgb[..., 1].astype(np.uint32) * 183
    y += rgb[..., 2].astype(np.uint32) * 19
    y += 128
    return (y >> 8).astype(np.uint8)


def _blur(y8: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a uint8 plane, as float32 in 0..1 (Pillow's fast blur)."""
    blurred = Image.fromarray(y8, "L").filter(ImageFilter.GaussianBlur(sigma))
    return np.asarray(blurred, dtype=np.float32) / 255


def _local_range(y8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """3×3 minimum and maximum of a uint8 plane (separable, edges repeated)."""

    def extreme(fn: Callable[..., np.ndarray]) -> np.ndarray:
        p = np.pad(y8, 1, mode="edge")
        rows = fn(fn(p[:-2], p[1:-1]), p[2:])
        return fn(fn(rows[:, :-2], rows[:, 1:-1]), rows[:, 2:])

    return extreme(np.minimum), extreme(np.maximum)


def _luma_pass(
    rgb: np.ndarray,
    out: np.ndarray,
    halo: int,
    delta: Callable[[np.ndarray, int, int, int], np.ndarray],
) -> None:
    """``out = rgb + delta`` (a luminance change added to every channel).

    Works in bands of rows; ``delta(luma8, a, b, y0)`` sees the band's uint8
    luminance with ``halo`` rows of context and returns the change (0..1
    units) for its rows ``a:b`` (image rows ``y0`` onwards). ``out`` may be
    ``rgb``: the context rows a band needs are kept from before the previous
    band overwrote them.
    """
    height, width = rgb.shape[:2]
    rows = max(2 * halo, 1, _CHUNK_PIXELS // max(width, 1))
    in_place = out is rgb
    saved: np.ndarray | None = None  # original rows just above the band
    for y0 in range(0, height, rows):
        y1 = min(height, y0 + rows)
        top, bottom = max(0, y0 - halo), min(height, y1 + halo)
        window = rgb[top:bottom]
        if in_place and top < y0 and saved is not None:
            window = np.concatenate([saved[len(saved) - (y0 - top) :], rgb[y0:bottom]])
        if in_place and halo:
            saved = rgb[max(y0, y1 - halo) : y1].copy()
        change = delta(_luma8(window), y0 - top, y1 - top, y0)
        x = window[y0 - top : y1 - top].astype(np.float32)
        x += (change * 255 + 0.5)[..., None]
        np.clip(x, 0, 255, out=x)
        out[y0:y1] = x.astype(np.uint8)


def _hash(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Stateless per-coordinate pseudo-random numbers in [0, 1)."""
    with np.errstate(over="ignore"):
        h = ix.astype(np.uint32) * np.uint32(0x8DA6B343)
        h ^= iy.astype(np.uint32) * np.uint32(0xD8163841)
        h ^= np.uint32(seed & 0xFFFFFFFF)
        h ^= h >> np.uint32(13)
        h *= np.uint32(0x5BD1E995)
        h ^= h >> np.uint32(15)
    return h.astype(np.float32) / np.float32(2**32)


def _noise(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance noise at integer coordinates."""
    n = _hash(ix, iy, _GRAIN_SEED) + _hash(ix, iy, _GRAIN_SEED * 7 + 1)
    return (n - 1) * np.float32(np.sqrt(6))


def _grain_field(y0: int, y1: int, width: int, size: float) -> np.ndarray:
    """Film-grain noise for image rows ``y0:y1``: deterministic and seamless
    between bands; ``size`` is the grain's size in pixels."""
    ys = np.arange(y0, y1, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    if size <= 1:
        return _noise(np.broadcast_to(xs, (y1 - y0, width)), np.broadcast_to(ys, (y1 - y0, width)))
    gx, gy = xs / size, ys / size
    ix, iy = np.floor(gx), np.floor(gy)
    fx, fy = gx - ix, gy - iy
    shape = (y1 - y0, width)
    ix, iy = np.broadcast_to(ix, shape), np.broadcast_to(iy, shape)
    top = _noise(ix, iy) * (1 - fx) + _noise(ix + 1, iy) * fx
    bottom = _noise(ix, iy + 1) * (1 - fx) + _noise(ix + 1, iy + 1) * fx
    # Bilinear interpolation lowers the variance to 4/9: restore it.
    return (top * (1 - fy) + bottom * fy) * np.float32(1.5)
