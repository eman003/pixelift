"""HDR Enhancement: HDR-like local tone mapping of the upscaled (SDR) image.

This widens the *perceived* dynamic range of an ordinary 8-bit photo — it
lifts shadows, compresses highlights and adds local contrast — and the result
is still an ordinary SDR image. It cannot recover detail the source never
recorded: clipped highlights stay white and crushed blacks stay black (the
curves keep 0 and 1 fixed), and lifting deep shadows only reveals what the
file's few dark levels held.

How it works (NumPy and Pillow only, deterministic):

1. ``analyse`` builds an edge-aware *base layer* of the picture's brightness
   with a guided filter (He et al.) on a small copy (at most ``ANALYSIS_SIDE``
   pixels). Only the filter's coefficients are kept; they are upsampled while
   the full-size image is processed, so the base follows strong edges at full
   resolution. That is what keeps halos away from the horizon and from
   silhouettes against a bright sky.
2. ``apply_hdr`` tone-maps the base (shadow lift, highlight compression) and
   adds the picture's own detail back on top: medium-scale detail is boosted
   (local contrast) and fine detail — texture and noise — keeps its strength,
   so lifted shadows do not get noisier. Boosts fade out for large steps
   (no harsh edges) and the total change is softly limited so nothing new
   clips or crushes.
3. Colour follows the brightness change with a gentler factor, so lifted
   shadows keep their hue without turning garish; chroma boosts stay inside
   the gamut. Greys stay grey.

Everything is processed in bands of rows, so the float working memory stays
bounded (tens of MB) however large the upscaled image is. Every radius is a
fraction of the picture, so the stage preview matches the export.

Intensity scales every effect towards zero; 0 % returns the input untouched.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image

from pixelift.core.camera_looks import _blur, _hash, _luma8
from pixelift.core.lighting import _CHUNK_PIXELS, _luma

NATURAL, BALANCED, DRAMATIC = "natural", "balanced", "dramatic"
ADVANCED_NAMES = ("shadows", "highlights", "local_contrast", "saturation")

ANALYSIS_SIDE = 512  # the base layer's analysis size (px, longest side)
_BASE_RADIUS = 0.04  # guided-filter radius, as a fraction of the longest side
_BASE_EPS = 0.012  # guided-filter edge threshold (variance): steps of ~0.11 are kept
_FINE_SIGMA = 0.0008  # fine detail (kept as is) is finer than this fraction of the side
_DETAIL_KNEE = 0.08  # medium-scale detail larger than this gets less boost
_FLOOR = np.float32(1.5 / 255)  # fine-detail boosts leave noise-sized wiggles alone
_MAX_COLOUR_GAIN = 2.0


@dataclass(frozen=True)
class HdrRecipe:
    """What the HDR stage does; 0 everywhere is the untouched image.

    shadows         0..1.5, lift of the dark parts of the base layer
    highlights      0..1.5, compression of the bright parts of the base layer
    local_contrast  0..1.5, boost of medium-scale detail (midtones most)
    detail          0..0.5, boost of fine detail above the noise floor
    colour          0..1, how far colour follows the brightness change
                    (1 keeps saturation, 0 keeps absolute chroma)
    vibrance        0..0.5, extra colour, gentler on already-saturated colours
    """

    shadows: float = 0.0
    highlights: float = 0.0
    local_contrast: float = 0.0
    detail: float = 0.0
    colour: float = 0.7
    vibrance: float = 0.0

    @property
    def is_neutral(self) -> bool:
        return (
            self.shadows == 0
            and self.highlights == 0
            and self.local_contrast == 0
            and self.detail == 0
            and self.vibrance == 0
        )

    def scaled(self, factor: float) -> HdrRecipe:
        """The recipe at ``factor`` (0..1) of its strength (colour is a ratio)."""
        if factor == 1:
            return self
        f = factor
        return dataclasses.replace(
            self,
            shadows=self.shadows * f,
            highlights=self.highlights * f,
            local_contrast=self.local_contrast * f,
            detail=self.detail * f,
            vibrance=self.vibrance * f,
        )

    def clamped(self) -> HdrRecipe:
        def clamp(v: float, high: float) -> float:
            v = float(v)
            return 0.0 if not math.isfinite(v) else min(high, max(0.0, v))

        return HdrRecipe(
            shadows=clamp(self.shadows, 1.5),
            highlights=clamp(self.highlights, 1.5),
            local_contrast=clamp(self.local_contrast, 1.5),
            detail=clamp(self.detail, 0.5),
            colour=clamp(self.colour, 1.0),
            vibrance=clamp(self.vibrance, 0.5),
        )


@dataclass(frozen=True)
class HdrProfile:
    id: str
    name: str
    description: str
    recipe: HdrRecipe
    default_intensity: int  # percent


_PROFILES: dict[str, HdrProfile] = {}


def register_profile(profile: HdrProfile) -> None:
    _PROFILES[profile.id] = profile


def all_profiles() -> list[HdrProfile]:
    return list(_PROFILES.values())


def get_profile(profile_id: str) -> HdrProfile:
    return _PROFILES[profile_id]


def has_profile(profile_id: str) -> bool:
    return profile_id in _PROFILES


for _profile in (
    HdrProfile(
        NATURAL,
        "Natural HDR",
        "Gentle shadow and highlight recovery; looks untouched",
        HdrRecipe(shadows=0.6, highlights=0.5, local_contrast=0.35, colour=0.5, vibrance=0.04),
        60,
    ),
    HdrProfile(
        BALANCED,
        "Balanced HDR",
        "Open shadows, protected highlights and crisper local contrast",
        HdrRecipe(
            shadows=0.9, highlights=0.8, local_contrast=0.6, detail=0.1, colour=0.55, vibrance=0.08
        ),
        65,
    ),
    HdrProfile(
        DRAMATIC,
        "Dramatic HDR",
        "Strong local contrast and deep tonal range for a bold look",
        HdrRecipe(
            shadows=1.2, highlights=1.1, local_contrast=1.1, detail=0.2, colour=0.6, vibrance=0.12
        ),
        70,
    ),
):
    register_profile(_profile)

DEFAULT_PROFILE = NATURAL


@dataclass(frozen=True)
class HdrSettings:
    """What the user picked: on/off, a profile, its intensity and the Advanced
    adjustments (-100..100 each; 0 is the profile as designed, -100 turns
    that part off, +100 doubles it)."""

    enabled: bool = False
    profile: str = DEFAULT_PROFILE
    intensity: int = 60  # percent, 0..100
    shadows: int = 0
    highlights: int = 0
    local_contrast: int = 0
    saturation: int = 0

    def recipe(self) -> HdrRecipe:
        """The effective recipe (intensity and Advanced applied)."""
        profile = _PROFILES.get(self.profile)
        if not self.enabled or profile is None:
            return HdrRecipe()

        def adj(name: str) -> float:
            value = getattr(self, name)
            return 1 + min(100, max(-100, value)) / 100

        base = profile.recipe
        recipe = dataclasses.replace(
            base,
            shadows=base.shadows * adj("shadows"),
            highlights=base.highlights * adj("highlights"),
            local_contrast=base.local_contrast * adj("local_contrast"),
            detail=base.detail * adj("local_contrast"),
            colour=base.colour * min(1.0, adj("saturation")),
            vibrance=base.vibrance * adj("saturation"),
        )
        return recipe.scaled(min(100, max(0, self.intensity)) / 100).clamped()

    @property
    def active(self) -> bool:
        return not self.recipe().is_neutral

    @property
    def advanced(self) -> tuple[int, ...]:
        return tuple(getattr(self, name) for name in ADVANCED_NAMES)

    def tag(self) -> str:
        """Short filename-safe label; "" when HDR changes nothing.

        Different effective settings give different tags, so an output made
        with other HDR settings is never mistaken for this one's.
        """
        if not self.active:
            return ""
        tag = f"hdr-{self.profile}-{min(100, max(0, self.intensity))}"
        if any(self.advanced):
            values = ",".join(str(v) for v in self.advanced)
            tag += "-" + hashlib.sha1(values.encode()).hexdigest()[:6]
        return tag


# --- analysis ----------------------------------------------------------------------
@dataclass(frozen=True)
class HdrPlan:
    """The base layer of one picture: guided-filter coefficients on a small grid.

    ``scale`` is the grid size in cells per picture (width, height), so a
    point at normalised position (u, v) is at cell coordinates
    (u * scale[0] - 0.5, v * scale[1] - 0.5).
    """

    a: np.ndarray  # float32 (h, w)
    b: np.ndarray  # float32 (h, w)
    scale: tuple[float, float]


def analyse(rgb: np.ndarray) -> HdrPlan:
    """The base layer of an (H, W, 3) ``uint8`` picture, read band by band."""
    _check(rgb)
    height, width = rgb.shape[:2]
    factor = max(1, math.ceil(max(height, width) / ANALYSIS_SIDE))
    rows = max(factor, (_CHUNK_PIXELS // max(width, 1)) // factor * factor)
    parts = []
    for y0 in range(0, height, rows):
        band = Image.fromarray(_luma8(rgb[y0 : y0 + rows]), "L")
        parts.append(np.asarray(band.reduce(factor) if factor > 1 else band))
    small = np.concatenate(parts).astype(np.float32) / 255
    radius = max(1, round(_BASE_RADIUS * max(small.shape)))
    a, b = _guided_coefficients(small, radius, _BASE_EPS)
    return HdrPlan(a, b, (width / factor, height / factor))


def _guided_coefficients(guide: np.ndarray, radius: int, eps: float) -> tuple[np.ndarray, ...]:
    """Self-guided filter coefficients, already averaged: base = a * guide + b."""
    mean = _box(guide, radius)
    var = np.maximum(_box(guide * guide, radius) - mean * mean, 0)
    a = var / (var + np.float32(eps))
    b = mean - a * mean
    return _box(a, radius), _box(b, radius)


def _box(x: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)² window with edges repeated (separable running sums)."""
    p = np.pad(x, r, mode="edge").astype(np.float64)
    c = np.cumsum(p, axis=0)
    c = np.concatenate([np.zeros((1, c.shape[1])), c])
    p = c[2 * r + 1 :] - c[: -2 * r - 1]
    c = np.cumsum(p, axis=1)
    c = np.concatenate([np.zeros((c.shape[0], 1)), c], axis=1)
    p = c[:, 2 * r + 1 :] - c[:, : -2 * r - 1]
    return (p / (2 * r + 1) ** 2).astype(np.float32)


# --- processing --------------------------------------------------------------------
Check = Callable[[], None]


def apply_hdr(
    rgb: np.ndarray,
    recipe: HdrRecipe,
    out: np.ndarray | None = None,
    plan: HdrPlan | None = None,
    frame: tuple[int, int] | None = None,
    region: tuple[float, float, float, float] | None = None,
    check: Check | None = None,
) -> np.ndarray:
    """HDR-enhance an (H, W, 3) ``uint8`` array; returns ``rgb`` itself when neutral.

    Pass ``out=rgb`` to work in place; ``rgb`` may be read-only (a memory-mapped
    cached upscale) when ``out`` is another array. For a crop of a larger
    picture (the zoomed-in preview) pass that picture's ``plan``, the crop's
    ``region`` in it as normalised (x0, y0, x1, y1) and the whole picture's
    size in ``rgb``'s pixels as ``frame``. ``check`` is called between bands
    (``JobControl.check``: pause and cancel).
    """
    _check(rgb)
    if out is not None and (out.shape != rgb.shape or out.dtype != np.uint8):
        raise ValueError("out must have the same shape as rgb")
    recipe = recipe.clamped()
    if recipe.is_neutral:
        if out is not None and out is not rgb:
            out[...] = rgb
            return out
        return rgb
    if plan is None:
        plan = analyse(rgb)
    if check:
        check()
    if out is None:
        out = np.empty_like(rgb)
    height, width = rgb.shape[:2]
    x0, y0n, x1, y1n = region or (0.0, 0.0, 1.0, 1.0)
    sigma = max(1.0, _FINE_SIGMA * max(frame or (width, height)))
    halo = int(3 * sigma) + 2
    # Column interpolation into the plan's grid, shared by every band.
    cols = _grid_positions(width, x0, x1, plan.scale[0], plan.a.shape[1])
    rows = max(2 * halo, 1, _CHUNK_PIXELS // max(width, 1))
    in_place = out is rgb
    saved: np.ndarray | None = None  # original rows just above the band
    for y0 in range(0, height, rows):
        y1 = min(height, y0 + rows)
        top, bottom = max(0, y0 - halo), min(height, y1 + halo)
        window = rgb[top:bottom]
        if in_place and top < y0 and saved is not None:
            window = np.concatenate([saved[len(saved) - (y0 - top) :], rgb[y0:bottom]])
        if in_place:
            saved = rgb[max(y0, y1 - halo) : y1].copy()
        a, b = y0 - top, y1 - top
        smooth = _blur(_luma8(window), sigma)[a:b]
        band_rows = _grid_positions(height, y0n, y1n, plan.scale[1], plan.a.shape[0], y0, y1)
        base = _upsample(plan.a, band_rows, cols) * smooth + _upsample(plan.b, band_rows, cols)
        out[y0:y1] = _tone_map(window[a:b], smooth, base, recipe, y0)
        if check:
            check()
    return out


def _tone_map(
    band: np.ndarray, smooth: np.ndarray, base: np.ndarray, r: HdrRecipe, y0: int
) -> np.ndarray:
    x = band.astype(np.float32)
    x /= 255
    v = _luma(x)
    np.clip(base, 0, 1, out=base)
    # Base tone curve: both terms keep 0 and 1 fixed and stay monotonic for
    # shadows, highlights <= 1.5 (as lighting's shadow/highlight curve).
    inv = 1 - base
    toned = base + np.float32(r.shadows) * base * inv * inv
    toned -= np.float32(r.highlights) * base * base * inv
    # Medium-scale detail: boosted in the midtones, less for large steps
    # (edges), so local contrast rises without halos or harsh outlines.
    # Noise-sized wiggles are left alone.
    mid = smooth - base
    weight = 4 * toned * (1 - toned)
    knee = np.float32(_DETAIL_KNEE)
    weight *= np.float32(r.local_contrast) / (1 + (mid / knee) ** 2)
    mid += weight * np.sign(mid) * np.maximum(np.abs(mid) - _FLOOR, 0)
    # Fine detail (texture, noise) keeps its strength: lifting the shadows
    # does not make them noisier. Only clear detail above the floor is boosted.
    fine = v - smooth
    if r.detail:
        fine += np.float32(r.detail) * np.sign(fine) * np.maximum(np.abs(fine) - _FLOOR, 0)
    change = toned + mid + fine - v
    del inv, toned, mid, fine, weight
    # Soft limit: the change approaches the room left before black or white
    # instead of clipping (no new clipped highlights or crushed shadows).
    room = np.where(change > 0, 1 - v, v)
    eps = np.float32(1e-6)
    q = (change / np.maximum(room, eps)) ** 2
    change /= np.sqrt(np.sqrt(1 + q * q))
    new_v = v + change
    # Colour follows the brightness change partly (``colour``), plus vibrance.
    ratio = (new_v + np.float32(0.02)) / (v + np.float32(0.02))
    gain = np.minimum(ratio ** np.float32(r.colour), np.float32(_MAX_COLOUR_GAIN))
    x -= v[..., None]  # chroma
    hi = x.max(axis=-1, keepdims=True)
    lo = x.min(axis=-1, keepdims=True)
    gain = gain[..., None]
    if r.vibrance:
        # As lighting's saturation: muted colours (and skin) gain more.
        gain *= 1 + np.float32(r.vibrance) * (1 - 0.5 * (hi - lo))
    lum = new_v[..., None]
    # The colour gain approaches the room left before a channel clips at the
    # new brightness smoothly (a soft minimum, as for the brightness change),
    # so saturated colours keep their hue and their detail instead of piling
    # up at 255.
    room = np.minimum((1 - lum) / np.maximum(hi, eps), lum / np.maximum(-lo, eps))
    q = (gain / np.maximum(room, eps)) ** 2
    x *= gain / np.sqrt(np.sqrt(1 + q * q))
    x += lum
    # Same triangular dither on every channel: no banding in smooth skies,
    # and greys stay grey.
    x *= 255
    x += (_dither(y0, x.shape[0], x.shape[1]) + 0.5)[..., None]
    np.clip(x, 0, 255, out=x)
    return x.astype(np.uint8)


def _grid_positions(
    size: int,
    start: float,
    stop: float,
    cells: float,
    n: int,
    i0: int = 0,
    i1: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolation indices and weights of pixels ``i0:i1`` (of ``size``
    spanning normalised ``start..stop``) in a grid of ``n`` cells."""
    i = np.arange(i0, size if i1 is None else i1, dtype=np.float64)
    u = start + (i + 0.5) / size * (stop - start)
    pos = np.clip(u * cells - 0.5, 0, n - 1)
    lo = np.floor(pos).astype(np.intp)
    hi = np.minimum(lo + 1, n - 1)
    return lo, hi, (pos - lo).astype(np.float32)


def _upsample(grid: np.ndarray, rows: tuple, cols: tuple) -> np.ndarray:
    """Bilinear samples of ``grid`` at the given row and column positions."""
    r0, r1, wr = rows
    c0, c1, wc = cols
    band = grid[r0] * (1 - wr)[:, None] + grid[r1] * wr[:, None]
    return band[:, c0] * (1 - wc) + band[:, c1] * wc


def _dither(y0: int, height: int, width: int) -> np.ndarray:
    """Deterministic triangular noise in -1..1 for image rows y0 onwards."""
    shape = (height, width)
    iy = np.broadcast_to(np.arange(y0, y0 + height, dtype=np.uint32)[:, None], shape)
    ix = np.broadcast_to(np.arange(width, dtype=np.uint32)[None, :], shape)
    return _hash(ix, iy, 0x4D5) + _hash(ix, iy, 0x1F3B) - 1


def _check(rgb: np.ndarray) -> None:
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected an (H, W, 3) uint8 array")
    if rgb.dtype != np.uint8:
        raise ValueError("expected an (H, W, 3) uint8 array")
    if rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise ValueError("the image is empty")
