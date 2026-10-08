"""Whole-image analysis on a small proxy: black-and-white detection, noise, levels."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# Rec. 601 luma, the usual weighting for photographic black-and-white conversion.
LUMA = (0.299, 0.587, 0.114)

ANALYSIS_SIDE = 768
# Colourfulness (mean residual chroma, 0..255 scale) below which a photo
# counts as black-and-white. Neutral greys measure ~0-1, toned/yellowed
# prints ~1-3 (their single hue is factored out), real colour photos > 6.
MONO_THRESHOLD = 3.5


def proxy(rgb: np.ndarray, max_side: int) -> np.ndarray:
    """Downscaled copy for global statistics (box filter, cheap)."""
    height, width = rgb.shape[:2]
    factor = max(height, width) / max_side
    if factor <= 1:
        return rgb
    size = (max(1, round(width / factor)), max(1, round(height / factor)))
    return np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BOX, reducing_gap=2.0))


@dataclass(frozen=True)
class MonoInfo:
    monochrome: bool
    toned: bool  # a single hue (sepia, cyanotype, yellowed paper) rather than neutral grey
    colorfulness: float


def detect_monochrome(rgb: np.ndarray) -> MonoInfo:
    """Is this a black-and-white photograph (possibly sepia-toned or yellowed)?

    A toned print's colour depends only on its brightness, so the chroma is
    predicted from the luma (per brightness bin) and only what is left over
    counts as real colour.
    """
    small = proxy(rgb, ANALYSIS_SIDE).astype(np.float32)
    y = small @ np.array(LUMA, dtype=np.float32)
    # Real colour is spatially coherent; grain and scanner noise (different in
    # each channel) average out over a few pixels, so measure smoothed chroma.
    cb = _smooth(small[..., 2] - y)
    cr = _smooth(small[..., 0] - y)
    bins = np.clip((y / 256 * 32).astype(np.int32), 0, 31).ravel()
    counts = np.bincount(bins, minlength=32).astype(np.float32)
    mean_cb = np.bincount(bins, cb.ravel(), minlength=32) / np.maximum(counts, 1)
    mean_cr = np.bincount(bins, cr.ravel(), minlength=32) / np.maximum(counts, 1)
    res_cb = cb.ravel() - mean_cb[bins]
    res_cr = cr.ravel() - mean_cr[bins]
    residual = np.hypot(res_cb, res_cr)
    # Robust: ignore the most colourful 1% (stains, coloured ink, scanner fringes).
    colorfulness = float(np.mean(np.sort(residual)[: max(1, int(residual.size * 0.99))]))
    tint = float(np.mean(np.hypot(cb, cr)))
    mono = colorfulness < MONO_THRESHOLD
    return MonoInfo(mono, mono and tint > 2.0, colorfulness)


def _smooth(plane: np.ndarray, radius: int = 2) -> np.ndarray:
    """Box mean over a (2r+1)² window (edges replicated), via an integral image."""
    k = 2 * radius + 1
    padded = np.pad(plane.astype(np.float64), radius + 1, mode="edge")
    integral = padded.cumsum(0).cumsum(1)
    h, w = plane.shape
    total = (
        integral[k : k + h, k : k + w]
        - integral[:h, k : k + w]
        - integral[k : k + h, :w]
        + integral[:h, :w]
    )
    return (total / (k * k)).astype(np.float32)


_MONO_CACHE: dict[tuple[str, int, int], MonoInfo] = {}
_MONO_CACHE_SIZE = 256
_mono_lock = threading.Lock()


def detect_monochrome_file(path: Path, decoded: Image.Image | None = None) -> MonoInfo:
    """``detect_monochrome`` from a fast reduced decode, cached per file version.

    ``decoded``: the file already decoded at ``ANALYSIS_SIDE`` (e.g. for a
    thumbnail), so it is not decoded again.
    """
    from pixelift.utils.image_utils import make_thumbnail

    stat = Path(path).stat()
    key = (str(Path(path).resolve()), stat.st_mtime_ns, stat.st_size)
    with _mono_lock:
        cached = _MONO_CACHE.get(key)
    if cached is not None:
        return cached
    image = decoded if decoded is not None else make_thumbnail(Path(path), ANALYSIS_SIDE)
    info = detect_monochrome(np.asarray(image.convert("RGB")))
    with _mono_lock:
        if len(_MONO_CACHE) >= _MONO_CACHE_SIZE:
            _MONO_CACHE.pop(next(iter(_MONO_CACHE)))  # oldest first
        _MONO_CACHE[key] = info
    return info


def estimate_noise(rgb: np.ndarray, crop: int = 1024) -> float:
    """Noise standard deviation (0..1 scale) of the luma, Immerkær's method.

    Measured on a full-resolution centre crop: downscaling would hide grain.
    """
    height, width = rgb.shape[:2]
    y0, x0 = max(0, (height - crop) // 2), max(0, (width - crop) // 2)
    patch = rgb[y0 : y0 + crop, x0 : x0 + crop].astype(np.float32) / 255.0
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return 0.0
    y = patch @ np.array(LUMA, dtype=np.float32)
    lap = (
        y[:-2, :-2] - 2 * y[:-2, 1:-1] + y[:-2, 2:]
        - 2 * y[1:-1, :-2] + 4 * y[1:-1, 1:-1] - 2 * y[1:-1, 2:]
        + y[2:, :-2] - 2 * y[2:, 1:-1] + y[2:, 2:]
    )  # fmt: skip
    # Edges inflate the estimate: use only the flatter 90% of pixels.
    mag = np.sort(np.abs(lap).ravel())
    mag = mag[: max(1, int(mag.size * 0.9))]
    return float(np.sqrt(np.pi / 2) * mag.mean() / 6.0)


@dataclass(frozen=True)
class Levels:
    """Per-channel black and white points and midtone, 0..1."""

    low: np.ndarray  # (3,)
    high: np.ndarray  # (3,)
    mid: np.ndarray  # (3,) median of the midtones after stretching
    shadows_clipped: float  # fraction of near-black pixels
    highlights_clipped: float  # fraction of near-white pixels


def measure_levels(rgb: np.ndarray, clip: float = 0.4) -> Levels:
    """Percentile levels on a proxy (``clip`` percent at each end)."""
    small = proxy(rgb, ANALYSIS_SIDE).reshape(-1, 3).astype(np.float32) / 255.0
    low = np.percentile(small, clip, axis=0)
    high = np.percentile(small, 100 - clip, axis=0)
    span = np.maximum(high - low, 1e-3)
    stretched = np.clip((small - low) / span, 0, 1)
    y = stretched @ np.array(LUMA, dtype=np.float32)
    midtones = stretched[(y > 0.2) & (y < 0.8)]
    mid = np.median(midtones, axis=0) if len(midtones) > 32 else np.median(stretched, axis=0)
    y_orig = small @ np.array(LUMA, dtype=np.float32)
    return Levels(
        low=low,
        high=high,
        mid=mid,
        shadows_clipped=float(np.mean(y_orig < 0.02)),
        highlights_clipped=float(np.mean(y_orig > 0.98)),
    )
