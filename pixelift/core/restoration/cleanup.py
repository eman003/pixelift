"""Scan cleanup with conventional image processing: dust, scratches, noise, sharpness.

No AI here — these defects are found reliably by mathematical morphology and
edge-preserving filters, which never invent content. Strengths are 0..100;
0 leaves the image untouched (the stage is skipped entirely).

Defects are detected on the brightness only and filled in colour from the
surrounding pixels; denoising treats brightness gently (keeping some film
grain) and colour noise more strongly.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from pixelift.core.control import JobControl
from pixelift.core.restoration import filters as fl

ANGLES = (0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5)
# Damage is sparse; a regular pattern (fabric, print screens, foliage) makes
# many detections close together, so above these densities they are kept.
# Measured over a 41 px window: 99 % of detections in a very dusty scan stay
# below 0.07, 90 % of a dotted tie's are above 0.1.
DUST_MAX_DENSITY = 0.085
SCRATCH_MAX_DENSITY = 0.15


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * min(1.0, max(0.0, t))


def _odd(value: float) -> int:
    n = max(1, round(value))
    return n if n % 2 else n + 1


def resolution_factor(width: int, height: int) -> float:
    """Defect sizes scale with the scan resolution (1.0 for a ~2000 px photo)."""
    return min(4.0, max(1.0, max(width, height) / 2000))


def remove_dust(
    rgb: np.ndarray, strength: int, noise: float, control: JobControl | None = None
) -> np.ndarray:
    """Remove small isolated specks (dust, mould spots, pinholes)."""
    if strength <= 0:
        return rgb
    t = strength / 100
    res = resolution_factor(rgb.shape[1], rgb.shape[0])
    # Specks up to ``speck`` px across are removed: a line must be longer
    # than a speck to survive the openings below.
    speck = _lerp(4, 9, t) * res
    size = _odd(speck + 2)
    floor = _lerp(8, 4, t) * noise  # never mistake grain for dust
    density_radius = round(20 * res)
    # Bright specks (dust on the negative prints white) are the common case;
    # dark ones are treated more cautiously, as they may be real detail.
    bright_threshold = max(_lerp(0.20, 0.07, t), floor)
    dark_threshold = max(_lerp(0.30, 0.10, t), floor)
    fill_sigma = speck / 3

    def run(x: torch.Tensor) -> torch.Tensor:
        y = fl.luma(x)
        # A speck is shorter than ``size`` in every direction, so openings
        # with lines of that length remove it whatever the direction; longer
        # structures (edges, lines, hair) survive in at least one direction.
        kept_bright = torch.stack([fl.line_opening(y, size, a) for a in ANGLES]).amax(0)
        kept_dark = torch.stack([fl.line_closing(y, size, a) for a in ANGLES]).amin(0)
        hits = ((y - kept_bright) > bright_threshold) | ((kept_dark - y) > dark_threshold)
        hits = _sparse(hits, density_radius, DUST_MAX_DENSITY)
        mask = fl.dilate(hits.float(), (3, 3))
        if not bool(mask.any()):
            return x
        return fl.inpaint(x, mask, fill_sigma, grain=noise)

    halo = size * 2 + density_radius + int(fill_sigma * 30)
    return fl.map_tiles(rgb, run, halo=halo, control=control)


def reduce_scratches(
    rgb: np.ndarray, strength: int, noise: float, control: JobControl | None = None
) -> np.ndarray:
    """Remove thin, long, high-contrast lines (emulsion scratches, cracks)."""
    if strength <= 0:
        return rgb
    t = strength / 100
    res = resolution_factor(rgb.shape[1], rgb.shape[0])
    width = _odd(_lerp(3, 7, t) * res)
    length = _odd(max(4 * width, 17 * res))
    floor = _lerp(7, 3.5, t) * noise
    # Scratches in prints are mostly white (bare paper); dark thin lines are
    # more often real detail such as hair, so they need more contrast.
    bright_threshold = max(_lerp(0.18, 0.06, t), floor)
    dark_threshold = max(_lerp(0.32, 0.12, t), floor)
    fill_sigma = width / 2

    def run(x: torch.Tensor) -> torch.Tensor:
        y = fl.luma(x)
        opened = fl.dilate(fl.erode(y, (width, width)), (width, width))
        closed = fl.erode(fl.dilate(y, (width, width)), (width, width))
        # Thin: removed by a ``width`` square. Long: survives a line opening
        # along its direction.
        openings = torch.stack([fl.line_opening(y, length, a) for a in ANGLES])
        closings = torch.stack([fl.line_closing(y, length, a) for a in ANGLES])
        bright = torch.minimum(y - opened, openings.amax(0) - opened)
        dark = torch.minimum(closed - y, closed - closings.amin(0))
        # A scratch lies on top of a surface: the pixels on its two sides
        # match. A highlight or shadow along an object's edge (a collar, a
        # window frame) has different surfaces on either side, so it stays.
        sides = torch.stack([_side_difference(y, a, width) for a in ANGLES])
        bright_sides = sides.gather(0, openings.argmax(0, keepdim=True))[0]
        dark_sides = sides.gather(0, closings.argmin(0, keepdim=True))[0]
        bright = torch.where(bright_sides < 0.5 * bright + 0.02, bright, torch.zeros_like(bright))
        dark = torch.where(dark_sides < 0.5 * dark + 0.02, dark, torch.zeros_like(dark))
        strong = (bright > bright_threshold) | (dark > dark_threshold)
        if not bool(strong.any()):
            return x
        # Follow each scratch through stretches where it is fainter (on
        # bright skin, say) instead of leaving dashes behind.
        weak = (bright > bright_threshold * 0.5) | (dark > dark_threshold * 0.5)
        hits = fl.grow(strong, weak, length)
        hits = _sparse(hits, length, SCRATCH_MAX_DENSITY)
        mask = fl.dilate(hits.float(), (3, 3))
        return fl.inpaint(x, mask, fill_sigma, grain=noise)

    halo = length * 3 + int(fill_sigma * 30)  # detection, hysteresis and density
    return fl.map_tiles(rgb, run, halo=halo, control=control)


def _side_difference(y: torch.Tensor, angle: float, distance: int) -> torch.Tensor:
    """|y(p + d·n) − y(p − d·n)| with n the normal of a line at ``angle`` degrees."""
    rad = math.radians(angle)
    # Lines run along (cos, -sin) in (x, y) image coordinates (see line_offsets).
    dx, dy = round(distance * math.sin(rad)), round(distance * math.cos(rad))
    r = max(abs(dx), abs(dy))
    padded = F.pad(y, (r, r, r, r), mode="replicate")
    h, w = y.shape[2:]
    plus = padded[:, :, r + dy : r + dy + h, r + dx : r + dx + w]
    minus = padded[:, :, r - dy : r - dy + h, r - dx : r - dx + w]
    return (plus - minus).abs()


def _sparse(hits: torch.Tensor, radius: int, max_density: float) -> torch.Tensor:
    """Drop detections where they are too dense to be damage (texture, patterns)."""
    if not bool(hits.any()):
        return hits
    density = fl.box_mean(hits.float(), radius)
    return hits & (density <= max_density)


def _guided(guide: torch.Tensor, src: torch.Tensor, radius: int, eps: float) -> torch.Tensor:
    """He et al. guided filter: edge-preserving smoothing of ``src`` along ``guide``."""
    mean_i = fl.box_mean(guide, radius)
    mean_p = fl.box_mean(src, radius)
    cov = fl.box_mean(guide * src, radius) - mean_i * mean_p
    var = fl.box_mean(guide * guide, radius) - mean_i * mean_i
    a = cov / (var + eps)
    b = mean_p - a * mean_i
    return fl.box_mean(a, radius) * guide + fl.box_mean(b, radius)


def denoise(
    rgb: np.ndarray,
    strength: int,
    noise: float,
    monochrome: bool,
    control: JobControl | None = None,
) -> np.ndarray:
    """Reduce grain and colour noise while keeping edges (and a little grain)."""
    if strength <= 0:
        return rgb
    t = strength / 100
    res = resolution_factor(rgb.shape[1], rgb.shape[0])
    radius = max(1, round(_lerp(1, 3, t) * res))
    sigma = max(noise, 0.004)
    eps = (_lerp(1.2, 3.0, t) * sigma) ** 2
    amount = _lerp(0.0, 0.85, t)  # never fully smooth: old photos have grain
    chroma_amount = _lerp(0.0, 1.0, t)

    def run(x: torch.Tensor) -> torch.Tensor:
        y = fl.luma(x)
        smooth = _guided(y, y, radius, eps)
        y_new = y + amount * (smooth - y)
        if monochrome:
            return y_new.expand_as(x)
        chroma = x - y
        chroma_smooth = _guided(y, chroma, radius * 2, 0.03**2)
        return y_new + chroma + chroma_amount * (chroma_smooth - chroma)

    return fl.map_tiles(rgb, run, halo=radius * 4 + 2, control=control)


def sharpen(
    rgb: np.ndarray, strength: int, noise: float, control: JobControl | None = None
) -> np.ndarray:
    """Noise-aware unsharp mask on brightness with overshoot (halo) limiting."""
    if strength <= 0:
        return rgb
    t = strength / 100
    res = resolution_factor(rgb.shape[1], rgb.shape[0])
    sigma = 0.9 * res
    amount = _lerp(0.0, 1.4, t)
    floor = 1.5 * noise
    margin = 0.03

    def run(x: torch.Tensor) -> torch.Tensor:
        y = fl.luma(x)
        detail = y - fl.gaussian_blur(y, sigma)
        # Soft threshold: leave grain-sized wiggles alone.
        detail = torch.sign(detail) * (detail.abs() - floor).clamp_min(0)
        sharp = y + amount * detail
        low = fl.erode(y, (3, 3)) - margin
        high = fl.dilate(y, (3, 3)) + margin
        sharp = torch.minimum(torch.maximum(sharp, low), high)
        return x + (sharp - y)

    return fl.map_tiles(rgb, run, halo=int(sigma * 3) + 3, control=control)


def clarity(rgb: np.ndarray, amount: float, control: JobControl | None = None) -> np.ndarray:
    """Local contrast in the midtones (large-radius unsharp mask on brightness)."""
    if amount <= 0:
        return rgb
    sigma = min(40.0, max(4.0, max(rgb.shape[:2]) / 150))

    def run(x: torch.Tensor) -> torch.Tensor:
        y = fl.luma(x)
        detail = y - fl.gaussian_blur(y, sigma)
        weight = 4 * y * (1 - y)  # strongest in the midtones, none at black/white
        return x + amount * weight * detail

    return fl.map_tiles(rgb, run, halo=int(sigma * 3) + 2, control=control)
