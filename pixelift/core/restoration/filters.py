"""Building blocks for the conventional restoration stages (PyTorch on the CPU).

Local filters run over the image in tiles with a halo of context, so the
working memory stays bounded however large the scan is. Global statistics
are computed on a small proxy instead (see ``analysis``).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image

from pixelift.core.control import JobControl
from pixelift.core.restoration.analysis import LUMA, proxy

__all__ = ["LUMA", "proxy"]  # re-exported for the stages
TILE = 768

TileFn = Callable[[torch.Tensor], torch.Tensor]  # (1, C, h, w) float 0..1 -> same


def map_tiles(
    image: np.ndarray,
    fn: TileFn,
    halo: int,
    control: JobControl | None = None,
    tile: int = TILE,
) -> np.ndarray:
    """Apply ``fn`` to ``image`` (H, W, C uint8) tile by tile; returns a new array.

    Each tile is given ``halo`` pixels of real context (reflected at the
    image border), so a filter whose reach is at most ``halo`` gives the same
    result as on the whole image.
    """
    height, width = image.shape[:2]
    out = np.empty_like(image)
    with torch.inference_mode():
        for y0 in range(0, height, tile):
            for x0 in range(0, width, tile):
                if control is not None:
                    control.check()
                y1, x1 = min(height, y0 + tile), min(width, x0 + tile)
                py0, px0 = max(0, y0 - halo), max(0, x0 - halo)
                py1, px1 = min(height, y1 + halo), min(width, x1 + halo)
                x = to_tensor(image[py0:py1, px0:px1])
                pads = (halo - (x0 - px0), halo - (px1 - x1), halo - (y0 - py0), halo - (py1 - y1))
                x = _pad(x, pads)
                y = fn(x)[:, :, halo : halo + (y1 - y0), halo : halo + (x1 - x0)]
                out[y0:y1, x0:x1] = to_uint8(y)
    return out


def _pad(x: torch.Tensor, pads: tuple[int, int, int, int]) -> torch.Tensor:
    if not any(pads):
        return x
    # Reflection needs the pad to be smaller than the size; replicate beyond that.
    h, w = x.shape[2:]
    if pads[0] < w and pads[1] < w and pads[2] < h and pads[3] < h:
        return F.pad(x, pads, mode="reflect")
    return F.pad(x, pads, mode="replicate")


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    """(H, W, C) uint8 -> (1, C, H, W) float32 in 0..1."""
    if not arr.flags.writeable or not arr.flags.c_contiguous:
        arr = np.array(arr)  # torch needs writable memory (this copies one tile)
    t = torch.from_numpy(arr)
    if t.ndim == 2:
        t = t[..., None]
    return t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)


def to_uint8(t: torch.Tensor) -> np.ndarray:
    """(1, C, H, W) float 0..1 -> (H, W, C) uint8."""
    return t[0].clamp(0, 1).mul(255).add_(0.5).to(torch.uint8).permute(1, 2, 0).numpy()


def luma(x: torch.Tensor) -> torch.Tensor:
    """(N, 3, H, W) -> (N, 1, H, W)."""
    r, g, b = LUMA
    return x[:, 0:1] * r + x[:, 1:2] * g + x[:, 2:3] * b


def luma_np(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (H, W) float32 in 0..255, a band of rows at a time."""
    out = np.empty(rgb.shape[:2], dtype=np.float32)
    rows = max(1, (1 << 20) // max(1, rgb.shape[1]))
    for y in range(0, rgb.shape[0], rows):
        band = rgb[y : y + rows].astype(np.float32)
        out[y : y + rows] = band[..., 0] * LUMA[0] + band[..., 1] * LUMA[1] + band[..., 2] * LUMA[2]
    return out


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """Neutral grey copy of ``rgb`` (H, W, 3 uint8), still three channels."""
    y = luma_np(rgb)
    np.add(y, 0.5, out=y)
    np.clip(y, 0, 255, out=y)
    return np.repeat(y.astype(np.uint8)[..., None], 3, axis=2)


def resize(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Lanczos resize of (H, W, 3) uint8 to ``size`` = (width, height)."""
    if (rgb.shape[1], rgb.shape[0]) == size:
        return rgb
    return np.array(Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS))  # writable


# --- separable kernels ------------------------------------------------------
def gaussian_kernel(sigma: float) -> torch.Tensor:
    radius = max(1, math.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(x * x) / (2 * sigma * sigma))
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of (N, C, H, W); edges replicated."""
    if sigma <= 0:
        return x
    k = gaussian_kernel(sigma).to(x.dtype)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    h, w = x.shape[2:]
    mode = "replicate" if r >= min(h, w) else "reflect"  # reflect needs r < size
    y = F.pad(x, (r, r, 0, 0), mode=mode)
    y = F.conv2d(y, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    y = F.pad(y, (0, 0, r, r), mode=mode)
    return F.conv2d(y, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def box_mean(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Mean over a (2r+1)² window, normalised at the borders."""
    if radius <= 0:
        return x
    return F.avg_pool2d(x, 2 * radius + 1, stride=1, padding=radius, count_include_pad=False)


def dilate(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Grey-level dilation with an (h, w) rectangle (odd sizes)."""
    kh, kw = size
    if kh <= 1 and kw <= 1:
        return x
    return F.max_pool2d(x, (kh, kw), stride=1, padding=(kh // 2, kw // 2))


def erode(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return -dilate(-x, size)


def line_offsets(length: int, angle: float) -> list[tuple[int, int]]:
    """Pixel offsets (dy, dx) of a centred digital line of ``length`` at ``angle`` degrees."""
    r = length // 2
    rad = math.radians(angle)
    offsets = []
    for i in range(-r, r + 1):
        offset = (round(-i * math.sin(rad)), round(i * math.cos(rad)))
        if offset not in offsets:
            offsets.append(offset)
    return offsets


def _line_extreme(x: torch.Tensor, offsets: list[tuple[int, int]], fn: Callable) -> torch.Tensor:
    """Running min/max over the pixels of a line structuring element."""
    r = max(max(abs(dy), abs(dx)) for dy, dx in offsets)
    padded = F.pad(x, (r, r, r, r), mode="replicate")
    h, w = x.shape[2:]
    result = None
    for dy, dx in offsets:
        view = padded[:, :, r + dy : r + dy + h, r + dx : r + dx + w]
        result = view if result is None else fn(result, view)
    assert result is not None
    return result


def line_opening(x: torch.Tensor, length: int, angle: float) -> torch.Tensor:
    """Opening with a line of ``length`` pixels at ``angle`` degrees.

    Keeps bright structures that contain such a line; removes thinner or
    shorter ones.
    """
    if angle % 180 == 0:
        return dilate(erode(x, (1, length)), (1, length))
    if angle % 180 == 90:
        return dilate(erode(x, (length, 1)), (length, 1))
    offsets = line_offsets(length, angle)
    mirrored = [(-dy, -dx) for dy, dx in offsets]
    return _line_extreme(_line_extreme(x, offsets, torch.minimum), mirrored, torch.maximum)


def line_closing(x: torch.Tensor, length: int, angle: float) -> torch.Tensor:
    return -line_opening(-x, length, angle)


def grow(seed: torch.Tensor, allowed: torch.Tensor, steps: int) -> torch.Tensor:
    """Hysteresis: extend ``seed`` (bool) into connected ``allowed`` pixels."""
    region = seed & allowed
    for _ in range(steps):
        grown = (dilate(region.float(), (3, 3)) > 0) & allowed
        if bool((grown == region).all()):
            break
        region = grown
    return region


def inpaint(x: torch.Tensor, mask: torch.Tensor, sigma: float, grain: float = 0.0) -> torch.Tensor:
    """Fill ``mask`` (1 = damaged) from the surrounding valid pixels.

    Normalised convolution at increasing scales: each pixel takes the
    weighted average of nearby undamaged pixels, so thin defects fill in with
    their surroundings' colour and brightness. ``grain`` (standard deviation,
    0..1 scale) adds film-grain-like noise to the fill so repairs do not show
    up as smooth patches in a grainy photo (deterministic per tile).
    """
    valid = 1.0 - mask
    filled = torch.zeros_like(x)
    support = torch.zeros_like(mask)
    s = max(0.8, sigma)
    for _ in range(4):
        num = gaussian_blur(x * valid, s)
        den = gaussian_blur(valid, s)
        take = (den > 0.05) & (support < 0.5)
        filled = torch.where(take, num / den.clamp_min(1e-6), filled)
        support = torch.where(take, torch.ones_like(support), support)
        if bool(support.min() >= 0.5):
            break
        s *= 2.5
    filled = torch.where(support > 0.5, filled, x)
    if grain > 0:
        generator = torch.Generator().manual_seed(x.shape[2] * 131 + x.shape[3])
        noise = torch.randn(mask.shape, generator=generator, dtype=x.dtype) * grain
        filled = filled + noise  # same on every channel: brightness grain
    return x * valid + filled * mask
