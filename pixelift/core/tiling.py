"""Tile layout and seam blending for tiled super-resolution.

Pure numpy — no PyTorch — so it can be unit-tested in isolation.

Tiles overlap by ``overlap`` input pixels. Tiles are pasted in raster order
directly into the final ``uint8`` buffer; inside the overlap the new tile is
cross-faded over what is already there with a linear ramp, so each tile's own
border pixels (where super-resolution networks are least accurate) get ~0 weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TILE_CHOICES = (1024, 768, 512, 384, 256, 192, 128, 96, 64)
MIN_TILE = 64


@dataclass(frozen=True)
class Tile:
    y0: int
    y1: int
    x0: int
    x1: int
    # Overlap (input px) with the previously pasted tile above / to the left.
    overlap_top: int
    overlap_left: int


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Evenly spaced tile start offsets covering ``length`` with >= ``overlap``."""
    if length <= tile:
        return [0]
    overlap = min(overlap, tile // 2)
    count = math.ceil((length - overlap) / (tile - overlap))
    count = max(count, 2)
    step = (length - tile) / (count - 1)
    return [round(i * step) for i in range(count)]


def plan_tiles(height: int, width: int, tile: int, overlap: int) -> list[Tile]:
    ys = tile_starts(height, tile, overlap)
    xs = tile_starts(width, tile, overlap)
    tiles: list[Tile] = []
    for yi, y0 in enumerate(ys):
        y1 = min(y0 + tile, height)
        top = max(0, min(ys[yi - 1] + tile, height) - y0) if yi else 0
        for xi, x0 in enumerate(xs):
            x1 = min(x0 + tile, width)
            left = max(0, min(xs[xi - 1] + tile, width) - x0) if xi else 0
            tiles.append(Tile(y0, y1, x0, x1, top, left))
    return tiles


def _ramp(length: int, overlap: int) -> np.ndarray:
    weights = np.ones(length, dtype=np.float32)
    if overlap:
        weights[:overlap] = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
    return weights


def paste_tile(out: np.ndarray, tile: np.ndarray, oy: int, ox: int, ov_y: int, ov_x: int) -> None:
    """Paste ``tile`` into ``out`` at (oy, ox), feathering the top/left overlaps."""
    th, tw = tile.shape[:2]
    region = out[oy : oy + th, ox : ox + tw]
    ov_y = min(ov_y, th)
    ov_x = min(ov_x, tw)
    region[ov_y:, ov_x:] = tile[ov_y:, ov_x:]
    if not ov_y and not ov_x:
        return
    wy = _ramp(th, ov_y)
    wx = _ramp(tw, ov_x)
    if ov_y:
        mask = (wy[:ov_y, None] * wx[None, :])[..., None]
        top = region[:ov_y]
        top[...] = (top * (1.0 - mask) + tile[:ov_y] * mask + 0.5).astype(np.uint8)
    if ov_x:
        mask = wx[None, :ov_x, None]
        left = region[ov_y:, :ov_x]
        left[...] = (left * (1.0 - mask) + tile[ov_y:, :ov_x] * mask + 0.5).astype(np.uint8)


def auto_tile_size(
    height: int,
    width: int,
    bytes_per_pixel: float,
    budget_bytes: int,
    cap: int = 1024,
) -> int:
    """Largest tile from TILE_CHOICES whose activations fit in ``budget_bytes``."""
    longest = max(height, width)
    for size in TILE_CHOICES:
        if size > cap:
            continue
        side = min(size, longest)
        if side * side * bytes_per_pixel <= budget_bytes:
            return size
    return MIN_TILE


def smaller_tile(size: int) -> int:
    for choice in TILE_CHOICES:
        if choice < size:
            return choice
    return MIN_TILE
