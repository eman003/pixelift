"""Black-and-white colorization with DeOldify, tuned for natural, historical colour.

The network predicts colour on a small square rendering of the photo (colour
needs far less resolution than detail); only its chroma is kept and laid
over the full-resolution brightness of the original — so every detail and,
with *Preserve Original Tones*, every tone of the original photo is kept.

The DeOldify Artistic model can be colourful. Its chroma passes through a soft
knee that tames saturated colours ("natural": early colour-film palette)
before the user's strength is applied: 0 % gives back the grey photo.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from pixelift.core import device_manager as dm
from pixelift.core.restoration.filters import LUMA, luma_np
from pixelift.models.deoldify_arch import IMAGENET_MEAN, IMAGENET_STD

RENDER_SIZE = 560  # DeOldify's default render_factor 35 × 16


def predict_color(net: torch.nn.Module, device: dm.DeviceInfo, gray: np.ndarray) -> np.ndarray:
    """(H, W) uint8 grey -> (RENDER_SIZE, RENDER_SIZE, 3) float32 RGB in 0..255."""
    small = Image.fromarray(gray).resize((RENDER_SIZE, RENDER_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    dtype = next(net.parameters()).dtype
    with torch.inference_mode():
        x = torch.from_numpy(arr).view(1, 1, RENDER_SIZE, RENDER_SIZE).expand(1, 3, -1, -1)
        x = ((x - mean) / std).to(dm.to_torch(device), dtype)
        y = net(x).float().cpu()
        y = (y * std + mean).clamp(0, 1) * 255.0
    return y[0].permute(1, 2, 0).numpy()


def _upscale_plane(plane: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Smooth float32 plane resize (chroma needs no sharp edges)."""
    return np.asarray(Image.fromarray(plane, "F").resize(size, Image.Resampling.BICUBIC))


def apply_color(
    gray: np.ndarray,
    predicted: np.ndarray,
    strength: int,
    vivid: int,
    preserve_tones: bool,
) -> np.ndarray:
    """Combine the full-resolution grey photo with the predicted colour.

    ``gray`` is (H, W, 3) uint8 (equal channels); returns (H, W, 3) uint8.
    """
    height, width = gray.shape[:2]
    y_full = luma_np(gray)
    pred_y = predicted @ np.array(LUMA, dtype=np.float32)
    cb = predicted[..., 2] - pred_y  # chroma as colour-minus-luma differences
    cr = predicted[..., 0] - pred_y
    # Soft knee on the chroma magnitude: natural keeps colours muted, vivid
    # lets the model's full saturation through.
    t = min(100, max(0, vivid)) / 100
    knee = 35.0 + 120.0 * t
    mag = np.hypot(cb, cr)
    gain = 1.0 / np.sqrt(1.0 + (mag / knee) ** 2)
    gain *= (0.9 + 0.25 * t) * min(100, max(0, strength)) / 100
    cb *= gain
    cr *= gain
    size = (width, height)
    cb_full = _upscale_plane(cb.astype(np.float32), size)
    cr_full = _upscale_plane(cr.astype(np.float32), size)
    if not preserve_tones:
        # Take the model's broad tonal rendering, keep the photo's detail.
        small_y = np.asarray(
            Image.fromarray(gray[..., 0]).resize(
                (predicted.shape[1], predicted.shape[0]), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        y_full += _upscale_plane((pred_y - small_y).astype(np.float32), size)
    out = np.empty_like(gray)
    rows = max(1, (1 << 20) // max(1, width))
    for y0 in range(0, height, rows):
        y = y_full[y0 : y0 + rows]
        r = y + cr_full[y0 : y0 + rows]
        b = y + cb_full[y0 : y0 + rows]
        g = (y - LUMA[0] * r - LUMA[2] * b) / LUMA[1]
        band = np.stack([r, g, b], axis=-1)
        np.clip(band + 0.5, 0, 255, out=band)
        out[y0 : y0 + rows] = band.astype(np.uint8)
    return out
