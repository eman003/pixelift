"""Metadata preservation: EXIF (with orientation normalised), ICC profile, DPI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from PIL import Image

log = logging.getLogger(__name__)

ORIENTATION_TAG = 0x0112


@dataclass
class ImageMetadata:
    exif: bytes | None = None
    icc_profile: bytes | None = None
    dpi: tuple[float, float] | None = None

    def scaled_dpi(self, scale: int) -> tuple[float, float] | None:
        """Keep the physical print size: the pixel count grew by ``scale``."""
        if not self.dpi:
            return None
        return (self.dpi[0] * scale, self.dpi[1] * scale)


def extract_metadata(img: Image.Image) -> ImageMetadata:
    """Read metadata from an image whose pixels are already EXIF-transposed.

    The orientation tag is forced to 1 ("normal") because the pixels were
    physically rotated on load; keeping the old value would rotate the output
    a second time in viewers.
    """
    meta = ImageMetadata()
    try:
        exif = img.getexif()
        if len(exif):
            if ORIENTATION_TAG in exif:
                exif[ORIENTATION_TAG] = 1
            meta.exif = exif.tobytes()
    except Exception:
        log.warning("Ignoring unreadable EXIF data", exc_info=True)
    icc = img.info.get("icc_profile")
    if isinstance(icc, bytes) and icc:
        meta.icc_profile = icc
    dpi = img.info.get("dpi")
    if isinstance(dpi, tuple) and len(dpi) == 2:
        try:
            meta.dpi = (float(dpi[0]), float(dpi[1]))
        except (TypeError, ValueError):
            pass
    return meta


def save_kwargs(
    fmt: str, meta: ImageMetadata | None, quality: int, scale: int = 1
) -> dict[str, Any]:
    """Pillow ``save`` keyword arguments for ``fmt`` including metadata."""
    kwargs: dict[str, Any] = {}
    if fmt == "JPEG":
        kwargs.update(quality=quality, optimize=True, subsampling=0 if quality >= 90 else 2)
    elif fmt == "WEBP":
        kwargs.update(quality=quality, method=4)
    elif fmt == "PNG":
        kwargs.update(compress_level=6)
    if meta is not None:
        if meta.exif:
            kwargs["exif"] = meta.exif
        if meta.icc_profile:
            kwargs["icc_profile"] = meta.icc_profile
        dpi = meta.scaled_dpi(scale)
        if dpi and fmt in ("JPEG", "PNG"):
            kwargs["dpi"] = dpi
    return kwargs
