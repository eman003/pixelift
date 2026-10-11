"""Metadata preservation: EXIF (with orientation normalised), ICC profile, DPI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from PIL import Image

log = logging.getLogger(__name__)

ORIENTATION_TAG = 0x0112
# A TIFF's EXIF is its first image directory, which also describes the pixel
# layout (size, strips, compression…). Only these descriptive tags carry over.
TIFF_DESCRIPTIVE_TAGS = frozenset(
    {
        0x010D,  # DocumentName
        0x010E,  # ImageDescription
        0x010F,  # Make
        0x0110,  # Model
        ORIENTATION_TAG,
        0x011A,  # XResolution
        0x011B,  # YResolution
        0x011D,  # PageName
        0x0128,  # ResolutionUnit
        0x0131,  # Software
        0x0132,  # DateTime
        0x013B,  # Artist
        0x013C,  # HostComputer
        0x8298,  # Copyright
        0x8769,  # Exif IFD
        0x8825,  # GPS IFD
    }
)


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
    """Read metadata from a freshly opened image (before EXIF-transposing it:
    the transposed copy of some formats, such as TIFF, has no EXIF).

    The orientation tag is forced to 1 ("normal") because the pixels are
    physically rotated on load; keeping the old value would rotate the output
    a second time in viewers. The image's own EXIF is left untouched, so
    ``exif_transpose`` still sees the original orientation.
    """
    meta = ImageMetadata()
    try:
        exif = Image.Exif()
        exif.load(img.getexif().tobytes())
        if img.format == "TIFF":
            for tag in [t for t in exif if t not in TIFF_DESCRIPTIVE_TAGS]:
                del exif[tag]
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
