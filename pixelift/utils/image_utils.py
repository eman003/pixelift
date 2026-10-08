"""Image loading, probing, thumbnails and output naming (no GUI, no PyTorch)."""

from __future__ import annotations

import os
import re
import string
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from pixelift.core.errors import InvalidImageError
from pixelift.utils.metadata import ImageMetadata, extract_metadata

# Large images are a core use case; keep Pillow's bomb check but raise the bar
# to ~1 gigapixel. Memory is checked separately before processing.
Image.MAX_IMAGE_PIXELS = 1_000_000_000

SUPPORTED_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".jpe", ".webp", ".tif", ".tiff", ".bmp"}
)
OUTPUT_FORMATS = {"png": ("PNG", ".png"), "jpeg": ("JPEG", ".jpg"), "webp": ("WEBP", ".webp")}
WEBP_MAX_DIMENSION = 16383
JPEG_MAX_DIMENSION = 65500
DEFAULT_TEMPLATE = "{name}_{scale}x"
_ORIENTATION_SWAPS = {5, 6, 7, 8}


@dataclass(frozen=True)
class ImageInfo:
    path: Path
    width: int  # after EXIF orientation
    height: int
    file_size: int
    format: str
    mode: str

    @property
    def has_alpha(self) -> bool:
        return self.mode in ("RGBA", "LA", "PA") or self.mode == "P"


@dataclass
class LoadedImage:
    rgb: np.ndarray  # (H, W, 3) uint8, orientation applied
    alpha: np.ndarray | None  # (H, W) uint8 or None
    grayscale: bool
    metadata: ImageMetadata
    source_format: str

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def collect_images(paths: Iterable[Path], recursive: bool = True) -> list[Path]:
    """Expand directories into supported image files (sorted, de-duplicated)."""
    found: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        path = Path(path)
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            children = sorted(p for p in path.glob(pattern) if p.is_file() and is_supported(p))
            # Never re-ingest our own output folders.
            children = [p for p in children if "upscaled" not in p.relative_to(path).parts[:-1]]
        elif path.is_file() and is_supported(path):
            children = [path]
        else:
            children = []
        for child in children:
            key = child.resolve()
            if key not in seen:
                seen.add(key)
                found.append(child)
    return found


def _open(path: Path) -> Image.Image:
    try:
        return Image.open(path)
    except FileNotFoundError:
        raise InvalidImageError(f"The file {path.name} no longer exists.") from None
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(
            f"{path.name} is not a supported image or is damaged.",
            ["Supported formats: PNG, JPEG, WebP, TIFF, BMP"],
        ) from exc


def probe_image(path: Path) -> ImageInfo:
    """Read dimensions/format from the header only (fast, low memory)."""
    path = Path(path)
    with _open(path) as img:
        width, height = img.size
        try:
            orientation = img.getexif().get(0x0112, 1)
        except Exception:  # noqa: BLE001
            orientation = 1
        if orientation in _ORIENTATION_SWAPS:
            width, height = height, width
        return ImageInfo(path, width, height, path.stat().st_size, img.format or "", img.mode)


def _to_8bit(img: Image.Image) -> Image.Image:
    """Convert 16/32-bit integer and float modes to 8-bit by scaling, not clipping."""
    if img.mode in ("I;16", "I;16L", "I;16B", "I;16N", "I"):
        arr = np.asarray(img, dtype=np.uint32)
        shift = 8 if img.mode.startswith("I;16") or arr.max(initial=0) > 255 else 0
        return Image.fromarray((arr >> shift).clip(0, 255).astype(np.uint8), "L")
    if img.mode == "F":
        arr = np.asarray(img, dtype=np.float32)
        peak = 1.0 if arr.max(initial=0) <= 1.0 else 255.0
        return Image.fromarray((arr / peak * 255).clip(0, 255).astype(np.uint8), "L")
    return img


def load_image(path: Path) -> LoadedImage:
    """Decode an image, apply EXIF orientation and split colour and alpha."""
    path = Path(path)
    with _open(path) as img:
        try:
            img.load()
        except (OSError, SyntaxError, ValueError) as exc:
            raise InvalidImageError(
                f"{path.name} could not be decoded. The file may be truncated or damaged."
            ) from exc
        source_format = img.format or ""
        # Transpose first: reading metadata resets the orientation tag on the
        # image's cached EXIF, which would make exif_transpose a no-op.
        img = ImageOps.exif_transpose(img) or img
        metadata = extract_metadata(img)
        img = _to_8bit(img)

        if img.mode == "P":
            img = img.convert("RGBA" if "transparency" in img.info else "RGB")
        if img.mode == "CMYK":
            img = img.convert("RGB")
            metadata.icc_profile = None  # a CMYK profile no longer applies
        grayscale = img.mode in ("1", "L", "LA")
        alpha: np.ndarray | None = None
        if img.mode in ("RGBA", "LA", "PA", "RGBa", "La"):
            alpha_band = np.asarray(img.getchannel("A"))
            if alpha_band.min() < 255:  # fully opaque alpha carries no information
                alpha = alpha_band
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        rgb = np.asarray(img)
        if not rgb.flags.writeable or not rgb.flags.c_contiguous:
            rgb = np.array(rgb, order="C")
    return LoadedImage(rgb, alpha, grayscale, metadata, source_format)


def make_thumbnail(path: Path, size: int = 128) -> Image.Image:
    """Small RGBA thumbnail with orientation applied; uses JPEG draft decoding."""
    with _open(Path(path)) as img:
        if img.format == "JPEG":
            img.draft("RGB", (size * 2, size * 2))
        img = ImageOps.exif_transpose(img) or img
        img = _to_8bit(img)
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        return img.convert("RGBA")


def load_preview(path: Path, max_side: int) -> tuple[Image.Image, tuple[int, int]]:
    """RGBA image no larger than ``max_side`` plus the full (oriented) size."""
    with _open(Path(path)) as img:
        if img.format == "JPEG":
            img.draft("RGB", (max_side, max_side))
        full = probe_image(Path(path))
        img = ImageOps.exif_transpose(img) or img
        img = _to_8bit(img)
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return img.convert("RGBA"), (full.width, full.height)


def load_region(
    path: Path, box: tuple[int, int, int, int], out_size: tuple[int, int]
) -> Image.Image:
    """Crop ``box`` (oriented image coordinates) and resize it to ``out_size``."""
    with _open(Path(path)) as img:
        img = ImageOps.exif_transpose(img) or img
        img = _to_8bit(img)
        region = img.crop(box)
        if region.size != out_size:
            region = region.resize(out_size, Image.Resampling.LANCZOS)
        return region.convert("RGBA")


_FIELD_RE = re.compile(r"[\\/\x00]")


def render_filename(
    template: str,
    source: Path,
    scale: int,
    model: str,
    width: int,
    height: int,
    ext: str,
    lighting: str = "",
) -> str:
    """Expand a filename template such as ``{name}_{scale}x`` and append ``ext``.

    Fields: {name} {ext} {scale} {model} {width} {height} {lighting}. Unknown
    fields raise ValueError so typos are caught when the setting is changed.
    A non-empty ``lighting`` tag the template does not place is appended as
    ``_<tag>``, so differently lit results never share a name.
    """
    template = template.strip() or DEFAULT_TEMPLATE
    values = {
        "name": source.stem,
        "ext": source.suffix.lstrip("."),
        "scale": scale,
        "model": model,
        "width": width,
        "height": height,
        "lighting": lighting,
    }
    fields = {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}
    for field_name in fields:
        if field_name not in values:
            raise ValueError(f"Unknown field {{{field_name}}} in filename template")
    if lighting and "lighting" not in fields:
        template += "_{lighting}"
    stem = _FIELD_RE.sub("_", template.format(**values)).strip(". ") or source.stem
    return stem + ext


def numbered_paths(path: Path) -> Iterator[Path]:
    """``photo.png``, ``photo (2).png``, ``photo (3).png``, …"""
    yield path
    for i in range(2, 10_000):
        yield path.with_name(f"{path.stem} ({i}){path.suffix}")


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1000 or unit == "GB":
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1000
    return f"{num_bytes:.1f} GB"


def default_output_dir(source: Path) -> Path:
    return source.parent / "upscaled"


def disk_free(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        stat = os.statvfs(probe)
        return stat.f_bavail * stat.f_frsize
    except OSError:
        return 1 << 62
