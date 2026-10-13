"""Image loading, probing, thumbnails and output naming (no GUI, no PyTorch)."""

from __future__ import annotations

import functools
import logging
import os
import re
import string
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from pixelift.core.errors import ImageTooLargeError, InvalidImageError
from pixelift.core.hdr_output import colour as cs
from pixelift.core.hdr_output.codecs import HdrCodecError, HdrSource, probe_size, sniff
from pixelift.core.hdr_output.pipeline import HdrImage, exif_orientation, load_hdr
from pixelift.utils.metadata import ImageMetadata, extract_metadata
from pixelift.utils.system import available_ram_bytes

# Large images are a core use case; keep Pillow's bomb check but raise the bar
# to ~1 gigapixel. Memory is checked separately before processing. Pillow only
# warns between this and twice it, so ``_open`` refuses anything above it.
log = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 1_000_000_000
# Decoded bytes per pixel when building a preview: Pillow keeps RGB as 4 bytes,
# and the 8-bit / orientation / RGBA conversions hold a second copy.
_PREVIEW_BYTES_PER_PIXEL = 8

SUPPORTED_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".jpe", ".webp", ".tif", ".tiff", ".bmp", ".avif", ".exr", ".hdr"}
)
# Decoding an HDR source holds the float32 image, ffmpeg's output and an SDR view.
_HDR_BYTES_PER_PIXEL = 40
OUTPUT_FORMATS = {"png": ("PNG", ".png"), "jpeg": ("JPEG", ".jpg"), "webp": ("WEBP", ".webp")}
WEBP_MAX_DIMENSION = 16383
JPEG_MAX_DIMENSION = 65500
DEFAULT_TEMPLATE = "{name}_{scale}x"
# Default output folders (next to the originals); never re-ingested as input.
UPSCALED_DIR, RESTORED_DIR = "upscaled", "restored"
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
    rgb: np.ndarray  # (H, W, 3) uint8, orientation applied (HDR sources: tone-mapped SDR view)
    alpha: np.ndarray | None  # (H, W) uint8 or None
    grayscale: bool
    metadata: ImageMetadata
    source_format: str
    hdr: HdrImage | None = None  # the HDR pixels of an HDR source (linear BT.2020)
    note: str = ""  # anything the job's result should mention

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
            children = [p for p in children if not _is_own_output(p.relative_to(path))]
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


def _is_own_output(relative: Path) -> bool:
    """Whether ``relative`` (inside a dropped folder) is one of Pixelift's results.

    Everything in an ``upscaled`` folder counts; in a ``restored`` folder only
    restoration results do (``<name>_restored…``), since people often keep
    their own restored scans in a folder of that name.
    """
    folders = relative.parts[:-1]
    if UPSCALED_DIR in folders:
        return True
    return RESTORED_DIR in folders and "_restored" in relative.stem


def _open(path: Path) -> Image.Image:
    try:
        img = Image.open(path)
    except FileNotFoundError:
        raise InvalidImageError(f"The file {path.name} no longer exists.") from None
    except Image.DecompressionBombError:
        raise _too_large(path) from None
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(
            f"{path.name} is not a supported image or is damaged.",
            ["Supported formats: PNG, JPEG, WebP, AVIF, TIFF, BMP, OpenEXR, Radiance HDR"],
        ) from exc
    width, height = img.size
    if width * height > Image.MAX_IMAGE_PIXELS:
        img.close()
        raise _too_large(path)
    return img


def _too_large(path: Path) -> ImageTooLargeError:
    return ImageTooLargeError(
        f"{path.name} has more than {Image.MAX_IMAGE_PIXELS // 1_000_000:,} megapixels.",
        ["Split the image into parts"],
    )


def _check_preview_memory(img: Image.Image, path: Path) -> None:
    """Refuse to decode ``img`` whole when that would exhaust memory.

    Thumbnails and previews of PNG, TIFF… decode every pixel; a small file can
    declare a huge canvas. Call after ``draft``, which shrinks JPEG decodes.
    """
    width, height = img.size
    needed = width * height * _PREVIEW_BYTES_PER_PIXEL
    available = available_ram_bytes()
    if needed > available // 2:
        raise ImageTooLargeError(
            f"{path.name} ({width}×{height}) needs about {human_size(needed)} of memory "
            f"to preview, but only {human_size(available)} is available.",
            ["Close other applications", "Split the image into parts"],
        )


def hdr_source(path: Path, gain_maps: bool = False) -> HdrSource | None:
    """The HDR description of ``path`` (PQ / HLG PNG or AVIF, EXR, RGBE) or None for SDR.

    Gain-map JPEGs (Ultra HDR) are complete SDR photos with optional HDR
    information: they count as HDR only with ``gain_maps`` (when HDR output
    asks for it). Otherwise their SDR picture is used as is, everywhere.
    """
    source = sniff(Path(path))
    if source is not None and source.encoding == "gainmap" and not gain_maps:
        return None
    return source


GAIN_MAP_FORMAT = "HDR JPEG (gain map)"  # ImageInfo.format of gain-map JPEGs


def probe_image(path: Path) -> ImageInfo:
    """Read dimensions/format from the header only (fast, low memory)."""
    path = Path(path)
    source = hdr_source(path)
    if source is not None:
        width, height, alpha = _hdr_call(lambda: probe_size(path))
        if exif_orientation(path) in _ORIENTATION_SWAPS:
            width, height = height, width
        mode = "RGBA" if alpha else "RGB"
        return ImageInfo(path, width, height, path.stat().st_size, source.describe(), mode)
    gain_map = hdr_source(path, gain_maps=True) is not None
    with _open(path) as img:
        width, height = img.size
        try:
            orientation = img.getexif().get(0x0112, 1)
        except Exception:  # noqa: BLE001
            orientation = 1
        if orientation in _ORIENTATION_SWAPS:
            width, height = height, width
        fmt = GAIN_MAP_FORMAT if gain_map else img.format or ""
        return ImageInfo(path, width, height, path.stat().st_size, fmt, img.mode)


def _to_8bit(img: Image.Image) -> Image.Image:
    """Convert 16/32-bit integer and float modes to 8-bit by scaling, not clipping,
    and CIELAB to sRGB (a plain ``convert`` reinterprets L*a*b* as RGB)."""
    if img.mode == "LAB":
        from PIL import ImageCms

        transform = ImageCms.buildTransform(
            ImageCms.createProfile("LAB"), ImageCms.createProfile("sRGB"), "LAB", "RGB"
        )
        return ImageCms.applyTransform(img, transform)
    if img.mode in ("I;16", "I;16L", "I;16B", "I;16N", "I"):
        arr = np.asarray(img, dtype=np.uint32)
        shift = 8 if img.mode.startswith("I;16") or arr.max(initial=0) > 255 else 0
        return Image.fromarray((arr >> shift).clip(0, 255).astype(np.uint8), "L")
    if img.mode == "F":
        arr = np.asarray(img, dtype=np.float32)
        peak = 1.0 if arr.max(initial=0) <= 1.0 else 255.0
        return Image.fromarray((arr / peak * 255).clip(0, 255).astype(np.uint8), "L")
    return img


def _hdr_call[T](func: Callable[[], T]) -> T:
    """Run an HDR codec call, turning its errors into user-facing image errors."""
    try:
        return func()
    except HdrCodecError as exc:
        raise InvalidImageError(str(exc), exc.suggestions) from exc


def load_hdr_image(path: Path, scene_white: float = 203.0) -> HdrImage:
    """Decode an HDR source to linear BT.2020 (checks memory and size first)."""
    path = Path(path)
    source = hdr_source(path, gain_maps=True)
    if source is None:
        raise InvalidImageError(f"{path.name} is not an HDR image.")
    width, height, _ = _hdr_call(lambda: probe_size(path))
    if width * height > Image.MAX_IMAGE_PIXELS:
        raise _too_large(path)
    needed = width * height * _HDR_BYTES_PER_PIXEL
    available = available_ram_bytes()
    if needed > available:
        raise ImageTooLargeError(
            f"{path.name} ({width}×{height}) needs about {human_size(needed)} of memory "
            f"to decode, but only {human_size(available)} is available.",
            ["Close other applications", "Split the image into parts"],
        )
    return _hdr_call(lambda: load_hdr(path, source, scene_white))


def _hdr_metadata(path: Path) -> ImageMetadata:
    try:
        with Image.open(path) as img:
            meta = extract_metadata(img)
    except Exception:  # noqa: BLE001 - EXR / RGBE: Pillow cannot open them
        return ImageMetadata()
    meta.icc_profile = None  # the colour description is cICP / nclx, not a profile
    return meta


def _load_hdr_as_loaded(path: Path, scene_white: float = 203.0) -> LoadedImage:
    hdr = load_hdr_image(path, scene_white)
    alpha = None
    if hdr.alpha is not None:
        alpha = np.rint(hdr.alpha * 255).astype(np.uint8)
    # The SDR view does not depend on ``scene_white`` (an HDR output setting):
    # SDR results, previews and cached upscales stay the same when it changes.
    relative = hdr.source.encoding in ("linear", "gainmap")
    sdr = hdr.sdr_view(cs.REFERENCE_WHITE / scene_white if relative else 1.0)
    return LoadedImage(sdr, alpha, False, _hdr_metadata(path), hdr.source.format, hdr)


# Full resolution (gigabytes for large sources): only the one being previewed
# is kept, for its zoomed crops; thumbnails decode uncached so as not to evict it.
@functools.lru_cache(maxsize=1)
def _hdr_sdr_image(path: Path, _stamp: tuple[int, int]) -> Image.Image:
    """Tone-mapped SDR rendition of an HDR source (cached by file identity)."""
    return _render_hdr_sdr(path)


def _render_hdr_sdr(path: Path) -> Image.Image:
    loaded = _load_hdr_as_loaded(path)
    img = Image.fromarray(loaded.rgb, "RGB")
    if loaded.alpha is not None:
        img.putalpha(Image.fromarray(loaded.alpha, "L"))
    return img


def _hdr_sdr_view(path: Path, cached: bool = True) -> Image.Image:
    path = Path(path)
    try:
        stat = path.stat()
    except FileNotFoundError:
        raise InvalidImageError(f"The file {path.name} no longer exists.") from None
    if not cached:
        return _render_hdr_sdr(path)
    # Shared with the cache: callers must not change it in place.
    return _hdr_sdr_image(path.resolve(), (stat.st_mtime_ns, stat.st_size))


def load_image(path: Path, scene_white: float = 203.0, gain_maps: bool = False) -> LoadedImage:
    """Decode an image, apply EXIF orientation and split colour and alpha.

    HDR sources (see ``hdr_source``) are decoded through ffmpeg: ``rgb`` is
    their tone-mapped SDR view and ``hdr`` holds the HDR pixels
    (``scene_white``: cd/m² of 1.0 in scene-linear EXR / RGBE files and of
    SDR white in gain-map JPEGs). Gain-map JPEGs are decoded as HDR only with
    ``gain_maps``; one whose gain map cannot be used is read as the SDR JPEG
    it also is (``note`` says why).
    """
    path = Path(path)
    source = hdr_source(path, gain_maps)
    if source is not None:
        if source.encoding != "gainmap":
            return _load_hdr_as_loaded(path, scene_white)
        try:
            return _load_hdr_as_loaded(path, scene_white)
        except InvalidImageError as exc:
            log.warning("Using the SDR picture of %s: %s", path.name, exc)
            loaded = load_image(path)
            loaded.note = f"Gain map not used ({exc}); read as an SDR photo"
            return loaded
    with _open(path) as img:
        try:
            img.load()
        except (OSError, SyntaxError, ValueError) as exc:
            raise InvalidImageError(
                f"{path.name} could not be decoded. The file may be truncated or damaged."
            ) from exc
        source_format = img.format or ""
        metadata = extract_metadata(img)
        img = ImageOps.exif_transpose(img) or img
        if img.mode in ("L", "RGB") and "transparency" in img.info:
            # A colour key (PNG tRNS): made an alpha band here, as the preview shows it.
            img = img.convert("LA" if img.mode == "L" else "RGBA")
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
    if hdr_source(Path(path)) is not None:
        img = _hdr_sdr_view(Path(path), cached=False)
        return _fit(img, size).convert("RGBA")
    with _open(Path(path)) as img:
        if img.format == "JPEG":
            img.draft("RGB", (size * 2, size * 2))
        _check_preview_memory(img, Path(path))
        img = ImageOps.exif_transpose(img) or img
        img = _to_8bit(img)
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        return img.convert("RGBA")


def load_preview(path: Path, max_side: int) -> tuple[Image.Image, tuple[int, int]]:
    """RGBA image no larger than ``max_side`` plus the full (oriented) size."""
    if hdr_source(Path(path)) is not None:
        img = _hdr_sdr_view(Path(path))
        return _fit(img, max_side).convert("RGBA"), img.size
    with _open(Path(path)) as img:
        if img.format == "JPEG":
            img.draft("RGB", (max_side, max_side))
        _check_preview_memory(img, Path(path))
        full = probe_image(Path(path))
        img = ImageOps.exif_transpose(img) or img
        img = _to_8bit(img)
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return img.convert("RGBA"), (full.width, full.height)


def _fit(img: Image.Image, side: int) -> Image.Image:
    """``img`` scaled to fit ``side`` (a new image; the shared cached one is not touched)."""
    if max(img.size) <= side:
        return img.copy()
    factor = side / max(img.size)
    size = (max(1, round(img.width * factor)), max(1, round(img.height * factor)))
    return img.resize(size, Image.Resampling.LANCZOS)


def load_region(
    path: Path, box: tuple[int, int, int, int], out_size: tuple[int, int]
) -> Image.Image:
    """Crop ``box`` (oriented image coordinates) and resize it to ``out_size``."""
    if hdr_source(Path(path)) is not None:
        region = _hdr_sdr_view(Path(path)).crop(box)
        if region.size != out_size:
            region = region.resize(out_size, Image.Resampling.LANCZOS)
        return region.convert("RGBA")
    with _open(Path(path)) as img:
        _check_preview_memory(img, Path(path))
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
    look: str = "",
    hdr: str = "",
) -> str:
    """Expand a filename template such as ``{name}_{scale}x`` and append ``ext``.

    Fields: {name} {ext} {scale} {model} {width} {height} {lighting} {look} {hdr}.
    Unknown fields raise ValueError so typos are caught when the setting is
    changed. A non-empty ``lighting``, ``look`` (camera look) or ``hdr`` tag
    the template does not place is appended as ``_<tag>``, so differently
    lit, graded or HDR-enhanced results never share a name.
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
        "look": look,
        "hdr": hdr,
    }
    fields = {name for _, name, _, _ in string.Formatter().parse(template) if name is not None}
    for field_name in fields:
        if field_name not in values:
            raise ValueError(f"Unknown field {{{field_name}}} in filename template")
    if lighting and "lighting" not in fields:
        template += "_{lighting}"
    if look and "look" not in fields:
        template += "_{look}"
    if hdr and "hdr" not in fields:
        template += "_{hdr}"
    stem = _FIELD_RE.sub("_", template.format(**values)).strip(". ") or source.stem
    return stem + ext


def check_template(template: str) -> None:
    """Raise ValueError if ``template`` cannot name a file (unknown field,
    unbalanced braces…), before a whole batch fails on it."""
    try:
        render_filename(template, Path("photo.jpg"), 4, "realesrgan", 7680, 4320, ".png")
    except (ValueError, IndexError, KeyError, AttributeError) as exc:
        raise ValueError(f"invalid filename template {template!r}: {exc}") from None


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


def default_output_dir(source: Path, restored: bool = False) -> Path:
    return source.parent / (RESTORED_DIR if restored else UPSCALED_DIR)


def disk_free(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        stat = os.statvfs(probe)
        return stat.f_bavail * stat.f_frsize
    except OSError:
        return 1 << 62
