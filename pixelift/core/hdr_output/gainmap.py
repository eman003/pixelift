"""Gain-map HDR JPEG ("Ultra HDR"): reading and writing.

Format (Adobe "Gain Map" specification 1.0, as used by Android Ultra HDR v1):

* The file is an ordinary SDR JPEG (the *base* rendition, sRGB here) that any
  viewer shows. After it comes a second JPEG, the *gain map*, located through
  a CIPA DC-007 Multi-Picture Format (MPF) ``APP2`` segment and an XMP
  ``Container:Directory`` in the primary image.
* The gain map's own XMP (namespace ``http://ns.adobe.com/hdr-gain-map/1.0/``,
  prefix ``hdrgm``) says how to rebuild the HDR rendition. For a gain-map
  sample ``g`` (0..1) and the linear SDR base ``sdr`` (1.0 = SDR white)::

      g'    = g ^ (1 / Gamma)
      log2b = GainMapMin · (1 - g') + GainMapMax · g'
      hdr   = (sdr + OffsetSDR) · 2^(log2b · w) - OffsetHDR

  ``w`` is 1 on a display with at least ``HDRCapacityMax`` stops of headroom
  (full HDR, what Pixelift decodes), 0 on an SDR display (the base as is).

What Pixelift writes: an sRGB base (the BT.2390 SDR rendition, or the SDR
original itself for SDR sources) and a three-channel (per R, G, B) gain map at
a quarter of the size in each direction (the Android default scale), with
``Gamma`` 1, offsets 1/64, ``HDRCapacityMin`` 0 and ``HDRCapacityMax`` the
picture's own range. Per-channel gains let the HDR rendition keep colours
the tone-mapped base had to desaturate; colours outside sRGB itself are
fitted into sRGB (the base's colour space). Decoders that only support
single-channel maps are outside the specification.

The ISO 21496-1 binary gain-map metadata is not read or written; files that
carry it usually carry the XMP too (Android does).
"""

from __future__ import annotations

import io
import math
import re
import struct
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from pixelift.core.hdr_output import colour as cs
from pixelift.core.hdr_output.codecs import FileReport, HdrCodecError

HDRGM_NS = "http://ns.adobe.com/hdr-gain-map/1.0/"
XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\0"
MPF_HEADER = b"MPF\0"
SCALE = 4  # gain map is 1/SCALE of the base per side
OFFSET = 1 / 64
_SNIFF_BYTES = 1 << 17


@dataclass(frozen=True)
class GainMapMetadata:
    """The ``hdrgm`` values (log2 units); one value per channel (1 or 3)."""

    gain_min: tuple[float, ...]
    gain_max: tuple[float, ...]
    gamma: tuple[float, ...]
    offset_sdr: tuple[float, ...]
    offset_hdr: tuple[float, ...]
    capacity_min: float
    capacity_max: float
    base_is_hdr: bool = False

    def xmp(self) -> bytes:
        def v(values: tuple[float, ...]) -> str:
            return f"{values[0]:.6f}"

        attrs = (
            f'hdrgm:Version="1.0" hdrgm:GainMapMin="{v(self.gain_min)}" '
            f'hdrgm:GainMapMax="{v(self.gain_max)}" hdrgm:Gamma="{v(self.gamma)}" '
            f'hdrgm:OffsetSDR="{v(self.offset_sdr)}" hdrgm:OffsetHDR="{v(self.offset_hdr)}" '
            f'hdrgm:HDRCapacityMin="{self.capacity_min:.6f}" '
            f'hdrgm:HDRCapacityMax="{self.capacity_max:.6f}" '
            f'hdrgm:BaseRenditionIsHDR="{"True" if self.base_is_hdr else "False"}"'
        )
        return _xmp_packet(f'xmlns:hdrgm="{HDRGM_NS}" {attrs}', "")


def _xmp_packet(description_attrs: str, body: str) -> bytes:
    xml = (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Pixelift">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f'<rdf:Description rdf:about="" {description_attrs}>{body}</rdf:Description>'
        "</rdf:RDF></x:xmpmeta>"
    )
    return XMP_HEADER + xml.encode()


def _primary_xmp(gain_map_length: int) -> bytes:
    attrs = (
        'xmlns:Container="http://ns.google.com/photos/1.0/container/" '
        'xmlns:Item="http://ns.google.com/photos/1.0/container/item/" '
        f'xmlns:hdrgm="{HDRGM_NS}" hdrgm:Version="1.0"'
    )
    body = (
        "<Container:Directory><rdf:Seq>"
        '<rdf:li rdf:parseType="Resource"><Container:Item Item:Semantic="Primary" '
        'Item:Mime="image/jpeg"/></rdf:li>'
        '<rdf:li rdf:parseType="Resource"><Container:Item Item:Semantic="GainMap" '
        f'Item:Mime="image/jpeg" Item:Length="{gain_map_length}"/></rdf:li>'
        "</rdf:Seq></Container:Directory>"
    )
    return _xmp_packet(attrs, body)


# --- JPEG segments --------------------------------------------------------------------
def _segments(data: bytes) -> Iterable[tuple[int, int, int]]:
    """(marker, start of payload, end of payload) of the segments before the scan."""
    if data[:2] != b"\xff\xd8":
        return
    pos = 2
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            return
        marker = data[pos + 1]
        if marker == 0xDA or marker == 0xD9:  # start of scan / end of image
            return
        length = struct.unpack(">H", data[pos + 2 : pos + 4])[0]
        yield marker, pos + 4, pos + 2 + length
        pos += 2 + length


def _xmp_of(data: bytes) -> str | None:
    for marker, s, e in _segments(data):
        if marker == 0xE1 and data[s : s + len(XMP_HEADER)] == XMP_HEADER:
            return data[s + len(XMP_HEADER) : e].decode("utf-8", "replace")
    return None


def is_gain_map_jpeg(path: Path) -> bool:
    """Whether ``path`` is a JPEG whose primary XMP announces an hdrgm gain map."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(_SNIFF_BYTES)
    except OSError:
        return False
    xmp = _xmp_of(head)
    return xmp is not None and HDRGM_NS in xmp and "GainMap" in xmp


def usable(path: Path) -> bool:
    """Whether :func:`decode` can use this JPEG's gain map, from its headers
    and metadata only (no pixels decoded). Output names and tags are fixed
    before loading, and a JPEG with an unusable gain map is read as SDR."""
    try:
        data = path.read_bytes()
        gm_bytes = _gain_map_bytes(data)
        if parse_metadata(_xmp_of(gm_bytes) or "").base_is_hdr:
            return False
        with Image.open(io.BytesIO(data)) as base, Image.open(io.BytesIO(gm_bytes)) as gm:
            return gm.width <= base.width and gm.height <= base.height
    except (OSError, HdrCodecError):
        return False


def _mpf_images(data: bytes) -> list[tuple[int, int]]:
    """(offset, size) of each image listed in the primary's MPF segment (absolute offsets)."""
    try:
        return _parse_mpf(data)
    except (struct.error, IndexError):
        return []  # damaged MPF: the caller falls back to the XMP directory


def _parse_mpf(data: bytes) -> list[tuple[int, int]]:
    for marker, s, _e in _segments(data):
        if marker != 0xE2 or data[s : s + 4] != MPF_HEADER:
            continue
        base = s + 4  # offsets are relative to the MP header (TIFF header)
        order = "<" if data[base : base + 2] == b"II" else ">"
        ifd = base + struct.unpack(order + "I", data[base + 4 : base + 8])[0]
        count = struct.unpack(order + "H", data[ifd : ifd + 2])[0]
        entries_at, n = None, 0
        for i in range(count):
            tag, _typ, cnt, val = struct.unpack(
                order + "HHII", data[ifd + 2 + 12 * i : ifd + 14 + 12 * i]
            )
            if tag == 0xB002:
                entries_at, n = base + val, cnt // 16
        if entries_at is None:
            return []
        images = []
        for i in range(n):
            _attr, size, offset, _d1, _d2 = struct.unpack(
                order + "IIIHH", data[entries_at + 16 * i : entries_at + 16 * i + 16]
            )
            images.append((0 if i == 0 else base + offset, size))
        return images
    return []


def _gain_map_bytes(data: bytes) -> bytes:
    images = _mpf_images(data)
    if len(images) >= 2:
        offset, size = images[1]
        part = data[offset : offset + size]
        if part[:2] == b"\xff\xd8" and len(part) == size:
            return part
    # No (usable) MPF: the XMP directory's Item:Length counts from the end.
    xmp = _xmp_of(data) or ""
    match = re.search(r'Item:Semantic="GainMap"[^>]*Item:Length="(\d+)"', xmp) or re.search(
        r'Item:Length="(\d+)"[^>]*Item:Semantic="GainMap"', xmp
    )
    if match and 0 < int(match.group(1)) < len(data):
        part = data[-int(match.group(1)) :]
        if part[:2] == b"\xff\xd8":
            return part
    raise HdrCodecError("The gain map image could not be found in this JPEG.")


_FIELDS = (
    "GainMapMin",
    "GainMapMax",
    "Gamma",
    "OffsetSDR",
    "OffsetHDR",
    "HDRCapacityMin",
    "HDRCapacityMax",
    "BaseRenditionIsHDR",
)


def parse_metadata(xmp: str) -> GainMapMetadata:
    """``hdrgm`` values from a gain map's XMP (attributes or elements, scalar or rdf:Seq)."""
    found: dict[str, list[str]] = {}
    for name in _FIELDS:
        attr = re.search(rf'hdrgm:{name}="([^"]*)"', xmp)
        if attr:
            found[name] = [attr.group(1)]
            continue
        elem = re.search(rf"<hdrgm:{name}>(.*?)</hdrgm:{name}>", xmp, re.S)
        if elem:
            items = re.findall(r"<rdf:li>([^<]*)</rdf:li>", elem.group(1))
            found[name] = items or [elem.group(1).strip()]
    if "GainMapMax" not in found or "HDRCapacityMax" not in found:
        raise HdrCodecError("The gain map metadata is incomplete (no GainMapMax).")

    def floats(name: str, default: float) -> tuple[float, ...]:
        try:
            return tuple(float(v) for v in found.get(name, [str(default)]))
        except ValueError as exc:
            raise HdrCodecError(f"Unreadable gain map value {name}.") from exc

    meta = GainMapMetadata(
        gain_min=floats("GainMapMin", 0.0),
        gain_max=floats("GainMapMax", 1.0),
        gamma=floats("Gamma", 1.0),
        offset_sdr=floats("OffsetSDR", OFFSET),
        offset_hdr=floats("OffsetHDR", OFFSET),
        capacity_min=floats("HDRCapacityMin", 0.0)[0],
        capacity_max=floats("HDRCapacityMax", 1.0)[0],
        base_is_hdr=found.get("BaseRenditionIsHDR", ["False"])[0].strip().lower() == "true",
    )
    _check_metadata(meta)
    return meta


# Gains are log2 boosts; real files stay within a few stops, and 2**32 is far past any display.
_MAX_STOPS = 32.0


def _check_metadata(meta: GainMapMetadata) -> None:
    """Reject values that would turn every pixel into NaN / inf (damaged or crafted files)."""
    values = (
        *meta.gain_min,
        *meta.gain_max,
        *meta.gamma,
        *meta.offset_sdr,
        *meta.offset_hdr,
        meta.capacity_min,
        meta.capacity_max,
    )
    if not all(math.isfinite(v) for v in values):
        raise HdrCodecError("The gain map metadata holds a non-finite value.")
    if any(g <= 0 for g in meta.gamma):
        raise HdrCodecError("The gain map metadata has an invalid Gamma.")
    if any(abs(v) > _MAX_STOPS for v in (*meta.gain_min, *meta.gain_max)):
        raise HdrCodecError("The gain map metadata has an implausible boost.")
    if any(
        hi < lo
        for lo, hi in zip(_per_channel(meta.gain_min), _per_channel(meta.gain_max), strict=True)
    ):
        raise HdrCodecError("The gain map metadata has GainMapMax below GainMapMin.")


def _per_channel(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(values[k] if k < len(values) else values[0] for k in range(3))


# --- reading ---------------------------------------------------------------------------
@dataclass
class GainMapImage:
    """A decoded gain-map JPEG: HDR rendition in linear BT.2020 (1.0 = SDR white)."""

    linear: np.ndarray
    metadata: GainMapMetadata
    icc_note: str


def decode(path: Path) -> GainMapImage:
    """Rebuild the full HDR rendition (weight 1) of a gain-map JPEG."""
    from pixelift.core.hdr_output.pipeline import sdr_colour_space

    data = path.read_bytes()
    gm_bytes = _gain_map_bytes(data)
    meta = parse_metadata(_xmp_of(gm_bytes) or "")
    if meta.base_is_hdr:
        raise HdrCodecError(
            f"{path.name} has an HDR base rendition (BaseRenditionIsHDR), which is not supported."
        )
    with Image.open(io.BytesIO(data)) as img:
        img.draft(None, None)
        icc = img.info.get("icc_profile")
        base = np.asarray(img.convert("RGB"))
    rgb, space, note = sdr_colour_space(base, icc)
    from pixelift.core.hdr_output.pipeline import _linearize

    sdr = _linearize(space, rgb.astype(np.float32) / np.float32(255))
    with Image.open(io.BytesIO(gm_bytes)) as gm_img:
        # Checked before decoding: the base already passed the size / memory checks,
        # and a gain map is never larger than its base (a bigger one is a bomb).
        if gm_img.width > base.shape[1] or gm_img.height > base.shape[0]:
            raise HdrCodecError("The gain map is larger than the image it belongs to.")
        channels = 3 if gm_img.mode not in ("L", "1", "P") else 1
        gm_img = gm_img.convert("RGB" if channels == 3 else "L")
        size = (base.shape[1], base.shape[0])
        planes = [
            np.asarray(
                Image.fromarray(np.asarray(band, np.float32) / 255, "F").resize(
                    size, Image.Resampling.BILINEAR
                )
            )
            for band in gm_img.split()
        ]
    hdr = np.empty_like(sdr)
    for c in range(3):
        k = c if len(planes) == 3 else 0

        def pick(values: tuple[float, ...], k: int = k) -> np.float32:
            return np.float32(values[k] if k < len(values) else values[0])

        g = np.clip(planes[k], 0, 1)
        gamma = pick(meta.gamma)
        if gamma != 1:
            g = np.power(g, 1 / gamma)
        log_boost = pick(meta.gain_min) * (1 - g) + pick(meta.gain_max) * g
        hdr[..., c] = (sdr[..., c] + pick(meta.offset_sdr)) * np.exp2(log_boost) - pick(
            meta.offset_hdr
        )
    linear = cs.apply_matrix(np.maximum(hdr, 0), space.to_bt2020())
    cs.fit_gamut(linear)
    return GainMapImage(linear, meta, note)


# --- writing ---------------------------------------------------------------------------
def _segment(marker: int, payload: bytes) -> bytes:
    if len(payload) + 2 > 0xFFFF:
        raise HdrCodecError("JPEG metadata segment too large.")
    return bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload


def _insert_after_app0(jpeg: bytes, segments: bytes) -> bytes:
    """Insert ``segments`` after SOI (and a JFIF APP0, which must stay first)."""
    pos = 2
    for marker, _s, e in _segments(jpeg):
        if marker == 0xE0:
            pos = e
        break
    return jpeg[:pos] + segments + jpeg[pos:]


def _mpf_payload(primary_size: int, gain_map_size: int, gain_map_offset: int) -> bytes:
    """MPF APP2 payload (little-endian) for two images; offset relative to the MP header."""
    entries = struct.pack("<IIIHH", 0x030000, primary_size, 0, 0, 0)
    entries += struct.pack("<IIIHH", 0x000000, gain_map_size, gain_map_offset, 0, 0)
    ifd_entries = 3
    ifd_size = 2 + 12 * ifd_entries + 4
    entries_offset = 8 + ifd_size
    ifd = struct.pack("<H", ifd_entries)
    ifd += struct.pack("<HHI4s", 0xB000, 7, 4, b"0100")  # MPFVersion
    ifd += struct.pack("<HHII", 0xB001, 4, 1, 2)  # NumberOfImages
    ifd += struct.pack("<HHII", 0xB002, 7, len(entries), entries_offset)  # MPEntry
    ifd += struct.pack("<I", 0)  # no next IFD
    return MPF_HEADER + b"II*\0" + struct.pack("<I", 8) + ifd + entries


def assemble(base_jpeg: bytes, gain_map_jpeg: bytes, meta: GainMapMetadata) -> bytes:
    """Combine an SDR base JPEG and a gain-map JPEG into one gain-map HDR JPEG."""
    gain_map = _insert_after_app0(gain_map_jpeg, _segment(0xE1, meta.xmp()))
    xmp = _segment(0xE1, _primary_xmp(len(gain_map)))
    mpf_len = len(_segment(0xE2, _mpf_payload(0, 0, 0)))
    primary_size = len(base_jpeg) + len(xmp) + mpf_len
    pos = 2  # the segments go after SOI and a JFIF APP0
    for marker, _s, e in _segments(base_jpeg):
        if marker == 0xE0:
            pos = e
        break
    # MPF offsets count from the MP header, just after "MPF\0" in the primary.
    mp_header = pos + len(xmp) + 4 + len(MPF_HEADER)  # marker(2) + length(2) + "MPF\0"
    mpf = _segment(0xE2, _mpf_payload(primary_size, len(gain_map), primary_size - mp_header))
    primary = base_jpeg[:pos] + xmp + mpf + base_jpeg[pos:]
    assert len(primary) == primary_size
    return primary + gain_map


def compute_gain_map(
    bands: Iterable[tuple[np.ndarray, np.ndarray]], width: int, height: int
) -> tuple[np.ndarray, float, float]:
    """Per-channel log2 ratios HDR/SDR at 1/``SCALE`` resolution: (map (h, w, 3), min, max).

    ``bands`` yields (hdr_linear_709, sdr_linear_709) row bands, 1.0 = SDR
    white, with heights that are multiples of ``SCALE`` (except the last).
    """
    gw, gh = -(-width // SCALE), -(-height // SCALE)
    out = np.zeros((gh, gw, 3), np.float32)
    y = 0
    for hdr, sdr in bands:
        ratio = np.log2((np.maximum(hdr, 0) + OFFSET) / (np.maximum(sdr, 0) + OFFSET))
        rows = ratio.shape[0]
        ph, pw = -rows % SCALE, -width % SCALE
        if ph or pw:
            ratio = np.pad(ratio, ((0, ph), (0, pw), (0, 0)), mode="edge")
        blocks = ratio.reshape(ratio.shape[0] // SCALE, SCALE, -1, SCALE, 3).mean(axis=(1, 3))
        out[y // SCALE : y // SCALE + blocks.shape[0]] = blocks
        y += rows
    low, high = float(out.min(initial=0.0)), float(out.max(initial=0.0))
    low = min(low, 0.0)
    if high - low < 1e-3:
        high = low + 1e-3
    return out, low, high


def encode_gain_map(ratios: np.ndarray, low: float, high: float, quality: int) -> bytes:
    norm = np.clip((ratios - low) / (high - low), 0, 1)
    img = Image.fromarray(np.rint(norm * 255).astype(np.uint8), "RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=max(quality, 85), subsampling=0)  # no chroma averaging
    return buf.getvalue()


def encode_base(rgb: np.ndarray, quality: int, exif: bytes | None) -> bytes:
    buf = io.BytesIO()
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    kwargs: dict[str, object] = {"quality": quality, "icc_profile": icc}
    kwargs["subsampling"] = 0 if quality >= 90 else 2
    if exif:
        kwargs["exif"] = exif
    Image.fromarray(rgb, "RGB").save(buf, "JPEG", **kwargs)
    return buf.getvalue()


def metadata_for(low: float, high: float) -> GainMapMetadata:
    return GainMapMetadata(
        gain_min=(low,),
        gain_max=(high,),
        gamma=(1.0,),
        offset_sdr=(OFFSET,),
        offset_hdr=(OFFSET,),
        capacity_min=0.0,
        capacity_max=max(high, 1e-3),
    )


# --- validation --------------------------------------------------------------------------
def inspect(path: Path) -> FileReport:
    """Check a gain-map JPEG's structure; raises ``HdrCodecError`` when it is inconsistent."""
    data = path.read_bytes()
    report = FileReport("JPEG gain map")
    xmp = _xmp_of(data)
    if xmp is None or HDRGM_NS not in xmp:
        raise HdrCodecError(f"{path.name} has no gain-map XMP.")
    images = _mpf_images(data)
    if len(images) != 2:
        raise HdrCodecError(f"{path.name}: the MPF segment does not list two images.")
    (_, primary_size), (offset, size) = images
    if offset != primary_size or offset + size != len(data):
        raise HdrCodecError(f"{path.name}: MPF offsets do not match the file layout.")
    gm = _gain_map_bytes(data)
    length = re.search(r'Item:Length="(\d+)"', xmp)
    if not length or int(length.group(1)) != len(gm):
        raise HdrCodecError(f"{path.name}: the XMP gain map length is wrong.")
    meta = parse_metadata(_xmp_of(gm) or "")
    with Image.open(io.BytesIO(data[:primary_size])) as base:
        report.width, report.height = base.size
        report.channels = len(base.getbands())
        report.has_icc = bool(base.info.get("icc_profile"))
    with Image.open(io.BytesIO(gm)) as gm_img:
        gm_img.load()
    report.bit_depth = 8
    report.headroom_stops = meta.capacity_max
    report.chunks = ["MPF", "XMP:Container", "XMP:hdrgm"]
    return report


Check = Callable[[], None]
