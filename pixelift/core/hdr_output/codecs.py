"""HDR file formats: writing, inspecting and decoding (no GUI, no PyTorch).

Output formats
--------------
* **PNG, 16 bits per channel, with a ``cICP`` chunk** (PNG Third Edition,
  W3C 2025). ``cICP`` carries ITU-T H.273 code points: colour primaries
  (9 = BT.2020, 12 = Display P3), transfer characteristics (16 = PQ),
  matrix coefficients (0 = RGB, required for PNG) and the full-range flag.
  Transfer 16 (PQ) or 18 (HLG); ``cLLi`` is written for PQ only.
  Written by ``write_png16`` with zlib only (no extra dependency), band by
  band, with alpha (RGBA) support. A ``cLLi`` chunk carries the content
  light level (MaxCLL / MaxFALL, measured from the pixels). No ``iCCP``,
  ``gAMA`` or ``sRGB`` chunk is written, so nothing contradicts ``cICP``; no
  mastering-display (``mDCv``) chunk is written, because Pixelift does not
  know the display the picture was graded on.
* **AVIF, 10 bits, 4:4:4 YCbCr** through ffmpeg's libaom-av1 encoder (an
  isolated subprocess; optional). The colour description is written as an
  ``nclx`` ``colr`` box and in the AV1 sequence header. Transparency is a
  10-bit alpha auxiliary item (``auxC``), fed to ffmpeg as a second raw input.
  Gain-map JPEGs are in ``gainmap``.

Every written file is read back with ``inspect_png`` / ``inspect_avif`` (our
own parsers; AVIF planes are also decoded back with ``read_avif_planes``)
before it is reported as a success, and with ``ffprobe`` when it is installed
(an independent implementation).

Decoding HDR sources
--------------------
``sniff`` recognises HDR sources by their colour description, not their
extension: PNG and AVIF whose ``cICP`` / ``nclx`` says PQ or HLG, OpenEXR and
Radiance RGBE (scene-linear). They are decoded with ffmpeg (``decode``),
because Pillow reduces 16-bit RGB to 8 bits and cannot read PQ, EXR or RGBE.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# ITU-T H.273 code points
CP_BT709, CP_BT2020, CP_P3 = 1, 9, 12
TC_LINEAR, TC_SRGB, TC_PQ, TC_HLG = 8, 13, 16, 18
MC_RGB, MC_BT709, MC_BT2020_NCL = 0, 1, 9
PRIMARIES_CODES = {"bt709": CP_BT709, "bt2020": CP_BT2020, "p3": CP_P3}
CODES_PRIMARIES = {v: k for k, v in PRIMARIES_CODES.items()}
_FFMPEG_TIMEOUT = 3600

Check = Callable[[], None]


class HdrCodecError(Exception):
    """An HDR file could not be written, read or validated (message is user-facing)."""

    def __init__(self, message: str, suggestions: list[str] | None = None) -> None:
        super().__init__(message)
        self.suggestions = suggestions or []


# --- tools -----------------------------------------------------------------------------
def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


def ff_file(path: Path) -> str:
    """``path`` as an ffmpeg / ffprobe URL: never an option (``-x.exr``) or protocol (``a:b``)."""
    return "file:" + str(Path(path).resolve())


_encoder_cache: dict[str, bool] = {}


def ffmpeg_has_encoder(name: str) -> bool:
    if name not in _encoder_cache:
        exe = ffmpeg_path()
        ok = False
        if exe:
            try:
                out = subprocess.run(
                    [exe, "-hide_banner", "-encoders"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                ).stdout
                ok = any(line.split()[1:2] == [name] for line in out.splitlines())
            except (OSError, subprocess.SubprocessError):
                ok = False
        _encoder_cache[name] = ok
    return _encoder_cache[name]


def avif_available() -> bool:
    """Whether 10-bit HDR AVIF can be written (ffmpeg with libaom-av1)."""
    return ffmpeg_has_encoder("libaom-av1")


def decoder_available() -> bool:
    return ffmpeg_path() is not None


# --- PNG writing -------------------------------------------------------------------------
def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _filter_band(raw: np.ndarray, prev: np.ndarray, bpp: int) -> bytes:
    """Paeth-filter ``raw`` (n, row_bytes) uint8 rows; ``prev`` is the row above (zeros at top).

    PNG filter type 4. Encoding only uses unfiltered neighbours, so it
    vectorises; rows go in small groups to bound the int16 temporaries.
    """
    parts = []
    step = max(1, _FILTER_BYTES // max(raw.shape[1], 1))
    for i in range(0, raw.shape[0], step):
        rows = raw[i : i + step]
        above = prev if i == 0 else raw[i - 1]
        parts.append(_paeth(rows, above, bpp))
    return b"".join(parts)


_FILTER_BYTES = 1 << 20


def _paeth(raw: np.ndarray, prev: np.ndarray, bpp: int) -> bytes:
    x = raw.astype(np.int16)
    b = np.concatenate([prev[None].astype(np.int16), x[:-1]])
    a = np.zeros_like(x)
    a[:, bpp:] = x[:, :-bpp]
    c = np.zeros_like(x)
    c[:, bpp:] = b[:, :-bpp]
    p = a + b - c
    pa, pb, pc = np.abs(p - a), np.abs(p - b), np.abs(p - c)
    pred = np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))
    out = np.empty((x.shape[0], x.shape[1] + 1), dtype=np.uint8)
    out[:, 0] = 4
    out[:, 1:] = (x - pred).astype(np.uint8)
    return out.tobytes()


@dataclass(frozen=True)
class LightLevel:
    """Content light level (CTA-861.3) in cd/m²: the brightest pixel's largest
    component (MaxCLL) and the picture's average of that (MaxFALL)."""

    max_cll: float
    max_fall: float


def write_png16(
    target: Path,
    width: int,
    height: int,
    channels: int,
    bands: Iterable[np.ndarray],
    cicp: tuple[int, int, int, int],
    light_level: Callable[[], LightLevel | None] | None = None,
    exif: bytes | None = None,
    dpi: tuple[float, float] | None = None,
    check: Check | None = None,
    level: int = 6,
) -> None:
    """Write a 16-bit RGB(A) PNG with a ``cICP`` chunk to ``target``.

    ``bands`` yields (rows, width, channels) ``uint16`` arrays top to bottom.
    ``light_level`` is called after the last band (when every pixel has been
    seen) for the ``cLLi`` chunk, which must precede the image data; the
    compressed data is therefore spooled to a temporary file next to
    ``target`` first.
    """
    if channels not in (3, 4):
        raise ValueError("channels must be 3 or 4")
    if cicp[2] != MC_RGB:
        raise ValueError("PNG requires matrix coefficients 0 (RGB) in cICP")
    bpp = 2 * channels
    prev = np.zeros(width * bpp, dtype=np.uint8)
    rows_seen = 0
    with tempfile.TemporaryFile(dir=target.parent) as spool:
        comp = zlib.compressobj(level)
        for band in bands:
            if band.dtype != np.uint16 or band.shape[1:] != (width, channels):
                raise ValueError("bands must be (rows, width, channels) uint16")
            raw = band.astype(">u2", copy=False).view(np.uint8).reshape(band.shape[0], -1)
            spool.write(comp.compress(_filter_band(raw, prev, bpp)))
            prev = raw[-1].copy()
            rows_seen += band.shape[0]
            if check:
                check()
        spool.write(comp.flush())
        if rows_seen != height:
            raise ValueError(f"got {rows_seen} rows, expected {height}")
        level_info = light_level() if light_level else None
        spool.seek(0)
        with open(target, "wb") as fh:
            fh.write(PNG_SIGNATURE)
            colour_type = 6 if channels == 4 else 2
            fh.write(
                _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 16, colour_type, 0, 0, 0))
            )
            fh.write(_chunk(b"cICP", bytes(cicp)))
            if level_info is not None:
                fh.write(
                    _chunk(
                        b"cLLi",
                        struct.pack(
                            ">II",
                            min(round(level_info.max_cll * 10000), 0xFFFFFFFF),
                            min(round(level_info.max_fall * 10000), 0xFFFFFFFF),
                        ),
                    )
                )
            if dpi:
                ppm = [max(1, round(d / 0.0254)) for d in dpi]
                fh.write(_chunk(b"pHYs", struct.pack(">IIB", ppm[0], ppm[1], 1)))
            if exif:
                fh.write(_chunk(b"eXIf", exif[6:] if exif.startswith(b"Exif\x00\x00") else exif))
            while True:
                data = spool.read(1 << 22)
                if not data:
                    break
                fh.write(_chunk(b"IDAT", data))
                if check:
                    check()
            fh.write(_chunk(b"IEND", b""))


# --- PNG inspection -------------------------------------------------------------------
@dataclass
class FileReport:
    """What an encoded file says about itself (read from its bytes)."""

    format: str
    width: int = 0
    height: int = 0
    bit_depth: int = 0
    channels: int = 0
    primaries: int | None = None  # H.273 code points
    transfer: int | None = None
    matrix: int | None = None
    full_range: bool | None = None
    max_cll: float | None = None  # cd/m²
    max_fall: float | None = None
    has_icc: bool = False
    has_alpha: bool = False  # AVIF: an alpha auxiliary image (PNG: see ``channels``)
    headroom_stops: float | None = None  # gain-map JPEG: HDRCapacityMax
    chunks: list[str] = field(default_factory=list)

    @property
    def is_pq(self) -> bool:
        return self.transfer == TC_PQ

    @property
    def is_hlg(self) -> bool:
        return self.transfer == TC_HLG

    def describe(self) -> str:
        if self.headroom_stops is not None:
            return (
                f"{self.format} {self.width}×{self.height}, sRGB base + gain map, "
                f"up to {2**self.headroom_stops:.1f}× SDR white"
            )
        names = {TC_PQ: "PQ", TC_HLG: "HLG", TC_SRGB: "sRGB", TC_LINEAR: "linear"}
        prim = CODES_PRIMARIES.get(self.primaries or -1, str(self.primaries))
        return (
            f"{self.format} {self.width}×{self.height}, {self.bit_depth}-bit, "
            f"{names.get(self.transfer or -1, self.transfer)}, {prim}"
        )


def inspect_png(path: Path, verify_data: bool = True) -> FileReport:
    """Parse a PNG's chunks (CRC-checked); with ``verify_data`` also inflate
    the image data and check it holds exactly the declared rows."""
    report = FileReport("PNG")
    idat = zlib.decompressobj() if verify_data else None
    inflated = 0
    with open(path, "rb") as fh:
        if fh.read(8) != PNG_SIGNATURE:
            raise HdrCodecError(f"{path.name} is not a PNG file.")
        while True:
            head = fh.read(8)
            if len(head) < 8:
                raise HdrCodecError(f"{path.name} is truncated (no IEND chunk).")
            length, kind = struct.unpack(">I4s", head)
            data = fh.read(length)
            crc = fh.read(4)
            if len(data) < length or len(crc) < 4:
                raise HdrCodecError(f"{path.name} is truncated.")
            if struct.unpack(">I", crc)[0] != zlib.crc32(kind + data):
                raise HdrCodecError(f"{path.name} is damaged (CRC mismatch in {kind!r}).")
            name = kind.decode("latin-1")
            report.chunks.append(name)
            if kind == b"IHDR":
                if len(data) != 13:
                    raise HdrCodecError(f"{path.name} is damaged (bad IHDR chunk).")
                w, h, depth, ctype = struct.unpack(">IIBB", data[:10])
                report.width, report.height, report.bit_depth = w, h, depth
                report.channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(ctype, 0)
            elif kind == b"cICP" and len(data) == 4:
                report.primaries, report.transfer, report.matrix = data[0], data[1], data[2]
                report.full_range = bool(data[3])
            elif kind == b"cLLi" and len(data) == 8:
                cll, fall = struct.unpack(">II", data)
                report.max_cll, report.max_fall = cll / 10000, fall / 10000
            elif kind == b"iCCP":
                report.has_icc = True
            elif kind == b"IDAT":
                if idat is not None:
                    inflated += len(idat.decompress(data, 1 << 20))
                    while idat.unconsumed_tail:
                        inflated += len(idat.decompress(idat.unconsumed_tail, 1 << 20))
                elif not verify_data:
                    break  # header is all that was asked for
            elif kind == b"IEND":
                break
    if verify_data and idat is not None:
        inflated += len(idat.flush())
        expected = report.height * (1 + report.width * report.channels * report.bit_depth // 8)
        if inflated != expected:
            raise HdrCodecError(f"{path.name}: image data is incomplete or damaged.")
    return report


def read_png_cicp(path: Path) -> tuple[int, int, int, int] | None:
    """The ``cICP`` code points of a PNG (header only), or None."""
    try:
        with open(path, "rb") as fh:
            if fh.read(8) != PNG_SIGNATURE:
                return None
            for _ in range(64):  # cICP must precede IDAT
                head = fh.read(8)
                if len(head) < 8:
                    return None
                length, kind = struct.unpack(">I4s", head)
                if kind in (b"IDAT", b"IEND"):
                    return None
                if kind == b"cICP" and length == 4:
                    data = fh.read(4)
                    return (data[0], data[1], data[2], data[3]) if len(data) == 4 else None
                fh.seek(length + 4, 1)
    except OSError:
        return None
    return None


# --- AVIF (ISOBMFF) inspection -----------------------------------------------------------
def _boxes(data: bytes, start: int, end: int) -> Iterable[tuple[bytes, int, int]]:
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack(">I4s", data[pos : pos + 8])
        header = 8
        if size == 1:
            size = struct.unpack(">Q", data[pos + 8 : pos + 16])[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            return
        yield kind, pos + header, pos + size
        pos += size


_AVIF_HEADER_BYTES = 1 << 20  # the meta box precedes the image data in practice


def inspect_avif(path: Path, header_only: bool = False) -> FileReport:
    """Read an AVIF's brand and image properties (``ispe``, ``pixi``, ``av1C``, ``colr``).

    ``header_only`` reads just the start of the file (for recognising HDR
    sources quickly) and does not check that the image data is complete.
    """
    with open(path, "rb") as fh:
        data = fh.read(_AVIF_HEADER_BYTES if header_only else -1)
    report = FileReport("AVIF")
    top: dict[bytes, tuple[int, int]] = {}
    pos = 0
    while pos + 8 <= len(data):  # top-level boxes; the last may be cut off in header mode
        size, kind = struct.unpack(">I4s", data[pos : pos + 8])
        header = 8
        if size == 1 and pos + 16 <= len(data):
            size, header = struct.unpack(">Q", data[pos + 8 : pos + 16])[0], 16
        elif size == 0:
            size = len(data) - pos
        if size < header:
            break
        top.setdefault(kind, (pos + header, min(pos + size, len(data))))
        pos += size
    if b"ftyp" not in top:
        raise HdrCodecError(f"{path.name} is not an AVIF file.")
    s, e = top[b"ftyp"]
    brands = {data[s : s + 4]} | {data[i : i + 4] for i in range(s + 8, e, 4)}
    if b"avif" not in brands:
        raise HdrCodecError(f"{path.name} is not an AVIF file (brand).")
    if not header_only:
        mdat = top.get(b"mdat")
        if mdat is None or mdat[1] <= mdat[0] or pos > len(data):
            raise HdrCodecError(f"{path.name} has no image data or is truncated.")

    if b"meta" in top:
        _read_avif_meta(data, top[b"meta"][0] + 4, top[b"meta"][1], report)
    return report


_ALPHA_URNS = (b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha", b"urn:mpeg:hevc:2015:auxid:1")


def _read_avif_meta(data: bytes, start: int, end: int, report: FileReport) -> None:
    """Fill ``report`` from the primary item's properties (``pitm`` + ``ipma``)."""
    props: list[tuple[bytes, int, int]] = []  # ipco, in order (1-based in ipma)
    assoc: dict[int, list[int]] = {}
    primary = 1

    def walk(start: int, end: int) -> None:
        nonlocal primary
        for kind, s, e in _boxes(data, start, end):
            report.chunks.append(kind.decode("latin-1"))
            if kind == b"iprp":
                walk(s, e)
            elif kind == b"ipco":
                props.extend(_boxes(data, s, e))
            elif kind == b"pitm" and e - s >= 6:
                primary = (
                    struct.unpack(">H", data[s + 4 : s + 6])[0]
                    if data[s] == 0
                    else (struct.unpack(">I", data[s + 4 : s + 8])[0])
                )
            elif kind == b"ipma" and e - s >= 8:
                version, flags = data[s], int.from_bytes(data[s + 1 : s + 4], "big")
                count = struct.unpack(">I", data[s + 4 : s + 8])[0]
                pos = s + 8
                id_size, index_size = (2 if version < 1 else 4), (2 if flags & 1 else 1)
                for _ in range(count):
                    if pos + id_size + 1 > e:
                        break  # truncated: keep what was read
                    if version < 1:
                        item = struct.unpack(">H", data[pos : pos + 2])[0]
                        pos += 2
                    else:
                        item = struct.unpack(">I", data[pos : pos + 4])[0]
                        pos += 4
                    n = data[pos]
                    pos += 1
                    if pos + n * index_size > e:
                        break
                    indices = []
                    for _ in range(n):
                        if flags & 1:
                            indices.append(struct.unpack(">H", data[pos : pos + 2])[0] & 0x7FFF)
                            pos += 2
                        else:
                            indices.append(data[pos] & 0x7F)
                            pos += 1
                    assoc[item] = indices

    walk(start, end)
    for item, indices in assoc.items():
        kinds = [props[i - 1] for i in indices if 0 < i <= len(props)]
        if item != primary:
            for kind, s, e in kinds:
                if kind == b"auxC" and data[s + 4 : e].rstrip(b"\0") in _ALPHA_URNS:
                    report.has_alpha = True
            continue
        for kind, s, e in kinds:
            if kind == b"ispe" and e - s >= 12:
                report.width, report.height = struct.unpack(">II", data[s + 4 : s + 12])
            elif kind == b"pixi" and e - s >= 6:
                report.channels = data[s + 4]
                report.bit_depth = data[s + 5] if report.channels else 0
            elif kind == b"av1C" and e - s >= 4 and not report.bit_depth:
                flags = data[s + 2]
                report.bit_depth = 12 if flags & 0x20 else (10 if flags & 0x40 else 8)
            elif kind == b"colr" and e - s >= 4:
                ctype = data[s : s + 4]
                if ctype == b"nclx" and e - s >= 11:
                    p, t, m = struct.unpack(">HHH", data[s + 4 : s + 10])
                    report.primaries, report.transfer, report.matrix = p, t, m
                    report.full_range = bool(data[s + 10] & 0x80)
                elif ctype in (b"prof", b"rICC"):
                    report.has_icc = True
            elif kind == b"clli" and e - s >= 4:
                cll, fall = struct.unpack(">HH", data[s : s + 4])
                report.max_cll, report.max_fall = float(cll), float(fall)


def read_avif_nclx(path: Path) -> tuple[int, int, int, int] | None:
    try:
        report = inspect_avif(path, header_only=True)
    except (OSError, HdrCodecError, struct.error, IndexError, ValueError):
        return None
    if report.transfer is None:
        return None
    return (
        report.primaries or 2,  # 0 is reserved
        report.transfer,
        report.matrix if report.matrix is not None else 2,  # 0 is RGB (identity)
        int(bool(report.full_range)),
    )


# --- ffprobe (independent check) -----------------------------------------------------
def ffprobe_stream(path: Path) -> dict | None:
    """ffprobe's view of the first video stream, or None without ffprobe."""
    exe = ffprobe_path()
    if not exe:
        return None
    try:
        out = subprocess.run(
            [
                exe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-of",
                "json",
                ff_file(path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return {"error": out.stderr.strip()[:300]}
    try:
        streams = json.loads(out.stdout).get("streams") or []
    except json.JSONDecodeError:
        return None
    return streams[0] if streams else {"error": "no video stream"}


# --- AVIF writing -----------------------------------------------------------------------
_MATRIX_KR_KB = {MC_BT709: (0.2126, 0.0722), MC_BT2020_NCL: (0.2627, 0.0593)}
_FF_NAMES = {
    "primaries": {CP_BT709: "bt709", CP_BT2020: "bt2020", CP_P3: "smpte432"},
    "transfer": {TC_PQ: "smpte2084", TC_HLG: "arib-std-b67"},
    "matrix": {MC_BT709: "bt709", MC_BT2020_NCL: "bt2020nc"},
}


def rgb_to_ycbcr10(rgb: np.ndarray, matrix: int) -> np.ndarray:
    """Non-linear (PQ-encoded) R'G'B' 0..1 -> full-range 10-bit Y'CbCr planes (3, H, W) uint16."""
    kr, kb = _MATRIX_KR_KB[matrix]
    kg = 1 - kr - kb
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = kr * r + kg * g + kb * b
    cb = (b - y) / (2 * (1 - kb))
    cr = (r - y) / (2 * (1 - kr))
    out = np.empty((3, *rgb.shape[:2]), dtype=np.uint16)
    out[0] = np.clip(np.rint(y * 1023), 0, 1023)
    out[1] = np.clip(np.rint(cb * 1023 + 512), 0, 1023)
    out[2] = np.clip(np.rint(cr * 1023 + 512), 0, 1023)
    return out


def ycbcr_to_rgb(planes: np.ndarray, matrix: int, full_range: bool, depth: int) -> np.ndarray:
    """Y'CbCr ``depth``-bit planes (3, H, W) -> R'G'B' 0..1 float32 (H, W, 3)."""
    y, cb, cr = (planes[i].astype(np.float32) for i in range(3))
    top = np.float32((1 << depth) - 1)
    half = np.float32(1 << (depth - 1))
    if full_range:
        y, cb, cr = y / top, (cb - half) / top, (cr - half) / top
    else:
        k = np.float32(1 << (depth - 8))
        y = (y - 16 * k) / (219 * k)
        cb, cr = (cb - 128 * k) / (224 * k), (cr - 128 * k) / (224 * k)
    kr, kb = _MATRIX_KR_KB.get(matrix, _MATRIX_KR_KB[MC_BT709])
    kg = 1 - kr - kb
    r = y + 2 * (1 - kr) * cr
    b = y + 2 * (1 - kb) * cb
    g = (y - kr * r - kb * b) / kg
    return np.clip(np.stack([r, g, b], axis=-1), 0, 1).astype(np.float32)


def avif_crf(quality: int) -> int:
    """Pixelift's 1..100 quality -> libaom's constant-quality level (63 worst, 0 lossless-ish)."""
    return round(63 * (1 - min(100, max(1, quality)) / 100) ** 0.7)


def write_avif10(
    target: Path,
    planes: np.ndarray,
    primaries: int,
    matrix: int,
    quality: int,
    check: Check | None = None,
    transfer: int = TC_PQ,
) -> None:
    """Encode full-range 10-bit 4:4:4 Y'CbCr ``planes`` as an HDR AVIF.

    ``planes`` is (3, H, W), or (4, H, W) with a 10-bit alpha plane, which
    becomes an AVIF alpha auxiliary image (``auxC``
    ``urn:mpeg:mpegB:cicp:systems:auxiliary:alpha``). ``transfer`` is the H.273
    transfer code point (16 PQ, 18 HLG).
    """
    exe = ffmpeg_path()
    if not exe or not avif_available():
        raise HdrCodecError(
            "HDR AVIF export needs ffmpeg with the libaom-av1 encoder, which was not found.",
            ["Choose 16-bit PNG as the HDR format", "Install ffmpeg (sudo apt install ffmpeg)"],
        )
    count, height, width = planes.shape
    alpha = count == 4
    colour = [
        "-color_primaries", _FF_NAMES["primaries"][primaries],
        "-color_trc", _FF_NAMES["transfer"][transfer],
        "-colorspace", _FF_NAMES["matrix"][matrix],
        "-color_range", "pc",
    ]  # fmt: skip
    alpha_file: Path | None = None
    if alpha:
        # The alpha plane goes in as a second raw input (no filters, which
        # could convert the colour planes' range), becoming the alpha item.
        fd, name = tempfile.mkstemp(prefix=".alpha-", suffix=".raw", dir=target.parent)
        alpha_file = Path(name)
        with os.fdopen(fd, "wb") as fh:
            fh.write(planes[3].astype("<u2", copy=False).tobytes())
        extra = ["-f", "rawvideo", "-pix_fmt", "gray10le", "-s", f"{width}x{height}"]
        extra += ["-color_range", "pc", "-i", ff_file(alpha_file)]
        maps = ["-map", "0:v", "-map", "1:v", "-color_range:1", "pc"]
        maps += ["-pix_fmt:0", "yuv444p10le", "-pix_fmt:1", "gray10le"]
    else:
        extra, maps = [], ["-pix_fmt", "yuv444p10le"]
    cmd = [
        exe, "-v", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "yuv444p10le", "-s", f"{width}x{height}", *colour,
        "-i", "pipe:0",
        *extra,
        *maps,
        "-frames:v", "1", "-c:v", "libaom-av1", "-still-picture", "1",
        "-crf", str(avif_crf(quality)), "-cpu-used", "6", "-row-mt", "1",
        *colour,
        "-f", "avif", ff_file(target),
    ]  # fmt: skip
    try:
        _run_encoder(cmd, planes[:3], height, check)
    finally:
        if alpha_file is not None:
            alpha_file.unlink(missing_ok=True)


def _run_encoder(cmd: list[str], planes: np.ndarray, height: int, check: Check | None) -> None:
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise HdrCodecError(f"Could not start ffmpeg: {exc}") from exc
    assert proc.stdin is not None and proc.stderr is not None
    # stderr is drained concurrently (ffmpeg can't stall on a full pipe), and a
    # watchdog kills a hung encoder even while a write to it is blocked.
    chunks: list[bytes] = []
    reader = threading.Thread(target=lambda: chunks.append(proc.stderr.read()), daemon=True)
    reader.start()
    timed_out = threading.Event()

    def expire() -> None:
        timed_out.set()
        proc.kill()

    watchdog = threading.Timer(_FFMPEG_TIMEOUT, expire)
    watchdog.daemon = True
    watchdog.start()
    try:
        try:
            for plane in planes:
                for y0 in range(0, height, 512):
                    proc.stdin.write(plane[y0 : y0 + 512].astype("<u2", copy=False).tobytes())
                    if check:
                        check()
            proc.stdin.close()
        except BrokenPipeError:
            pass  # ffmpeg exited early; its exit status and stderr say why
        proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        watchdog.cancel()
        with contextlib.suppress(OSError):
            proc.stdin.close()
        reader.join(timeout=10)
    if proc.returncode != 0:
        err = b"".join(chunks)
        detail = err.decode(errors="replace").strip().splitlines()[-1:] if err else []
        if timed_out.is_set():
            detail = ["timed out"]
        raise HdrCodecError(
            "The AVIF encoder failed" + (f": {detail[0][:200]}" if detail else "."),
            ["Choose 16-bit PNG as the HDR format"],
        )


# --- HDR sources ------------------------------------------------------------------------
@dataclass(frozen=True)
class HdrSource:
    """An HDR source file as recognised by ``sniff``.

    ``encoding``: "pq" or "hlg" (display-referred signals), "linear"
    (scene-linear EXR / RGBE, where 1.0 is mapped to the SDR white level) or
    "gainmap" (an SDR JPEG plus a gain map, see ``gainmap``).
    ``chromaticities``: an EXR's own primaries and white (rx, ry, gx, gy, bx,
    by, wx, wy), which override ``primaries``.
    """

    format: str  # "PNG", "AVIF", "EXR", "RGBE", "JPEG"
    encoding: str
    primaries: str  # "bt2020", "p3", "bt709"
    matrix: int = MC_RGB
    full_range: bool = True
    chromaticities: tuple[float, ...] | None = None

    def describe(self) -> str:
        enc = {"pq": "PQ", "hlg": "HLG", "linear": "scene-linear", "gainmap": "gain map"}[
            self.encoding
        ]
        if self.chromaticities is not None:
            prim = "custom primaries"
        else:
            prim = {"bt2020": "BT.2020", "p3": "Display P3", "bt709": "BT.709"}[self.primaries]
        return f"HDR {self.format} ({enc}, {prim})"


HDR_ONLY_EXTENSIONS = frozenset({".exr", ".hdr"})


def sniff(path: Path) -> HdrSource | None:
    """The HDR description of ``path``, or None for SDR (and unreadable) files."""
    path = Path(path)
    try:
        with open(path, "rb") as fh:
            head = fh.read(16)
    except OSError:
        return None
    if head.startswith(PNG_SIGNATURE):
        cicp = read_png_cicp(path)
        return _from_cicp("PNG", cicp) if cicp else None
    if head[4:8] == b"ftyp" and head[8:12] in (b"avif", b"avis", b"mif1", b"heic", b"msf1"):
        nclx = read_avif_nclx(path)
        return _from_cicp("AVIF", nclx) if nclx else None
    if head.startswith(b"\x76\x2f\x31\x01"):
        return HdrSource("EXR", "linear", "bt709", chromaticities=read_exr_chromaticities(path))
    if head.startswith(b"\xff\xd8"):
        from pixelift.core.hdr_output import gainmap

        return HdrSource("JPEG", "gainmap", "bt709") if gainmap.is_gain_map_jpeg(path) else None
    if head.startswith((b"#?RADIANCE", b"#?RGBE")):
        return HdrSource("RGBE", "linear", "bt709")
    return None


_D65_REC709 = (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290)


def read_exr_chromaticities(path: Path) -> tuple[float, ...] | None:
    """An OpenEXR header's ``chromaticities`` attribute (None: absent, i.e. Rec. 709 / D65)."""
    try:
        with open(path, "rb") as fh:
            data = fh.read(1 << 16)  # headers are small
    except OSError:
        return None
    pos = 8  # magic + version
    for _ in range(256):
        name_end = data.find(b"\0", pos)
        if name_end <= pos:  # empty name: end of header (or damaged)
            return None
        type_end = data.find(b"\0", name_end + 1)
        if type_end < 0 or type_end + 5 > len(data):
            return None
        name = data[pos:name_end]
        size = struct.unpack("<i", data[type_end + 1 : type_end + 5])[0]
        value = data[type_end + 5 : type_end + 5 + size]
        if size < 0 or len(value) < size:
            return None
        if name == b"chromaticities" and size == 32:
            chroma = struct.unpack("<8f", value)
            if not all(0 < v < 1 for v in chroma):
                return None
            # Rec. 709 / D65 is what EXR means without the attribute anyway.
            if all(abs(a - b) < 1e-4 for a, b in zip(chroma, _D65_REC709, strict=True)):
                return None
            return tuple(float(v) for v in chroma)
        pos = type_end + 5 + size
    return None


def _from_cicp(fmt: str, cicp: tuple[int, int, int, int]) -> HdrSource | None:
    primaries, transfer, matrix, full = cicp
    encoding = {TC_PQ: "pq", TC_HLG: "hlg"}.get(transfer)
    if encoding is None:
        return None
    return HdrSource(fmt, encoding, CODES_PRIMARIES.get(primaries, "bt2020"), matrix, bool(full))


def probe_size(path: Path) -> tuple[int, int, bool]:
    """(width, height, has_alpha) of an HDR source (as stored, before EXIF orientation)."""
    source = sniff(path)
    if source is not None and source.format == "JPEG":
        from PIL import Image

        with Image.open(path) as img:
            return img.width, img.height, False
    stream = ffprobe_stream(path)
    if stream is None:
        raise _no_decoder(path)
    if "error" in stream:
        raise HdrCodecError(f"{path.name} could not be read: {stream['error']}")
    pix = str(stream.get("pix_fmt", ""))
    alpha = _has_alpha(pix)
    if not alpha and stream.get("codec_name") == "av1":
        with contextlib.suppress(OSError, HdrCodecError, struct.error):
            alpha = inspect_avif(path, header_only=True).has_alpha  # an alpha item
    return int(stream["width"]), int(stream["height"]), alpha


def _has_alpha(pix_fmt: str) -> bool:
    return pix_fmt.startswith(("rgba", "bgra", "gbrap", "ya", "yuva", "argb", "abgr"))


def _no_decoder(path: Path) -> HdrCodecError:
    return HdrCodecError(
        f"{path.name} is an HDR image; reading it needs ffmpeg, which was not found.",
        ["Install ffmpeg (sudo apt install ffmpeg)"],
    )


def decode(path: Path, source: HdrSource) -> tuple[np.ndarray, np.ndarray | None]:
    """Decode an HDR source -> (signal, alpha).

    ``signal`` is (H, W, 3) float32: the PQ / HLG signal 0..1 in R'G'B' for
    PQ and HLG sources, linear values for EXR and RGBE. ``alpha`` is (H, W)
    float32 0..1 or None.
    """
    exe = ffmpeg_path()
    if not exe:
        raise _no_decoder(path)
    width, height, alpha = probe_size(path)
    if source.format == "AVIF" and source.matrix != MC_RGB:
        planes, depth = read_avif_planes(path, width, height)
        rgb = ycbcr_to_rgb(planes, source.matrix, source.full_range, depth)
        return rgb, (_avif_alpha(path, (width, height)) if alpha else None)
    n = 4 if alpha else 3
    if source.encoding == "linear":  # floating-point sources
        fmt, dtype, size, top = ("gbrapf32le" if alpha else "gbrpf32le"), "<f4", 4, 1.0
    else:  # integer signals: exact 16-bit samples
        fmt, dtype, size, top = ("gbrap16le" if alpha else "gbrp16le"), "<u2", 2, 65535.0
    raw = _run_decode(exe, path, fmt, width, height, n, size)
    planes = raw.view(dtype).reshape(n, height, width).astype(np.float32)
    if top != 1.0:
        planes /= np.float32(top)
    rgb = np.stack([planes[2], planes[0], planes[1]], axis=-1)  # G B R planes -> RGB
    alpha_plane = np.clip(planes[3], 0, 1).copy() if alpha else None
    return np.ascontiguousarray(rgb, dtype=np.float32), alpha_plane


def read_avif_planes(
    path: Path, width: int | None = None, height: int | None = None
) -> tuple[np.ndarray, int]:
    """An AVIF's colour planes exactly as stored, chroma upsampled: ((3, H, W) uint16, depth).

    ffmpeg's own pixel-format conversions may assume BT.601 or rescale the
    range, so the native planar format is read and converted here.
    """
    exe = ffmpeg_path()
    if not exe:
        raise _no_decoder(path)
    stream = ffprobe_stream(path) or {}
    if "error" in stream:
        raise HdrCodecError(f"{path.name} could not be read: {stream['error']}")
    width = width or int(stream.get("width", 0))
    height = height or int(stream.get("height", 0))
    layout = _YUV_LAYOUT.match(str(stream.get("pix_fmt", "")))
    if not layout or not width or not height:
        raise HdrCodecError(f"{path.name}: unsupported AVIF pixel format.")
    sub, depth = layout.group(1), int(layout.group(2) or 8)
    sx, sy = {"444": (1, 1), "422": (2, 1), "420": (2, 2)}[sub]
    cw, ch = -(-width // sx), -(-height // sy)
    size = 2 if depth > 8 else 1
    raw = _run_decode_raw(exe, path, layout.group(0), size * (width * height + 2 * cw * ch))
    samples = raw.view("<u2" if size == 2 else np.uint8)
    luma = samples[: width * height].reshape(height, width)
    chroma = samples[width * height :].reshape(2, ch, cw)
    chroma = chroma.repeat(sy, axis=1).repeat(sx, axis=2)[:, :height, :width]
    return np.concatenate([luma[None], chroma]).astype(np.uint16), depth


def _avif_alpha(path: Path, size: tuple[int, int]) -> np.ndarray | None:
    """An AVIF's alpha item via Pillow's libavif (ffmpeg's demuxer only reads the colour)."""
    from PIL import Image

    try:
        with Image.open(path) as img:
            if img.mode not in ("RGBA", "LA") or img.size != size:
                return None
            return np.asarray(img.getchannel("A"), dtype=np.float32) / np.float32(255)
    except (OSError, ValueError):
        log.warning("Could not read the alpha channel of %s", path.name, exc_info=True)
        return None


_YUV_LAYOUT = re.compile(r"yuv(444|422|420)p(?:(10|12)le)?$")


def _run_decode(
    exe: str, path: Path, pix_fmt: str, width: int, height: int, planes: int, size: int
) -> np.ndarray:
    return _run_decode_raw(exe, path, pix_fmt, width * height * planes * size)


def _run_decode_raw(exe: str, path: Path, pix_fmt: str, expected: int) -> np.ndarray:
    cmd = [exe, "-v", "error", "-nostdin", "-i", ff_file(path), "-frames:v", "1"]
    cmd += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=_FFMPEG_TIMEOUT, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise HdrCodecError(f"Could not run ffmpeg to read {path.name}: {exc}") from exc
    if out.returncode != 0 or len(out.stdout) != expected:
        detail = out.stderr.decode(errors="replace").strip().splitlines()[-1:]
        raise HdrCodecError(
            f"{path.name} could not be decoded" + (f": {detail[0][:200]}" if detail else "."),
        )
    return np.frombuffer(out.stdout, dtype=np.uint8)


def remove_quietly(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()
