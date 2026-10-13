"""HDR output pipeline: settings, SDR→HDR conversion, native HDR sources, encoding.

Data flow (see the README's "HDR output" section):

* **SDR source → HDR output**: the usual Upscale pipeline (lighting, look,
  AI model, HDR Enhancement, look finish) produces the 8-bit SDR result; it
  is then linearised with the source's ICC profile (sRGB without one),
  converted to BT.2020, scaled so SDR white sits at ``sdr_white`` cd/m²
  (203 by default, ITU-R BT.2408) and, only if asked, its brightest tones
  are expanded ("Expand highlights"). This is a *conversion*: no highlight or
  shadow detail is recovered.
* **HDR source → HDR output**: the HDR values are decoded to linear BT.2020.
  The AI model only accepts 0..1 values, so it receives an invertible proxy
  (``colour.ProxyCurve``: highlights compressed, 16 bits per channel); its
  output is mapped back through the exact inverse. The SDR-only tools
  (lighting, camera look, HDR Enhancement) are skipped for these sources.
* Both: highlights above the chosen peak are rolled off (BT.2390), the
  primaries converted to the output's (BT.2020 or Display P3) with out-of-
  gamut colours desaturated, then PQ-encoded and written band by band, and
  the written file is inspected before it counts as a success.
"""

from __future__ import annotations

import contextlib
import functools
import io
import logging
import os
import struct
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from pixelift.core.hdr_output import codecs
from pixelift.core.hdr_output import colour as cs

log = logging.getLogger(__name__)

MODES = ("sdr", "hdr")
TRANSFERS = {"pq": "PQ", "hlg": "HLG"}
FORMATS = {
    "png": ("16-bit PNG", ".png"),
    "avif": ("10-bit AVIF", ".avif"),
    "jpeg": ("Ultra HDR JPEG", ".jpg"),
}
OUTPUT_PRIMARIES = {"bt2020": "Rec. 2020", "p3": "Display P3"}
PEAK_RANGE = (400, 10000)
SDR_WHITE_RANGE = (100, 400)
EXPAND_KNEE = 0.6  # SDR linear; tones above it are stretched by "Expand highlights"
EXPAND_MAX = 4.0  # at most this many times the SDR white level
_BAND_PIXELS = 1 << 19  # pixels per processing band (~6 MB of float32 RGB)

Check = Callable[[], None]


@dataclass(frozen=True)
class HdrOutputSettings:
    """How results are written: SDR (as before) or HDR.

    HDR is PQ or HLG in a 16-bit PNG or 10-bit AVIF, or a gain-map JPEG
    (``format == "jpeg"``: an sRGB base plus a gain map; no transfer function).
    HLG is defined for Rec. 2020 and a 1000 cd/m² reference display
    (ITU-R BT.2100), so it always uses those (``effective_*``).
    """

    mode: str = "sdr"  # "sdr" | "hdr"
    format: str = "png"  # "png" | "avif" | "jpeg"
    primaries: str = "bt2020"  # "bt2020" | "p3"
    peak_nits: int = 1000  # brightest luminance written (cd/m²)
    sdr_white: int = int(cs.REFERENCE_WHITE)  # where SDR white / scene 1.0 lands (cd/m²)
    expand_highlights: bool = False  # SDR sources: stretch the brightest tones
    transfer: str = "pq"  # "pq" | "hlg" (PNG and AVIF)

    @property
    def enabled(self) -> bool:
        return self.mode == "hdr"

    @property
    def gain_map(self) -> bool:
        return self.format == "jpeg"

    @property
    def hlg(self) -> bool:
        return self.transfer == "hlg" and not self.gain_map

    @property
    def effective_peak(self) -> int:
        return int(cs.HLG_NOMINAL_PEAK) if self.hlg else self.peak_nits

    @property
    def effective_primaries(self) -> str:
        if self.gain_map:
            return "bt709"  # the sRGB base
        return "bt2020" if self.hlg else self.primaries

    @property
    def extension(self) -> str:
        return FORMATS[self.format][1]

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown output mode: {self.mode}")
        if self.format not in FORMATS:
            raise ValueError(f"unknown HDR format: {self.format}")
        if self.transfer not in TRANSFERS:
            raise ValueError(f"unknown HDR transfer function: {self.transfer}")
        if self.primaries not in OUTPUT_PRIMARIES:
            raise ValueError(f"unknown HDR colour space: {self.primaries}")
        if not PEAK_RANGE[0] <= self.peak_nits <= PEAK_RANGE[1]:
            raise ValueError(f"peak luminance must be {PEAK_RANGE[0]}–{PEAK_RANGE[1]} cd/m²")
        if not SDR_WHITE_RANGE[0] <= self.sdr_white <= SDR_WHITE_RANGE[1]:
            raise ValueError(f"SDR white must be {SDR_WHITE_RANGE[0]}–{SDR_WHITE_RANGE[1]} cd/m²")

    def tag(self) -> str:
        """Filename tag of HDR outputs ("" for SDR): ``pq1000``, ``hlg``, ``uhdr1000``…"""
        if not self.enabled:
            return ""
        if self.gain_map:
            tag = f"uhdr{self.peak_nits}"
        elif self.hlg:
            tag = "hlg"
        else:
            tag = f"pq{self.peak_nits}"
        if self.effective_primaries == "p3":
            tag += "-p3"
        if self.sdr_white != int(cs.REFERENCE_WHITE):
            tag += f"-w{self.sdr_white}"
        if self.expand_highlights:
            tag += "-x"
        return tag

    def describe(self) -> str:
        if not self.enabled:
            return "SDR"
        if self.gain_map:
            return f"HDR gain-map JPEG (Ultra HDR), sRGB base, peak {self.peak_nits} cd/m²"
        prim = OUTPUT_PRIMARIES[self.effective_primaries]
        peak = f"peak {self.effective_peak} cd/m²" + (" (HLG reference)" if self.hlg else "")
        return f"HDR {TRANSFERS[self.transfer]}, {prim}, {FORMATS[self.format][0]}, {peak}"


def conversion_status(source_hdr: bool, settings: HdrOutputSettings, gain_map: bool = False) -> str:
    """One-line description of what happens to the dynamic range.

    ``gain_map``: the source is a gain-map JPEG, whose SDR picture is used as
    is for SDR output.
    """
    if gain_map and not settings.enabled:
        return "Ultra HDR source → SDR (its own SDR picture)"
    if source_hdr:
        return "HDR source → HDR output" if settings.enabled else "HDR source → SDR (tone-mapped)"
    return "SDR source → HDR conversion" if settings.enabled else "SDR source → SDR output"


# --- source colour spaces ---------------------------------------------------------------
SRGB = cs.IccColourSpace(cs.rgb_to_xyz("bt709"), (("srgb",),) * 3)  # type: ignore[arg-type]


@functools.lru_cache(maxsize=16)
def _parse_icc_cached(icc: bytes) -> cs.IccColourSpace:
    return cs.parse_icc(icc)


def sdr_colour_space(
    rgb: np.ndarray, icc: bytes | None
) -> tuple[np.ndarray, cs.IccColourSpace, str]:
    """How to linearise ``rgb`` (uint8): (pixels, colour space, note).

    Matrix/TRC profiles (sRGB, Display P3, Adobe RGB, ProPhoto…) are read
    directly, so wide-gamut colours survive into BT.2020. Other profiles are
    converted to sRGB with LittleCMS first; no profile means sRGB.
    """
    if not icc:
        return rgb, SRGB, ""
    try:
        return rgb, _parse_icc_cached(icc), ""
    except (cs.IccUnsupportedError, struct.error, ValueError, IndexError):
        pass
    try:
        from PIL import ImageCms

        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        dst = ImageCms.createProfile("sRGB")
        img = ImageCms.profileToProfile(Image.fromarray(rgb, "RGB"), src, dst, outputMode="RGB")
        assert img is not None
        return np.asarray(img), SRGB, "ICC profile converted to sRGB first"
    except Exception:
        log.warning("Unusable ICC profile; assuming sRGB", exc_info=True)
        return rgb, SRGB, "Unreadable ICC profile; assumed sRGB"


def _linearize(space: cs.IccColourSpace, encoded: np.ndarray) -> np.ndarray:
    if space.curves[0] == ("srgb",):
        return cs.srgb_eotf(encoded)
    return space.linearize(encoded)


def expand_highlights(rgb: np.ndarray, headroom: float) -> np.ndarray:
    """Stretch SDR-linear tones above ``EXPAND_KNEE`` so 1.0 reaches ``headroom``, in place.

    ``m' = m + (headroom - 1)·t³`` with ``t = (m - knee) / (1 - knee)`` on the
    largest component, applied as a ratio (hue kept). Continuous slope at the
    knee, monotonic, nothing below the knee changes. An inverse tone-mapping
    choice, not a recovery: clipped areas become bright, flat areas.
    """
    if headroom <= 1:
        return rgb
    m = rgb.max(axis=-1)
    t = np.clip((m - EXPAND_KNEE) / (1 - EXPAND_KNEE), 0, 1)
    new = m + np.float32(headroom - 1) * t * t * t
    ratio = np.where(m > 0, new / np.maximum(m, np.float32(1e-12)), 1).astype(np.float32)
    rgb *= ratio[..., None]
    return rgb


# --- linear-light sources ---------------------------------------------------------------
class LinearImage:
    """A full-size picture produced band by band as linear BT.2020 (1.0 = 203 cd/m²)."""

    width: int
    height: int
    peak: float  # brightest value bands can hold (relative)

    def band(self, y0: int, y1: int) -> np.ndarray:
        raise NotImplementedError


class SdrConverted(LinearImage):
    """SDR ``uint8`` pixels placed in HDR (``HdrOutputSettings`` decides how)."""

    def __init__(
        self, rgb: np.ndarray, space: cs.IccColourSpace, settings: HdrOutputSettings
    ) -> None:
        self.rgb = rgb
        self.height, self.width = rgb.shape[:2]
        self.matrix = space.to_bt2020()
        self.space = space
        self.white = settings.sdr_white / cs.REFERENCE_WHITE
        self.headroom = (
            min(EXPAND_MAX, settings.peak_nits / settings.sdr_white)
            if settings.expand_highlights
            else 1.0
        )
        self.peak = self.white * self.headroom

    def band(self, y0: int, y1: int) -> np.ndarray:
        encoded = self.rgb[y0:y1].astype(np.float32) * np.float32(1 / 255)
        lin = cs.apply_matrix(_linearize(self.space, encoded), self.matrix)
        cs.fit_gamut(lin)  # profiles wider than BT.2020 (ProPhoto)
        expand_highlights(lin, self.headroom)
        lin *= np.float32(self.white)
        return lin


class ProxyReconstructed(LinearImage):
    """The AI model's 16-bit proxy output mapped back to HDR (``ProxyCurve.inverse``)."""

    def __init__(self, proxy: np.ndarray, curve: cs.ProxyCurve) -> None:
        self.proxy = proxy
        self.height, self.width = proxy.shape[:2]
        self.curve = curve
        self.peak = max(curve.peak, 1.0)

    def band(self, y0: int, y1: int) -> np.ndarray:
        encoded = self.proxy[y0:y1].astype(np.float32) * np.float32(1 / 65535)
        return self.curve.inverse(cs.srgb_eotf(encoded))


def make_proxy(linear: np.ndarray, curve: cs.ProxyCurve) -> np.ndarray:
    """Linear HDR (H, W, 3) -> the 16-bit sRGB-encoded proxy the AI model upscales."""
    out = np.empty(linear.shape, dtype=np.uint16)
    rows = max(1, _BAND_PIXELS // max(linear.shape[1], 1))
    for y0 in range(0, linear.shape[0], rows):
        enc = cs.srgb_oetf(curve.forward(linear[y0 : y0 + rows]))
        out[y0 : y0 + rows] = np.rint(enc * 65535)
    return out


# --- native HDR sources ---------------------------------------------------------------------
@dataclass
class HdrImage:
    """A decoded HDR source: linear BT.2020, 1.0 = 203 cd/m², never clipped."""

    linear: np.ndarray  # (H, W, 3) float32
    alpha: np.ndarray | None  # (H, W) float32 0..1
    source: codecs.HdrSource

    @property
    def peak(self) -> float:
        """The brightest component in the picture (relative; 1.0 = 203 cd/m²)."""
        return float(self.linear.max(initial=0.0))

    @property
    def peak_nits(self) -> float:
        return self.peak * cs.REFERENCE_WHITE

    def sdr_view(self, scale: float = 1.0) -> np.ndarray:
        """Tone-mapped SDR rendition, (H, W, 3) uint8 sRGB (for previews and SDR output).

        ``scale`` multiplies the linear values first (to undo a scene-white
        setting, so the SDR view does not depend on it).
        """
        out = np.empty(self.linear.shape, dtype=np.uint8)
        rows = max(1, _BAND_PIXELS // max(self.linear.shape[1], 1))
        peak = self.peak_nits * scale
        for y0 in range(0, self.linear.shape[0], rows):
            band = self.linear[y0 : y0 + rows]
            if scale != 1.0:
                band = band * np.float32(scale)
            enc = cs.to_sdr_preview(band, peak)
            out[y0 : y0 + rows] = np.rint(enc * 255)
        return out


def load_hdr(path: Path, source: codecs.HdrSource, scene_white: float = 203.0) -> HdrImage:
    """Decode an HDR source to linear BT.2020.

    ``scene_white``: cd/m² of 1.0 in scene-linear EXR / RGBE files and of SDR
    white in gain-map JPEGs (whose HDR rendition is relative to SDR white).
    """
    alpha = None
    if source.encoding == "gainmap":
        from pixelift.core.hdr_output import gainmap

        linear = gainmap.decode(path).linear
        linear *= np.float32(scene_white / cs.REFERENCE_WHITE)
    else:
        signal, alpha = codecs.decode(path, source)
        if source.encoding == "pq":
            nits = cs.pq_decode32(signal)
        elif source.encoding == "hlg":
            nits = cs.hlg_decode(signal)
        else:
            nits = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0) * np.float32(scene_white)
        linear = nits * np.float32(1 / cs.REFERENCE_WHITE)
        del nits, signal
        if source.chromaticities is not None:
            linear = cs.apply_matrix(linear, cs.chromaticities_to_bt2020(source.chromaticities))
        elif source.primaries != "bt2020":
            linear = cs.apply_matrix(linear, cs.conversion_matrix(source.primaries, "bt2020"))
    cs.fit_gamut(linear)
    np.minimum(linear, cs.PQ_MAX_NITS / cs.REFERENCE_WHITE, out=linear)
    if alpha is not None and float(alpha.min(initial=1.0)) >= 1.0:
        alpha = None  # fully opaque: no transparency to keep
    orientation = exif_orientation(path)
    linear = _orient(linear, orientation)
    if alpha is not None:
        alpha = _orient(alpha, orientation)
    return HdrImage(np.ascontiguousarray(linear), alpha, source)


def exif_orientation(path: Path) -> int:
    with contextlib.suppress(Exception), Image.open(path) as img:
        return int(img.getexif().get(0x0112, 1))
    return 1


def _orient(a: np.ndarray, orientation: int) -> np.ndarray:
    """Apply an EXIF orientation as ``ImageOps.exif_transpose`` does."""
    ops: dict[int, Callable[[np.ndarray], np.ndarray]] = {
        2: lambda x: x[:, ::-1],
        3: lambda x: x[::-1, ::-1],
        4: lambda x: x[::-1],
        5: lambda x: x.swapaxes(0, 1),
        6: lambda x: np.rot90(x, -1),
        7: lambda x: np.rot90(x, 2).swapaxes(0, 1),
        8: lambda x: np.rot90(x, 1),
    }
    op = ops.get(orientation)
    return np.ascontiguousarray(op(a)) if op else a


# --- encoding ----------------------------------------------------------------------------
@dataclass
class _LightMeter:
    max_cll: float = 0.0
    total: float = 0.0
    pixels: int = 0

    def add(self, nits_rgb: np.ndarray) -> None:
        m = nits_rgb.max(axis=-1)
        self.max_cll = max(self.max_cll, float(m.max(initial=0.0)))
        self.total += float(m.sum(dtype=np.float64))
        self.pixels += m.size

    def result(self) -> codecs.LightLevel:
        return codecs.LightLevel(self.max_cll, self.total / max(self.pixels, 1))


def _linear_bands(
    image: LinearImage, peak: float, rows: int, check: Check | None
) -> Iterator[tuple[int, np.ndarray]]:
    """Linear BT.2020 bands (1.0 = 203 cd/m²) with highlights rolled off to ``peak`` cd/m²."""
    eetf = cs.Eetf(image.peak * cs.REFERENCE_WHITE, float(peak))
    for y0 in range(0, image.height, rows):
        lin = image.band(y0, min(image.height, y0 + rows))
        eetf(lin)
        yield y0, lin
        if check:
            check()


def _band_rows(width: int, multiple: int = 1) -> int:
    return max(multiple, _BAND_PIXELS // max(width, 1) // multiple * multiple)


def _signal_bands(
    image: LinearImage, settings: HdrOutputSettings, meter: _LightMeter, check: Check | None
) -> Iterator[tuple[int, np.ndarray]]:
    """PQ or HLG R'G'B' bands (float 0..1) in the output primaries."""
    primaries = settings.effective_primaries
    peak = settings.effective_peak
    to_out = cs.conversion_matrix("bt2020", primaries) if primaries != "bt2020" else None
    luma = tuple(cs.rgb_to_xyz(primaries)[1])
    for y0, lin in _linear_bands(image, peak, _band_rows(image.width), check):
        if to_out is not None:
            lin = cs.apply_matrix(lin, to_out)
            cs.fit_gamut(lin, luma)  # type: ignore[arg-type]
        nits = lin * np.float32(cs.REFERENCE_WHITE)
        np.minimum(nits, np.float32(peak), out=nits)
        meter.add(nits)
        yield y0, (cs.hlg_encode(nits, peak) if settings.hlg else cs.pq_encode32(nits))


def _alpha_u16(alpha: np.ndarray | None, size: tuple[int, int]) -> np.ndarray | None:
    """Alpha resized to ``size`` (w, h) as uint16 (8-bit or float sources)."""
    if alpha is None:
        return None
    if alpha.dtype == np.uint8:
        img = Image.fromarray(alpha, "L").resize(size, Image.Resampling.LANCZOS)
        return np.asarray(img).astype(np.uint16) * 257
    img = Image.fromarray(alpha.astype(np.float32), "F").resize(size, Image.Resampling.LANCZOS)
    return np.rint(np.clip(np.asarray(img), 0, 1) * 65535).astype(np.uint16)


@dataclass(frozen=True)
class HdrWriteResult:
    report: codecs.FileReport
    light: codecs.LightLevel | None  # PQ only
    alpha_dropped: bool
    independent_check: bool  # ffprobe / Pillow confirmed the file

    def summary(self) -> str:
        parts = [self.report.describe()]
        if self.light is not None:
            parts.append(f"MaxCLL {self.light.max_cll:.0f} cd/m²")
        if self.alpha_dropped:
            parts.append("Transparency dropped (JPEG has no alpha channel)")
        return " · ".join(parts)


def write_hdr(
    target: Path,
    image: LinearImage,
    settings: HdrOutputSettings,
    alpha: np.ndarray | None = None,
    exif: bytes | None = None,
    dpi: tuple[float, float] | None = None,
    quality: int = 92,
    check: Check | None = None,
) -> HdrWriteResult:
    """Encode ``image`` as an HDR file at ``target`` and verify it; raises ``HdrCodecError``.

    ``target`` should be a temporary path; on failure it is removed.
    """
    settings.validate()
    if not settings.enabled:
        raise ValueError("write_hdr needs an HDR output mode")
    meter = _LightMeter()
    prim_code = codecs.PRIMARIES_CODES[settings.effective_primaries]
    transfer = codecs.TC_HLG if settings.hlg else codecs.TC_PQ
    alpha_dropped = False
    light: codecs.LightLevel | None = None
    try:
        if settings.gain_map:
            alpha_dropped = alpha is not None
            report = _write_gain_map(target, image, settings, exif, quality, check)
            if (report.width, report.height) != (image.width, image.height):
                raise codecs.HdrCodecError("The written gain-map JPEG has the wrong size.")
            independent = True  # Pillow parsed the MPF container and decoded both images
        else:
            a16 = _alpha_u16(alpha, (image.width, image.height))
            if settings.format == "png":
                channels = 4 if a16 is not None else 3

                def bands() -> Iterator[np.ndarray]:
                    for y0, sig in _signal_bands(image, settings, meter, check):
                        u16 = np.rint(sig * 65535).astype(np.uint16)
                        if a16 is not None:
                            u16 = np.concatenate([u16, a16[y0 : y0 + len(u16), :, None]], axis=-1)
                        yield u16

                codecs.write_png16(
                    target,
                    image.width,
                    image.height,
                    channels,
                    bands(),
                    (prim_code, transfer, codecs.MC_RGB, 1),
                    None if settings.hlg else meter.result,  # CTA-861.3 levels are for PQ
                    exif=exif,
                    dpi=dpi,
                    check=check,
                )
                report = codecs.inspect_png(target)
                expected = (16, channels, False)
            else:
                matrix = (
                    codecs.MC_BT2020_NCL
                    if settings.effective_primaries == "bt2020"
                    else codecs.MC_BT709
                )
                count = 4 if a16 is not None else 3
                try:
                    planes = np.empty((count, image.height, image.width), dtype=np.uint16)
                except MemoryError as exc:
                    raise codecs.HdrCodecError(
                        "There is not enough memory to encode the AVIF.", ["Choose 16-bit PNG"]
                    ) from exc
                for y0, sig in _signal_bands(image, settings, meter, check):
                    planes[:3, y0 : y0 + len(sig)] = codecs.rgb_to_ycbcr10(sig, matrix)
                if a16 is not None:
                    planes[3] = (a16.astype(np.uint32) * 1023 + 32767) // 65535
                codecs.write_avif10(target, planes, prim_code, matrix, quality, check, transfer)
                report = codecs.inspect_avif(target)
                expected = (10, 3, a16 is not None)
                _verify_avif_pixels(target, planes[:3], quality)
                del planes
            _verify(target, report, settings, image, expected)
            independent = _verify_ffprobe(target, settings)
            if not settings.hlg:
                light = meter.result()
    except BaseException:
        codecs.remove_quietly(target)
        raise
    return HdrWriteResult(report, light, alpha_dropped, independent)


def _write_gain_map(
    target: Path,
    image: LinearImage,
    settings: HdrOutputSettings,
    exif: bytes | None,
    quality: int,
    check: Check | None,
) -> codecs.FileReport:
    """Gain-map JPEG: sRGB base (SDR rendition) + quarter-size luminance gain map."""
    from pixelift.core.hdr_output import gainmap

    white = settings.sdr_white / cs.REFERENCE_WHITE  # SDR white in linear units
    to709 = cs.conversion_matrix("bt2020", "bt709")
    base = np.empty((image.height, image.width, 3), dtype=np.uint8)
    # The picture's brightest value after the roll-off, relative to SDR white: an
    # SDR picture (peak 1) gets an untouched base.
    peak_rel = min(image.peak * cs.REFERENCE_WHITE, settings.peak_nits) / settings.sdr_white

    def bands() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        rows = _band_rows(image.width, gainmap.SCALE)
        for y0, lin in _linear_bands(image, settings.peak_nits, rows, check):
            rel = lin * np.float32(1 / white)  # 1.0 = SDR white
            enc = cs.to_sdr_preview(rel, peak_rel * cs.REFERENCE_WHITE)
            rows_u8 = np.rint(enc * 255).astype(np.uint8)
            base[y0 : y0 + len(rows_u8)] = rows_u8
            sdr = cs.srgb_eotf(rows_u8.astype(np.float32) * np.float32(1 / 255))
            hdr = cs.apply_matrix(rel, to709)
            cs.fit_gamut(hdr, cs.BT709_LUMA)
            yield hdr, sdr

    ratios, low, high = gainmap.compute_gain_map(bands(), image.width, image.height)
    meta = gainmap.metadata_for(low, high)
    gain_jpeg = gainmap.encode_gain_map(ratios, low, high, quality)
    base_jpeg = gainmap.encode_base(base, quality, exif)
    if check:
        check()
    target.write_bytes(gainmap.assemble(base_jpeg, gain_jpeg, meta))
    report = gainmap.inspect(target)
    log.info("Verified %s: %s", target.name, report.describe())
    return report


def _verify(
    target: Path,
    report: codecs.FileReport,
    settings: HdrOutputSettings,
    image: LinearImage,
    expected: tuple[int, int, bool],
) -> None:
    problems = []
    if (report.width, report.height) != (image.width, image.height):
        problems.append("size")
    if report.bit_depth != expected[0]:
        problems.append(f"bit depth {report.bit_depth}")
    if report.channels and report.channels != expected[1]:
        problems.append(f"{report.channels} channels")
    if report.has_alpha != expected[2]:
        problems.append("alpha")
    want = codecs.TC_HLG if settings.hlg else codecs.TC_PQ
    if report.transfer != want:
        problems.append(f"transfer function is not {TRANSFERS[settings.transfer]}")
    if report.primaries != codecs.PRIMARIES_CODES[settings.effective_primaries]:
        problems.append("colour primaries")
    if not report.full_range:
        problems.append("range")
    if report.has_icc:
        problems.append("contradictory ICC profile")
    if problems:
        raise codecs.HdrCodecError(
            f"The written HDR file failed validation ({', '.join(problems)}).",
            ["Report this as a bug", "Choose the other HDR format"],
        )
    log.info("Verified %s: %s", target.name, report.describe())


def _verify_avif_pixels(target: Path, planes: np.ndarray, quality: int) -> None:
    """Decode the written AVIF and compare its planes with the ones encoded.

    The codec is lossy, so small differences are expected; a range or matrix
    mix-up shows as a large average error or a shifted mean.
    """
    decoded, depth = codecs.read_avif_planes(target, planes.shape[2], planes.shape[1])
    if depth != 10:
        raise codecs.HdrCodecError(f"The written AVIF decodes as {depth}-bit, not 10-bit.")
    step = max(1, int(np.sqrt(planes[0].size / 1_000_000)))  # a ~1 MP sample is enough
    a = planes[:, ::step, ::step].astype(np.float32)
    b = decoded[:, ::step, ::step].astype(np.float32)
    error = float(np.abs(a - b).mean())
    bias = float(np.abs(a.mean(axis=(1, 2)) - b.mean(axis=(1, 2))).max())
    limit = 4 + (100 - quality) * 0.4  # 10-bit codes
    if error > limit or bias > limit / 2:
        raise codecs.HdrCodecError(
            f"The written AVIF does not decode to the encoded pixels (mean error {error:.1f} "
            f"codes, bias {bias:.1f}).",
            ["Report this as a bug", "Choose 16-bit PNG as the HDR format"],
        )


_FF_PRIMARIES = {"bt2020": "bt2020", "p3": "smpte432"}
_FF_TRANSFERS = {"pq": "smpte2084", "hlg": "arib-std-b67"}


def _verify_ffprobe(target: Path, settings: HdrOutputSettings) -> bool:
    """Cross-check with ffprobe (an independent decoder) when it is installed."""
    if os.environ.get("PIXELIFT_SKIP_FFPROBE"):
        return False
    stream = codecs.ffprobe_stream(target)
    if stream is None:
        return False
    if "error" in stream:
        raise codecs.HdrCodecError(f"ffprobe cannot read the written file: {stream['error']}")
    transfer, primaries = stream.get("color_transfer"), stream.get("color_primaries")
    want = _FF_TRANSFERS[settings.transfer if settings.hlg else "pq"]
    if transfer != want or primaries != _FF_PRIMARIES[settings.effective_primaries]:
        raise codecs.HdrCodecError(
            f"ffprobe reads the written file as {transfer}/{primaries}, not "
            f"{TRANSFERS[settings.transfer]}.",
            ["Report this as a bug"],
        )
    return True
