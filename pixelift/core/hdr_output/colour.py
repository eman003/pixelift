"""Colour science for HDR output: transfer functions, primaries, gamut and tone mapping.

Conventions (used by every function here and documented in the README):

* **Linear light** values are relative to the HDR reference white: ``1.0`` is
  ``REFERENCE_WHITE`` = 203 cd/m² (ITU-R BT.2408, "HDR Reference White",
  where SDR diffuse white is placed in an HDR signal). ``nits = value * 203``.
  Linear values are never clipped to 1: an HDR highlight of 1000 cd/m² is
  ``4.93``.
* **Encoded** values (sRGB gamma, PQ, HLG signals) are kept apart from linear
  values; only ``*_eotf`` functions turn encoded values into linear light and
  only ``*_oetf`` / ``pq_encode`` turn linear light into encoded values.
* The working colour space is linear-light **ITU-R BT.2020** RGB with a D65
  white point. All maths is float32 (float64 for the reference constants).

Standards implemented:

* SMPTE ST 2084 (PQ), ITU-R BT.2100 table 4: ``pq_encode`` / ``pq_decode``.
* ITU-R BT.2100 HLG inverse OETF and OOTF (decoding HLG sources only;
  Pixelift does not write HLG).
* IEC 61966-2-1 sRGB transfer function.
* RGB↔XYZ matrices derived from the primaries (SMPTE RP 177), Bradford
  chromatic adaptation for ICC (D50) profiles.
* ITU-R BT.2390 EETF (hermite-spline roll-off in the PQ domain) for
  compressing highlights above a target peak, and for SDR previews.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np

REFERENCE_WHITE = 203.0  # cd/m², ITU-R BT.2408 HDR reference white (= linear 1.0)
PQ_MAX_NITS = 10000.0

# --- SMPTE ST 2084 (PQ) --------------------------------------------------------------
PQ_M1 = 2610 / 16384
PQ_M2 = 2523 / 4096 * 128
PQ_C1 = 3424 / 4096
PQ_C2 = 2413 / 4096 * 32
PQ_C3 = 2392 / 4096 * 32


def pq_encode(nits: np.ndarray | float) -> np.ndarray:
    """PQ inverse EOTF: absolute luminance in cd/m² (0..10000) -> signal 0..1.

    Values outside 0..10000 cd/m² are clamped (PQ cannot represent them).
    """
    y = np.clip(np.asarray(nits, dtype=np.float64) / PQ_MAX_NITS, 0.0, 1.0)
    p = np.power(y, PQ_M1)
    return np.power((PQ_C1 + PQ_C2 * p) / (1 + PQ_C3 * p), PQ_M2)


def pq_decode(signal: np.ndarray | float) -> np.ndarray:
    """PQ EOTF: signal 0..1 -> absolute luminance in cd/m² (0..10000)."""
    e = np.power(np.clip(np.asarray(signal, dtype=np.float64), 0.0, 1.0), 1 / PQ_M2)
    return PQ_MAX_NITS * np.power(np.maximum(e - PQ_C1, 0) / (PQ_C2 - PQ_C3 * e), 1 / PQ_M1)


def pq_encode32(nits: np.ndarray) -> np.ndarray:
    """``pq_encode`` returning float32 (computed in float64: the m2 = 78.8
    exponent would amplify float32 rounding to about one 16-bit code)."""
    return pq_encode(nits).astype(np.float32)


def pq_decode32(signal: np.ndarray) -> np.ndarray:
    """``pq_decode`` returning float32 (computed in float64)."""
    return pq_decode(signal).astype(np.float32)


# --- ITU-R BT.2100 HLG (decoding only) -----------------------------------------------
HLG_A = 0.17883277
HLG_B = 1 - 4 * HLG_A
HLG_C = 0.5 - HLG_A * math.log(4 * HLG_A)
HLG_NOMINAL_PEAK = 1000.0  # cd/m², the BT.2100 reference display for the OOTF


def hlg_decode(signal: np.ndarray, peak: float = HLG_NOMINAL_PEAK) -> np.ndarray:
    """HLG (BT.2020 RGB) signal -> display light in cd/m² on a ``peak`` display.

    Inverse OETF, then the BT.2100 OOTF with gamma 1.2 at 1000 cd/m² (adjusted
    for other peaks as BT.2100 note 5f), black level 0.
    """
    e = np.clip(signal.astype(np.float32, copy=False), 0, 1)
    low = e * e / 3
    high = (np.exp((e - np.float32(HLG_C)) / np.float32(HLG_A)) + np.float32(HLG_B)) / 12
    scene = np.where(e <= 0.5, low, high).astype(np.float32)
    gamma = 1.2 + 0.42 * math.log10(peak / 1000)
    ys = scene @ np.asarray(BT2020_LUMA, dtype=np.float32)
    gain = np.power(np.maximum(ys, 1e-12), np.float32(gamma - 1))
    return scene * (gain * np.float32(peak))[..., None]


def hlg_oetf(scene: np.ndarray) -> np.ndarray:
    """BT.2100 HLG OETF: scene light 0..1 -> signal 0..1."""
    e = np.clip(scene.astype(np.float64, copy=False), 0, 1)
    low = np.sqrt(3 * e)
    high = HLG_A * np.log(np.maximum(12 * e - HLG_B, 1e-12)) + HLG_C
    return np.where(e <= 1 / 12, low, high).astype(np.float32)


def hlg_encode(nits: np.ndarray, peak: float = HLG_NOMINAL_PEAK) -> np.ndarray:
    """Display light in cd/m² (BT.2020 RGB, at most ``peak``) -> HLG signal 0..1.

    The inverse of ``hlg_decode``: inverse BT.2100 OOTF for a ``peak``
    display (gamma 1.2 at 1000 cd/m²), then the OETF. This is the BT.2408
    display-light conversion, so 203 cd/m² grey becomes the 75 % HLG signal.
    """
    fd = np.clip(nits.astype(np.float64, copy=False), 0, peak)
    gamma = 1.2 + 0.42 * math.log10(peak / 1000)
    yd = fd @ np.asarray(BT2020_LUMA)
    ys = np.power(np.maximum(yd / peak, 1e-12), 1 / gamma)
    scene = np.where(yd[..., None] > 0, fd / (peak * np.power(ys, gamma - 1))[..., None], 0)
    # Bright saturated colours can exceed HLG's scene range (a component > 1):
    # desaturate them towards their luminance instead of clipping each
    # channel, so luminance and hue are kept.
    high = scene.max(axis=-1)
    over = high > 1
    if over.any():
        y = ys[over][:, None]
        t = (1 - y) / np.maximum(high[over][:, None] - y, 1e-12)
        scene[over] = y + t * (scene[over] - y)
    return hlg_oetf(scene)


# --- sRGB (IEC 61966-2-1) --------------------------------------------------------------
def srgb_eotf(encoded: np.ndarray) -> np.ndarray:
    """sRGB-encoded 0..1 -> linear 0..1 (SDR-relative: 1.0 is SDR white)."""
    v = encoded.astype(np.float32, copy=False)
    return np.where(
        v <= 0.04045, v / np.float32(12.92), np.power((v + 0.055) / np.float32(1.055), 2.4)
    ).astype(np.float32)


def srgb_oetf(linear: np.ndarray) -> np.ndarray:
    """Linear 0..1 -> sRGB-encoded 0..1 (values outside are clamped)."""
    v = np.clip(linear.astype(np.float32, copy=False), 0, 1)
    return np.where(
        v <= 0.0031308, v * np.float32(12.92), 1.055 * np.power(v, np.float32(1 / 2.4)) - 0.055
    ).astype(np.float32)


# --- primaries ---------------------------------------------------------------------------
D65 = (0.3127, 0.3290)
D50_XYZ = (0.9642, 1.0, 0.8249)  # ICC profile connection space white
PRIMARIES = {
    "bt709": ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)),  # also sRGB
    "bt2020": ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046)),
    "p3": ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)),  # Display P3 / P3-D65
}
BT2020_LUMA = (0.2627, 0.6780, 0.0593)  # Y of BT.2020 RGB (BT.2100 table 4)


def rgb_to_xyz(name: str) -> np.ndarray:
    """3x3 matrix from linear RGB with ``name`` primaries (D65) to CIE XYZ."""
    return _rgb_to_xyz(PRIMARIES[name], D65)


def _rgb_to_xyz(prim: tuple, white: tuple[float, float]) -> np.ndarray:
    def xyz(x: float, y: float) -> np.ndarray:
        return np.array([x / y, 1.0, (1 - x - y) / y])

    m = np.stack([xyz(*p) for p in prim], axis=1)
    s = np.linalg.solve(m, xyz(*white))
    return m * s


def chromaticities_to_bt2020(chroma: tuple[float, ...]) -> np.ndarray:
    """3x3 matrix: linear RGB with the given primaries and white (rx, ry, gx, gy,
    bx, by, wx, wy; e.g. an OpenEXR ``chromaticities`` attribute) -> BT.2020.

    A white point other than D65 is adapted to D65 with Bradford.
    """
    prim = ((chroma[0], chroma[1]), (chroma[2], chroma[3]), (chroma[4], chroma[5]))
    white = (chroma[6], chroma[7])
    to_xyz = _rgb_to_xyz(prim, white)
    adapt = _bradford(_xyz_from_xy(*white), _xyz_from_xy(*D65))
    return np.linalg.inv(rgb_to_xyz("bt2020")) @ adapt @ to_xyz


def conversion_matrix(src: str, dst: str) -> np.ndarray:
    """Linear RGB ``src`` primaries -> linear RGB ``dst`` primaries (both D65)."""
    return np.linalg.inv(rgb_to_xyz(dst)) @ rgb_to_xyz(src)


def apply_matrix(rgb: np.ndarray, matrix: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """``rgb`` (..., 3) float32 multiplied by a 3x3 matrix, per pixel."""
    return np.matmul(rgb, matrix.T.astype(np.float32), out=out)


def fit_gamut(rgb: np.ndarray, luma: tuple[float, float, float] = BT2020_LUMA) -> np.ndarray:
    """Bring negative components (colours outside the primaries) into gamut, in place.

    Each out-of-gamut colour is desaturated towards its own luminance just far
    enough that its smallest component is 0: luminance and hue angle are kept,
    only saturation is reduced. Colours inside the gamut are untouched.
    """
    low = rgb.min(axis=-1)
    bad = low < 0
    if not bad.any():
        return rgb
    sel = rgb[bad]
    y = np.maximum(sel @ np.asarray(luma, dtype=np.float32), 0)[:, None]
    t = y / np.maximum(y - low[bad][:, None], np.float32(1e-12))
    rgb[bad] = y + t * (sel - y)
    np.maximum(rgb, 0, out=rgb)  # rounding residue
    return rgb


def fit_unit_cube(rgb: np.ndarray, luma: tuple[float, float, float]) -> np.ndarray:
    """``fit_gamut``, then pull components above 1 back towards luminance (for SDR), in place."""
    fit_gamut(rgb, luma)
    high = rgb.max(axis=-1)
    over = high > 1
    if over.any():
        sel = rgb[over]
        y = np.clip(sel @ np.asarray(luma, dtype=np.float32), 0, 1)[:, None]
        t = (1 - y) / np.maximum(high[over][:, None] - y, np.float32(1e-12))
        rgb[over] = y + t * (sel - y)
    return np.clip(rgb, 0, 1, out=rgb)


# --- tone mapping (ITU-R BT.2390 EETF) ------------------------------------------------
@dataclass(frozen=True)
class Eetf:
    """BT.2390 roll-off of luminances from ``source_peak`` down to ``target_peak`` (cd/m²).

    Applied to the largest RGB component in the PQ domain, then as a ratio to
    all three components, so hue and saturation are kept. Below the knee
    (about half the target's PQ signal) nothing changes.
    """

    source_peak: float
    target_peak: float

    @property
    def is_identity(self) -> bool:
        return self.source_peak <= self.target_peak

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        """``rgb`` linear (1.0 = 203 cd/m²), scaled in place; returns it."""
        if self.is_identity:
            return rgb
        src_pq = float(pq_encode(self.source_peak))
        max_lum = float(pq_encode(self.target_peak)) / src_pq
        ks = 1.5 * max_lum - 0.5
        m = rgb.max(axis=-1)
        e1 = pq_encode32(m * np.float32(REFERENCE_WHITE)) / np.float32(src_pq)
        t = np.clip((e1 - ks) / (1 - ks), 0, 1)
        t2, t3 = t * t, t * t * t
        p = (2 * t3 - 3 * t2 + 1) * ks + (t3 - 2 * t2 + t) * (1 - ks) + (-2 * t3 + 3 * t2) * max_lum
        e2 = np.where(e1 < ks, e1, p) * np.float32(src_pq)
        new = pq_decode32(e2) / np.float32(REFERENCE_WHITE)
        ratio = np.where(m > 0, new / np.maximum(m, np.float32(1e-12)), 1).astype(np.float32)
        rgb *= ratio[..., None]
        return rgb


def to_sdr_preview(rgb2020: np.ndarray, source_peak: float) -> np.ndarray:
    """Linear BT.2020 HDR (1.0 = 203 cd/m²) -> sRGB-encoded float 0..1 for SDR screens.

    BT.2390 roll-off of everything above 203 cd/m² (the SDR white), BT.2020 ->
    BT.709 primaries, gamut fit, sRGB encoding. For previews and SDR exports
    of HDR sources only: it discards the HDR range.
    """
    rgb = np.array(rgb2020, dtype=np.float32)
    Eetf(max(source_peak, REFERENCE_WHITE), REFERENCE_WHITE)(rgb)
    rgb = apply_matrix(rgb, conversion_matrix("bt2020", "bt709"))
    fit_unit_cube(rgb, BT709_LUMA)
    return srgb_oetf(rgb)


BT709_LUMA = (0.2126, 0.7152, 0.0722)


# --- AI-model proxy curve ---------------------------------------------------------------
@dataclass(frozen=True)
class ProxyCurve:
    """Invertible mapping of HDR linear light into the 0..1 range an SDR model accepts.

    For the largest RGB component ``m`` (1.0 = 203 cd/m²) and the picture's
    ``peak``: ``y = a·m`` up to ``KNEE``, then ``y = a·KNEE·(1 + ln(m / KNEE))``,
    continuous with a continuous slope, and ``y(peak) = 1``. All three
    components are scaled by ``y / m``, so hue is kept. When ``peak <= 1``
    (nothing above SDR white) the curve is the identity. ``inverse`` undoes it
    exactly, so an upscaler that changed nothing returns the original HDR
    values (up to 16-bit quantisation).
    """

    peak: float

    KNEE = 0.5  # linear, ~100 cd/m²

    @property
    def is_identity(self) -> bool:
        return self.peak <= 1

    @property
    def gain(self) -> float:
        if self.is_identity:
            return 1.0
        return 1 / (self.KNEE * (1 + math.log(self.peak / self.KNEE)))

    def forward(self, rgb: np.ndarray) -> np.ndarray:
        """Linear HDR -> linear 0..1 proxy (new array)."""
        rgb = np.maximum(rgb.astype(np.float32, copy=False), 0)
        if self.is_identity:
            return np.minimum(rgb, 1)
        a, k = np.float32(self.gain), np.float32(self.KNEE)
        m = rgb.max(axis=-1)
        y = np.where(m <= k, a * m, a * k * (1 + np.log(np.maximum(m, k) / k)))
        y = np.minimum(y, 1)
        ratio = np.where(m > 0, y / np.maximum(m, np.float32(1e-12)), 0).astype(np.float32)
        return rgb * ratio[..., None]

    def inverse(self, proxy: np.ndarray) -> np.ndarray:
        """Linear 0..1 proxy -> linear HDR (new array)."""
        proxy = np.clip(proxy.astype(np.float32, copy=False), 0, 1)
        if self.is_identity:
            return proxy.copy()
        a, k = np.float32(self.gain), np.float32(self.KNEE)
        y = proxy.max(axis=-1)
        m = np.where(y <= a * k, y / a, k * np.exp(np.maximum(y, a * k) / (a * k) - 1))
        ratio = np.where(y > 0, m / np.maximum(y, np.float32(1e-12)), 0).astype(np.float32)
        return proxy * ratio[..., None]


# --- ICC profiles ---------------------------------------------------------------------------
_BRADFORD = np.array(
    [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]]
)


class IccUnsupportedError(ValueError):
    """The ICC profile is not a matrix/TRC (or grey TRC) profile this module reads."""


@dataclass(frozen=True)
class IccColourSpace:
    """What a matrix/TRC ICC profile says about RGB (or grey) pixel values."""

    to_xyz_d65: np.ndarray  # 3x3, linear RGB -> XYZ (D65 adapted); identity-ish for grey
    curves: tuple[tuple, tuple, tuple]  # per channel: ("gamma", g) | ("para", ...) | ("table", t)
    grey: bool = False

    def linearize(self, encoded: np.ndarray) -> np.ndarray:
        """Encoded 0..1 (H, W, 3) -> linear 0..1 per the profile's tone curves."""
        out = np.empty(encoded.shape, dtype=np.float32)
        for c in range(3):
            out[..., c] = _apply_curve(self.curves[c], encoded[..., c].astype(np.float32))
        return out

    def to_bt2020(self) -> np.ndarray:
        """3x3 matrix: linear profile RGB -> linear BT.2020 RGB."""
        return np.linalg.inv(rgb_to_xyz("bt2020")) @ self.to_xyz_d65


def parse_icc(icc: bytes) -> IccColourSpace:
    """Read a matrix/TRC RGB or grey-TRC ICC profile (v2 or v4).

    LUT-based profiles (A2B0 only), CMYK, Lab and others raise
    ``IccUnsupportedError``; the caller then falls back to a LittleCMS
    conversion to sRGB.
    """
    if len(icc) < 132:
        raise IccUnsupportedError("truncated ICC profile")
    space = icc[16:20]
    count = struct.unpack(">I", icc[128:132])[0]
    tags: dict[bytes, bytes] = {}
    for i in range(min(count, 200)):
        base = 132 + 12 * i
        if base + 12 > len(icc):
            break
        sig, off, size = struct.unpack(">4sII", icc[base : base + 12])
        if off + size <= len(icc):
            tags[sig] = icc[off : off + size]
    if space == b"GRAY":
        curve = _read_curve(tags.get(b"kTRC"))
        return IccColourSpace(rgb_to_xyz("bt709"), (curve, curve, curve), grey=True)
    if space != b"RGB ":
        raise IccUnsupportedError(f"ICC colour space {space!r} is not RGB")
    try:
        cols = [_read_xyz(tags[t]) for t in (b"rXYZ", b"gXYZ", b"bXYZ")]
        curves = tuple(_read_curve(tags.get(t)) for t in (b"rTRC", b"gTRC", b"bTRC"))
    except KeyError as exc:
        raise IccUnsupportedError("not a matrix/TRC profile") from exc
    to_d50 = np.stack(cols, axis=1)
    # Columns are D50-adapted (ICC PCS); adapt back to D65 with Bradford.
    d65 = _xyz_from_xy(*D65)
    adapt = _bradford(np.asarray(D50_XYZ), d65)
    return IccColourSpace(adapt @ to_d50, curves)  # type: ignore[arg-type]


def _xyz_from_xy(x: float, y: float) -> np.ndarray:
    return np.array([x / y, 1.0, (1 - x - y) / y])


def _bradford(src_white: np.ndarray, dst_white: np.ndarray) -> np.ndarray:
    s = _BRADFORD @ src_white
    d = _BRADFORD @ dst_white
    return np.linalg.inv(_BRADFORD) @ np.diag(d / s) @ _BRADFORD


def _s15(data: bytes, i: int) -> float:
    return struct.unpack(">i", data[i : i + 4])[0] / 65536


def _read_xyz(data: bytes) -> np.ndarray:
    if data[:4] != b"XYZ " or len(data) < 20:
        raise IccUnsupportedError("bad XYZ tag")
    return np.array([_s15(data, 8), _s15(data, 12), _s15(data, 16)])


_PARA_COUNTS = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}


def _read_curve(data: bytes | None) -> tuple:
    if data is None:
        raise IccUnsupportedError("missing tone curve")
    kind = data[:4]
    if kind == b"curv":
        n = struct.unpack(">I", data[8:12])[0]
        if n == 0:
            return ("gamma", 1.0)
        if n == 1:
            return ("gamma", struct.unpack(">H", data[12:14])[0] / 256)
        table = np.frombuffer(data[12 : 12 + 2 * n], dtype=">u2").astype(np.float32) / 65535
        if len(table) != n:
            raise IccUnsupportedError("truncated tone curve")
        return ("table", tuple(table.tolist()))
    if kind == b"para":
        func = struct.unpack(">H", data[8:10])[0]
        if func not in _PARA_COUNTS:
            raise IccUnsupportedError("unknown parametric curve")
        params = tuple(_s15(data, 12 + 4 * i) for i in range(_PARA_COUNTS[func]))
        return ("para", func, *params)
    raise IccUnsupportedError(f"unsupported tone curve {kind!r}")


def _apply_curve(curve: tuple, x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    if curve[0] == "gamma":
        return np.power(x, np.float32(curve[1]))
    if curve[0] == "table":
        table = np.asarray(curve[1], dtype=np.float32)
        return np.interp(x, np.linspace(0, 1, len(table), dtype=np.float32), table).astype(
            np.float32
        )
    func, *p = curve[1:]
    g = np.float32(p[0])
    if func == 0:
        return np.power(x, g)
    a, b = np.float32(p[1]), np.float32(p[2])
    if func == 1:
        return np.where(x >= -b / a, np.power(np.maximum(a * x + b, 0), g), 0).astype(np.float32)
    c = np.float32(p[3])
    if func == 2:
        return np.where(x >= -b / a, np.power(np.maximum(a * x + b, 0), g) + c, c).astype(
            np.float32
        )
    d = np.float32(p[4])
    if func == 3:
        return np.where(x >= d, np.power(np.maximum(a * x + b, 0), g), c * x).astype(np.float32)
    e, f = np.float32(p[5]), np.float32(p[6])
    return np.where(x >= d, np.power(np.maximum(a * x + b, 0), g) + e, c * x + f).astype(np.float32)
