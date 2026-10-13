"""HDR output: transfer functions, colour, SDR→HDR, native HDR, encoding and validation.

Standards-based values come from the documents themselves (ITU-R BT.2100,
BT.2408, BT.2087, IEC 61966-2-1) and from a high-precision ``decimal``
evaluation of the SMPTE ST 2084 formula. Written files are checked by
parsing their bytes, with Pillow (an independent PNG decoder) and, when
installed, with ffmpeg / ffprobe. Tests needing ffmpeg skip without it.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import struct
import tracemalloc
from decimal import Decimal, getcontext
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms

from pixelift import cli
from pixelift.core import upscale_cache
from pixelift.core.control import JobControl
from pixelift.core.errors import CancelledError, InvalidImageError, OutputError
from pixelift.core.hdr_output import codecs, gainmap
from pixelift.core.hdr_output import colour as cs
from pixelift.core.hdr_output import pipeline as hp
from pixelift.core.hdr_output.pipeline import HdrOutputSettings
from pixelift.core.image_processor import ProcessingOptions, keeps_hdr, process_image
from pixelift.core.restoration.settings import RestorationSettings
from pixelift.core.upscaler import Upscaler
from pixelift.storage.settings import Settings
from pixelift.utils import image_utils as iu

needs_ffmpeg = pytest.mark.skipif(
    codecs.ffmpeg_path() is None or codecs.ffprobe_path() is None, reason="needs ffmpeg/ffprobe"
)
needs_avif = pytest.mark.skipif(not codecs.avif_available(), reason="needs ffmpeg libaom-av1")


class NearestUpscaler(Upscaler):
    """Identity "model" (nearest-neighbour): any change in values comes from the pipeline."""

    supports_uint16 = True

    def __init__(self) -> None:
        self.calls = 0
        self.dtypes: list[np.dtype] = []

    def upscale(self, image, scale, model, *, progress=None, control=None):
        self.calls += 1
        self.dtypes.append(image.dtype)
        return np.repeat(np.repeat(image, scale, 0), scale, 1)

    def cache_identity(self, model, scale, size=None, wide=False):
        return ("nearest",)


def hdr_options(**kwargs) -> ProcessingOptions:
    out = kwargs.pop("hdr_output", HdrOutputSettings(mode="hdr"))
    return ProcessingOptions(scale=2, model="x", hdr_output=out, existing="overwrite", **kwargs)


def write_pq_png(path: Path, nits: np.ndarray, alpha: np.ndarray | None = None) -> Path:
    sig = np.rint(cs.pq_encode32(nits) * 65535).astype(np.uint16)
    if alpha is not None:
        sig = np.concatenate([sig, alpha[..., None]], axis=-1)
    h, w, c = sig.shape
    codecs.write_png16(path, w, h, c, [sig], (9, 16, 0, 1))
    return path


# --- SMPTE ST 2084 (PQ) ---------------------------------------------------------------
def _pq_decimal(nits: float) -> Decimal:
    """ST 2084 inverse EOTF evaluated with 50 significant digits."""
    getcontext().prec = 50
    m1 = Decimal(2610) / 16384
    m2 = Decimal(2523) / 4096 * 128
    c1 = Decimal(3424) / 4096
    c2 = Decimal(2413) / 4096 * 32
    c3 = Decimal(2392) / 4096 * 32
    y = Decimal(nits) / 10000
    p = (y.ln() * m1).exp() if y > 0 else Decimal(0)
    return (((c1 + c2 * p) / (1 + c3 * p)).ln() * m2).exp()


@pytest.mark.parametrize("nits", [0.001, 0.1, 1, 26, 100, 203, 400, 1000, 4000, 10000])
def test_pq_matches_high_precision_reference(nits):
    assert float(cs.pq_encode(nits)) == pytest.approx(float(_pq_decimal(nits)), abs=1e-12)
    assert float(cs.pq_decode(cs.pq_encode(nits))) == pytest.approx(nits, rel=1e-9)


def test_pq_published_values():
    # ITU-R BT.2408 table 1: reference white 203 cd/m² is 58 % PQ, 18 % grey
    # (26 cd/m²) is 38 %; 100 cd/m² is 10-bit full-range code 520, 1000 is 769.
    assert float(cs.pq_encode(203)) == pytest.approx(0.58, abs=0.005)
    assert float(cs.pq_encode(26)) == pytest.approx(0.38, abs=0.005)
    assert round(float(cs.pq_encode(100)) * 1023) == 520
    assert round(float(cs.pq_encode(1000)) * 1023) == 769
    assert float(cs.pq_encode(10000)) == pytest.approx(1.0)
    assert float(cs.pq_encode(0)) == pytest.approx(cs.PQ_C1**cs.PQ_M2)
    assert float(cs.pq_encode(20000)) == pytest.approx(1.0)  # clamped, never > 1


def test_pq_float32_matches_float64():
    nits = np.geomspace(0.001, 10000, 2000).astype(np.float32)
    assert np.abs(cs.pq_encode32(nits) - cs.pq_encode(nits)).max() < 1e-7
    sig = np.linspace(0, 1, 2000, dtype=np.float32)
    rel = np.abs(cs.pq_decode32(sig) - cs.pq_decode(sig)) / np.maximum(cs.pq_decode(sig), 1e-3)
    assert rel.max() < 1e-4
    assert np.all(np.diff(cs.pq_encode32(nits)) >= 0)  # monotonic


def test_hlg_reference_white():
    # BT.2408: 75 % HLG is reference white, 203 cd/m² on a 1000 cd/m² display.
    grey = np.full((1, 1, 3), 0.75, np.float32)
    assert float(cs.hlg_decode(grey)[0, 0, 0]) == pytest.approx(203, rel=0.02)
    assert float(cs.hlg_decode(np.ones((1, 1, 3), np.float32))[0, 0, 0]) == pytest.approx(1000)


def test_srgb_transfer_reference_values():
    v = np.array([0.0, 0.04045, 0.5, 1.0], np.float32)
    assert cs.srgb_eotf(v) == pytest.approx([0.0, 0.0031308, 0.2140411, 1.0], abs=1e-6)
    x = np.linspace(0, 1, 1001, dtype=np.float32)
    assert np.abs(cs.srgb_oetf(cs.srgb_eotf(x)) - x).max() < 1e-5


# --- colour spaces ---------------------------------------------------------------------
def test_bt709_to_bt2020_matches_bt2087():
    published = [[0.6274, 0.3293, 0.0433], [0.0691, 0.9195, 0.0114], [0.0164, 0.0880, 0.8956]]
    assert cs.conversion_matrix("bt709", "bt2020") == pytest.approx(np.array(published), abs=1e-4)


@pytest.mark.parametrize("src", ["bt709", "p3", "bt2020"])
def test_conversions_keep_white_and_round_trip(src):
    m = cs.conversion_matrix(src, "bt2020")
    assert m @ np.ones(3) == pytest.approx(np.ones(3), abs=1e-9)  # D65 white stays white
    back = cs.conversion_matrix("bt2020", src)
    assert back @ m == pytest.approx(np.eye(3), abs=1e-9)


def test_fit_gamut_desaturates_only_out_of_gamut_colours():
    rgb = np.array([[0.5, 0.2, 0.1], [1.2, -0.1, 0.05], [-0.05, -0.02, 0.3]], np.float32)
    luma = np.array(cs.BT2020_LUMA, np.float32)
    y_before = rgb @ luma
    out = cs.fit_gamut(rgb.copy())
    assert out[0] == pytest.approx(rgb[0])
    assert out.min() >= 0
    assert out[1] @ luma == pytest.approx(y_before[1], abs=1e-6)  # luminance kept
    assert out[1, 0] > out[1, 2] > out[1, 1]  # hue order kept
    assert y_before[2] < 0 and out[2] == pytest.approx([0, 0, 0])  # no light: black


def _matrix_trc_icc(primaries: str, gamma: float = 2.2) -> bytes:
    """A minimal v2 matrix/TRC ICC profile for ``primaries`` (D50-adapted columns)."""
    to_xyz = cs.rgb_to_xyz(primaries)
    adapt = cs._bradford(cs._xyz_from_xy(*cs.D65), np.asarray(cs.D50_XYZ))
    cols = adapt @ to_xyz

    def xyz_tag(v):
        return b"XYZ \0\0\0\0" + b"".join(struct.pack(">i", round(x * 65536)) for x in v)

    curve = b"curv\0\0\0\0" + struct.pack(">IH", 1, round(gamma * 256)) + b"\0\0"
    tags = [(b"rXYZ", xyz_tag(cols[:, 0])), (b"gXYZ", xyz_tag(cols[:, 1]))]
    tags += [(b"bXYZ", xyz_tag(cols[:, 2])), (b"rTRC", curve), (b"gTRC", curve)]
    tags += [(b"bTRC", curve)]
    offset = 128 + 4 + 12 * len(tags)
    table, data = b"", b""
    for sig, body in tags:
        table += struct.pack(">4sII", sig, offset + len(data), len(body))
        data += body
    header = bytearray(128)
    header[12:16], header[16:20], header[20:24] = b"mntr", b"RGB ", b"XYZ "
    body = bytes(header) + struct.pack(">I", len(tags)) + table + data
    return struct.pack(">I", len(body)) + body[4:]


def test_icc_srgb_profile_parsed():
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    space = cs.parse_icc(icc)
    assert space.to_xyz_d65 == pytest.approx(cs.rgb_to_xyz("bt709"), abs=2e-3)
    x = np.linspace(0, 1, 50, dtype=np.float32)
    lin = space.linearize(np.stack([x, x, x], -1)[None])[0, :, 0]
    assert np.abs(lin - cs.srgb_eotf(x)).max() < 1e-4


def test_wide_gamut_icc_colours_survive_into_bt2020():
    icc = _matrix_trc_icc("p3", 2.2)
    space = cs.parse_icc(icc)
    red = space.to_bt2020() @ np.array([1.0, 0, 0])
    assert red == pytest.approx(cs.conversion_matrix("p3", "bt2020") @ [1.0, 0, 0], abs=2e-3)
    # It is outside BT.709: converted to 709 it has a negative component.
    assert (cs.conversion_matrix("bt2020", "bt709") @ red).min() < -0.01
    _rgb, parsed, note = hp.sdr_colour_space(np.zeros((1, 1, 3), np.uint8), icc)
    assert note == "" and parsed.to_xyz_d65 == pytest.approx(space.to_xyz_d65)


def test_unparseable_icc_falls_back_to_srgb():
    _rgb, space, note = hp.sdr_colour_space(np.zeros((2, 2, 3), np.uint8), b"garbage" * 40)
    assert space is hp.SRGB and "sRGB" in note


# --- tone mapping and the AI proxy --------------------------------------------------
def test_eetf_rolls_off_to_target_peak_and_keeps_midtones():
    m = np.geomspace(0.01, 4000, 500, dtype=np.float32) / cs.REFERENCE_WHITE
    rgb = np.stack([m, m * 0.5, m * 0.25], -1)
    out = cs.Eetf(4000, 1000)(rgb.copy())
    nits = out.max(-1) * cs.REFERENCE_WHITE
    assert nits.max() == pytest.approx(1000, rel=1e-3)
    assert np.all(np.diff(nits) >= -1e-3)  # monotonic
    low = m * cs.REFERENCE_WHITE < 100
    assert out[low] == pytest.approx(rgb[low], rel=1e-4)  # below the knee: untouched
    assert out[:, 1] / out[:, 0] == pytest.approx(0.5, rel=1e-4)  # ratios (hue) kept
    assert cs.Eetf(800, 1000).is_identity


def test_proxy_curve_is_exactly_invertible():
    curve = cs.ProxyCurve(4000 / cs.REFERENCE_WHITE)
    rng = np.random.default_rng(1)
    rgb = (rng.random((20000, 3)) ** 3 * curve.peak).astype(np.float32)
    proxy = curve.forward(rgb)
    assert proxy.max() <= 1 and proxy.min() >= 0
    assert curve.inverse(proxy) == pytest.approx(rgb, rel=1e-4, abs=1e-6)
    # Through 16-bit sRGB encoding (what the model sees and returns).
    q = np.rint(cs.srgb_oetf(proxy) * 65535) / 65535
    back = curve.inverse(cs.srgb_eotf(q.astype(np.float32)))
    rel = np.abs(back - rgb) / np.maximum(rgb, 0.01)
    assert np.percentile(rel, 99.9) < 2e-3
    sdr = cs.ProxyCurve(0.9)
    assert sdr.is_identity and sdr.forward(rgb[:5].clip(0, 0.9)) == pytest.approx(
        rgb[:5].clip(0, 0.9)
    )


# --- SDR → HDR conversion -----------------------------------------------------------------
def _convert(rgb: np.ndarray, **kwargs) -> np.ndarray:
    settings = HdrOutputSettings(mode="hdr", **kwargs)
    return hp.SdrConverted(rgb, hp.SRGB, settings).band(0, rgb.shape[0]) * cs.REFERENCE_WHITE


def test_sdr_white_lands_on_reference_white():
    nits = _convert(np.full((2, 2, 3), 255, np.uint8))
    assert nits == pytest.approx(203, rel=1e-4)
    assert _convert(np.full((1, 1, 3), 255, np.uint8), sdr_white=100) == pytest.approx(
        100, rel=1e-4
    )
    grey = _convert(np.full((1, 1, 3), 128, np.uint8))
    assert grey[0, 0, 0] == pytest.approx(grey[0, 0, 1]) == pytest.approx(grey[0, 0, 2])
    assert float(grey[0, 0, 0]) == pytest.approx(203 * float(cs.srgb_eotf(np.float32(128 / 255))))


def test_sdr_conversion_never_exceeds_sdr_white_without_expansion():
    rng = np.random.default_rng(2)
    rgb = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    assert _convert(rgb).max() <= 203 * 1.0001


def test_expand_highlights_only_touches_the_brightest_tones():
    ramp = np.arange(256, dtype=np.uint8)[None, :, None].repeat(3, axis=2)
    plain = _convert(ramp)[0, :, 0]
    expanded = _convert(ramp, expand_highlights=True, peak_nits=1000)[0, :, 0]
    knee = plain / 203 <= hp.EXPAND_KNEE
    assert expanded[knee] == pytest.approx(plain[knee])
    assert expanded[-1] == pytest.approx(203 * hp.EXPAND_MAX, rel=1e-4)  # 4× SDR white
    assert np.all(np.diff(expanded) > 0)
    capped = _convert(ramp, expand_highlights=True, peak_nits=400)[0, -1, 0]
    assert capped == pytest.approx(400, rel=1e-4)  # never above the peak


# --- PNG (cICP) ---------------------------------------------------------------------------
def test_png16_is_valid_and_correctly_identified(tmp_path):
    rng = np.random.default_rng(3)
    data = rng.integers(0, 65536, (37, 53, 3), dtype=np.uint16)
    data[:5] = 65535  # rows the Paeth predictor sees as "up" neighbours
    path = tmp_path / "x.png"
    codecs.write_png16(
        path, 53, 37, 3, [data[:10], data[10:11], data[11:]], (9, 16, 0, 1),
        lambda: codecs.LightLevel(812.4, 120.5), exif=None, dpi=(300, 300),
    )  # fmt: skip
    report = codecs.inspect_png(path)
    assert (report.width, report.height, report.bit_depth, report.channels) == (53, 37, 16, 3)
    assert (report.primaries, report.transfer, report.matrix, report.full_range) == (
        9, 16, 0, True,
    )  # fmt: skip
    assert report.max_cll == pytest.approx(812.4) and report.max_fall == pytest.approx(120.5)
    assert report.chunks.index("cICP") < report.chunks.index("IDAT")
    assert report.chunks.index("cLLi") < report.chunks.index("IDAT")
    assert not {"iCCP", "gAMA", "sRGB"} & set(report.chunks)  # nothing contradicts cICP
    # Pillow (an independent decoder) reads the pixels back: its 8-bit view
    # is the high byte of each sample, which checks our filtering.
    with Image.open(path) as img:
        assert img.size == (53, 37)
        np.testing.assert_array_equal(np.asarray(img.convert("RGB")), (data >> 8).astype(np.uint8))


@needs_ffmpeg
def test_png16_read_by_ffmpeg_exactly(tmp_path):
    rng = np.random.default_rng(4)
    nits = rng.random((24, 32, 3)).astype(np.float32) * 3000
    path = write_pq_png(tmp_path / "x.png", nits)
    stream = codecs.ffprobe_stream(path)
    assert stream["color_transfer"] == "smpte2084"
    assert stream["color_primaries"] == "bt2020"
    assert stream["pix_fmt"] == "rgb48be"
    signal, _ = codecs.decode(path, codecs.sniff(path))
    expected = np.rint(cs.pq_encode32(nits) * 65535) / 65535
    assert np.abs(signal - expected).max() < 1e-6


def test_png_inspection_rejects_damage(tmp_path):
    path = write_pq_png(tmp_path / "x.png", np.full((8, 8, 3), 100, np.float32))
    data = bytearray(path.read_bytes())
    bad = tmp_path / "bad.png"
    data[40] ^= 0xFF  # inside a chunk: CRC mismatch
    bad.write_bytes(bytes(data))
    with pytest.raises(codecs.HdrCodecError, match="damaged"):
        codecs.inspect_png(bad)
    short = tmp_path / "short.png"
    short.write_bytes(path.read_bytes()[:-20])
    with pytest.raises(codecs.HdrCodecError):
        codecs.inspect_png(short)


def test_sniff_tells_hdr_from_sdr(tmp_path):
    sdr = tmp_path / "sdr.png"
    Image.new("RGB", (4, 4)).save(sdr)
    assert codecs.sniff(sdr) is None
    hdr = write_pq_png(tmp_path / "hdr.png", np.full((4, 4, 3), 100, np.float32))
    assert codecs.sniff(hdr) == codecs.HdrSource("PNG", "pq", "bt2020")
    srgb_cicp = tmp_path / "srgb.png"
    codecs.write_png16(srgb_cicp, 1, 1, 3, [np.zeros((1, 1, 3), np.uint16)], (1, 13, 0, 1))
    assert codecs.sniff(srgb_cicp) is None  # cICP, but an SDR transfer function
    (tmp_path / "x.exr").write_bytes(b"\x76\x2f\x31\x01" + b"\0" * 20)
    assert codecs.sniff(tmp_path / "x.exr").encoding == "linear"


# --- AVIF ---------------------------------------------------------------------------------
@needs_avif
def test_avif10_is_valid_and_round_trips(tmp_path):
    y, x = np.mgrid[0:48, 0:64]
    nits = np.stack([x * 15.0, y * 8.0, np.full_like(x, 40.0)], -1).astype(np.float32)
    sig = cs.pq_encode32(nits)
    path = tmp_path / "x.avif"
    codecs.write_avif10(path, codecs.rgb_to_ycbcr10(sig, codecs.MC_BT2020_NCL), 9, 9, 95)
    report = codecs.inspect_avif(path)
    assert (report.width, report.height, report.bit_depth) == (64, 48, 10)
    assert (report.primaries, report.transfer, report.matrix, report.full_range) == (
        9, 16, 9, True,
    )  # fmt: skip
    stream = codecs.ffprobe_stream(path)
    assert stream["pix_fmt"] == "yuv444p10le"
    assert stream["color_transfer"] == "smpte2084" and stream["color_space"] == "bt2020nc"
    decoded, _ = codecs.decode(path, codecs.sniff(path))
    assert np.abs(decoded - sig).mean() < 0.003  # lossy, but the same picture
    cut = tmp_path / "cut.avif"
    cut.write_bytes(path.read_bytes()[:-50])
    with pytest.raises(codecs.HdrCodecError, match="truncated"):
        codecs.inspect_avif(cut)


def test_ycbcr_conversion_round_trips():
    rng = np.random.default_rng(5)
    rgb = rng.random((16, 16, 3)).astype(np.float32)
    for matrix in (codecs.MC_BT2020_NCL, codecs.MC_BT709):
        planes = codecs.rgb_to_ycbcr10(rgb, matrix)
        back = codecs.ycbcr_to_rgb(planes, matrix, True, 10)
        assert np.abs(back - rgb).max() < 2.5 / 1023


# --- the whole pipeline ------------------------------------------------------------------
def _sdr_photo(path: Path, size=(40, 30)) -> Path:
    w, h = size
    y, x = np.mgrid[0:h, 0:w]
    rgb = np.dstack([x * 255 // (w - 1), y * 255 // (h - 1), np.full_like(x, 90)])
    rgb[:4, :4] = 255  # SDR white
    Image.fromarray(rgb.astype(np.uint8)).save(path)
    return path


def test_sdr_source_exported_as_verified_hdr_png(tmp_path):
    src = _sdr_photo(tmp_path / "photo.png")
    result = process_image(src, hdr_options(), NearestUpscaler())
    assert result.output.name == "photo_2x_pq1000.png"
    assert "Converted from SDR" in result.note and "PQ" in result.note
    report = codecs.inspect_png(result.output)
    assert (report.bit_depth, report.transfer, report.primaries) == (16, 16, 9)
    assert report.max_cll == pytest.approx(203, rel=1e-3)
    back = iu.load_hdr_image(result.output)
    assert back.linear[0, 0] * cs.REFERENCE_WHITE == pytest.approx(203, rel=1e-3)
    assert back.linear.shape == (60, 80, 3)
    assert not list(result.output.parent.glob(".upscaling-*"))  # no temporaries left


def test_hdr_values_above_sdr_survive_upscaling(tmp_path):
    nits = np.full((20, 24, 3), 50, np.float32)
    nits[4:8, 4:8] = 2000  # a highlight far above SDR white
    nits[12:16, 12:20] = [800, 100, 30]  # a bright saturated colour
    src = write_pq_png(tmp_path / "scene.png", nits)
    up = NearestUpscaler()
    result = process_image(
        src, hdr_options(hdr_output=HdrOutputSettings("hdr", peak_nits=4000)), up
    )
    assert up.dtypes == [np.uint16]  # the model saw the 16-bit proxy, not 8 bits
    back = iu.load_hdr_image(result.output).linear * cs.REFERENCE_WHITE
    assert back[10, 10] == pytest.approx([2000] * 3, rel=2e-3)  # not clipped to SDR
    assert back[28, 30] == pytest.approx([800, 100, 30], rel=3e-3)
    assert back[0, 0] == pytest.approx([50] * 3, rel=2e-3)
    assert codecs.inspect_png(result.output).max_cll == pytest.approx(2000, rel=2e-3)


def test_highlights_above_peak_are_rolled_off_not_clipped(tmp_path):
    nits = np.full((10, 10, 3), 20, np.float32)
    nits[:, 5:] = np.linspace(500, 3000, 10, dtype=np.float32)[:, None, None]
    src = write_pq_png(tmp_path / "bright.png", nits)
    result = process_image(src, hdr_options(), NearestUpscaler())
    back = iu.load_hdr_image(result.output).linear * cs.REFERENCE_WHITE
    column = back[::2, 15, 0]
    assert column.max() <= 1000 * 1.001
    assert np.all(np.diff(column) > 0)  # gradation kept: a roll-off, not a clip
    assert back[0, 0, 0] == pytest.approx(20, rel=2e-3)


def test_output_settings_reuse_the_cached_upscale(tmp_path):
    src = _sdr_photo(tmp_path / "photo.png")
    up = NearestUpscaler()
    for out in (
        HdrOutputSettings("hdr"),
        HdrOutputSettings("hdr", peak_nits=2000),
        HdrOutputSettings("hdr", primaries="p3", sdr_white=250),
    ):
        process_image(src, hdr_options(hdr_output=out), up)
    assert up.calls == 1
    p3 = codecs.inspect_png(tmp_path / "upscaled" / "photo_2x_pq1000-p3-w250.png")
    assert p3.primaries == codecs.CP_P3


def test_hdr_source_cached_upscale_is_16_bit(tmp_path):
    src = write_pq_png(tmp_path / "s.png", np.full((8, 8, 3), 1500, np.float32))
    up = NearestUpscaler()
    process_image(src, hdr_options(), up)
    process_image(src, hdr_options(hdr_output=HdrOutputSettings("hdr", peak_nits=2000)), up)
    assert up.calls == 1
    cached = list(upscale_cache.cache_dir().glob("*.npy"))
    assert cached and np.load(cached[0]).dtype == np.uint16


def test_alpha_kept_in_hdr_png(tmp_path):
    src = tmp_path / "logo.png"
    rgba = np.zeros((10, 12, 4), np.uint8)
    rgba[..., 0] = 200
    rgba[..., 3] = np.linspace(0, 255, 12, dtype=np.uint8)
    Image.fromarray(rgba, "RGBA").save(src)
    result = process_image(src, hdr_options(), NearestUpscaler())
    assert codecs.inspect_png(result.output).channels == 4
    back = iu.load_hdr_image(result.output)
    assert (
        back.alpha is not None and back.alpha[:, 0].max() < 0.05 and back.alpha[:, -1].min() > 0.95
    )


def test_hdr_source_alpha_kept(tmp_path):
    alpha = np.full((6, 6), 65535, np.uint16)
    alpha[:, :3] = 0
    src = write_pq_png(tmp_path / "a.png", np.full((6, 6, 3), 600, np.float32), alpha)
    result = process_image(src, hdr_options(), NearestUpscaler())
    back = iu.load_hdr_image(result.output)
    assert back.alpha[:, :4].max() < 0.1 and back.alpha[:, 8:].min() > 0.9


@needs_avif
def test_avif_export_verified(tmp_path):
    src = _sdr_photo(tmp_path / "photo.png")
    result = process_image(
        src, hdr_options(hdr_output=HdrOutputSettings("hdr", format="avif")), NearestUpscaler()
    )
    assert result.output.suffix == ".avif"
    report = codecs.inspect_avif(result.output)
    assert (report.bit_depth, report.transfer, report.primaries) == (10, 16, 9)


def test_hdr_source_written_as_sdr_is_tone_mapped_and_says_so(tmp_path):
    nits = np.full((8, 8, 3), 3000, np.float32)
    nits[:4] = 5
    src = write_pq_png(tmp_path / "h.png", nits)
    result = process_image(src, ProcessingOptions(scale=2, model="x"), NearestUpscaler())
    assert "tone-mapped to SDR" in result.note
    with Image.open(result.output) as img:
        assert img.mode == "RGB"
        px = np.asarray(img)
    assert codecs.sniff(result.output) is None  # an ordinary SDR file
    # 5 cd/m² is 2.5 % of SDR white (sRGB code ~43); 3000 cd/m² rolls off to white.
    assert px[12, 0].max() == 255 and 35 < px[0, 0].max() < 50


def test_hdr_sources_previewed_through_tone_mapping(tmp_path):
    nits = np.zeros((10, 20, 3), np.float32)
    nits[:, 10:] = 1000
    src = write_pq_png(tmp_path / "p.png", nits)
    info = iu.probe_image(src)
    assert info.format.startswith("HDR ") and (info.width, info.height) == (20, 10)
    preview, full = iu.load_preview(src, 8)
    assert full == (20, 10) and max(preview.size) == 8
    thumb = np.asarray(iu.make_thumbnail(src, 16))
    assert thumb[..., :3].max() == 255 and thumb[2, 1, :3].max() == 0
    assert iu.is_supported(Path("x.exr")) and iu.is_supported(Path("x.hdr"))


@needs_ffmpeg
def test_radiance_hdr_source_scaled_by_sdr_white(tmp_path):
    rgbe = tmp_path / "scene.hdr"
    # A 2×1 Radiance RGBE file: (1, 1, 1) and (4, 4, 4) in scene-linear units.
    header = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 2\n"
    rgbe.write_bytes(header + bytes([128, 128, 128, 129, 128, 128, 128, 131]))
    image = iu.load_hdr_image(rgbe, scene_white=203)
    assert image.linear[0, 0] == pytest.approx([1, 1, 1], rel=1e-3)
    assert image.linear[0, 1] == pytest.approx([4, 4, 4], rel=1e-3)
    assert image.source.encoding == "linear"


def test_invalid_hdr_file_is_a_clear_error(tmp_path):
    bad = tmp_path / "broken.exr"
    bad.write_bytes(b"\x76\x2f\x31\x01" + b"\0" * 50)
    with pytest.raises(InvalidImageError):
        iu.load_image(bad)


def test_missing_avif_encoder_fails_before_work(tmp_path, monkeypatch):
    monkeypatch.setattr(codecs, "avif_available", lambda: False)
    src = _sdr_photo(tmp_path / "photo.png")
    up = NearestUpscaler()
    with pytest.raises(OutputError, match="libaom"):
        process_image(src, hdr_options(hdr_output=HdrOutputSettings("hdr", format="avif")), up)
    assert up.calls == 0
    assert not (tmp_path / "upscaled").exists() or not any((tmp_path / "upscaled").iterdir())


def test_encoder_failure_leaves_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(codecs, "avif_available", lambda: True)

    def fail(*_a, **_k):
        raise codecs.HdrCodecError("The AVIF encoder failed: boom")

    monkeypatch.setattr(codecs, "write_avif10", fail)
    src = _sdr_photo(tmp_path / "photo.png")
    with pytest.raises(OutputError, match="boom"):
        process_image(
            src, hdr_options(hdr_output=HdrOutputSettings("hdr", format="avif")), NearestUpscaler()
        )
    assert list((tmp_path / "upscaled").iterdir()) == []


def test_failed_validation_is_never_reported_as_success(tmp_path, monkeypatch):
    real = codecs.inspect_png

    def sdr_lookalike(path, verify_data=True):
        report = real(path, verify_data)
        report.transfer = codecs.TC_SRGB  # as if the encoder had written SDR
        return report

    monkeypatch.setattr(codecs, "inspect_png", sdr_lookalike)
    src = _sdr_photo(tmp_path / "photo.png")
    with pytest.raises(OutputError, match="not PQ"):
        process_image(src, hdr_options(), NearestUpscaler())
    assert list((tmp_path / "upscaled").iterdir()) == []


def test_cancel_during_encoding_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(hp, "_BAND_PIXELS", 200)  # many bands
    control = JobControl()
    calls = {"n": 0}
    original = control.check

    def check():
        calls["n"] += 1
        if calls["n"] > 8:
            control.cancel()
        original()

    control.check = check  # type: ignore[method-assign]
    src = _sdr_photo(tmp_path / "photo.png", (60, 50))
    with pytest.raises(CancelledError):
        process_image(src, hdr_options(), NearestUpscaler(), control=control)
    assert list((tmp_path / "upscaled").iterdir()) == []


def test_restoration_writes_sdr():
    options = ProcessingOptions(
        hdr_output=HdrOutputSettings("hdr"), restoration=RestorationSettings()
    )
    assert not options.effective_hdr_output().enabled
    assert options.effective_hdr_output().tag() == ""


def test_sdr_exports_unchanged(tmp_path):
    src = _sdr_photo(tmp_path / "photo.png")
    for fmt, ext in (("png", ".png"), ("jpeg", ".jpg"), ("webp", ".webp")):
        result = process_image(
            src, ProcessingOptions(scale=2, model="x", output_format=fmt), NearestUpscaler()
        )
        assert result.output.name == f"photo_2x{ext}"
        assert codecs.sniff(result.output) is None
        with Image.open(result.output) as img:
            assert img.mode == "RGB" and img.size == (80, 60)


def test_settings_validation_and_tags():
    assert HdrOutputSettings().tag() == "" and not HdrOutputSettings().enabled
    assert HdrOutputSettings("hdr", peak_nits=4000, expand_highlights=True).tag() == "pq4000-x"
    for bad in (
        HdrOutputSettings("hdr", peak_nits=100),
        HdrOutputSettings("hlg"),
        HdrOutputSettings("hdr", format="webp"),
        HdrOutputSettings("hdr", transfer="gamma"),
    ):
        with pytest.raises(ValueError):
            bad.validate()
    s = Settings(output_range="bogus", hdr_peak_nits=99999, hdr_sdr_white=5).normalise()
    assert (s.output_range, s.hdr_peak_nits, s.hdr_sdr_white) == ("sdr", 10000, 100)
    assert Settings(output_range="pq").normalise().output_range == "hdr"  # older builds
    hlg = HdrOutputSettings("hdr", transfer="hlg", primaries="p3", peak_nits=4000)
    assert (hlg.tag(), hlg.effective_peak, hlg.effective_primaries) == ("hlg", 1000, "bt2020")
    uhdr = HdrOutputSettings("hdr", format="jpeg", primaries="p3", transfer="hlg")
    assert (uhdr.tag(), uhdr.effective_primaries, uhdr.extension) == ("uhdr1000", "bt709", ".jpg")
    assert hp.conversion_status(False, HdrOutputSettings("hdr")) == "SDR source → HDR conversion"
    assert hp.conversion_status(True, HdrOutputSettings()) == "HDR source → SDR (tone-mapped)"


def test_cli_hdr_export_and_inspect(tmp_path, tiny_specs, capsys):
    src = _sdr_photo(tmp_path / "photo.png", (12, 10))
    out = tmp_path / "out"
    code = cli.run_cli(
        [str(src), "-o", str(out), "-s", "4", "-m", "test-x4", "--device", "cpu",
         "--dynamic-range", "hdr", "--hdr-peak", "2000"]
    )  # fmt: skip
    assert code == 0
    written = out / "photo_4x_pq2000.png"
    assert codecs.inspect_png(written).is_pq
    assert cli.run_cli(["--inspect-hdr", str(written)]) == 0
    assert "PQ" in capsys.readouterr().out
    assert cli.run_cli(["--inspect-hdr", str(src)]) == 3


def test_torch_upscaler_16_bit_matches_8_bit(upscaler):
    rng = np.random.default_rng(6)
    img8 = rng.integers(0, 256, (12, 14, 3), dtype=np.uint8)
    out8 = upscaler.upscale(img8, 4, "test-x4")
    out16 = upscaler.upscale(img8.astype(np.uint16) * 257, 4, "test-x4")
    assert out16.dtype == np.uint16 and out16.shape == out8.shape
    assert np.abs(out16 / 257.0 - out8).max() <= 1.0


def test_large_export_memory_is_bounded(tmp_path):
    """Encoding works in bands: memory grows with the picture only by the
    upscaled uint8 image (3 bytes/pixel), not by a float copy (12+ bytes)."""
    peaks = []
    for size in ((1000, 750), (1500, 1125)):  # 2× -> 3 and 6.75 megapixels
        src = _sdr_photo(tmp_path / f"big{size[0]}.png", size)
        tracemalloc.start()
        try:
            result = process_image(src, hdr_options(), NearestUpscaler())
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
        assert codecs.inspect_png(result.output).height == size[1] * 2
    extra_pixels = (3000 * 2250) - (2000 * 1500)
    assert (peaks[1] - peaks[0]) / extra_pixels < 6
    assert peaks[1] < 160 * 2**20


def test_png_writer_rejects_inconsistent_input(tmp_path):
    with pytest.raises(ValueError):
        codecs.write_png16(
            tmp_path / "x.png", 4, 4, 3, [np.zeros((2, 4, 3), np.uint16)], (9, 16, 0, 1)
        )
    with pytest.raises(ValueError):
        codecs.write_png16(
            tmp_path / "y.png", 4, 4, 3, [np.zeros((4, 4, 3), np.uint16)], (9, 16, 9, 1)
        )


def test_hdr_source_through_the_torch_upscaler(tmp_path, upscaler):
    """The real inference path (tiling, 16-bit proxy) produces a valid PQ file."""
    nits = np.full((14, 18, 3), 80, np.float32)
    nits[3:7, 3:9] = 1500
    src = write_pq_png(tmp_path / "scene.png", nits)
    options = ProcessingOptions(
        scale=4, model="test-x4", hdr_output=HdrOutputSettings("hdr", peak_nits=2000)
    )
    result = process_image(src, options, upscaler)
    report = codecs.inspect_png(result.output)
    assert (report.width, report.height, report.transfer, report.bit_depth) == (72, 56, 16, 16)
    linear = iu.load_hdr_image(result.output).linear
    assert np.isfinite(linear).all() and linear.max() * cs.REFERENCE_WHITE <= 2000 * 1.001


# --- HLG output --------------------------------------------------------------------------
def test_hlg_encode_reference_values_and_round_trip():
    # BT.2408: 203 cd/m² (reference white) is 75 % HLG; the 1000 cd/m² peak is 100 %.
    grey = cs.hlg_encode(np.full((1, 3), 203.0))
    assert float(grey[0, 0]) == pytest.approx(0.75, abs=0.002)
    assert float(cs.hlg_encode(np.full((1, 3), 1000.0))[0, 0]) == pytest.approx(1.0)
    assert float(cs.hlg_oetf(np.array([1 / 12]))[0]) == pytest.approx(0.5)  # BT.2100 knee
    rng = np.random.default_rng(7)
    grey_ramp = np.repeat(rng.random((2000, 1)) * 1000, 3, axis=1).astype(np.float32)
    assert cs.hlg_decode(cs.hlg_encode(grey_ramp)) == pytest.approx(grey_ramp, rel=1e-4, abs=1e-3)


def test_hlg_out_of_range_colours_keep_luminance_and_hue():
    bright_red = np.array([[1000.0, 0.0, 0.0], [800.0, 200.0, 50.0]], np.float32)
    back = cs.hlg_decode(cs.hlg_encode(bright_red))
    luma = np.asarray(cs.BT2020_LUMA)
    assert back @ luma == pytest.approx(bright_red @ luma, rel=1e-4)
    assert back[0, 0] > back[0, 1] == pytest.approx(back[0, 2])  # still red, desaturated
    assert back[1] == pytest.approx(bright_red[1], rel=1e-4)  # in range: exact


def test_hlg_png_export(tmp_path):
    nits = np.full((10, 12, 3), 60, np.float32)
    nits[2:6, 2:6] = 900
    src = write_pq_png(tmp_path / "s.png", nits)
    out = HdrOutputSettings("hdr", transfer="hlg", primaries="p3", peak_nits=4000)
    result = process_image(src, hdr_options(hdr_output=out), NearestUpscaler())
    assert result.output.name == "s_2x_hlg.png"
    report = codecs.inspect_png(result.output)
    assert (report.transfer, report.primaries, report.bit_depth) == (18, 9, 16)  # always 2020
    assert report.max_cll is None and "cLLi" not in report.chunks  # CTA-861.3 is for PQ
    if codecs.ffprobe_path():
        assert codecs.ffprobe_stream(result.output)["color_transfer"] == "arib-std-b67"
    back = iu.load_hdr_image(result.output).linear * cs.REFERENCE_WHITE
    assert back[6, 6] == pytest.approx([900] * 3, rel=1e-3)
    assert back[0, 0] == pytest.approx([60] * 3, rel=1e-3)


def test_hlg_rolls_off_above_its_1000_nit_reference(tmp_path):
    src = write_pq_png(tmp_path / "b.png", np.full((4, 4, 3), 3000, np.float32))
    out = HdrOutputSettings("hdr", transfer="hlg")
    back = iu.load_hdr_image(
        process_image(src, hdr_options(hdr_output=out), NearestUpscaler()).output
    )
    assert back.linear.max() * cs.REFERENCE_WHITE == pytest.approx(1000, rel=1e-3)


# --- AVIF transparency and pixel validation --------------------------------------------
@needs_avif
@pytest.mark.parametrize("transfer", ["pq", "hlg"])
def test_avif_keeps_alpha_and_values(tmp_path, transfer):
    alpha = np.full((10, 12), 65535, np.uint16)
    alpha[:, :4] = 0
    nits = np.full((10, 12, 3), 80, np.float32)
    nits[4:8, 6:10] = [700, 90, 40]
    src = write_pq_png(tmp_path / "a.png", nits, alpha)
    out = HdrOutputSettings("hdr", format="avif", transfer=transfer)
    result = process_image(src, hdr_options(hdr_output=out, quality=95), NearestUpscaler())
    report = codecs.inspect_avif(result.output)
    assert report.has_alpha and (report.bit_depth, report.channels) == (10, 3)
    assert report.transfer == (18 if transfer == "hlg" else 16)
    with Image.open(result.output) as img:  # libavif, an independent decoder
        assert img.mode == "RGBA"
        a = np.asarray(img.getchannel("A"))
    assert a[:, :6].max() < 10 and a[:, 10:].min() > 245
    back = iu.load_hdr_image(result.output)
    assert back.alpha is not None
    assert back.linear[12, 16] * cs.REFERENCE_WHITE == pytest.approx([700, 90, 40], rel=0.05)
    assert back.linear[1, 18] * cs.REFERENCE_WHITE == pytest.approx([80] * 3, rel=0.05)


@needs_avif
def test_avif_pixel_validation_catches_wrong_values(tmp_path, monkeypatch):
    real = codecs.read_avif_planes

    def range_bug(path, width=None, height=None):
        planes, depth = real(path, width, height)
        return (planes * 219 // 255 + 64).astype(np.uint16), depth  # full read as limited

    monkeypatch.setattr(codecs, "read_avif_planes", range_bug)
    src = _sdr_photo(tmp_path / "photo.png")
    out = HdrOutputSettings("hdr", format="avif")
    with pytest.raises(OutputError, match="does not decode"):
        process_image(src, hdr_options(hdr_output=out), NearestUpscaler())
    assert list((tmp_path / "upscaled").iterdir()) == []


# --- Ultra HDR gain-map JPEG -------------------------------------------------------------
def test_gain_map_jpeg_written_and_read_back(tmp_path):
    nits = np.full((16, 20, 3), 60, np.float32)
    nits[4:12, 4:8] = 800
    nits[4:12, 12:16] = [500, 90, 40]
    src = write_pq_png(tmp_path / "scene.png", nits)
    out = HdrOutputSettings("hdr", format="jpeg")
    result = process_image(src, hdr_options(hdr_output=out), NearestUpscaler())
    assert result.output.name == "scene_2x_uhdr1000.jpg"
    report = gainmap.inspect(result.output)
    assert (report.width, report.height) == (40, 32)
    assert report.headroom_stops == pytest.approx(np.log2(800 / 203), abs=0.05)
    # Pillow's own MPF parser (independent of ours) finds the gain map where the
    # primary image ends, and both images decode.
    data = result.output.read_bytes()
    with Image.open(result.output) as img:
        mp = img._getmp()
        assert img.size == (40, 32) and img.info.get("icc_profile")
        assert np.asarray(img.convert("RGB"))[0, 0].max() > 100  # an ordinary SDR picture
    primary, second = mp[0xB002]
    mp_header = img.info["mpoffset"]
    assert mp[0xB001] == 2 and primary["Size"] == mp_header + second["DataOffset"]
    assert primary["Size"] + second["Size"] == len(data)
    with Image.open(io.BytesIO(data[primary["Size"] :])) as gm:
        assert gm.size == (10, 8) and gm.mode == "RGB"
        assert b'hdrgm:GainMapMax="' in gm.info.get("xmp", b"")
    # Our decoder rebuilds the HDR rendition (weight 1).
    back = iu.load_hdr_image(result.output).linear * cs.REFERENCE_WHITE
    assert codecs.sniff(result.output).encoding == "gainmap"
    assert back[16, 12] == pytest.approx([800] * 3, rel=0.06)
    assert back[16, 28] == pytest.approx([500, 90, 40], rel=0.1, abs=4)
    assert back[0, 0] == pytest.approx([60] * 3, rel=0.06)


def test_gain_map_from_sdr_source_keeps_the_original_as_base(tmp_path):
    src = _sdr_photo(tmp_path / "photo.png")
    out = HdrOutputSettings("hdr", format="jpeg")
    result = process_image(src, hdr_options(hdr_output=out), NearestUpscaler())
    with Image.open(result.output) as img:
        base = np.asarray(img.convert("RGB")).astype(int)
    with Image.open(src) as img:
        original = np.asarray(img.convert("RGB").resize((80, 60), Image.Resampling.NEAREST))
    assert np.abs(base - original).mean() < 3  # JPEG noise only
    assert gainmap.inspect(result.output).headroom_stops < 0.01  # nothing to add


def test_gain_map_metadata_parsing_variants():
    seq = (
        "<hdrgm:GainMapMax><rdf:Seq><rdf:li>2.0</rdf:li><rdf:li>2.5</rdf:li>"
        "<rdf:li>3.0</rdf:li></rdf:Seq></hdrgm:GainMapMax>"
        '<x hdrgm:HDRCapacityMax="3.0" hdrgm:Gamma="1.2"/>'
    )
    meta = gainmap.parse_metadata(seq)
    assert meta.gain_max == (2.0, 2.5, 3.0) and meta.gamma == (1.2,)
    assert meta.offset_sdr == (1 / 64,) and meta.gain_min == (0.0,)  # spec defaults
    with pytest.raises(codecs.HdrCodecError):
        gainmap.parse_metadata('hdrgm:Version="1.0"')


def test_plain_jpeg_is_not_hdr_and_damaged_gain_map_is_an_error(tmp_path):
    plain = tmp_path / "plain.jpg"
    Image.new("RGB", (8, 8), "red").save(plain)
    assert codecs.sniff(plain) is None
    src = _sdr_photo(tmp_path / "photo.png")
    out = HdrOutputSettings("hdr", format="jpeg", expand_highlights=True)
    good = process_image(src, hdr_options(hdr_output=out), NearestUpscaler()).output
    cut = tmp_path / "cut.jpg"
    cut.write_bytes(good.read_bytes()[:-100])
    with pytest.raises(codecs.HdrCodecError):
        gainmap.inspect(cut)
    with pytest.raises(InvalidImageError):
        iu.load_hdr_image(cut)


# --- OpenEXR chromaticities ------------------------------------------------------------
def _exr_with_chromaticities(path: Path, rgb: np.ndarray, chroma: tuple[float, ...]) -> None:
    """Write an EXR with ffmpeg, then add a ``chromaticities`` header attribute
    (moving the scan-line offset table, whose offsets are absolute)."""
    import subprocess

    h, w, _ = rgb.shape
    planes = np.stack([rgb[..., 1], rgb[..., 2], rgb[..., 0]]).astype("<f4")  # G B R
    raw = path.with_suffix(".raw")
    raw.write_bytes(planes.tobytes())
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gbrpf32le",
         "-s", f"{w}x{h}", "-i", str(raw), "-frames:v", "1", "-c:v", "exr", str(path)],
        check=True,
    )  # fmt: skip
    data = bytearray(path.read_bytes())
    pos = 8
    while data[pos] != 0:  # walk the attributes to the header's terminating null
        name_end = data.index(0, pos)
        type_end = data.index(0, name_end + 1)
        size = struct.unpack("<i", data[type_end + 1 : type_end + 5])[0]
        pos = type_end + 5 + size
    attr = b"chromaticities\0chromaticities\0" + struct.pack("<i", 32) + struct.pack("<8f", *chroma)
    table_start = pos + 1
    offsets = np.frombuffer(bytes(data[table_start : table_start + 8 * h]), "<u8") + len(attr)
    data[table_start : table_start + 8 * h] = offsets.astype("<u8").tobytes()
    path.write_bytes(bytes(data[:pos] + attr + data[pos:]))


@needs_ffmpeg
def test_exr_chromaticities_are_applied(tmp_path):
    p3 = (0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.3127, 0.3290)
    rgb = np.zeros((4, 4, 3), np.float32)
    rgb[..., 0] = 1.0  # pure P3 red, scene-linear 1.0
    path = tmp_path / "p3.exr"
    _exr_with_chromaticities(path, rgb, p3)
    source = codecs.sniff(path)
    assert source.chromaticities == pytest.approx(p3)
    image = iu.load_hdr_image(path, scene_white=203)
    expected = (cs.conversion_matrix("p3", "bt2020") @ np.array([1.0, 0, 0]))[None]
    expected = cs.fit_gamut(expected.astype(np.float32))[0]  # P3 red is just outside 2020
    assert image.linear[0, 0] == pytest.approx(expected, abs=1e-4)
    assert image.linear[0, 0, 0] > 0.7  # not read as Rec. 709 red (0.627 in Rec. 2020)
    plain = tmp_path / "plain.exr"
    _exr_with_chromaticities(plain, rgb, (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290))
    assert codecs.sniff(plain).chromaticities is None  # Rec. 709: the default anyway


def test_chromaticities_matrix_handles_other_white_points():
    aces = (0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, 0.32168, 0.33767)  # ACES AP0
    m = cs.chromaticities_to_bt2020(aces)
    assert m @ np.ones(3) == pytest.approx(np.ones(3), abs=1e-3)  # its white -> D65 white


def test_cli_hlg_and_gain_map(tmp_path, tiny_specs, capsys):
    src = _sdr_photo(tmp_path / "photo.png", (12, 10))
    out = tmp_path / "out"
    base = [str(src), "-o", str(out), "-s", "4", "-m", "test-x4", "--device", "cpu"]
    assert cli.run_cli([*base, "--dynamic-range", "hdr", "--hdr-transfer", "hlg"]) == 0
    assert codecs.inspect_png(out / "photo_4x_hlg.png").is_hlg
    assert cli.run_cli([*base, "--dynamic-range", "hdr", "--hdr-format", "jpeg"]) == 0
    capsys.readouterr()
    assert cli.run_cli(["--inspect-hdr", str(out / "photo_4x_uhdr1000.jpg")]) == 0
    assert "gain map" in capsys.readouterr().out
    plain = tmp_path / "plain.jpg"
    Image.new("RGB", (4, 4)).save(plain)
    assert cli.run_cli(["--inspect-hdr", str(plain)]) == 3


@pytest.mark.skipif(
    not (os.environ.get("PIXELIFT_TEST_CHROME") and shutil.which("google-chrome")),
    reason="opt-in: PIXELIFT_TEST_CHROME=1 and google-chrome",
)
def test_chrome_decodes_the_colour_description(tmp_path):
    """Headless Chrome (an independent viewer) honours cICP / nclx.

    Each file has a 26, a 203 and a 1000 cd/m² patch; Chrome tone-maps HDR to
    the SDR canvas, so we compare luminance ratios, not absolute values. A
    viewer ignoring cICP would show the raw PQ signal (ratio 0.45, not 0.13).
    """
    import http.server
    import json
    import subprocess
    import threading
    import time
    import urllib.parse

    nits = np.full((32, 64, 3), 203, np.float32)
    nits[:, 32:] = 1000
    nits[:16, :16] = 26
    src = write_pq_png(tmp_path / "src.png", nits)
    files = []
    for out in (
        HdrOutputSettings("hdr"),
        HdrOutputSettings("hdr", transfer="hlg"),
        *([HdrOutputSettings("hdr", format="avif")] if codecs.avif_available() else []),
    ):
        result = process_image(src, hdr_options(hdr_output=out), NearestUpscaler())
        files.append(result.output.relative_to(tmp_path).as_posix())
    script = """(async()=>{const r={};for(const f of FILES){
      const im=new Image();im.src=f;await im.decode();const c=document.createElement('canvas');
      c.width=im.naturalWidth;c.height=im.naturalHeight;const x=c.getContext('2d');
      x.drawImage(im,0,0);r[f]=[[8,8],[40,50],[100,50]].map(([a,b])=>x.getImageData(a,b,1,1).data[0]);}
      fetch('/r?'+encodeURIComponent(JSON.stringify(r)));})();"""
    page = "<!doctype html><script>" + script.replace("FILES", json.dumps(files)) + "</script>"
    (tmp_path / "index.html").write_text(page)
    got: dict = {}

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(tmp_path), **k)

        def do_GET(self):
            if self.path.startswith("/r?"):
                got.update(json.loads(urllib.parse.unquote(self.path[3:])))
            return super().do_GET()

        def log_message(self, *_a):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    chrome = subprocess.Popen(
        ["google-chrome", "--headless=new", "--no-first-run", "--disable-gpu",
         f"--user-data-dir={tmp_path / 'profile'}",
         f"http://127.0.0.1:{server.server_port}/index.html"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    try:
        deadline = time.monotonic() + 60
        while not got and time.monotonic() < deadline and chrome.poll() is None:
            time.sleep(0.2)
    finally:
        chrome.terminate()
        chrome.wait(10)
        server.shutdown()
    assert set(got) == set(files)
    for grey26, ref203, _peak in got.values():
        lin = cs.srgb_eotf(np.array([grey26, ref203], np.float32) / 255)
        assert lin[0] / lin[1] == pytest.approx(26 / 203, rel=0.1)


# --- review fixes ------------------------------------------------------------------------
def _ultra_hdr_photo(tmp_path: Path) -> Path:
    """An Ultra HDR JPEG as a phone writes it (here: made by Pixelift)."""
    nits = np.full((16, 20, 3), 60, np.float32)
    nits[4:12, 4:8] = 800
    src = write_pq_png(tmp_path / "phone_src.png", nits)
    out = HdrOutputSettings("hdr", format="jpeg")
    made = process_image(src, hdr_options(hdr_output=out), NearestUpscaler()).output
    return made.rename(tmp_path / "phone.jpg")


def test_gain_map_jpeg_with_sdr_output_is_the_original_photo(tmp_path):
    photo = _ultra_hdr_photo(tmp_path)
    assert iu.hdr_source(photo) is None and iu.hdr_source(photo, gain_maps=True) is not None
    assert iu.probe_image(photo).format == iu.GAIN_MAP_FORMAT
    loaded = iu.load_image(photo)
    assert loaded.hdr is None
    with Image.open(photo) as img:
        base = np.asarray(img.convert("RGB"))
    np.testing.assert_array_equal(loaded.rgb, base)  # the JPEG's own picture
    result = process_image(photo, ProcessingOptions(scale=2, model="x"), NearestUpscaler())
    assert "tone-mapped" not in result.note
    preview, _ = iu.load_preview(photo, 64)
    np.testing.assert_array_equal(np.asarray(preview.convert("RGB")), base)


def test_unusable_gain_map_falls_back_to_the_sdr_photo(tmp_path, monkeypatch):
    photo = _ultra_hdr_photo(tmp_path)

    def base_is_hdr(_path):
        raise codecs.HdrCodecError("BaseRenditionIsHDR is not supported")

    monkeypatch.setattr(gainmap, "decode", base_is_hdr)
    result = process_image(photo, hdr_options(), NearestUpscaler())
    assert "Gain map not used" in result.note and "Converted from SDR" in result.note
    assert codecs.inspect_png(result.output).is_pq


def test_scene_white_does_not_change_the_sdr_view(tmp_path):
    rgbe = tmp_path / "scene.hdr"
    header = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 2\n"
    rgbe.write_bytes(header + bytes([128, 128, 128, 129, 128, 128, 128, 131]))
    if codecs.ffmpeg_path() is None:
        pytest.skip("needs ffmpeg")
    a = iu.load_image(rgbe, scene_white=203)
    b = iu.load_image(rgbe, scene_white=350)
    np.testing.assert_array_equal(a.rgb, b.rgb)  # SDR results and caches stay valid
    assert b.hdr.linear[0, 0, 0] == pytest.approx(350 / 203, rel=1e-3)  # HDR uses it


def test_opaque_hdr_alpha_is_dropped(tmp_path):
    alpha = np.full((6, 6), 65535, np.uint16)
    src = write_pq_png(tmp_path / "a.png", np.full((6, 6, 3), 300, np.float32), alpha)
    result = process_image(src, hdr_options(), NearestUpscaler())
    assert codecs.inspect_png(result.output).channels == 3
    out = HdrOutputSettings("hdr", format="jpeg")
    result = process_image(src, hdr_options(hdr_output=out), NearestUpscaler())
    assert "Transparency" not in result.note


def test_skipped_sdr_tools_are_not_in_hdr_source_names(tmp_path):
    from pixelift.core.hdr import HdrSettings
    from pixelift.core.lighting import LightingSettings

    src = write_pq_png(tmp_path / "h.png", np.full((6, 6, 3), 300, np.float32))
    options = hdr_options(
        lighting=LightingSettings("golden-hour", 60), hdr=HdrSettings(True, "natural", 60)
    )
    result = process_image(src, options, NearestUpscaler())
    assert result.output.name == "h_2x_pq1000.png"
    assert (result.lighting, result.look, result.hdr) == ("", "", "pq1000")
    assert "skipped" in result.note
    sdr = _sdr_photo(tmp_path / "s.png")
    named = process_image(sdr, options, NearestUpscaler()).output.name
    assert "golden-hour" in named and "hdr-natural-60" in named and "pq1000" in named


def test_16_bit_input_never_runs_in_float16():
    from pixelift.core import device_manager as dm
    from pixelift.core.upscaler import _half_for

    gpu = dm.DeviceInfo("cuda:0", "Test GPU", "CUDA")
    assert _half_for(gpu, wide=False) and not _half_for(gpu, wide=True)
    assert not _half_for(dm.CPU_DEVICE, wide=False)


def test_inspect_hdr_reports_a_malformed_png_instead_of_crashing(tmp_path, capsys):
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    bad = tmp_path / "bad.png"
    bad.write_bytes(codecs.PNG_SIGNATURE + chunk(b"IHDR", b"\0\0") + chunk(b"IEND", b""))
    assert cli.run_cli(["--inspect-hdr", str(bad)]) == 1
    assert "damaged" in capsys.readouterr().err


def test_unusable_gain_map_is_named_and_tagged_as_the_sdr_photo_it_is_read_as(tmp_path):
    photo = _ultra_hdr_photo(tmp_path)
    assert keeps_hdr(photo, hdr_options())
    data = photo.read_bytes()
    match = re.search(rb'hdrgm:GainMapMax="([^"]*)"', data)
    assert match
    damaged = b"nan".ljust(len(match.group(1)))  # same length: offsets stay valid
    photo.write_bytes(data[: match.start(1)] + damaged + data[match.end(1) :])
    assert iu.hdr_source(photo, gain_maps=True) is not None  # still looks like Ultra HDR
    assert not keeps_hdr(photo, hdr_options())  # but is read (and named) as SDR


def test_ffmpeg_paths_are_never_options_or_protocols(tmp_path):
    for name in ("-x.exr", "a:b.png"):
        arg = codecs.ff_file(Path(name))
        assert arg.startswith("file:/") and arg.endswith(name)


@needs_ffmpeg
def test_ffprobe_reads_awkward_file_names(tmp_path):
    src = write_pq_png(tmp_path / "-a:b.png", np.full((6, 8, 3), 100, np.float32))
    stream = codecs.ffprobe_stream(src)
    assert "error" not in stream and (stream["width"], stream["height"]) == (8, 6)


def _fake_encoder(script: str) -> list[str]:
    import sys

    return [sys.executable, "-c", script]


def test_avif_encoder_exiting_early_reports_its_stderr():
    planes = np.zeros((3, 2048, 2048), np.uint16)  # far more than a pipe buffer
    cmd = _fake_encoder("import sys; sys.stderr.write('bad option\\n'); sys.exit(3)")
    with pytest.raises(codecs.HdrCodecError, match="bad option"):
        codecs._run_encoder(cmd, planes, 2048, None)


def test_hung_avif_encoder_times_out(monkeypatch):
    import time

    monkeypatch.setattr(codecs, "_FFMPEG_TIMEOUT", 0.5)
    planes = np.zeros((3, 2048, 2048), np.uint16)
    start = time.monotonic()
    with pytest.raises(codecs.HdrCodecError, match="timed out"):
        codecs._run_encoder(_fake_encoder("import time; time.sleep(60)"), planes, 2048, None)
    assert time.monotonic() - start < 30


def test_thumbnails_do_not_evict_the_cached_hdr_preview(tmp_path):
    nits = np.full((10, 20, 3), 300, np.float32)
    shown = write_pq_png(tmp_path / "shown.png", nits)
    other = write_pq_png(tmp_path / "other.png", nits)
    iu._hdr_sdr_image.cache_clear()
    iu.load_preview(shown, 8)
    iu.make_thumbnail(other, 8)
    iu.make_thumbnail(shown, 8)
    assert iu._hdr_sdr_image.cache_info().currsize == 1
    iu.load_region(shown, (0, 0, 4, 4), (4, 4))
    assert iu._hdr_sdr_image.cache_info().hits == 1
