"""AI photo restoration: settings, conventional stages, pipeline, files, batch, CLI.

These tests need no downloaded models: the AI stages are exercised through
small stand-ins (see ``fake_faces``) and the tiny Real-ESRGAN test networks
from conftest. ``test_restoration_models.py`` runs the real models.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pytest
from conftest import make_image
from PIL import Image, ImageDraw

from pixelift.core.batch_processor import BatchProcessor, ItemStatus, QueueItem
from pixelift.core.errors import ModelNotInstalledError
from pixelift.core.image_processor import ProcessingOptions, output_path_for, process_image
from pixelift.core.lighting import Adjustments, LightingSettings
from pixelift.core.restoration import cleanup, faces, tones
from pixelift.core.restoration import filters as fl
from pixelift.core.restoration import settings as rs
from pixelift.core.restoration.analysis import (
    MonoInfo,
    detect_monochrome,
    detect_monochrome_file,
    estimate_noise,
)
from pixelift.core.restoration.colorize import apply_color
from pixelift.core.restoration.pipeline import Restorer, required_models, upscale_weight
from pixelift.storage.settings import Settings

ORIENTATION = 0x0112
# Every stage off: restoration must leave the pixels alone.
NOTHING = rs.RestorationSettings(level=rs.CUSTOM, custom=rs.Stages(), modern=rs.MODERN_OFF)
# Conventional stages only (no AI models needed).
CONVENTIONAL = rs.RestorationSettings(
    level=rs.CUSTOM,
    custom=rs.Stages(dust=60, scratches=50, noise=40, fading=50, sharpness=30, auto_color=True),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def photo(size=(240, 180), seed=0) -> np.ndarray:
    """A smooth, photo-like colour image with soft shapes and mild grain."""
    rng = np.random.default_rng(seed)
    w, h = size
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 120 + 60 * np.sin(x / 37) + 30 * np.cos(y / 23)
    g = 110 + 50 * np.cos(x / 29 + y / 41)
    b = 90 + 40 * np.sin(y / 31)
    img = np.dstack([r, g, b]) + rng.normal(0, 3, (h, w, 3))
    return img.clip(0, 255).astype(np.uint8)


def gray(rgb: np.ndarray) -> np.ndarray:
    return fl.to_gray(rgb)


def add_specks(rgb: np.ndarray, count=15, seed=1) -> tuple[np.ndarray, list[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    h, w = rgb.shape[:2]
    spots = []
    for _ in range(count):
        cx, cy = int(rng.integers(10, w - 10)), int(rng.integers(10, h - 10))
        r = float(rng.uniform(1.0, 2.5))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255))
        spots.append((cx, cy))
    return np.asarray(img), spots


def add_scratch(rgb: np.ndarray, x: int) -> np.ndarray:
    img = Image.fromarray(rgb)
    ImageDraw.Draw(img).line([(x, 0), (x + 4, rgb.shape[0] - 1)], fill=(250, 250, 250), width=2)
    return np.asarray(img)


def to_file(rgb: np.ndarray, path: Path, mode: str | None = None, **kw) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(rgb)
    if mode:
        img = img.convert(mode)
    img.save(path, **kw)
    return path


# --- settings ----------------------------------------------------------------
def test_level_presets_and_custom():
    assert rs.RestorationSettings(level=rs.LIGHT).stages().face == rs.FACE_OFF
    assert rs.RestorationSettings(level=rs.STANDARD).stages().face == rs.FACE_NATURAL
    assert rs.RestorationSettings(level=rs.HEAVY).stages().detail
    light, standard, heavy = (rs.LEVEL_STAGES[k] for k in (rs.LIGHT, rs.STANDARD, rs.HEAVY))
    for name in ("dust", "scratches", "noise", "fading"):
        assert getattr(light, name) <= getattr(standard, name) <= getattr(heavy, name)
    custom = rs.RestorationSettings(level=rs.CUSTOM, custom=rs.Stages(dust=500, face="bogus"))
    assert custom.stages().dust == 100 and custom.stages().face == rs.FACE_OFF


def test_identity_detection():
    assert NOTHING.is_identity()
    assert not rs.RestorationSettings().is_identity()
    assert not dataclasses.replace(NOTHING, scale=2).is_identity()
    assert not dataclasses.replace(NOTHING, colorize=True).is_identity()
    assert dataclasses.replace(NOTHING, colorize=True, colorize_strength=0).is_identity()


def test_tags_name_different_settings_differently():
    default = rs.RestorationSettings()
    assert default.tag() == "" and dataclasses.replace(default, colorize=True).tag() == ""
    assert rs.RestorationSettings(level=rs.HEAVY).tag() == "heavy"
    tags = {
        dataclasses.replace(default, fidelity=70).tag(),
        dataclasses.replace(default, modern="vivid").tag(),
        dataclasses.replace(default, color=Adjustments(temperature=10)).tag(),
        dataclasses.replace(default, colorize=True, colorize_strength=40).tag(),
        rs.RestorationSettings(level=rs.CUSTOM, custom=rs.Stages(dust=1)).tag(),
    }
    assert len(tags) == 5 and "" not in tags


def test_colorize_settings_only_tag_colorized_results():
    plain = rs.RestorationSettings(colorize=True)
    stronger = dataclasses.replace(plain, colorize_strength=40, colorize_vivid=30)
    # Colour photos are never colorized: the colour settings must not rename them.
    assert stronger.tag(colorized=False) == plain.tag(colorized=False) == ""
    assert stronger.tag(colorized=True) != plain.tag(colorized=True)


def test_ai_scale():
    assert rs.RestorationSettings().ai_scale() == 0
    assert rs.RestorationSettings(scale=4).ai_scale() == 4
    assert rs.RestorationSettings(level=rs.HEAVY).ai_scale() == 2  # detail reconstruction
    assert rs.RestorationSettings(level=rs.HEAVY, scale=4).ai_scale() == 4


def test_memory_check_counts_detail_reconstruction(tmp_path, monkeypatch):
    from pixelift.core import image_processor as ip
    from pixelift.core.errors import ImageTooLargeError

    target = tmp_path / "x.png"
    standard = restore_options(restoration=rs.RestorationSettings())
    heavy = restore_options(restoration=rs.RestorationSettings(level=rs.HEAVY))
    monkeypatch.setattr(ip.iu, "disk_free", lambda _p: 1 << 62)
    # Just enough memory for Standard at 1×: Heavy also needs its 2× intermediate.
    needed = 4000 * 3000 * (4 * 2 + 4 + 3 * 4 + 3 * 2)
    monkeypatch.setattr(ip, "available_ram_bytes", lambda: needed)
    ip.check_feasible(4000, 3000, standard, target)
    with pytest.raises(ImageTooLargeError, match="memory"):
        ip.check_feasible(4000, 3000, heavy, target)


def test_mono_detection_reuses_a_decode(tmp_path):
    src = to_file(photo((60, 40)), tmp_path / "colour.png")
    # The caller's decode is used instead of reading the file again (and cached).
    assert detect_monochrome_file(src, Image.fromarray(gray(photo((60, 40))))).monochrome
    assert detect_monochrome_file(src).monochrome


def test_validate_rejects_bad_values():
    for bad in ({"level": "x"}, {"modern": "x"}, {"scale": 3}):
        with pytest.raises(ValueError):
            rs.RestorationSettings(**bad).validate()


def test_presets_switch_colorize_and_upscale():
    assert rs.preset_flags(rs.PRESET_RESTORE) == (False, False)
    assert rs.preset_flags(rs.PRESET_COLORIZE) == (True, False)
    assert rs.preset_flags(rs.PRESET_UPSCALE) == (False, True)
    assert rs.preset_flags(rs.PRESET_FULL) == (True, True)


def test_app_settings_round_trip(tmp_path):
    from pixelift.storage.settings import load_settings, save_settings

    settings = Settings(mode="restore", restore_preset=rs.PRESET_FULL, restore_scale=4)
    settings.restore_level = rs.CUSTOM
    settings.restore_dust = 77
    settings.restore_face = rs.FACE_STRONG
    settings.restore_temperature = -20
    path = tmp_path / "s.json"
    save_settings(settings, path)
    loaded = load_settings(path)
    options = loaded.processing_options()
    restoration = options.restoration
    assert restoration is not None and options.output_scale == 4
    assert restoration.colorize and restoration.stages().dust == 77
    assert restoration.stages().face == rs.FACE_STRONG
    assert restoration.color.temperature == -20
    loaded.mode = "upscale"
    assert loaded.processing_options().restoration is None


def test_app_settings_normalise_bad_values():
    settings = Settings(
        mode="x", restore_level="x", restore_face="x", restore_modern="x", restore_scale=3
    )
    settings.restore_dust = 900
    settings.restore_tint = -500
    settings.normalise()
    assert (settings.mode, settings.restore_level, settings.restore_scale) == (
        "upscale",
        rs.STANDARD,
        2,
    )
    assert settings.restore_dust == 100 and settings.restore_tint == -100


# --- black-and-white detection ----------------------------------------------
def test_detects_black_and_white_sepia_and_colour():
    color = photo()
    bw = gray(color)
    sepia = (bw.astype(np.float32) * np.array([1.07, 0.95, 0.75]) + [8, 4, 0]).clip(0, 255)
    sepia = sepia.astype(np.uint8)
    assert detect_monochrome(bw).monochrome and not detect_monochrome(bw).toned
    info = detect_monochrome(sepia)
    assert info.monochrome and info.toned
    assert not detect_monochrome(color).monochrome
    # Grain on each channel separately does not make a B&W photo "colour".
    noisy = (bw.astype(np.float32) + np.random.default_rng(3).normal(0, 4, bw.shape)).clip(0, 255)
    assert detect_monochrome(noisy.astype(np.uint8)).monochrome


def test_detection_from_file_is_cached_per_version(tmp_path):
    path = to_file(gray(photo()), tmp_path / "bw.png")
    assert detect_monochrome_file(path).monochrome
    to_file(photo(), path)  # same name, new content
    assert not detect_monochrome_file(path).monochrome


# --- conventional stages ------------------------------------------------------
def test_strength_zero_changes_nothing():
    img = photo()
    for fn in (cleanup.remove_dust, cleanup.reduce_scratches, cleanup.sharpen):
        assert fn(img, 0, 0.01) is img
    assert cleanup.denoise(img, 0, 0.01, False) is img
    assert tones.restore_tones(img, 0, False, False) is img


def test_dense_patterns_are_not_dust():
    """A dotted fabric is many specks close together: texture, not damage."""
    img = Image.new("RGB", (200, 200), (40, 50, 120))
    draw = ImageDraw.Draw(img)
    for y in range(4, 200, 9):
        for x in range(4, 200, 9):
            draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=(230, 230, 240))
    pattern = np.asarray(img)
    out = cleanup.remove_dust(pattern, 80, 0.005)
    assert np.abs(out.astype(int) - pattern).mean() < 1.0


def test_dust_removal_removes_specks():
    clean = gray(photo())
    dusty, spots = add_specks(clean)
    out = cleanup.remove_dust(dusty, 60, estimate_noise(clean))
    before = np.mean([abs(int(dusty[y, x, 0]) - int(clean[y, x, 0])) for x, y in spots])
    after = np.mean([abs(int(out[y, x, 0]) - int(clean[y, x, 0])) for x, y in spots])
    assert before > 60 and after < 15


def test_dust_strength_controls_speck_size():
    base = np.full((120, 200, 3), 120, np.uint8)
    img = Image.fromarray(base)
    ImageDraw.Draw(img).ellipse([96, 56, 104, 64], fill=(255, 255, 255))  # 9 px speck
    img = np.asarray(img)
    weak = cleanup.remove_dust(img, 10, 0.0)
    strong = cleanup.remove_dust(img, 100, 0.0)
    assert weak[60, 100, 0] > 200  # too big for a light touch: kept
    assert abs(int(strong[60, 100, 0]) - 120) < 10  # removed at full strength


def test_scratch_removal_and_edges_survive():
    clean = gray(photo((300, 200)))
    # A long dark edge (real detail) must survive scratch removal.
    clean[:, 150:] = (clean[:, 150:].astype(np.int16) - 60).clip(0, 255).astype(np.uint8)
    scratched = add_scratch(clean, 60)
    out = cleanup.reduce_scratches(scratched, 60, estimate_noise(clean))
    col = slice(58, 68)
    err_before = np.abs(scratched[:, col].astype(int) - clean[:, col]).mean()
    err_after = np.abs(out[:, col].astype(int) - clean[:, col]).mean()
    assert err_after < err_before * 0.3
    edge_drop = int(out[100, 140, 0]) - int(out[100, 160, 0])
    assert edge_drop > 40


def test_denoise_reduces_noise_but_keeps_edges():
    rng = np.random.default_rng(5)
    base = np.full((160, 160), 90, np.float32)
    base[:, 80:] = 170
    noisy = (base + rng.normal(0, 12, base.shape)).clip(0, 255).astype(np.uint8)
    rgb = np.repeat(noisy[..., None], 3, 2)
    out = cleanup.denoise(rgb, 80, estimate_noise(rgb), monochrome=True)
    assert out[:, 10:70, 0].std() < noisy[:, 10:70].std() * 0.6
    assert out[:, 85:150, 0].mean() - out[:, 10:75, 0].mean() > 70  # edge contrast kept
    # Colour noise on a colour photo is reduced.
    color = photo(seed=2).astype(np.float32)
    color += rng.normal(0, 10, color.shape)
    color = color.clip(0, 255).astype(np.uint8)
    out = cleanup.denoise(color, 80, estimate_noise(color), monochrome=False)
    chroma = lambda a: (a[..., 0].astype(float) - a[..., 1]).std()  # noqa: E731
    assert chroma(out) < chroma(color)


def test_sharpen_adds_detail_without_halos():
    img = Image.fromarray(gray(photo())).filter(__import__("PIL.ImageFilter").ImageFilter.BLUR)
    soft = np.asarray(img)
    out = cleanup.sharpen(soft, 80, 0.002)
    lap = lambda a: np.abs(np.diff(a[..., 0].astype(float), axis=1)).mean()  # noqa: E731
    assert lap(out) > lap(soft)
    assert out.max() <= soft.max() + 10 and out.min() >= soft.min() - 10


def test_tiled_processing_matches_whole_image():
    img = add_specks(gray(photo((300, 220))))[0]
    run = lambda x: fl.inpaint(x, (fl.luma(x) > 0.9).float(), 1.5)  # noqa: E731
    whole = fl.map_tiles(img, run, halo=60, tile=1024)
    tiled = fl.map_tiles(img, run, halo=60, tile=64)
    assert np.abs(whole.astype(int) - tiled).max() <= 1


def test_tiny_images_are_processed():
    tiny = photo((12, 9))
    for fn in (cleanup.remove_dust, cleanup.reduce_scratches, cleanup.sharpen):
        assert fn(tiny, 80, 0.01).shape == tiny.shape


def test_fade_recovery_restores_contrast_and_removes_cast():
    original = photo().astype(np.float32)
    faded = 70 + 0.45 * original
    faded[..., 0] += 25  # red/yellow cast from aged dyes
    faded[..., 2] -= 15
    faded = faded.clip(0, 255).astype(np.uint8)
    out = tones.restore_tones(faded, 70, auto_color=True, monochrome=False).astype(np.float32)
    assert out.std() > faded.std() * 1.5
    cast_before = faded[..., 0].mean() - faded[..., 2].mean()
    cast_after = out[..., 0].mean() - out[..., 2].mean()
    true_cast = original[..., 0].mean() - original[..., 2].mean()
    assert abs(cast_after - true_cast) < abs(cast_before - true_cast)


def test_fade_recovery_keeps_black_and_white_neutral():
    faded = (60 + 0.5 * gray(photo()).astype(np.float32)).astype(np.uint8)
    out = tones.restore_tones(faded, 70, auto_color=True, monochrome=True)
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])
    assert out.std() > faded.std() * 1.4


# --- colorization blending -----------------------------------------------------
def test_colorize_strength_and_tones():
    bw = gray(photo())
    predicted = np.zeros((64, 64, 3), np.float32)
    predicted[...] = (170, 120, 90)  # a warm colour everywhere
    off = apply_color(bw, predicted, strength=0, vivid=0, preserve_tones=True)
    assert np.abs(off.astype(int) - bw).max() <= 1  # 0% = the grey photo
    subtle = apply_color(bw, predicted, 25, 0, True).astype(float)
    full = apply_color(bw, predicted, 100, 0, True).astype(float)
    chroma = lambda a: np.abs(a[..., 0] - a[..., 2]).mean()  # noqa: E731
    assert 0 < chroma(subtle) < chroma(full)
    # Preserve tones: brightness unchanged wherever no channel clips.
    y_out = full @ np.array(fl.LUMA)
    ok = (full.max(2) < 254) & (full.min(2) > 1)
    assert np.abs(y_out[ok] - bw[..., 0][ok]).mean() < 1.0
    natural = apply_color(bw, predicted, 100, 0, True).astype(float)
    vivid = apply_color(bw, predicted, 100, 100, True).astype(float)
    assert chroma(vivid) > chroma(natural)


# --- faces: geometry and identity safeguards ------------------------------------
def test_similarity_transform_recovers_known_transform():
    angle, scale, shift = 0.3, 1.7, np.array([12.0, -40.0])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    src = faces.TEMPLATE / 3 + 50
    dst = scale * src @ rot.T + shift
    m = faces.similarity_transform(src, dst)
    assert np.allclose(m[:, :2], scale * rot, atol=1e-9) and np.allclose(m[:, 2], shift)
    assert np.allclose(faces.invert(faces.invert(m)), m)


def test_crop_and_paste_round_trip():
    img = photo((400, 400))
    landmarks = faces.TEMPLATE * 0.4 + 90  # a face ~200 px wide
    m = faces.similarity_transform(landmarks, faces.TEMPLATE)
    crop = faces.crop_face(img, m)
    assert crop.shape == (512, 512, 3)
    out = img.copy()
    faces.paste_face(out, np.zeros_like(crop), m, faces.ellipse_mask())
    assert np.array_equal(out, img)  # no change: the photo keeps all its own detail
    # Brightening the aligned crop to white brightens the facial area to white.
    changed = img.copy()
    faces.paste_face(changed, 255.0 - crop, m, faces.ellipse_mask())
    assert changed[200, 200].mean() > 240  # facial area replaced
    assert np.array_equal(changed[5, 5], img[5, 5])  # far outside the face: untouched


def test_identity_guard_and_weights():
    import torch

    rng = np.random.default_rng(0)
    face = torch.from_numpy(rng.uniform(0, 255, (1, 3, 512, 512)).astype(np.float32))
    face = fl.gaussian_blur(face, 8)
    mask = faces.ellipse_mask()
    assert faces.identity_factor(face, face, mask) == 1.0
    other = torch.flip(face, dims=[2, 3])
    assert faces.identity_factor(face, other, mask) < 0.5
    flat = torch.full_like(face, 128)
    assert faces.identity_factor(flat, face, mask) <= 0.25
    assert faces.base_weight(rs.FACE_OFF, 100) == 0
    for fidelity in (0, 50, 100):
        assert faces.base_weight(rs.FACE_NATURAL, fidelity) < faces.base_weight(
            rs.FACE_STRONG, fidelity
        )
    assert faces.base_weight(rs.FACE_NATURAL, 0) < faces.base_weight(rs.FACE_NATURAL, 100)
    assert faces.size_factor(5) < faces.size_factor(10) < faces.size_factor(40) == 1.0
    assert upscale_weight(0) < upscale_weight(100) == 1.0


def test_harmonise_keeps_original_tones():
    import torch

    original = torch.full((1, 3, 512, 512), 100.0)
    restored = torch.full((1, 3, 512, 512), 160.0)  # GFPGAN shifted the skin tone
    restored[..., 250:260, :] += 50  # plus some fine detail
    for mode in (rs.FACE_NATURAL, rs.FACE_STRONG):
        out = faces.harmonise(original, restored, mode)
        assert abs(float(out[..., 100:200, 100:200].mean()) - 100) < 3  # tone kept
        assert float(out[..., 255, 300].mean()) > float(out[..., 150, 300].mean()) + 20


# --- pipeline -----------------------------------------------------------------------
@pytest.fixture
def restorer(upscaler):
    return Restorer(upscaler)


def test_original_mode_changes_nothing(restorer):
    img = photo()
    sepia = (gray(img).astype(np.float32) * [1.05, 0.95, 0.8]).astype(np.uint8)
    for src in (img, gray(img), sepia):
        result = restorer.restore(src, NOTHING, model="test-family")
        assert np.array_equal(result.rgb, src) and result.rgb is not src
        assert not result.colorized
    assert not restorer.restore(sepia, NOTHING, model="test-family").monochrome  # stays toned


def test_conventional_restoration_keeps_colour_photo_colour(restorer):
    src = photo()
    result = restorer.restore(src, CONVENTIONAL, model="test-family")
    assert result.rgb.shape == src.shape and not result.monochrome and not result.colorized
    assert not detect_monochrome(result.rgb).monochrome


def test_black_and_white_restored_in_black_and_white(restorer):
    sepia = (gray(photo()).astype(np.float32) * [1.06, 0.96, 0.78]).astype(np.uint8)
    result = restorer.restore(sepia, CONVENTIONAL, model="test-family")
    assert result.monochrome and not result.colorized
    assert np.array_equal(result.rgb[..., 0], result.rgb[..., 2])  # neutral grey


def test_colour_photos_are_never_colorized(restorer):
    settings = dataclasses.replace(CONVENTIONAL, colorize=True)
    # No colorization model is installed, yet a colour photo needs none.
    result = restorer.restore(photo(), settings, model="test-family")
    assert not result.colorized
    with pytest.raises(ModelNotInstalledError):
        restorer.restore(gray(photo()), settings, model="test-family")


def test_colorization_can_be_disabled(restorer):
    bw = gray(photo())
    result = restorer.restore(bw, CONVENTIONAL, model="test-family")
    assert not result.colorized and result.monochrome


def test_missing_face_models_are_reported(restorer):
    with pytest.raises(ModelNotInstalledError) as err:
        restorer.restore(photo(), rs.RestorationSettings(), model="test-family")
    assert "GFPGAN" in err.value.reason or "RetinaFace" in err.value.reason
    assert required_models(NOTHING) == []
    assert set(required_models(rs.RestorationSettings())) == {"gfpgan-v1.4", "retinaface-resnet50"}
    with_colour = dataclasses.replace(NOTHING, colorize=True)
    assert required_models(with_colour, monochrome=False) == []
    assert required_models(with_colour, monochrome=True) == ["deoldify-artistic"]


def test_restore_and_upscale(restorer):
    src = photo((48, 36))
    settings = dataclasses.replace(CONVENTIONAL, scale=4)
    result = restorer.restore(src, settings, model="test-family")
    assert result.rgb.shape == (144, 192, 3)
    detail = dataclasses.replace(
        CONVENTIONAL, custom=dataclasses.replace(CONVENTIONAL.custom, detail=True)
    )
    assert restorer.restore(src, detail, model="test-family").rgb.shape == src.shape


def test_restoration_intensity(restorer):
    clean = gray(photo())
    dusty, spots = add_specks(clean)

    def error(strength: int) -> float:
        stages = rs.Stages(dust=strength)
        settings = dataclasses.replace(NOTHING, custom=stages)
        out = restorer.restore(dusty, settings, model="test-family").rgb
        return float(np.mean([abs(int(out[y, x, 0]) - int(clean[y, x, 0])) for x, y in spots]))

    assert error(0) > error(30) + 30  # specks removed…
    assert error(90) <= error(30) + 1  # …and stronger never does worse


def test_progress_and_cancel(restorer):
    from pixelift.core.control import JobControl
    from pixelift.core.errors import CancelledError

    stages: list[str] = []
    restorer.restore(
        photo(), CONVENTIONAL, model="test-family", progress=lambda f, s: stages.append(s)
    )
    assert {"Removing dust", "Reducing scratches", "Reducing noise"} <= set(stages)
    control = JobControl()
    control.cancel()
    with pytest.raises(CancelledError):
        restorer.restore(photo(), CONVENTIONAL, model="test-family", control=control)


@pytest.fixture
def fake_faces(monkeypatch, models_dir):
    """Face stage with stand-ins: one known face, and a 'restoration' that brightens.

    Model files exist (empty placeholders) so the pipeline's checks pass; the
    networks themselves are never loaded.
    """
    from pixelift.core.upscaler import TorchUpscaler
    from pixelift.models.restoration import GFPGAN_V14, RETINAFACE

    for spec in (GFPGAN_V14, RETINAFACE):
        (models_dir / spec.filename).write_bytes(b"placeholder")
    calls = {"detect": 0, "restore": 0}

    def fake_run_model(self, spec, fn, control=None, release=True):
        return fn(spec.id, None)

    def fake_detect(_net, _device, rgb):
        calls["detect"] += 1
        h, w = rgb.shape[:2]
        scale = min(h, w) / 512 * 0.8
        landmarks = faces.TEMPLATE * scale + [w / 2 - 256 * scale, h / 2 - 290 * scale]
        return [faces.Face((0, 0, w, h), 0.999, landmarks)]

    def fake_gfpgan(_net, _device, face):
        calls["restore"] += 1
        out = face.copy()
        out[250:262] = 255.0  # "restored detail": a bright band across the face
        return out

    monkeypatch.setattr(TorchUpscaler, "run_model", fake_run_model)
    monkeypatch.setattr(faces, "detect_faces", fake_detect)
    monkeypatch.setattr(faces, "run_gfpgan", fake_gfpgan)
    return calls


def test_face_restoration_modes(restorer, fake_faces):
    src = gray(photo((320, 320)))
    off = dataclasses.replace(NOTHING, custom=rs.Stages(face=rs.FACE_OFF))
    natural = dataclasses.replace(NOTHING, custom=rs.Stages(face=rs.FACE_NATURAL))
    strong = dataclasses.replace(NOTHING, custom=rs.Stages(face=rs.FACE_STRONG))
    assert np.array_equal(restorer.restore(src, off, model="test-family").rgb, src)
    assert fake_faces["detect"] == 0
    changes = []
    for settings in (natural, strong):
        result = restorer.restore(src, settings, model="test-family")
        assert result.faces.found == 1 and result.faces.restored == 1
        assert result.monochrome  # a B&W photo stays B&W
        changes.append(np.abs(result.rgb.astype(int) - src).mean())
        assert np.array_equal(result.rgb[:5, :5], src[:5, :5])  # background untouched
    assert 0 < changes[0] < changes[1]  # Natural changes less than Strong
    low = dataclasses.replace(natural, fidelity=0)
    high = dataclasses.replace(natural, fidelity=100)

    def diff(settings):
        return np.abs(
            restorer.restore(src, settings, model="test-family").rgb.astype(int) - src
        ).mean()

    assert diff(low) < diff(high)  # fidelity: lower keeps more of the original


def test_face_restoration_after_upscaling(restorer, fake_faces):
    src = photo((96, 96))
    settings = dataclasses.replace(NOTHING, scale=4, custom=rs.Stages(face=rs.FACE_NATURAL))
    result = restorer.restore(src, settings, model="test-family")
    assert result.rgb.shape == (384, 384, 3) and result.faces.restored == 1


# --- files: formats, naming, metadata, originals -------------------------------
def restore_options(**kw) -> ProcessingOptions:
    restoration = kw.pop("restoration", CONVENTIONAL)
    return ProcessingOptions(model="test-family", restoration=restoration, **kw)


def test_output_names(tmp_path):
    bw = to_file(gray(photo()), tmp_path / "grandma_1962.jpg", quality=95)
    color = to_file(photo(), tmp_path / "beach.jpg", quality=95)
    std = rs.RestorationSettings()
    name = lambda src, **kw: output_path_for(src, 240, 180, restore_options(**kw)).name  # noqa: E731
    assert name(bw, restoration=std, output_format="jpeg") == "grandma_1962_restored.jpg"
    colorize = dataclasses.replace(std, colorize=True)
    assert name(bw, restoration=colorize, output_format="jpeg") == (
        "grandma_1962_restored_colorized.jpg"
    )
    assert name(color, restoration=colorize, output_format="jpeg") == "beach_restored.jpg"
    upscale = dataclasses.replace(std, scale=4)
    assert name(bw, restoration=upscale, output_format="jpeg") == "grandma_1962_restored_4x.jpg"
    full = dataclasses.replace(std, colorize=True, scale=2)
    assert name(bw, restoration=full) == "grandma_1962_restored_colorized_2x.png"
    heavy = rs.RestorationSettings(level=rs.HEAVY)
    assert name(bw, restoration=heavy) == "grandma_1962_restored-heavy.png"
    lit = LightingSettings("golden-hour")
    assert name(bw, restoration=std, lighting=lit) == "grandma_1962_restored_golden-hour.png"
    assert output_path_for(bw, 1, 1, restore_options()).parent == tmp_path / "restored"


@pytest.mark.parametrize(
    ("name", "mode", "fmt"),
    [
        ("a.png", "RGB", "png"),
        ("b.jpg", "RGB", "jpeg"),
        ("c.png", "RGBA", "png"),
        ("d.webp", "RGB", "webp"),
    ],
)
def test_restore_formats(tmp_path, upscaler, name, mode, fmt):
    src = make_image(tmp_path / "in" / name, size=(64, 48), mode=mode)
    before = sha(src)
    result = process_image(src, restore_options(output_format=fmt), upscaler)
    assert sha(src) == before  # the original is never modified
    with Image.open(result.output) as out:
        assert out.size == (64, 48)
        assert out.mode == ("RGBA" if mode == "RGBA" and fmt != "jpeg" else "RGB")
    assert result.restored and not result.colorized


def test_rgba_alpha_preserved_and_flattened_for_jpeg(tmp_path, upscaler):
    src = make_image(tmp_path / "in" / "logo.png", size=(64, 48), mode="RGBA")
    out = process_image(src, restore_options(), upscaler).output
    with Image.open(src) as a, Image.open(out) as b:
        assert np.array_equal(np.asarray(a.getchannel("A")), np.asarray(b.getchannel("A")))
    result = process_image(src, restore_options(output_format="jpeg"), upscaler)
    assert "flattened" in result.note


def test_black_and_white_saved_as_greyscale(tmp_path, upscaler):
    src = to_file(gray(photo()), tmp_path / "old.png", mode="L")
    result = process_image(src, restore_options(), upscaler)
    with Image.open(result.output) as out:
        assert out.mode == "L"
    assert "Black & white" in result.note


def test_metadata_and_orientation_preserved(tmp_path, upscaler):
    img = Image.fromarray(photo((60, 40)))
    exif = Image.Exif()
    exif[ORIENTATION] = 6  # rotated 90° by the camera
    exif[0x010F] = "Kodak"  # Make
    exif[0x9003] = "1962:07:04 12:00:00"  # DateTimeOriginal (in IFD0 for the test)
    src = tmp_path / "scan.jpg"
    from PIL import ImageCms

    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    img.save(src, exif=exif, icc_profile=icc, dpi=(300, 300), quality=95)
    options = restore_options(
        restoration=dataclasses.replace(CONVENTIONAL, scale=2), output_format="jpeg"
    )
    result = process_image(src, options, upscaler)
    with Image.open(result.output) as out:
        assert out.size == (80, 120)  # orientation applied to the pixels
        out_exif = out.getexif()
        # …and reset, so viewers don't rotate again.
        assert out_exif.get(ORIENTATION) in (None, 1)
        assert out_exif.get(0x010F) == "Kodak" and out_exif.get(0x9003) == "1962:07:04 12:00:00"
        assert out.info.get("icc_profile") == icc
        assert round(out.info["dpi"][0]) == 600  # same print size at twice the pixels


def test_metadata_can_be_dropped(tmp_path, upscaler):
    src = tmp_path / "scan.jpg"
    exif = Image.Exif()
    exif[0x010F] = "Kodak"
    Image.fromarray(photo((40, 30))).save(src, exif=exif)
    result = process_image(src, restore_options(preserve_metadata=False), upscaler)
    with Image.open(result.output) as out:
        assert 0x010F not in out.getexif()


def test_original_never_overwritten(tmp_path, upscaler):
    src = to_file(photo((40, 30)), tmp_path / "photo_restored.png")
    before = sha(src)
    # Output folder = the original's folder, file name colliding with the source.
    options = restore_options(output_dir=tmp_path, existing="overwrite")
    options.restoration = NOTHING
    trick = tmp_path / "photo.png"
    to_file(photo((40, 30), seed=9), trick)
    result = process_image(trick, options, upscaler)
    assert result.output != src or sha(src) == before
    # A source whose own name equals its output name is never overwritten.
    options.restoration = NOTHING
    for policy in ("overwrite", "skip", "rename"):
        options.existing = policy
        own = output_path_for(trick, 40, 30, options)
        to_file(photo((40, 30), seed=3), own)
        guarded = sha(own)
        result = process_image(own, options, upscaler)
        assert sha(own) == guarded and result.output != own


def test_upscaling_cannot_overwrite_the_original(tmp_path, upscaler):
    src = make_image(tmp_path / "pic.png")
    before = sha(src)
    options = ProcessingOptions(
        scale=4,
        model="test-family",
        output_dir=tmp_path,
        filename_template="{name}",
        existing="overwrite",
    )
    result = process_image(src, options, upscaler)
    assert sha(src) == before and result.output == tmp_path / "pic (2).png"


def test_skip_existing_restored_output(tmp_path, upscaler):
    src = make_image(tmp_path / "in" / "a.png")
    first = process_image(src, restore_options(), upscaler)
    again = process_image(src, restore_options(), upscaler)
    assert again.skipped and again.output == first.output
    assert again.restoration == first.restoration


def test_high_resolution_scan(tmp_path, upscaler):
    rng = np.random.default_rng(0)
    big = (rng.normal(128, 20, (1800, 2600, 3))).clip(0, 255).astype(np.uint8)
    src = to_file(big, tmp_path / "scan.png")
    result = process_image(src, restore_options(), upscaler)
    assert result.output_size == (2600, 1800)


def test_low_resolution_image(tmp_path, upscaler):
    src = make_image(tmp_path / "tiny.png", size=(16, 12))
    options = restore_options(restoration=dataclasses.replace(CONVENTIONAL, scale=4))
    assert process_image(src, options, upscaler).output_size == (64, 48)


# --- batch -------------------------------------------------------------------------
def test_batch_restoration_shares_models(tmp_path, upscaler, fake_faces, monkeypatch):
    sources = [make_image(tmp_path / "in" / f"photo_{i:02d}.png", size=(64, 48)) for i in range(4)]
    sources.append(to_file(gray(photo((64, 48))), tmp_path / "in" / "old_bw.png", mode="L"))
    hashes = [sha(p) for p in sources]
    loads = []
    original_load = type(upscaler).load

    def counting_load(self, spec):
        if spec.id not in self._nets:
            loads.append(spec.id)  # a real load, not a cache hit
        return original_load(self, spec)

    monkeypatch.setattr(type(upscaler), "load", counting_load)
    options = restore_options(
        restoration=dataclasses.replace(
            CONVENTIONAL,
            scale=2,
            custom=dataclasses.replace(CONVENTIONAL.custom, face=rs.FACE_NATURAL),
        )
    )
    stages = set()
    processor = BatchProcessor(
        upscaler,
        options,
        lambda e: e.item is not None and stages.add(e.item.stage),
        progress_interval=0,
    )
    summary = processor.run([QueueItem(p) for p in sources])
    assert summary.done == 5 and summary.failed == 0
    assert [sha(p) for p in sources] == hashes
    assert processor.restorer is not None
    assert len(set(loads)) == len(loads) == 1  # the upscaling model was loaded once
    assert fake_faces["detect"] == 5
    assert any(s.startswith("Restoring faces") or s == "Finding faces" for s in stages)
    names = sorted(p.name for p in (tmp_path / "in" / "restored").iterdir())
    assert len(names) == 5
    assert names[0].startswith("old_bw_restored-custom-") and names[0].endswith("_2x.png")


def test_batch_reports_missing_models_per_image(tmp_path, upscaler):
    src = make_image(tmp_path / "a.png")
    options = restore_options(restoration=rs.RestorationSettings())  # needs face models
    item = QueueItem(src)
    BatchProcessor(upscaler, options).run([item])
    assert item.status is ItemStatus.FAILED
    assert isinstance(item.error, ModelNotInstalledError)


# --- command line -------------------------------------------------------------------
def test_cli_restore(tmp_path, tiny_specs, monkeypatch, capsys):
    from pixelift.cli import run_cli

    src = to_file(gray(photo((60, 40))), tmp_path / "old.jpg", quality=95)
    before = sha(src)
    code = run_cli([str(src), "--restore", "--face", "off", "-m", "test-family", "--device", "cpu"])
    out = capsys.readouterr().out
    assert code == 0, out
    # Standard with faces off is a custom restoration: tagged so it never
    # overwrites (or is mistaken for) a Standard result.
    first = list((tmp_path / "restored").glob("old_restored-custom-*.png"))
    assert len(first) == 1
    assert "1 restored" in out and sha(src) == before
    code = run_cli(
        [str(src), "--restore", "--face", "off", "--upscale", "-s", "2", "-m", "test-family"]
    )
    assert code == 0
    assert (tmp_path / "restored" / f"{first[0].stem}_2x.png").exists()


def test_cli_restore_needs_models(tmp_path, tiny_specs, capsys):
    from pixelift.cli import run_cli

    src = make_image(tmp_path / "a.png")
    assert run_cli([str(src), "--restore", "-m", "test-family"]) == 1
    err = capsys.readouterr().err
    assert "--download-model gfpgan-v1.4" in err


def test_cli_restore_heavy_checks_the_upscaler_first(tmp_path, tiny_specs, capsys):
    from pixelift.cli import run_cli

    src = make_image(tmp_path / "a.png")
    # Heavy reconstructs detail with Real-ESRGAN even without --upscale.
    args = [str(src), "--restore", "--restore-level", "heavy", "--face", "off"]
    assert run_cli([*args, "-m", "realesrgan"]) == 1
    assert "not installed" in capsys.readouterr().err.lower()
    assert not (tmp_path / "restored").exists()  # refused before any work
    assert run_cli([*args, "-m", "test-family", "--device", "cpu"]) == 0


def test_cli_colorize_needs_no_model_for_colour_photos(tmp_path, tiny_specs, capsys):
    from pixelift.cli import run_cli

    src = to_file(photo((60, 40)), tmp_path / "colour.png")
    args = [str(src), "--restore", "--face", "off", "--colorize", "-m", "test-family"]
    assert run_cli([*args, "--device", "cpu"]) == 0, capsys.readouterr().err
    bw = to_file(gray(photo((60, 40))), tmp_path / "bw" / "old.png")
    assert run_cli([str(bw), *args[1:]]) == 1
    assert "--download-model deoldify-artistic" in capsys.readouterr().err


def test_cli_restore_options_need_restore(tmp_path):
    from pixelift.cli import run_cli

    with pytest.raises(SystemExit):
        run_cli([str(tmp_path), "--colorize"])


def test_cli_lists_restoration_models(capsys):
    from pixelift.cli import run_cli

    assert run_cli(["--list-models"]) == 0
    out = capsys.readouterr().out
    assert "Face Restoration models" in out and "gfpgan-v1.4" in out and "Apache-2.0" in out
    assert "Photo Colorization models" in out and "deoldify-artistic" in out


# --- GPU -------------------------------------------------------------------------------
@pytest.mark.gpu
def test_restoration_on_gpu(manager):
    from pixelift.core import device_manager as dm
    from pixelift.core.upscaler import TorchUpscaler

    report = dm.detect_devices()
    if not report.best.is_gpu:
        pytest.skip("no compatible GPU")
    restorer = Restorer(TorchUpscaler(manager, report.best))
    settings = dataclasses.replace(CONVENTIONAL, scale=2)
    result = restorer.restore(photo((64, 48)), settings, model="test-family")
    assert result.rgb.shape == (96, 128, 3)


def test_mono_info_override_is_respected(restorer):
    """The pipeline follows the caller's B&W decision (the one the file was named by)."""
    result = restorer.restore(photo(), NOTHING, model="test-family", mono=MonoInfo(True, False, 0))
    assert result.monochrome is False  # identity: nothing converted, so not neutral grey
