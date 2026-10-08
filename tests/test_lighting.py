from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from PIL import Image

from pixelift import cli
from pixelift.core import lighting as lt
from pixelift.core.image_processor import ProcessingOptions, process_image
from pixelift.core.lighting import (
    ADJUSTMENT_NAMES,
    CUSTOM,
    ORIGINAL,
    Adjustments,
    LightingProfile,
    LightingSettings,
    apply_lighting,
    apply_to_pil,
)
from pixelift.core.upscaler import Upscaler
from pixelift.storage.settings import Settings, load_settings, save_settings

BUILT_IN = [
    "original",
    "natural-daylight",
    "bright-clean",
    "golden-hour",
    "studio",
    "cinematic",
    "low-light-recovery",
    "cool-daylight",
    "vivid",
    "high-contrast",
    "custom",
]
PRESETS = [p for p in BUILT_IN if p not in (ORIGINAL, CUSTOM)]


class NearestUpscaler(Upscaler):
    """Deterministic stand-in for the AI model, so outputs can be predicted."""

    def upscale(self, image, scale, model, *, progress=None, control=None):
        return np.repeat(np.repeat(image, scale, 0), scale, 1)


def photo(h: int = 64, w: int = 96, seed: int = 0) -> np.ndarray:
    """Smooth gradients plus a skin-like patch and some noise: a stand-in photo."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    rgb = np.dstack([x * 255 / (w - 1), y * 255 / (h - 1), (x + y) * 255 / (w + h - 2)])
    rgb[: h // 4, : w // 4] = (224, 172, 140)  # skin tone
    rgb += rng.normal(0, 4, rgb.shape)
    return rgb.clip(0, 255).astype(np.uint8)


def grey_ramp() -> np.ndarray:
    return np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (2, 1, 3))


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])


def chroma(rgb: np.ndarray) -> float:
    rgb = rgb.astype(np.float64)
    return float((rgb.max(-1) - rgb.min(-1)).mean())


def magenta(rgb: np.ndarray) -> float:
    rgb = rgb.astype(np.float64)
    return float(((rgb[..., 0] + rgb[..., 2]) / 2 - rgb[..., 1]).mean())


# --- profiles -----------------------------------------------------------------
def test_all_built_in_profiles_registered_in_order():
    assert [p.id for p in lt.all_profiles()] == BUILT_IN
    assert lt.get_profile("golden-hour").name == "Golden Hour"
    assert all(p.description for p in lt.all_profiles())


@pytest.mark.parametrize("profile", PRESETS)
def test_every_profile_changes_the_image_safely(profile):
    src = photo()
    before = src.copy()
    adj = LightingSettings(profile, 100).adjustments()
    out = apply_lighting(src, adj)
    assert np.array_equal(src, before)  # input untouched without out=
    assert out.shape == src.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - src).mean() > 1.5
    # Tone curves are monotonic and keep black at black: no banding reversals
    # and no washed-out blacks.
    ramp = apply_lighting(grey_ramp(), adj)
    assert (np.diff(luma(ramp[0])) >= -0.5).all()
    assert ramp[0, 0].max() == 0
    # Highlights roll off rather than flattening into a block of 255s.
    assert len(np.unique(luma(ramp[0, 200:]).round())) > 30


def test_profiles_have_their_character():
    src = photo()

    def run(profile: str) -> np.ndarray:
        return apply_lighting(src, LightingSettings(profile, 100).adjustments())

    def warmth(rgb: np.ndarray) -> float:
        return float(rgb[..., 0].astype(float).mean() - rgb[..., 2].astype(float).mean())

    assert warmth(run("golden-hour")) > warmth(src) + 5
    assert warmth(run("cool-daylight")) < warmth(src) - 5
    assert luma(run("bright-clean")).mean() > luma(src).mean() + 5
    dark = src // 3
    assert luma(apply_lighting(dark, lt.get_profile("low-light-recovery").adjustments)).mean() > (
        luma(dark).mean() + 10
    )
    assert chroma(run("vivid")) > chroma(src) * 1.1
    assert luma(run("high-contrast")).std() > luma(src).std() * 1.1
    assert luma(run("cinematic"))[luma(src) < 80].mean() < luma(src)[luma(src) < 80].mean()


# --- original & intensity -------------------------------------------------------
@pytest.mark.parametrize("intensity", [0, 100])
def test_original_returns_source_unchanged(intensity):
    src = photo()
    settings = LightingSettings(ORIGINAL, intensity)
    assert not settings.active
    assert apply_lighting(src, settings.adjustments()) is src


@pytest.mark.parametrize("profile", PRESETS)
def test_zero_intensity_has_no_effect(profile):
    src = photo()
    settings = LightingSettings(profile, 0)
    assert not settings.active
    assert np.array_equal(apply_lighting(src, settings.adjustments()), src)


@pytest.mark.parametrize("profile", PRESETS)
def test_full_intensity_is_the_full_profile(profile):
    assert LightingSettings(profile, 100).adjustments() == lt.get_profile(profile).adjustments


def test_intensity_interpolates_the_adjustments():
    full = lt.get_profile("golden-hour").adjustments
    half = LightingSettings("golden-hour", 50).adjustments()
    for name in ADJUSTMENT_NAMES:
        assert getattr(half, name) == pytest.approx(getattr(full, name) / 2)
    src = photo()
    warm = [
        float(
            np.mean(apply_lighting(src, LightingSettings("golden-hour", i).adjustments())[..., 0])
        )
        - float(
            np.mean(apply_lighting(src, LightingSettings("golden-hour", i).adjustments())[..., 2])
        )
        for i in (0, 25, 50, 75, 100)
    ]
    assert warm == sorted(warm) and warm[0] < warm[-1]


def test_intensity_out_of_range_is_clamped():
    full = lt.get_profile("vivid").adjustments
    assert LightingSettings("vivid", 250).adjustments() == full
    assert LightingSettings("vivid", -5).adjustments().is_neutral


def test_unknown_profile_is_neutral():
    assert not LightingSettings("does-not-exist", 100).active


# --- custom -------------------------------------------------------------------
def custom(**values: float) -> Adjustments:
    return LightingSettings(CUSTOM, 0, Adjustments(**values)).adjustments()


def test_custom_ignores_intensity_and_neutral_custom_is_identity():
    assert custom(exposure=40).exposure == 40
    src = photo()
    assert apply_lighting(src, custom()) is src


@pytest.mark.parametrize(
    ("name", "check"),
    [
        ("exposure", lambda s, o: luma(o).mean() > luma(s).mean() + 5),
        ("brightness", lambda s, o: luma(o).mean() > luma(s).mean() + 5),
        ("contrast", lambda s, o: luma(o).std() > luma(s).std() * 1.05),
        (
            "highlights",
            lambda s, o: luma(o)[luma(s) > 170].mean() > luma(s)[luma(s) > 170].mean() + 3,
        ),
        ("shadows", lambda s, o: luma(o)[luma(s) < 85].mean() > luma(s)[luma(s) < 85].mean() + 3),
        (
            "temperature",
            lambda s, o: (
                (o[..., 0].mean() - o[..., 2].mean()) > (s[..., 0].mean() - s[..., 2].mean()) + 5
            ),
        ),
        ("tint", lambda s, o: magenta(o) > magenta(s) + 5),
        ("saturation", lambda s, o: chroma(o) > chroma(s) * 1.1),
    ],
)
def test_each_custom_adjustment(name, check):
    src = photo()
    raised = apply_lighting(src, custom(**{name: 50}))
    lowered = apply_lighting(src, custom(**{name: -50}))
    assert check(src, raised)
    assert not check(src, lowered)
    assert not np.array_equal(raised, lowered)


def test_custom_saturation_minimum_is_greyscale():
    out = apply_lighting(photo(), custom(saturation=-100)).astype(int)
    assert (out.max(-1) - out.min(-1)).max() <= 1


def test_extreme_custom_values_stay_in_range_and_protect_highlights():
    src = photo()
    extreme = Adjustments(**dict.fromkeys(ADJUSTMENT_NAMES, 100))
    for adj in (extreme, extreme.scaled(-1), Adjustments(exposure=500)):
        out = apply_lighting(src, LightingSettings(CUSTOM, 100, adj).adjustments())
        assert out.dtype == np.uint8
    # +2 EV on a grey ramp: highlights compress but never flatten into a
    # block of pure white, and every output level stays in use.
    ramp = apply_lighting(grey_ramp(), custom(exposure=100))[0, :, 0].astype(int)
    assert (ramp == 255).sum() <= 2
    assert len(np.unique(ramp[192:])) == 256 - ramp[192]


def test_saturation_boost_keeps_hue_and_skin_natural():
    skin = np.full((4, 4, 3), (224, 172, 140), np.uint8)
    out = apply_lighting(skin, custom(saturation=100))[0, 0].astype(int)
    assert out[0] > out[1] > out[2]  # same hue order
    assert out.max() < 255  # not pushed into clipping


# --- image integrity ----------------------------------------------------------
def test_in_place_and_out_parameter():
    src = photo()
    expected = apply_lighting(src, lt.get_profile("vivid").adjustments)
    buffer = src.copy()
    assert apply_lighting(buffer, lt.get_profile("vivid").adjustments, out=buffer) is buffer
    assert np.array_equal(buffer, expected)


def test_large_image_is_processed_in_bands(monkeypatch):
    src = photo(600, 1000, seed=3)
    adj = lt.get_profile("golden-hour").adjustments
    whole = apply_lighting(src, adj)
    monkeypatch.setattr(lt, "_CHUNK_PIXELS", 7_000)  # many bands, one partial
    assert np.array_equal(apply_lighting(src, adj), whole)


def test_big_image_dimensions_preserved():
    src = np.random.default_rng(1).integers(0, 256, (3000, 4000, 3), dtype=np.uint8)
    out = apply_lighting(src, lt.get_profile("low-light-recovery").adjustments, out=src)
    assert out is src and out.shape == (3000, 4000, 3)


def test_rejects_non_rgb_arrays():
    with pytest.raises(ValueError):
        apply_lighting(np.zeros((4, 4), np.uint8), custom(exposure=10))


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_apply_to_pil_preserves_alpha(mode):
    img = Image.fromarray(photo(), "RGB")
    if mode == "RGBA":
        img.putalpha(Image.fromarray(np.arange(64 * 96).reshape(64, 96).astype(np.uint8)))
    out = apply_to_pil(img, lt.get_profile("studio").adjustments)
    assert out.mode == mode and out.size == img.size
    if mode == "RGBA":
        assert np.array_equal(np.asarray(out.getchannel("A")), np.asarray(img.getchannel("A")))
    assert apply_to_pil(img, Adjustments()) is img


# --- pipeline & export -----------------------------------------------------------
def _process(path, profile, fmt="png", intensity=100, **kw):
    options = ProcessingOptions(
        scale=2,
        output_format=fmt,
        quality=95,
        existing="overwrite",
        lighting=LightingSettings(profile, intensity),
        **kw,
    )
    return process_image(path, options, NearestUpscaler())


def _pixels(path, mode="RGB"):
    with Image.open(path) as img:
        return np.asarray(img.convert(mode))


def test_export_png_contains_the_lighting(image_factory):
    path = image_factory("p.png", size=(48, 32))
    result = _process(path, "golden-hour")
    with Image.open(path) as src:
        expected = apply_lighting(
            np.asarray(src.convert("RGB")), lt.get_profile("golden-hour").adjustments
        )
    expected = np.repeat(np.repeat(expected, 2, 0), 2, 1)
    assert result.output_size == (96, 64)
    assert np.array_equal(_pixels(result.output), expected)


@pytest.mark.parametrize(("ext", "fmt"), [("jpg", "jpeg"), ("webp", "webp"), ("png", "jpeg")])
def test_export_lossy_formats_contain_the_lighting(image_factory, ext, fmt):
    path = image_factory(f"p.{ext}", size=(48, 32))
    plain = _pixels(_process(path, ORIGINAL, fmt).output).astype(float)
    lit = _pixels(_process(path, "bright-clean", fmt).output).astype(float)
    assert lit.shape == plain.shape == (64, 96, 3)
    assert luma(lit).mean() > luma(plain).mean() + 8


def test_export_rgba_keeps_alpha(image_factory):
    path = image_factory("a.png", size=(48, 32), mode="RGBA")
    plain = _pixels(_process(path, ORIGINAL).output, "RGBA")
    lit = _pixels(_process(path, "vivid").output, "RGBA")
    assert np.array_equal(lit[..., 3], plain[..., 3])
    assert not np.array_equal(lit[..., :3], plain[..., :3])


def test_export_grayscale_stays_grayscale(image_factory):
    path = image_factory("g.png", mode="L")
    assert not lt.get_profile("high-contrast").adjustments.changes_colour
    with Image.open(_process(path, "high-contrast").output) as out:
        assert out.mode == "L"


def test_export_grayscale_keeps_a_colour_shift(image_factory):
    """Golden Hour warms a grey image in the preview, so the file is warm too."""
    path = image_factory("g.png", mode="L")
    with Image.open(_process(path, "golden-hour").output) as out:
        assert out.mode == "RGB"
        rgb = np.asarray(out).astype(float)
    assert rgb[..., 0].mean() > rgb[..., 2].mean() + 5


def test_changed_lighting_is_not_skipped(image_factory):
    path = image_factory("p.png")

    def run(profile, intensity=100):
        options = ProcessingOptions(scale=2, lighting=LightingSettings(profile, intensity))
        return process_image(path, options, NearestUpscaler())

    plain = run(ORIGINAL)
    warm = run("golden-hour")
    half = run("golden-hour", 50)
    assert not warm.skipped and not half.skipped
    assert plain.output.name == "p_2x.png"
    assert warm.output.name == "p_2x_golden-hour.png" and warm.lighting == "golden-hour"
    assert half.output.name == "p_2x_golden-hour-50.png"
    again = run("golden-hour")
    assert again.skipped and again.output == warm.output


def test_lighting_tag():
    assert LightingSettings().tag() == ""
    assert LightingSettings("vivid", 0).tag() == ""
    assert LightingSettings("vivid").tag() == "vivid"
    assert LightingSettings("vivid", 30).tag() == "vivid-30"
    a = LightingSettings(CUSTOM, custom=Adjustments(contrast=10)).tag()
    b = LightingSettings(CUSTOM, 40, custom=Adjustments(contrast=10)).tag()
    c = LightingSettings(CUSTOM, custom=Adjustments(contrast=11)).tag()
    assert a == b and a != c and a.startswith("custom-")
    assert LightingSettings(CUSTOM).tag() == ""


def test_export_keeps_metadata(image_factory):
    from PIL import ImageCms

    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    path = image_factory("icc.jpg", icc_profile=icc, dpi=(72, 72))
    with Image.open(_process(path, "cinematic").output) as out:
        assert out.info.get("icc_profile") == icc
        assert round(out.info["dpi"][0]) == 144


@pytest.mark.parametrize("fmt", ["png", "jpeg", "webp"])
def test_original_profile_output_identical_to_default(image_factory, tmp_path, fmt):
    """Selecting Original must behave exactly as Pixelift did before lighting existed."""
    path = image_factory("same.jpg", size=(48, 32))
    default = ProcessingOptions(scale=2, output_format=fmt, output_dir=tmp_path / "a")
    original = ProcessingOptions(
        scale=2,
        output_format=fmt,
        output_dir=tmp_path / "b",
        lighting=LightingSettings(ORIGINAL, 100),
    )
    a = process_image(path, default, NearestUpscaler()).output
    b = process_image(path, original, NearestUpscaler()).output
    assert a.read_bytes() == b.read_bytes()


def test_default_processing_options_have_no_lighting():
    assert ProcessingOptions().lighting == LightingSettings()
    assert LightingSettings() == LightingSettings(ORIGINAL, 100)
    assert not ProcessingOptions().lighting.active


def test_progress_reports_lighting_stage(image_factory):
    stages = []
    path = image_factory("s.png")
    options = ProcessingOptions(scale=2, lighting=LightingSettings("vivid", 60))
    process_image(path, options, NearestUpscaler(), progress=lambda _f, s: stages.append(s))
    assert "Adjusting lighting" in stages
    stages.clear()
    process_image(
        path,
        ProcessingOptions(scale=2, existing="overwrite"),
        NearestUpscaler(),
        lambda _f, s: stages.append(s),
    )
    assert "Adjusting lighting" not in stages


def test_real_model_pipeline_with_lighting(image_factory, upscaler, tiny_specs):
    path = image_factory("m.png")
    options = ProcessingOptions(scale=4, model="test-x4", lighting=LightingSettings("studio", 80))
    result = process_image(path, options, upscaler)
    assert result.output_size == (160, 120)


# --- settings & CLI ---------------------------------------------------------------
def test_settings_default_is_original_full_intensity():
    s = Settings()
    assert (s.lighting_profile, s.lighting_intensity) == (ORIGINAL, 100)
    assert not s.lighting().active
    assert s.processing_options().lighting == LightingSettings()


def test_settings_lighting_roundtrip(tmp_path):
    path = tmp_path / "s.json"
    s = Settings(lighting_profile=CUSTOM, lighting_intensity=40, lighting_tint=-12)
    s.lighting_shadows = 33
    save_settings(s, path)
    loaded = load_settings(path)
    assert loaded == s
    lighting = loaded.processing_options().lighting
    assert lighting.profile == CUSTOM and lighting.intensity == 40
    assert lighting.custom == Adjustments(tint=-12, shadows=33)


def test_settings_lighting_invalid_values_normalised(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps(
            {"lighting_profile": "nope", "lighting_intensity": 900, "lighting_exposure": -400}
        )
    )
    s = load_settings(path)
    assert (s.lighting_profile, s.lighting_intensity, s.lighting_exposure) == (ORIGINAL, 100, -100)


def test_old_settings_files_get_lighting_defaults(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"scale": 2, "output_format": "webp"}))
    s = load_settings(path)
    assert s.scale == 2 and s.lighting_profile == ORIGINAL and not s.lighting().active


def test_cli_lighting_option(image_factory, tiny_specs, tmp_path, capsys):
    src = image_factory("photo.png")
    args = [str(src), "-s", "4", "-m", "test-x4", "--device", "cpu"]
    assert cli.run_cli([*args, "-o", str(tmp_path / "plain")]) == 0
    assert "· lighting" not in capsys.readouterr().out
    lit_args = [*args, "-o", str(tmp_path / "lit"), "--lighting", "bright-clean"]
    assert cli.run_cli([*lit_args, "--lighting-intensity", "100"]) == 0
    assert "lighting Bright & Clean 100%" in capsys.readouterr().out
    plain = _pixels(tmp_path / "plain" / "photo_4x.png").astype(float)
    lit = _pixels(tmp_path / "lit" / "photo_4x_bright-clean.png").astype(float)
    assert luma(lit).mean() > luma(plain).mean() + 5


@pytest.mark.parametrize(
    "extra",
    [
        ["--lighting-intensity", "250"],
        ["--lighting-intensity", "-1"],
        ["--lighting-intensity", "x"],
        ["--lighting", "custom", "--lighting-intensity", "50"],
    ],
)
def test_cli_rejects_bad_lighting_intensity(image_factory, extra, capsys):
    src = image_factory("photo.png")
    with pytest.raises(SystemExit) as exc:
        cli.run_cli([str(src), "-l", "vivid", *extra])
    assert exc.value.code == 2
    assert "lighting-intensity" in capsys.readouterr().err


# --- extensibility ----------------------------------------------------------------
def test_new_profiles_plug_in_without_other_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(lt, "_PROFILES", dict(lt._PROFILES))
    lt.register_profile(
        LightingProfile(
            "moody", "Moody", "Dark and desaturated", Adjustments(exposure=-20, saturation=-30)
        )
    )
    ids = [p.id for p in lt.all_profiles()]
    assert ids[-2:] == ["moody", CUSTOM]  # Custom stays last
    assert LightingSettings("moody", 50).adjustments().saturation == -15
    path = tmp_path / "s.json"
    save_settings(Settings(lighting_profile="moody"), path)
    assert load_settings(path).lighting_profile == "moody"


def test_adjustments_helpers():
    adj = Adjustments(exposure=40, tint=-200)
    assert adj.scaled(0.5) == Adjustments(exposure=20, tint=-100)
    assert adj.clamped().tint == -100
    assert Adjustments().is_neutral and not adj.is_neutral
    assert tuple(f.name for f in dataclasses.fields(Adjustments)) == ADJUSTMENT_NAMES
