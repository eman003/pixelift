from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import make_image
from PIL import Image

from pixelift import cli
from pixelift.core import hdr, upscale_cache
from pixelift.core.control import JobControl
from pixelift.core.errors import CancelledError
from pixelift.core.hdr import HdrRecipe, HdrSettings, analyse, apply_hdr
from pixelift.core.image_processor import ProcessingOptions, process_image
from pixelift.core.lighting import LightingSettings
from pixelift.core.upscaler import Upscaler
from pixelift.storage.settings import Settings, load_settings, save_settings

PROFILES = ["natural", "balanced", "dramatic"]
LUMA = np.array([0.2126, 0.7152, 0.0722])


class CountingUpscaler(Upscaler):
    """Deterministic stand-in for the AI model that counts its runs."""

    def __init__(self, identity=("counting",)):
        self.calls = 0
        self.identity = identity

    def upscale(self, image, scale, model, *, progress=None, control=None):
        self.calls += 1
        return np.repeat(np.repeat(image, scale, 0), scale, 1)

    def cache_identity(self, model, scale, size=None, wide=False):
        return self.identity


def scene(h: int = 120, w: int = 160, seed: int = 0) -> np.ndarray:
    """Dark foreground, bright sky, a skin-like patch, saturated colour and noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w] / np.array([h - 1, w - 1])[:, None, None]
    sky = np.stack([0.55 + 0.3 * y, 0.7 + 0.2 * y, 0.95 + 0 * y], -1)
    ground = np.stack([0.06 + 0.1 * x, 0.05 + 0.08 * x, 0.04 + 0 * x], -1)
    rgb = np.where((y > 0.5)[..., None], ground, sky)
    rgb[: h // 5, : w // 5] = (0.88, 0.67, 0.55)  # skin
    rgb[-h // 5 :, -w // 5 :] = (0.85, 0.25, 0.05)  # saturated orange
    rgb = rgb * 255 + rng.normal(0, 2, rgb.shape)
    return rgb.clip(0, 255).astype(np.uint8)


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float64) @ LUMA


def settings(profile: str = "balanced", intensity: int | None = None, **advanced) -> HdrSettings:
    if intensity is None:
        intensity = hdr.get_profile(profile).default_intensity
    return HdrSettings(True, profile, intensity, **advanced)


# --- profiles ------------------------------------------------------------------------
def test_profiles_registered_with_sensible_defaults():
    assert [p.id for p in hdr.all_profiles()] == PROFILES
    assert [p.name for p in hdr.all_profiles()] == ["Natural HDR", "Balanced HDR", "Dramatic HDR"]
    defaults = [p.default_intensity for p in hdr.all_profiles()]
    assert all(40 <= d <= 80 for d in defaults)
    # Default settings: off, Natural at its own intensity.
    default = HdrSettings()
    assert not default.enabled and not default.active and default.tag() == ""
    assert default.profile == "natural" and default.intensity == defaults[0]


@pytest.mark.parametrize("profile", PROFILES)
def test_each_profile_recovers_shadows_and_protects_highlights(profile):
    rgb = scene()
    out = apply_hdr(rgb, settings(profile).recipe())
    before, after = luma(rgb), luma(out)
    dark, bright = before < 40, before > 200
    assert after[dark].mean() > before[dark].mean() + 3  # shadows lifted
    assert after[bright].mean() < before[bright].mean()  # highlights compressed
    # Nothing new clipped or crushed.
    assert not ((out >= 255).any(-1) & ~(rgb >= 255).any(-1)).any()
    assert not ((after <= 1) & (before > 3)).any()


def test_profiles_are_ordered_by_strength():
    rgb = scene()
    changes = [
        np.abs(apply_hdr(rgb, settings(p, 100).recipe()).astype(int) - rgb).mean() for p in PROFILES
    ]
    assert changes[0] < changes[1] < changes[2]


@pytest.mark.parametrize("profile", PROFILES)
def test_profiles_keep_skin_tones_natural(profile):
    rgb = scene()
    out = apply_hdr(rgb, settings(profile).recipe())
    skin_in = rgb[:20, :28].reshape(-1, 3).astype(float).mean(0)
    skin_out = out[:20, :28].reshape(-1, 3).astype(float).mean(0)
    # Same hue order and only a small change in the colour balance.
    assert skin_out[0] > skin_out[1] > skin_out[2]
    ratio_in = (skin_in[0] - skin_in[2]) / skin_in.mean()
    ratio_out = (skin_out[0] - skin_out[2]) / skin_out.mean()
    assert abs(ratio_out - ratio_in) < 0.12


def test_does_not_amplify_noise_in_lifted_shadows():
    flat = np.full((200, 200, 3), 20, np.uint8)
    noisy = (flat + np.random.default_rng(3).normal(0, 3, flat.shape)).clip(0, 255)
    noisy = noisy.astype(np.uint8)
    out = apply_hdr(noisy, settings("dramatic", 100).recipe())
    assert luma(out).mean() > luma(noisy).mean() + 5  # lifted
    assert luma(out).std() < luma(noisy).std() * 1.3  # noise kept near its strength


def test_no_banding_in_smooth_gradient():
    ramp = np.tile(np.linspace(40, 230, 512)[None, :, None], (64, 1, 3)).astype(np.uint8)
    out = apply_hdr(ramp, settings("dramatic", 100).recipe())
    row = luma(out)[32]
    assert np.abs(np.diff(row)).max() <= 4  # no steps beyond dither size
    # Still increasing over any 16-pixel span (monotonic tone curve).
    assert (row[16:] - row[:-16] > -2).all()


def test_no_halo_at_strong_edge():
    """A dark object against a bright sky: no bright rim or dark ring at the edge."""
    rgb = np.full((200, 200, 3), 235, np.uint8)
    rgb[:, 100:] = 25
    out = luma(apply_hdr(rgb, settings("dramatic", 100).recipe()))
    bright_side, dark_side = out[:, :95], out[:, 105:]
    assert bright_side.max() - bright_side.min() < 6
    assert dark_side.max() - dark_side.min() < 6


def test_grey_stays_grey():
    rgb = np.repeat(luma(scene()).astype(np.uint8)[..., None], 3, -1)
    out = apply_hdr(rgb, settings("dramatic", 100).recipe())
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])


def test_deterministic():
    rgb = scene()
    recipe = settings().recipe()
    assert np.array_equal(apply_hdr(rgb, recipe), apply_hdr(rgb, recipe))


# --- intensity and settings ---------------------------------------------------------------
@pytest.mark.parametrize("profile", PROFILES)
def test_zero_intensity_returns_the_input_untouched(profile):
    rgb = scene()
    recipe = settings(profile, 0).recipe()
    assert recipe.is_neutral
    assert apply_hdr(rgb, recipe) is rgb
    out = np.zeros_like(rgb)
    assert apply_hdr(rgb, recipe, out=out) is out and np.array_equal(out, rgb)
    assert settings(profile, 0).tag() == "" and not settings(profile, 0).active


def test_disabled_returns_the_input_untouched():
    rgb = scene()
    off = HdrSettings(False, "dramatic", 100)
    assert off.recipe().is_neutral and apply_hdr(rgb, off.recipe()) is rgb


def test_intensity_scales_the_effect_and_is_clamped():
    rgb = scene()

    def change(intensity):
        return np.abs(apply_hdr(rgb, settings("balanced", intensity).recipe()) - rgb.astype(int))

    small, half, full = change(5).mean(), change(50).mean(), change(100).mean()
    assert small < half < full
    assert small < 2  # a small intensity is a small change: no jump from 0 %
    assert settings("balanced", 250).recipe() == settings("balanced", 100).recipe()
    assert settings("balanced", -5).recipe().is_neutral


def test_advanced_adjustments():
    base = settings("balanced").recipe()
    more = settings("balanced", shadows=100, local_contrast=50).recipe()
    off = settings("balanced", highlights=-100).recipe()
    assert more.shadows == pytest.approx(min(1.5, base.shadows * 2))
    assert more.local_contrast == pytest.approx(base.local_contrast * 1.5)
    assert off.highlights == 0
    assert settings("balanced", saturation=-100).recipe().vibrance == 0
    # Out-of-range values are clamped; unknown profiles do nothing.
    assert settings("balanced", shadows=500).recipe() == settings("balanced", shadows=100).recipe()
    assert HdrSettings(True, "nope", 80).recipe().is_neutral


def test_recipe_clamps_hostile_values():
    recipe = HdrRecipe(shadows=float("nan"), highlights=99, local_contrast=-3, colour=7).clamped()
    assert recipe.shadows == 0 and recipe.highlights == 1.5 and recipe.local_contrast == 0
    assert recipe.colour == 1


def test_tags_identify_effective_settings():
    assert settings("natural", 60).tag() == "hdr-natural-60"
    assert settings("dramatic", 100).tag() == "hdr-dramatic-100"
    tuned = settings("natural", 60, shadows=20).tag()
    assert tuned.startswith("hdr-natural-60-") and tuned != settings("natural", 60).tag()
    assert tuned != settings("natural", 60, shadows=30).tag()


# --- input validation -------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((4, 4), np.uint8),
        np.zeros((4, 4, 4), np.uint8),
        np.zeros((4, 4, 3), np.float32),
        np.zeros((0, 4, 3), np.uint8),
        [[1, 2, 3]],
    ],
)
def test_rejects_invalid_input(bad):
    with pytest.raises(ValueError):
        apply_hdr(bad, settings().recipe())


def test_rejects_mismatched_out():
    with pytest.raises(ValueError):
        apply_hdr(scene(), settings().recipe(), out=np.zeros((5, 5, 3), np.uint8))


def test_tiny_images():
    for shape in ((1, 1, 3), (1, 300, 3), (300, 1, 3), (2, 2, 3)):
        rgb = np.full(shape, 60, np.uint8)
        assert apply_hdr(rgb, settings("dramatic", 100).recipe()).shape == shape


# --- alpha ----------------------------------------------------------------------------
def test_preview_keeps_alpha_untouched():
    """The preview path: camera_looks.apply_to_pil with HDR between grade and finish."""
    from pixelift.core import camera_looks as cl

    rgb = scene()
    alpha = np.tile(np.linspace(0, 255, rgb.shape[1]).astype(np.uint8), (rgb.shape[0], 1))
    img = Image.fromarray(rgb)
    img.putalpha(Image.fromarray(alpha))
    recipe = settings().recipe()
    out = cl.apply_to_pil(img, cl.LookRecipe(), between=lambda a: apply_hdr(a, recipe, out=a))
    assert out.mode == "RGBA"
    assert np.array_equal(np.asarray(out)[..., 3], alpha)
    assert np.array_equal(np.asarray(out)[..., :3], apply_hdr(rgb, recipe))


# --- banded processing, crops and large images ------------------------------------------------
def test_in_place_matches_a_new_array_across_bands(monkeypatch):
    monkeypatch.setattr(hdr, "_CHUNK_PIXELS", 160 * 7)  # many small bands
    rgb = scene()
    expected = apply_hdr(rgb, settings().recipe())
    work = rgb.copy()
    assert apply_hdr(work, settings().recipe(), out=work) is work
    assert np.array_equal(work, expected)


def test_band_size_does_not_change_the_result(monkeypatch):
    rgb = scene()
    whole = apply_hdr(rgb, settings().recipe())
    monkeypatch.setattr(hdr, "_CHUNK_PIXELS", 160 * 9)
    assert np.array_equal(apply_hdr(rgb, settings().recipe()), whole)


def test_read_only_input_with_separate_output():
    rgb = scene()
    rgb.flags.writeable = False
    out = apply_hdr(rgb, settings().recipe(), out=np.empty_like(rgb))
    assert out.shape == rgb.shape


def test_crop_with_the_whole_pictures_plan_matches_the_whole():
    """The zoomed-in preview: a crop rendered with the whole picture's plan."""
    rgb = scene(240, 320)
    plan = analyse(rgb)
    recipe = settings().recipe()
    whole = apply_hdr(rgb, recipe, plan=plan).astype(int)
    crop = rgb[60:180, 80:240]
    part = apply_hdr(crop, recipe, plan=plan, frame=(320, 240), region=(0.25, 0.25, 0.75, 0.75))
    inner = np.abs(part.astype(int) - whole[60:180, 80:240])[8:-8, 8:-8]
    assert inner.mean() < 1.5


def test_preview_size_matches_export_size():
    """Radii are fractions of the picture: a downscaled render looks the same."""
    rgb = scene(240, 320)
    big = np.asarray(Image.fromarray(rgb).resize((1280, 960), Image.Resampling.LANCZOS))
    recipe = settings().recipe()
    small_out = apply_hdr(rgb, recipe).astype(float)
    big_out = apply_hdr(big, recipe)
    big_down = np.asarray(Image.fromarray(big_out).resize((320, 240), Image.Resampling.BOX))
    assert np.abs(big_down.astype(float) - small_out).mean() < 3


def test_large_image_is_processed_in_bounded_memory():
    """The working memory is a few bands of rows: it does not grow with the image."""
    import tracemalloc

    def peak(height):
        rgb = np.zeros((height, 3200, 3), np.uint8)
        rgb[..., 0] = np.linspace(0, 255, 3200, dtype=np.uint8)
        rgb[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
        tracemalloc.start()
        apply_hdr(rgb, settings().recipe(), out=rgb)
        _current, top = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return top

    small, large = peak(1200), peak(4800)  # 3.8 MP and 15.4 MP
    assert large < small * 1.2
    assert large < 15.4e6 * 3 * 4 / 2  # well below one float copy of the image


def test_check_is_called_between_bands_and_can_cancel(monkeypatch):
    monkeypatch.setattr(hdr, "_CHUNK_PIXELS", 160 * 20)
    control = JobControl()
    calls = []

    def check():
        calls.append(1)
        if len(calls) == 3:
            control.cancel()
        control.check()

    with pytest.raises(CancelledError):
        apply_hdr(scene(), settings().recipe(), check=check)
    assert len(calls) == 3


# --- pipeline: upscaling, cache, export ----------------------------------------------------
def _options(hdr_settings=None, **kwargs) -> ProcessingOptions:
    return ProcessingOptions(scale=2, hdr=hdr_settings or HdrSettings(), **kwargs)


def _pixels(path, mode="RGB"):
    with Image.open(path) as img:
        return np.asarray(img.convert(mode))


def test_hdr_runs_after_upscaling(image_factory):
    path = image_factory("p.png", size=(64, 48))
    upscaler = CountingUpscaler()
    result = process_image(path, _options(settings()), upscaler)
    with Image.open(path) as src:
        upscaled = np.repeat(np.repeat(np.asarray(src.convert("RGB")), 2, 0), 2, 1)
    expected = apply_hdr(upscaled, settings().recipe())
    assert result.output_size == (128, 96)
    assert result.output.name == "p_2x_hdr-balanced-65.png" and result.hdr == "hdr-balanced-65"
    assert np.array_equal(_pixels(result.output), expected)


def test_hdr_off_output_unchanged(image_factory):
    path = image_factory("p.png")
    upscaler = CountingUpscaler()
    result = process_image(path, _options(HdrSettings(True, "dramatic", 0)), upscaler)
    with Image.open(path) as src:
        upscaled = np.repeat(np.repeat(np.asarray(src.convert("RGB")), 2, 0), 2, 1)
    assert result.output.name == "p_2x.png" and result.hdr == ""
    assert np.array_equal(_pixels(result.output), upscaled)
    assert upscale_cache.total_bytes() == 0  # nothing cached without HDR


def test_changing_hdr_reuses_the_cached_upscale(image_factory):
    path = image_factory("p.png", size=(64, 48))
    upscaler = CountingUpscaler()
    first = process_image(path, _options(settings("natural")), upscaler)
    assert upscaler.calls == 1 and upscale_cache.total_bytes() > 0
    stages = []
    second = process_image(
        path, _options(settings("dramatic", 90)), upscaler, progress=lambda f, s: stages.append(s)
    )
    third = process_image(path, _options(settings("natural", 30, shadows=40)), upscaler)
    assert upscaler.calls == 1  # the AI model ran once
    assert "Using the cached upscale" in stages and "Enhancing dynamic range" in stages
    assert len({first.output, second.output, third.output}) == 3
    # Switching HDR off again gives the plain upscale, also from the cache.
    plain = process_image(path, _options(), upscaler)
    assert upscaler.calls == 1 and plain.output.name == "p_2x.png"
    with Image.open(path) as src:
        upscaled = np.repeat(np.repeat(np.asarray(src.convert("RGB")), 2, 0), 2, 1)
    assert np.array_equal(_pixels(plain.output), upscaled)
    # The cached result is exactly what a fresh upscale gives.
    fresh = CountingUpscaler()
    upscale_cache.clear()
    again = process_image(path, _options(settings("dramatic", 90), existing="overwrite"), fresh)
    assert fresh.calls == 1 and np.array_equal(_pixels(again.output), _pixels(second.output))


def test_cache_misses_when_upscale_inputs_change(image_factory):
    path = image_factory("p.png", size=(64, 48))
    upscaler = CountingUpscaler()
    process_image(path, _options(settings()), upscaler)
    process_image(path, _options(settings(), lighting=LightingSettings("vivid")), upscaler)
    assert upscaler.calls == 2
    options = _options(settings())
    options.scale = 4
    process_image(path, options, upscaler)
    assert upscaler.calls == 3
    make_image(path, size=(64, 48), quality=50)  # the source file changed
    path.write_bytes(path.read_bytes())
    import os

    os.utime(path, ns=(1, 1))
    process_image(path, _options(settings(), existing="overwrite"), upscaler)
    assert upscaler.calls == 4


def test_damaged_cache_entry_is_ignored(image_factory):
    path = image_factory("p.png", size=(64, 48))
    upscaler = CountingUpscaler()
    process_image(path, _options(settings()), upscaler)
    for entry in upscale_cache.cache_dir().glob("*.npy"):
        entry.write_bytes(b"not an array")
    result = process_image(path, _options(settings("dramatic")), upscaler)
    assert upscaler.calls == 2 and result.output.exists()


def test_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(upscale_cache, "MAX_BYTES", 3000)
    arrays = [np.full((20, 20, 3), i, np.uint8) for i in range(4)]  # 1.2 kB each
    for i, array in enumerate(arrays):
        assert upscale_cache.store(f"k{i}", array)
    assert upscale_cache.total_bytes() <= 3000
    assert upscale_cache.load("k0", (20, 20, 3)) is None  # oldest evicted
    assert np.array_equal(upscale_cache.load("k3", (20, 20, 3)), arrays[3])
    assert upscale_cache.load("k3", (10, 10, 3)) is None  # wrong size: discarded
    assert not upscale_cache.store("huge", np.zeros((40, 40, 3), np.uint8))


def test_cached_upscale_is_read_only_and_original_untouched(image_factory):
    path = image_factory("p.png", size=(64, 48))
    before = path.read_bytes()
    upscaler = CountingUpscaler()
    process_image(path, _options(settings()), upscaler)
    process_image(path, _options(settings("dramatic")), upscaler)
    assert path.read_bytes() == before
    (entry,) = upscale_cache.cache_dir().glob("*.npy")
    cached = np.load(entry, mmap_mode="r")
    assert not cached.flags.writeable


def test_cancel_during_hdr(image_factory):
    path = image_factory("p.png", size=(64, 48))
    control = JobControl()

    def progress(_fraction, stage):
        if stage == "Enhancing dynamic range":
            control.cancel()

    with pytest.raises(CancelledError):
        process_image(path, _options(settings()), CountingUpscaler(), progress, control)
    assert not list(path.parent.parent.rglob("*_hdr-*"))  # nothing half-written


def test_restoration_ignores_hdr():
    options = _options(settings())
    from pixelift.core.image_processor import hdr_tag
    from pixelift.core.restoration.settings import RestorationSettings

    assert hdr_tag(options) == "hdr-balanced-65"
    options.restoration = RestorationSettings()
    assert hdr_tag(options) == "" and not options.effective_hdr().active


@pytest.mark.parametrize(("fmt", "ext"), [("png", ".png"), ("jpeg", ".jpg"), ("webp", ".webp")])
def test_export_formats(image_factory, fmt, ext):
    path = image_factory("p.png", size=(64, 48))
    plain = process_image(path, _options(output_format=fmt), CountingUpscaler())
    enhanced = process_image(
        path, _options(settings("dramatic"), output_format=fmt), CountingUpscaler()
    )
    assert enhanced.output.suffix == ext
    with Image.open(enhanced.output) as img:
        assert img.mode == "RGB" and img.size == (128, 96)
        # An ordinary SDR file: 8-bit, no HDR transfer-function markers.
        assert not any("hdr" in str(k).lower() or k == "cicp" for k in img.info)
    assert not np.array_equal(_pixels(enhanced.output), _pixels(plain.output))


def test_export_keeps_alpha(image_factory):
    path = image_factory("a.png", size=(64, 48), mode="RGBA")
    plain = _pixels(process_image(path, _options(), CountingUpscaler()).output, "RGBA")
    out = _pixels(process_image(path, _options(settings()), CountingUpscaler()).output, "RGBA")
    assert np.array_equal(out[..., 3], plain[..., 3])
    assert not np.array_equal(out[..., :3], plain[..., :3])


def test_export_grayscale_stays_grayscale(image_factory):
    path = image_factory("g.png", mode="L", size=(64, 48))
    with Image.open(process_image(path, _options(settings()), CountingUpscaler()).output) as out:
        assert out.mode == "L"


def test_export_keeps_metadata(image_factory):
    exif = Image.Exif()
    exif[0x010F] = "TestCam"
    path = image_factory("m.jpg", size=(64, 48), exif=exif.tobytes())
    result = process_image(path, _options(settings(), output_format="jpeg"), CountingUpscaler())
    with Image.open(result.output) as img:
        assert img.getexif().get(0x010F) == "TestCam"


def test_upscalers_that_cannot_identify_their_output_are_not_cached(image_factory):
    path = image_factory("p.png", size=(64, 48))
    upscaler = CountingUpscaler(identity=None)
    process_image(path, _options(settings()), upscaler)
    process_image(path, _options(settings("dramatic")), upscaler)
    assert upscaler.calls == 2 and upscale_cache.total_bytes() == 0


def test_cache_key_follows_the_upscalers_identity(image_factory):
    """Other weights, tile size or device: the cached upscale is not reused."""
    path = image_factory("p.png", size=(64, 48))
    upscaler = CountingUpscaler(identity=("weights-a", 0))
    process_image(path, _options(settings()), upscaler)
    upscaler.identity = ("weights-b", 0)  # e.g. the model file was replaced
    process_image(path, _options(settings("dramatic")), upscaler)
    upscaler.identity = ("weights-b", 512)  # another tile size
    process_image(path, _options(settings("natural")), upscaler)
    assert upscaler.calls == 3


def test_photo_edited_during_the_upscale_is_not_cached_under_its_new_identity(image_factory):
    path = image_factory("p.png", size=(64, 48))

    class EditingUpscaler(CountingUpscaler):
        def upscale(self, image, scale, model, *, progress=None, control=None):
            if self.calls == 0:  # the user saves an edit while the model runs
                Image.fromarray(scene(48, 64, seed=7)).save(path)
            return super().upscale(image, scale, model, progress=progress, control=control)

    upscaler = EditingUpscaler()
    process_image(path, _options(settings()), upscaler)
    process_image(path, _options(settings("dramatic")), upscaler)
    assert upscaler.calls == 2  # the edited photo is upscaled, not the stale cache


def test_torch_upscaler_identity(upscaler, tiny_specs):
    # Automatic tiling: the tile size depends on the input and the free memory.
    assert upscaler.cache_identity("test-x4", 4) is None
    identity = upscaler.cache_identity("test-x4", 4, (30, 40))
    assert tiny_specs["x4"].sha256 in identity and "cpu" in identity
    upscaler.tile_size = 256
    assert upscaler.cache_identity("test-x4", 4, (30, 40)) != identity
    assert upscaler.cache_identity("no-such-model", 4, (30, 40)) is None


def test_torch_upscaler_identity_uses_the_tile_actually_used(upscaler, tiny_specs):
    image = np.zeros((40, 60, 3), np.uint8)
    upscaler.tile_size = 0
    upscaler.choose_tile_size = lambda *a, **k: 32  # e.g. little free memory
    upscaler.upscale(image, 4, "test-x4")
    used = upscaler.cache_identity("test-x4", 4, (40, 60))
    upscaler.choose_tile_size = lambda *a, **k: 512  # more memory later
    assert upscaler.cache_identity("test-x4", 4, (40, 60)) == used  # what this run used
    assert 32 in used
    assert upscaler.cache_identity("test-x4", 4, (41, 60)) != used  # another input: planned
    assert 512 in upscaler.cache_identity("test-x4", 4, (41, 60))


def test_cache_upscale_off_reads_but_does_not_write(image_factory):
    path = image_factory("p.png", size=(64, 48))
    upscaler = CountingUpscaler()
    process_image(path, _options(settings(), cache_upscale=False), upscaler)
    assert upscale_cache.total_bytes() == 0
    process_image(path, _options(settings("natural")), upscaler)  # caches
    process_image(path, _options(settings("dramatic"), cache_upscale=False), upscaler)
    assert upscaler.calls == 2  # the last run read the cache


def test_progress_never_goes_backwards(image_factory):
    path = image_factory("p.png", size=(64, 48))
    for cache in (True, False):
        fractions = []
        process_image(
            path,
            _options(settings(), cache_upscale=cache, existing="overwrite"),
            CountingUpscaler(),
            progress=lambda f, _s, out=fractions: out.append(f),
        )
        upscale_cache.clear()
        assert fractions == sorted(fractions)


def test_interrupted_writes_are_cleaned_up_and_counted():
    import os
    import time

    folder = upscale_cache.cache_dir()
    folder.mkdir(parents=True)
    stale = folder / ".tmp-crashed.npy"
    stale.write_bytes(b"x" * 1000)
    old = time.time() - 7 * 3600
    os.utime(stale, (old, old))
    fresh = folder / ".tmp-writing.npy"  # another job's write in progress
    fresh.write_bytes(b"x" * 500)
    assert upscale_cache.total_bytes() == 1500
    assert upscale_cache.store("k", np.zeros((4, 4, 3), np.uint8))
    assert not stale.exists() and fresh.exists()
    upscale_cache.clear()
    assert upscale_cache.total_bytes() == 0 and not fresh.exists()


# --- settings and CLI -------------------------------------------------------------------
def test_settings_roundtrip_and_normalise(tmp_path):
    path = tmp_path / "s.json"
    s = Settings(hdr_enabled=True, hdr_profile="dramatic", hdr_intensity=40, hdr_shadows=-20)
    save_settings(s, path)
    loaded = load_settings(path)
    assert loaded.hdr() == HdrSettings(True, "dramatic", 40, shadows=-20)
    assert loaded.processing_options().hdr == loaded.hdr()
    loaded.mode = "restore"
    assert not loaded.processing_options().effective_hdr().active
    path.write_text(
        json.dumps({"hdr_profile": "nope", "hdr_intensity": 400, "hdr_saturation": -900})
    )
    bad = load_settings(path)
    assert (bad.hdr_profile, bad.hdr_intensity, bad.hdr_saturation) == ("natural", 100, -100)
    assert not bad.hdr_enabled


def test_old_settings_files_get_hdr_defaults(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"scale": 2}))
    assert load_settings(path).hdr() == HdrSettings()


def test_cli_hdr_options():
    parser = cli.build_parser()
    args = parser.parse_args(["x.png", "--hdr", "dramatic"])
    assert args.hdr == "dramatic" and args.hdr_intensity is None
    args = parser.parse_args(["x.png", "--hdr", "natural", "--hdr-intensity", "35"])
    assert args.hdr_intensity == 35
    assert parser.parse_args(["x.png"]).hdr == "off"
    with pytest.raises(SystemExit):
        parser.parse_args(["x.png", "--hdr", "extreme"])
    with pytest.raises(SystemExit):
        parser.parse_args(["x.png", "--hdr-intensity", "120"])


def test_cli_hdr_intensity_needs_a_profile(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.run_cli(["x.png", "--hdr-intensity", "50"])
    assert exc.value.code == 2 and "--hdr-intensity needs" in capsys.readouterr().err


def test_cli_uses_other_profiles_as_designed(image_factory, tiny_specs, tmp_path):
    """The app's own profile keeps its Advanced tuning; another one does not."""
    save_settings(Settings(hdr_enabled=True, hdr_profile="natural", hdr_local_contrast=100))
    src = image_factory("photo.png")
    out_dir = tmp_path / "out"
    args = [str(src), "-s", "4", "-m", "test-x4", "-o", str(out_dir), "--device", "cpu"]
    assert cli.run_cli([*args, "--hdr", "dramatic"]) == 0
    assert (out_dir / "photo_4x_hdr-dramatic-70.png").exists()  # no Advanced hash
    assert cli.run_cli(args) == 0  # the app's profile, with its tuning
    tuned = [p.name for p in out_dir.iterdir() if p.name.startswith("photo_4x_hdr-natural-60-")]
    assert len(tuned) == 1
    assert upscale_cache.total_bytes() == 0  # the CLI does not fill the cache


def test_cli_upscale_with_hdr(image_factory, tiny_specs, tmp_path, capsys):
    src = image_factory("photo.png")
    out_dir = tmp_path / "out"
    args = [str(src), "-s", "4", "-m", "test-x4", "-o", str(out_dir), "--device", "cpu"]
    assert cli.run_cli([*args, "--hdr", "balanced"]) == 0
    assert (out_dir / "photo_4x_hdr-balanced-65.png").exists()
    assert "Balanced HDR 65%" in capsys.readouterr().out
    assert cli.run_cli([*args, "--hdr", "natural", "--hdr-intensity", "30"]) == 0
    assert (out_dir / "photo_4x_hdr-natural-30.png").exists()
    assert cli.run_cli([*args, "--hdr", "off"]) == 0
    assert (out_dir / "photo_4x.png").exists()


def test_torch_upscaler_identity_shared_across_threads_and_oom_retries(upscaler, tiny_specs):
    import threading

    image = np.zeros((40, 60, 3), np.uint8)
    run_tiled = upscaler._run_tiled

    def oom_above_128(spec, device, img, out, scale, tile, progress, control):
        if tile > 128:
            raise RuntimeError("CUDA out of memory")
        run_tiled(spec, device, img, out, scale, tile, progress, control)

    upscaler._run_tiled = oom_above_128
    for configured in (0, 256):  # automatic, then a fixed size retried smaller
        upscaler.tile_size = configured
        upscaler.choose_tile_size = lambda *a, **k: 256
        worker = threading.Thread(target=upscaler.upscale, args=(image, 4, "test-x4"))
        worker.start()  # a batch's worker thread, gone before the next lookup
        worker.join()
        identity = upscaler.cache_identity("test-x4", 4, (40, 60))
        assert 256 not in identity and 128 in identity
