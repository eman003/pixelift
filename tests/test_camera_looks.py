"""Camera looks: processing, profiles, intensity, pipelines, export, settings and CLI."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time

import numpy as np
import pytest
from PIL import Image

from pixelift import cli
from pixelift.core import camera_looks as cl
from pixelift.core import lighting as lt
from pixelift.core.batch_processor import BatchProcessor, ItemStatus, QueueItem
from pixelift.core.camera_looks import (
    CUSTOM,
    ORIGINAL,
    Band,
    CameraLook,
    CameraLookSettings,
    LookRecipe,
    Toning,
    apply_look,
    apply_to_pil,
    custom_recipe,
    finish_look,
)
from pixelift.core.image_processor import ProcessingOptions, process_image
from pixelift.core.lighting import Adjustments, LightingSettings, apply_lighting
from pixelift.core.restoration import settings as rs
from pixelift.core.restoration.pipeline import Restorer
from pixelift.core.upscaler import Upscaler
from pixelift.storage.settings import Settings, load_settings, save_settings

REQUIRED = {
    "sony": ["Sony Natural", "Sony Vivid", "Sony Portrait", "Sony Cinematic"],
    "canon": ["Canon Natural", "Canon Standard", "Canon Portrait", "Canon Landscape"],
    "nikon": ["Nikon Neutral", "Nikon Standard", "Nikon Portrait", "Nikon Landscape"],
    "fujifilm": [
        "Fujifilm Classic",
        "Fujifilm Provia",
        "Fujifilm Velvia",
        "Fujifilm Astia",
        "Fujifilm Classic Negative",
    ],
    "leica": ["Leica Natural", "Leica Monochrome", "Leica Filmic"],
    "hasselblad": ["Hasselblad Natural", "Hasselblad Filmic"],
    "film": [
        "Kodak-inspired",
        "Portra-inspired",
        "Ektar-inspired",
        "Classic Film",
        "Modern Film",
        "Cinematic",
    ],
    "monochrome": ["Black & White"],
}
LOOKS = [look.id for look in cl.all_looks() if look.id not in (ORIGINAL, CUSTOM)]
DIGITAL = [
    look.id
    for look in cl.all_looks()
    if look.category in ("sony", "canon", "nikon", "hasselblad") and "filmic" not in look.id
]
PORTRAIT_LOOKS = ["canon-portrait", "sony-portrait", "nikon-portrait", "fujifilm-astia"]


class NearestUpscaler(Upscaler):
    """Deterministic stand-in for the AI model, so outputs can be predicted."""

    def upscale(self, image, scale, model, *, progress=None, control=None):
        return np.repeat(np.repeat(image, scale, 0), scale, 1)


# --- test images ------------------------------------------------------------------
SKIN = (214, 160, 128)


def landscape(w: int = 192, h: int = 128) -> np.ndarray:
    """Sky, foliage and a grey ramp, with a little noise: a stand-in landscape."""
    rgb = np.array(cl.sample_image(w, h), dtype=np.float32)
    rgb += np.random.default_rng(0).normal(0, 3, rgb.shape)
    return rgb.clip(0, 255).astype(np.uint8)


def portrait(w: int = 128, h: int = 160) -> np.ndarray:
    """A shaded skin-coloured face on a neutral-grey background."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    rgb = np.full((h, w, 3), 128, np.float32)
    face = ((x - w / 2) / (w * 0.3)) ** 2 + ((y - h / 2) / (h * 0.35)) ** 2 < 1
    shade = (0.75 + 0.25 * (1 - y / h))[..., None]
    rgb = np.where(face[..., None], np.array(SKIN, np.float32) * shade, rgb)
    rgb += np.random.default_rng(1).normal(0, 2, rgb.shape)
    return rgb.clip(0, 255).astype(np.uint8), face


def full(look_id: str, intensity: int = 100, grain: str = cl.GRAIN_AUTO) -> LookRecipe:
    return CameraLookSettings(look_id, intensity, grain).recipe()


def render(rgb: np.ndarray, recipe: LookRecipe) -> np.ndarray:
    return finish_look(apply_look(rgb, recipe), recipe)


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def chroma(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32)
    return x.max(-1) - x.min(-1)


def diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


# --- registry ---------------------------------------------------------------------
def test_all_required_looks_are_registered_and_grouped():
    names = {look.name: look for look in cl.all_looks()}
    for category, wanted in REQUIRED.items():
        for name in wanted:
            assert name in names, name
            assert names[name].category == category, name
    ids = [look.id for look in cl.all_looks()]
    assert ids[0] == ORIGINAL and ids[-1] == CUSTOM
    assert len(set(ids)) == len(ids)


def test_brand_looks_are_labelled_as_inspired():
    for look in cl.all_looks():
        category = next((c for c in cl.CATEGORIES if c.id == look.category), None)
        if category is not None and category.brand:
            assert look.category_label == f"{category.name}-inspired"
    assert "not official" in cl.DISCLAIMER and "endorsement" in cl.DISCLAIMER


def test_category_filter_lists_monochrome_looks_from_brands():
    mono = [look.id for look in cl.all_looks() if look.in_category("monochrome")]
    assert set(mono) == {"leica-monochrome", "black-white"}
    assert [look.id for look in cl.all_looks() if look.in_category("leica")] == [
        "leica-natural",
        "leica-monochrome",
        "leica-filmic",
    ]


def test_look_tags_never_clash_with_lighting_tags():
    """Both tags go into file names: a lighting and a look must not be confused."""
    lighting_ids = {p.id for p in lt.all_profiles()}
    assert not lighting_ids & set(LOOKS)


def test_new_looks_plug_in_without_other_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(cl, "_LOOKS", dict(cl._LOOKS))
    cl.register_look(
        CameraLook("moody", "Moody", "film", "Dark and soft", LookRecipe(A(exposure=-20), fade=20))
    )
    assert cl.all_looks()[-1].id == CUSTOM  # Custom stays last
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"camera_look": "moody"}))
    assert load_settings(path).camera_look == "moody"
    assert full("moody").fade == 20
    with pytest.raises(ValueError):
        cl.register_look(CameraLook("x", "X", "no-such-category", ""))


A = Adjustments


# --- Original and intensity -------------------------------------------------------
@pytest.mark.parametrize("intensity", [0, 50, 100])
@pytest.mark.parametrize("grain", cl.GRAIN_CHOICES)
def test_original_is_the_untouched_image(intensity, grain):
    recipe = full(ORIGINAL, intensity, grain)
    assert recipe.is_neutral
    src = landscape()
    assert apply_look(src, recipe) is src and finish_look(src, recipe) is src
    assert apply_to_pil(Image.fromarray(src), recipe) is not None
    assert not CameraLookSettings(ORIGINAL, intensity, grain).active
    assert CameraLookSettings(ORIGINAL, intensity, grain).tag() == ""


@pytest.mark.parametrize("look", LOOKS)
def test_zero_intensity_has_no_effect(look):
    recipe = full(look, 0, "high")
    assert recipe.is_neutral
    src = landscape()
    assert apply_look(src, recipe) is src and finish_look(src, recipe) is src


@pytest.mark.parametrize("look", LOOKS)
def test_intensity_scales_the_look(look):
    src = landscape()
    d25, d50, d100 = (diff(render(src, full(look, i)), src) for i in (25, 50, 100))
    assert 0 < d25 < d50 < d100
    assert full(look, 100) == cl.get_look(look).recipe


def test_intensity_scales_every_parameter_linearly():
    recipe = cl.get_look("fujifilm-classic-negative").recipe
    half = full("fujifilm-classic-negative", 50)
    assert half.tone == recipe.tone.scaled(0.5)
    assert half.bands == tuple(b.scaled(0.5) for b in recipe.bands)
    assert half.shadow_tint == Toning(recipe.shadow_tint.hue, recipe.shadow_tint.amount / 2)
    assert (half.rolloff, half.fade, half.grain) == (
        recipe.rolloff / 2,
        recipe.fade / 2,
        recipe.grain / 2,
    )
    assert half.skin == recipe.skin  # protection is not weakened by a gentler look


def test_intensity_out_of_range_is_clamped():
    assert full("sony-vivid", 250) == full("sony-vivid", 100)
    assert full("sony-vivid", -5).is_neutral


def test_unknown_look_is_neutral():
    assert full("no-such-look").is_neutral and CameraLookSettings("no-such-look").tag() == ""


def test_default_settings_are_original_at_half_intensity():
    assert CameraLookSettings() == CameraLookSettings(ORIGINAL, 50, cl.GRAIN_AUTO)
    assert ProcessingOptions().camera_look == CameraLookSettings()


# --- every built-in look ----------------------------------------------------------------
@pytest.mark.parametrize("look", LOOKS)
def test_every_look_is_photographic_not_a_cheap_filter(look):
    """No crushed blacks, clipped highlights, neon colours or extreme shifts."""
    for src in (landscape(), portrait()[0]):
        out = render(src, full(look))
        assert out.shape == src.shape and out.dtype == np.uint8
        # Clipping: at most a sliver more than the source had.
        assert (out >= 254).mean() <= (src >= 254).mean() + 0.01
        assert (out <= 1).mean() <= (src <= 1).mean() + 0.01
        # Not a heavy-handed filter: tones stay close, colour does not explode.
        assert abs(luma(out).mean() - luma(src).mean()) < 25
        assert chroma(out).mean() < chroma(src).mean() * 1.6 + 5
        assert diff(out, src) < 40


@pytest.mark.parametrize("look", LOOKS)
def test_every_look_keeps_a_grey_ramp_monotonic(look):
    """Tonal order is preserved (no banding or inverted tones from the curves)."""
    ramp = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (4, 1, 3))
    recipe = dataclasses.replace(full(look), grain=0, sharpen=0, clarity=0)
    out = luma(apply_look(ramp, recipe))[0]
    assert np.all(np.diff(out) >= -1.0)


def test_looks_are_distinct_from_each_other():
    src = landscape()
    outputs = {look: render(src, full(look)) for look in LOOKS}
    for i, a in enumerate(LOOKS):
        for b in LOOKS[i + 1 :]:
            assert diff(outputs[a], outputs[b]) > 0.8, (a, b)


def test_looks_have_their_character():
    src = landscape()
    sat = {look: chroma(render(src, full(look))).mean() for look in LOOKS}
    base = chroma(src).mean()
    # Fujifilm-inspired: Velvia rich, Provia moderate, Classic muted.
    assert sat["fujifilm-velvia"] > sat["fujifilm-provia"] > sat["fujifilm-classic"]
    assert sat["fujifilm-classic"] < base
    assert sat["canon-landscape"] > base and sat["nikon-landscape"] > base
    assert sat["sony-vivid"] > sat["sony-natural"]
    # Monochrome looks are neutral black and white.
    for look in ("black-white", "leica-monochrome"):
        assert sat[look] < 1.5
    # Black-and-white channel mixes differ: Leica's renders reds lighter.
    red = np.zeros((8, 8, 3), np.uint8) + np.array([200, 40, 40], np.uint8)
    assert (
        luma(render(red, full("leica-monochrome"))).mean()
        > luma(render(red, full("black-white"))).mean()
    )


def test_canon_is_warmer_and_sony_cooler():
    grey = np.full((16, 16, 3), 128, np.uint8)

    def warmth(look: str) -> float:
        out = apply_look(grey, dataclasses.replace(full(look), grain=0)).astype(float)
        return float((out[..., 0] - out[..., 2]).mean())

    assert warmth("canon-standard") > 1.5 and warmth("canon-portrait") > warmth("canon-standard")
    assert warmth("sony-natural") < 0
    assert abs(warmth("nikon-neutral")) < 1


def test_cinematic_looks_split_tone_shadows_and_highlights():
    ramp = np.tile(np.linspace(0, 255, 256).astype(np.uint8)[None, :, None], (4, 1, 3))
    out = apply_look(ramp, full("cinematic-film")).astype(float)
    shadows, highlights = out[:, 50:90], out[:, 170:210]
    # Teal shadows (blue/green over red), warm highlights (red over blue).
    assert (shadows[..., 2] - shadows[..., 0]).mean() > 3
    assert (highlights[..., 0] - highlights[..., 2]).mean() > 3
    # Pure black and white stay neutral.
    assert np.ptp(out[:, 0]) <= 2 and np.ptp(out[:, 255]) <= 2


def test_film_looks_have_softer_highlights_and_lifted_blacks():
    ramp = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (4, 1, 3))
    out = luma(apply_look(ramp, full("classic-film")))[0]
    assert out[0] > 5  # matte, faded blacks
    assert out[255] >= 250  # highlights stay clean
    # Highlight roll-off: the slope approaching white is gentler than the midtones'.
    assert out[250] - out[230] < out[140] - out[120]


def test_grain_defaults_only_on_film_looks():
    for look in DIGITAL:
        assert cl.get_look(look).recipe.grain == 0, look
    for look in ("kodak", "portra", "classic-film", "fujifilm-classic-negative"):
        assert cl.get_look(look).recipe.grain > 0, look


# --- skin tones -----------------------------------------------------------------------
@pytest.mark.parametrize("look", [*PORTRAIT_LOOKS, "leica-natural", "fujifilm-velvia", "ektar"])
def test_skin_tones_stay_natural(look):
    src, face = portrait()
    out = render(src, full(look))
    skin_in = src[face].astype(float).mean(0)
    skin_out = out[face].astype(float).mean(0)

    def hue(rgb):
        r, g, b = rgb
        return np.degrees(np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)) % 360

    assert abs(hue(skin_out) - hue(skin_in)) < 8
    assert chroma(out[face]).mean() < chroma(src[face]).mean() * 1.25


def test_skin_protection_holds_back_colour_changes():
    src, face = portrait()
    recipe = full("fujifilm-velvia")
    unprotected = dataclasses.replace(recipe, skin=0)
    change = chroma(render(src, recipe)[face]).mean()
    change_unprotected = chroma(render(src, unprotected)[face]).mean()
    assert change < change_unprotected
    # Pinkish skin (hue just below 360°) is protected too.
    pink = np.zeros((8, 8, 3), np.uint8) + np.array([205, 150, 158], np.uint8)
    assert chroma(render(pink, recipe)).mean() < chroma(render(pink, unprotected)).mean()


# --- the processing engine -------------------------------------------------------------
def test_lookup_table_matches_the_reference_grade():
    rng = np.random.default_rng(3)
    src = np.clip(rng.normal(128, 60, (64, 96, 3)), 0, 255).astype(np.uint8)
    for look in ("fujifilm-classic-negative", "portra", "black-white", "sony-cinematic"):
        recipe = dataclasses.replace(full(look).clamped(), clarity=0)
        reference = np.clip(cl._GradePlan(recipe).run(src), 0, 1) * 255
        error = np.abs(apply_look(src, recipe).astype(float) - reference)
        assert error.mean() < 0.6 and np.percentile(error, 99) <= 3, look


def test_hue_band_controls_affect_only_their_colours():
    swatches = np.array(
        [[[200, 40, 40], [40, 160, 40], [40, 60, 200], [128, 128, 128]]], dtype=np.uint8
    )
    greens = LookRecipe(bands=cl.bands(green=(0, 60, 0)))
    out = apply_look(swatches, greens).astype(int)
    assert chroma(out[0, 1:2]) > chroma(swatches[0, 1:2]) + 10  # green boosted
    assert np.abs(out[0, [0, 2, 3]] - swatches[0, [0, 2, 3]]).max() <= 2  # others kept
    # Hue shift: red towards orange/yellow raises green.
    warm_reds = LookRecipe(bands=cl.bands(red=(25, 0, 0)))
    assert apply_look(swatches, warm_reds)[0, 0, 1] > swatches[0, 0, 1] + 10
    # Luminance: darker blues.
    dark_blue = LookRecipe(bands=cl.bands(blue=(0, 0, -60)))
    assert luma(apply_look(swatches, dark_blue)[0, 2:3]) < luma(swatches[0, 2:3]) - 5


def test_bands_reject_unknown_names():
    with pytest.raises(ValueError):
        cl.bands(teal=(0, 1, 0))


@pytest.mark.parametrize("look", ["fujifilm-classic-negative", "canon-landscape", "black-white"])
def test_banded_processing_matches_whole_image(monkeypatch, look):
    """Clarity, sharpening and grain need neighbours: bands must be seamless."""
    src = landscape(160, 120)
    whole = render(src, full(look))
    monkeypatch.setattr(cl, "_CHUNK_PIXELS", 160 * 7)  # many small bands
    banded = render(src, full(look))
    assert np.array_equal(whole, banded)
    in_place = src.copy()
    recipe = full(look)
    apply_look(in_place, recipe, out=in_place)
    finish_look(in_place, recipe, out=in_place)
    assert np.array_equal(whole, in_place)


def test_grain_is_deterministic_neutral_and_follows_the_choice():
    flat = np.full((64, 64, 3), 128, np.uint8)
    recipe = full("classic-film")
    a, b = finish_look(flat, recipe), finish_look(flat, recipe)
    assert np.array_equal(a, b)  # the same image always gets the same grain
    assert abs(float(a.mean()) - 128) < 1.5  # zero-mean
    assert np.array_equal(a[..., 0], a[..., 2])  # luminance grain: grey stays grey
    spread = {
        g: float(finish_look(flat, full("sony-natural", 100, g)).std())
        for g in ("off", "low", "medium", "high")
    }
    assert spread["off"] < 0.5 < spread["low"] < spread["medium"] < spread["high"]
    assert finish_look(flat, full("classic-film", 100, "off")).std() < 0.5


def test_grain_size_follows_the_picture():
    """A big export gets proportionally bigger grain, so it looks like the preview."""
    flat = np.full((32, 32, 3), 128, np.uint8)
    recipe = LookRecipe(grain=80)

    def roughness(frame):
        out = finish_look(flat, recipe, frame=frame).astype(float)[..., 0]
        return float(np.abs(np.diff(out, axis=1)).mean())

    assert roughness((12000, 8000)) < roughness(None) * 0.6


def test_sharpening_adds_detail_without_halos():
    edge = np.zeros((32, 32, 3), np.uint8)
    edge[:, 16:] = 200
    edge[:, :16] = 50
    out = finish_look(edge, LookRecipe(sharpen=100)).astype(int)
    assert out[:, 15].mean() < 50 or out[:, 16].mean() > 200  # crisper edge
    assert out.min() >= 50 - 9 and out.max() <= 200 + 9  # overshoot limited


def test_clarity_raises_midtone_contrast():
    _y, x = np.mgrid[0:96, 0:96]
    soft = (128 + 30 * np.sin(x / 2.0))[..., None].repeat(3, -1).astype(np.uint8)
    out = apply_look(soft, LookRecipe(clarity=100))
    assert out.astype(float).std() > soft.astype(float).std() * 1.05
    softer = apply_look(soft, LookRecipe(clarity=-100))
    assert softer.astype(float).std() < soft.astype(float).std()


def test_apply_to_pil_keeps_alpha_size_and_combines_lighting():
    src = landscape(64, 48)
    img = Image.fromarray(src).convert("RGBA")
    alpha = Image.fromarray(np.tile(np.arange(64, dtype=np.uint8)[None] * 4, (48, 1)), "L")
    img.putalpha(alpha)
    recipe, lighting = full("portra"), Adjustments(exposure=20)
    out = apply_to_pil(img, recipe, lighting)
    assert out.mode == "RGBA" and out.size == img.size
    assert np.array_equal(np.asarray(out.getchannel("A")), np.asarray(alpha))
    expected = render(apply_lighting(src, lighting), recipe)
    assert np.array_equal(np.asarray(out.convert("RGB")), expected)
    quick = np.asarray(apply_to_pil(img, recipe, lighting, quick=True).convert("RGB"))
    assert diff(quick, expected) < 1.0


def test_rejects_non_rgb_arrays():
    with pytest.raises(ValueError):
        apply_look(np.zeros((4, 4), np.uint8), full("portra"))
    with pytest.raises(ValueError):
        finish_look(np.zeros((4, 4, 4), np.uint8), full("portra"))


def test_lighting_and_look_combine_without_clipping():
    src = landscape()
    for profile in ("bright-clean", "high-contrast", "vivid", "golden-hour"):
        lit = apply_lighting(src, LightingSettings(profile).adjustments())
        for look in ("fujifilm-velvia", "ektar", "sony-vivid", "cinematic-film"):
            out = render(lit, full(look))
            # No crushed blacks or blown highlights (a saturated colour may
            # reach the gamut edge in one channel; that is not clipping).
            assert (luma(out) >= 252).mean() <= (luma(lit) >= 252).mean() + 0.005
            assert (luma(out) <= 3).mean() <= (luma(lit) <= 3).mean() + 0.005
            assert chroma(out).mean() < chroma(lit).mean() * 1.6 + 5


def test_preview_render_is_fast():
    """Changing a look or its intensity must update a preview in well under a second."""
    base = cl.sample_image(1600, 1067)
    apply_to_pil(base, full("fujifilm-classic-negative", 61), quick=True)  # warm up
    started = time.perf_counter()
    apply_to_pil(base, full("fujifilm-classic-negative", 62), quick=True)
    assert time.perf_counter() - started < 1.5


# --- Custom ------------------------------------------------------------------------------
def test_custom_neutral_is_identity_and_ignores_intensity():
    assert custom_recipe({}).is_neutral
    recipe = custom_recipe({"contrast": 40})
    settings = CameraLookSettings(CUSTOM, 10, cl.GRAIN_AUTO, recipe)
    assert settings.recipe().tone.contrast == 40  # not scaled by the intensity
    assert CameraLookSettings(CUSTOM, 50, cl.GRAIN_AUTO, custom_recipe({})).tag() == ""


def hue_wheel(w: int = 180, h: int = 24) -> np.ndarray:
    """Every hue at several saturations."""
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = np.linspace(0, 255, w).astype(np.uint8)[None]
    hsv[..., 1] = np.linspace(60, 220, h).astype(np.uint8)[:, None]
    hsv[..., 2] = 190
    return np.asarray(Image.fromarray(hsv, "HSV").convert("RGB"))


@pytest.mark.parametrize("name", cl.CUSTOM_NAMES)
def test_each_custom_control_changes_the_image(name):
    src = np.concatenate([landscape(180, 96), hue_wheel()])
    _low, high = cl.custom_range(name)
    recipe = custom_recipe({name: high})
    assert not recipe.is_neutral
    assert diff(render(src, recipe), src) > 0.05, name


def test_custom_values_are_clamped():
    recipe = custom_recipe({"contrast": 900, "sharpness": -50, "green": "x"})
    assert recipe.tone.contrast == 100 and recipe.sharpen == 0
    assert recipe.bands[cl.BANDS.index("green")] == Band()


def test_custom_grain_comes_from_the_grain_choice():
    custom = custom_recipe({"contrast": 10})
    assert CameraLookSettings(CUSTOM, 50, "medium", custom).recipe().grain == 55


# --- tags ----------------------------------------------------------------------------------
def test_tags_name_different_looks_differently():
    tags = {
        CameraLookSettings("portra", 100).tag(),
        CameraLookSettings("portra", 50).tag(),
        CameraLookSettings("portra", 50, "high").tag(),
        CameraLookSettings("ektar", 50).tag(),
        CameraLookSettings(CUSTOM, 50, cl.GRAIN_AUTO, custom_recipe({"tint": 5})).tag(),
        CameraLookSettings(CUSTOM, 50, cl.GRAIN_AUTO, custom_recipe({"tint": 6})).tag(),
        CameraLookSettings("user:mine", 50, cl.GRAIN_AUTO, custom_recipe({"tint": 6})).tag(),
    }
    assert len(tags) == 7 and "" not in tags
    assert CameraLookSettings("portra", 100).tag() == "portra"
    assert CameraLookSettings("portra", 50, "high").tag() == "portra-50-grain-high"


# --- export pipeline -------------------------------------------------------------------------
def _process(path, look="portra", fmt="png", intensity=100, grain=cl.GRAIN_AUTO, **kw):
    options = ProcessingOptions(
        scale=2,
        output_format=fmt,
        quality=95,
        existing="overwrite",
        camera_look=CameraLookSettings(look, intensity, grain),
        **kw,
    )
    return process_image(path, options, NearestUpscaler())


def _pixels(path, mode="RGB"):
    with Image.open(path) as img:
        return np.asarray(img.convert(mode))


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_export_applies_the_look_at_full_resolution(image_factory):
    path = image_factory("p.png", size=(48, 32))
    before = _sha(path)
    result = _process(path, "fujifilm-classic-negative", intensity=60)
    assert result.output.name == "p_2x_fujifilm-classic-negative-60.png"
    assert result.look == "fujifilm-classic-negative-60"
    assert result.output_size == (96, 64)
    with Image.open(path) as src:
        rgb = np.asarray(src.convert("RGB")).copy()
    recipe = full("fujifilm-classic-negative", 60)
    # Grade before upscaling, finish (sharpening, grain) on the upscaled image.
    graded = apply_look(rgb, recipe)
    expected = finish_look(np.repeat(np.repeat(graded, 2, 0), 2, 1), recipe)
    assert np.array_equal(_pixels(result.output), expected)
    assert _sha(path) == before  # the original is never modified


def test_grain_is_applied_after_upscaling(image_factory):
    """Grain made before a 2× upscale would come in 2×2 blocks."""
    path = image_factory("g.png", size=(48, 32))
    flat = np.full((32, 48, 3), 128, np.uint8)
    Image.fromarray(flat).save(path)
    out = _pixels(_process(path, "classic-film").output).astype(int)
    assert np.abs(out[0::2, 0::2] - out[1::2, 0::2]).mean() > 0.5


@pytest.mark.parametrize(("ext", "fmt"), [("jpg", "jpeg"), ("png", "png"), ("png", "jpeg")])
def test_export_formats_contain_the_look(image_factory, ext, fmt):
    path = image_factory(f"f.{ext}", size=(48, 32))
    plain = _process(path, ORIGINAL, fmt=fmt)
    graded = _process(path, "fujifilm-velvia", fmt=fmt)
    assert plain.output != graded.output
    assert diff(_pixels(graded.output), _pixels(plain.output)) > 2


def test_export_rgba_keeps_alpha(image_factory):
    path = image_factory("a.png", size=(48, 32), mode="RGBA")
    result = _process(path, "portra")
    with Image.open(result.output) as out, Image.open(path) as src:
        assert out.mode == "RGBA" and out.size == (96, 64)
        expected = np.asarray(src.getchannel("A").resize((96, 64), Image.Resampling.LANCZOS))
        assert np.array_equal(np.asarray(out.getchannel("A")), expected)


def test_export_grayscale(image_factory):
    path = image_factory("g.png", size=(48, 32), mode="L")
    # A neutral black-and-white look keeps a greyscale file greyscale…
    with Image.open(_process(path, "black-white").output) as out:
        assert out.mode == "L"
    # …a look that tints (split toning, warmth) makes it colour, as previewed.
    with Image.open(_process(path, "cinematic-film").output) as out:
        assert out.mode == "RGB"


def test_export_keeps_metadata(image_factory):
    from PIL import ImageCms

    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    path = image_factory("icc.jpg", icc_profile=icc, dpi=(72, 72))
    with Image.open(_process(path, "leica-natural").output) as out:
        assert out.info.get("icc_profile") == icc
        assert round(out.info["dpi"][0]) == 144


def test_export_keeps_orientation(image_factory):
    path = image_factory("o.jpg", size=(48, 32))
    with Image.open(path) as img:
        exif = img.getexif()
        exif[0x0112] = 6  # rotated 90°
        img.save(path, exif=exif)
    result = _process(path, "kodak")
    assert result.output_size == (64, 96)  # upright, as for any upscale


@pytest.mark.parametrize("fmt", ["png", "jpeg", "webp"])
def test_original_at_zero_intensity_is_byte_identical_to_before(image_factory, tmp_path, fmt):
    """Users who never touch Camera Looks get exactly the previous output."""
    path = image_factory("same.jpg", size=(48, 32))
    default = ProcessingOptions(scale=2, output_format=fmt, output_dir=tmp_path / "a")
    untouched = ProcessingOptions(
        scale=2,
        output_format=fmt,
        output_dir=tmp_path / "b",
        camera_look=CameraLookSettings(ORIGINAL, 0),
    )
    zero = ProcessingOptions(
        scale=2,
        output_format=fmt,
        output_dir=tmp_path / "c",
        camera_look=CameraLookSettings("fujifilm-velvia", 0, "high"),
    )
    a = process_image(path, default, NearestUpscaler()).output
    b = process_image(path, untouched, NearestUpscaler()).output
    c = process_image(path, zero, NearestUpscaler()).output
    assert a.name == b.name == c.name == f"same_2x.{'jpg' if fmt == 'jpeg' else fmt}"
    assert a.read_bytes() == b.read_bytes() == c.read_bytes()


def test_lighting_and_look_in_the_pipeline(image_factory):
    path = image_factory("both.png", size=(48, 32))
    options = ProcessingOptions(
        scale=2,
        existing="overwrite",
        lighting=LightingSettings("golden-hour", 100),
        camera_look=CameraLookSettings("portra", 50),
    )
    result = process_image(path, options, NearestUpscaler())
    assert result.output.name == "both_2x_golden-hour_portra-50.png"
    with Image.open(path) as src:
        rgb = np.asarray(src.convert("RGB")).copy()
    lit = apply_lighting(rgb, LightingSettings("golden-hour", 100).adjustments())
    recipe = full("portra", 50)
    expected = finish_look(np.repeat(np.repeat(apply_look(lit, recipe), 2, 0), 2, 1), recipe)
    assert np.array_equal(_pixels(result.output), expected)


def test_changed_look_is_not_skipped(image_factory):
    path = image_factory("s.png", size=(48, 32))
    first = _process(path, "portra")
    options = ProcessingOptions(scale=2, camera_look=CameraLookSettings("ektar", 100))
    second = process_image(path, options, NearestUpscaler())
    assert not second.skipped and second.output != first.output
    again = process_image(path, options, NearestUpscaler())
    assert again.skipped and again.look == "ektar"


def test_progress_reports_look_stages(image_factory):
    path = image_factory("prog.png", size=(48, 32))
    stages = []
    options = ProcessingOptions(scale=2, camera_look=CameraLookSettings("classic-film", 100))
    process_image(path, options, NearestUpscaler(), progress=lambda _f, s: stages.append(s))
    assert "Applying camera look" in stages and "Finishing camera look" in stages
    assert stages.index("Applying camera look") < stages.index("Upscaling")
    assert stages.index("Finishing camera look") > stages.index("Upscaling")


def test_batch_processing_applies_the_look(image_factory, tmp_path):
    paths = [image_factory(f"b{i}.png", size=(24, 16)) for i in range(3)]
    options = ProcessingOptions(
        scale=2, output_dir=tmp_path / "out", camera_look=CameraLookSettings("ektar", 70)
    )
    items = [QueueItem(p) for p in paths]
    summary = BatchProcessor(NearestUpscaler(), options, lambda _e: None, workers=2).run(items)
    assert summary.done == 3
    for item in items:
        assert item.status == ItemStatus.DONE
        assert item.result.output.name.endswith("_2x_ektar-70.png")


def test_real_model_pipeline_with_a_look(image_factory, upscaler, tiny_specs):
    path = image_factory("real.png", size=(32, 24))
    options = ProcessingOptions(
        scale=4, model="test-family", camera_look=CameraLookSettings("leica-filmic", 80)
    )
    result = process_image(path, options, upscaler)
    assert result.output_size == (128, 96) and result.output.name == "real_4x_leica-filmic-80.png"


# --- restoration ------------------------------------------------------------------------------
NOTHING = rs.RestorationSettings(level=rs.CUSTOM, custom=rs.Stages(), modern=rs.MODERN_OFF)
CONVENTIONAL = rs.RestorationSettings(
    level=rs.CUSTOM,
    custom=rs.Stages(dust=40, noise=30, fading=50, sharpness=20, auto_color=True),
)


def test_restoration_without_a_look_is_unchanged(upscaler):
    src = landscape()
    restorer = Restorer(upscaler)
    a = restorer.restore(src, CONVENTIONAL, model="test-family")
    b = restorer.restore(src, CONVENTIONAL, model="test-family", look=full(ORIGINAL))
    assert np.array_equal(a.rgb, b.rgb)


def test_restored_photo_gets_the_look_after_restoration(upscaler):
    src = landscape()
    restorer = Restorer(upscaler)
    plain = restorer.restore(src, NOTHING, model="test-family")
    recipe = full("kodak", 70)
    graded = restorer.restore(src, NOTHING, model="test-family", look=recipe)
    assert np.array_equal(graded.rgb, render(plain.rgb, recipe))
    assert np.array_equal(src, landscape())  # the input is not modified


def test_black_and_white_restoration_with_looks(upscaler):
    bw = np.repeat(luma(landscape()).astype(np.uint8)[..., None], 3, -1)
    restorer = Restorer(upscaler)
    mono = restorer.restore(bw, CONVENTIONAL, model="test-family", look=full("black-white"))
    assert mono.monochrome and not mono.colorized  # still saved as black and white
    tinted = restorer.restore(bw, CONVENTIONAL, model="test-family", look=full("classic-film"))
    assert not tinted.monochrome  # a toned look turns it into a (toned) colour image


def test_colorized_photo_gets_the_look_after_colorization(upscaler, models_dir, monkeypatch):
    """The look grades the colorized result; the colorizer never sees the look."""
    from pixelift.core.restoration import colorize
    from pixelift.core.upscaler import TorchUpscaler
    from pixelift.models.restoration import DEOLDIFY_ARTISTIC

    (models_dir / DEOLDIFY_ARTISTIC.filename).write_bytes(b"placeholder")
    seen: dict[str, np.ndarray] = {}

    def fake_run_model(self, spec, fn, control=None, release=True):
        return fn(spec.id, None)

    def fake_predict(_net, _device, gray):
        seen["colorizer_input"] = gray.copy()
        return np.full((colorize.RENDER_SIZE, colorize.RENDER_SIZE, 3), (170, 120, 90), np.float32)

    original_apply = cl.apply_look

    def spy_apply(rgb, recipe, **kw):
        seen["look_input"] = rgb.copy()
        return original_apply(rgb, recipe, **kw)

    monkeypatch.setattr(TorchUpscaler, "run_model", fake_run_model)
    monkeypatch.setattr(colorize, "predict_color", fake_predict)
    monkeypatch.setattr(cl, "apply_look", spy_apply)
    bw = np.repeat(luma(landscape()).astype(np.uint8)[..., None], 3, -1)
    settings = dataclasses.replace(NOTHING, colorize=True)
    result = Restorer(upscaler).restore(bw, settings, model="test-family", look=full("portra"))
    assert result.colorized
    assert chroma(seen["look_input"]).mean() > 5  # the look saw the colorized photo
    assert np.array_equal(seen["colorizer_input"], bw[..., 0])  # the colorizer did not


def test_restore_pipeline_through_process_image(image_factory, upscaler, tiny_specs):
    path = image_factory("old.png", size=(48, 32))
    options = ProcessingOptions(
        model="test-family",
        existing="overwrite",
        restoration=NOTHING,
        camera_look=CameraLookSettings("hasselblad-filmic", 100),
    )
    result = process_image(path, options, upscaler)
    assert result.restored and result.look == "hasselblad-filmic"
    assert result.output.name.endswith("_hasselblad-filmic.png")


# --- settings -------------------------------------------------------------------------------------
def test_settings_default_is_original():
    s = Settings()
    assert s.camera_look == ORIGINAL and s.camera_look_intensity == 50
    assert not s.camera_look_settings().active and s.processing_options().camera_look.tag() == ""


def test_settings_round_trip_with_favorites_and_saved_looks(tmp_path):
    path = tmp_path / "settings.json"
    s = Settings(camera_look="fujifilm-astia", camera_look_intensity=35, camera_look_grain="low")
    s.camera_look_favorites = ["fujifilm-astia", "kodak"]
    s.look_green = 40
    s.camera_look_saved = {"Greens": {**s.look_values()}}
    save_settings(s, path)
    loaded = load_settings(path)
    assert loaded.camera_look_settings() == s.camera_look_settings()
    assert loaded.camera_look_favorites == ["fujifilm-astia", "kodak"]
    assert loaded.camera_look_saved["Greens"]["green"] == 40
    loaded.camera_look = "user:Greens"
    saved = loaded.camera_look_settings()
    assert saved.user_look and saved.recipe().bands[3].saturation == 40 * 0.35
    assert saved.name() == "Greens"


def test_settings_invalid_values_are_normalised(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "camera_look": "user:missing",
                "camera_look_intensity": 900,
                "camera_look_grain": "extreme",
                "camera_look_favorites": ["kodak", "nope", 3, "kodak", "original"],
                "camera_look_saved": {"ok": {"contrast": 500, "x": 1}, "": {}, "bad": 5},
                "look_sharpness": -20,
                "look_red": 999,
            }
        )
    )
    s = load_settings(path)
    assert s.camera_look == ORIGINAL and s.camera_look_intensity == 100
    assert s.camera_look_grain == cl.GRAIN_AUTO
    assert s.camera_look_favorites == ["kodak"]
    assert list(s.camera_look_saved) == ["ok"] and s.camera_look_saved["ok"]["contrast"] == 100
    assert s.look_sharpness == 0 and s.look_red == 100
    path.write_text(json.dumps({"camera_look_favorites": "kodak", "camera_look_saved": []}))
    s = load_settings(path)
    assert s.camera_look_favorites == [] and s.camera_look_saved == {}


def test_old_settings_files_get_camera_look_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"scale": 2, "lighting_profile": "vivid"}))
    s = load_settings(path)
    assert s.camera_look == ORIGINAL and not s.camera_look_settings().active


def test_custom_settings_build_the_custom_recipe():
    s = Settings(camera_look=CUSTOM, look_contrast=30, look_blue=-20, look_sharpness=40)
    recipe = s.camera_look_settings().recipe()
    assert recipe.tone.contrast == 30 and recipe.sharpen == 40
    assert recipe.bands[cl.BANDS.index("blue")].saturation == -20


# --- CLI ------------------------------------------------------------------------------------------
def test_cli_look_options(image_factory, tiny_specs, tmp_path, capsys):
    src = image_factory("photo.png")
    args = [str(src), "-s", "4", "-m", "test-x4", "--device", "cpu"]
    assert cli.run_cli([*args, "-o", str(tmp_path / "plain")]) == 0
    assert "· look" not in capsys.readouterr().out
    look_args = [*args, "-o", str(tmp_path / "look"), "--look", "fujifilm-velvia"]
    assert cli.run_cli([*look_args, "--look-intensity", "80", "--grain", "low"]) == 0
    assert "look Fujifilm Velvia 80%" in capsys.readouterr().out
    plain = _pixels(tmp_path / "plain" / "photo_4x.png")
    graded = _pixels(tmp_path / "look" / "photo_4x_fujifilm-velvia-80-grain-low.png")
    assert diff(plain, graded) > 2


def test_cli_lists_looks(capsys):
    assert cli.run_cli(["--list-looks"]) == 0
    out = capsys.readouterr().out
    assert "fujifilm-classic-negative" in out and "Fujifilm-inspired" in out
    assert "not official" in out


@pytest.mark.parametrize(
    "extra",
    [
        ["--look-intensity", "101"],
        ["--look", "custom", "--look-intensity", "50"],
        ["--look", "no-such-look"],
        ["--grain", "huge"],
    ],
)
def test_cli_rejects_bad_look_options(image_factory, extra, capsys):
    src = image_factory("photo.png")
    with pytest.raises(SystemExit) as exc:
        cli.run_cli([str(src), *extra])
    assert exc.value.code == 2


# --- review fixes ---------------------------------------------------------------------------
def test_preview_grain_and_sharpening_match_the_export():
    """A preview smaller than the export shows the export's grain as it looks
    scaled down, not coarser grain at preview pixels."""
    flat = np.full((1600, 1600, 3), 128, np.uint8)
    recipe = LookRecipe(grain=85)
    export = finish_look(flat, recipe)
    seen = np.asarray(Image.fromarray(export).resize((400, 400), Image.Resampling.BOX))
    preview = finish_look(flat[:400, :400], recipe, output=(1600, 1600))
    naive = finish_look(flat[:400, :400], recipe)
    target = seen.astype(float).std()
    assert abs(preview.astype(float).std() - target) < target * 0.5
    assert naive.astype(float).std() > target * 2  # what the preview used to show
    # Without an export size, the export path is unchanged.
    assert np.array_equal(finish_look(flat[:64, :64], recipe), export[:64, :64])
    edge = np.zeros((32, 32, 3), np.uint8)
    edge[:, 16:] = 200
    full_sharp = finish_look(edge, LookRecipe(sharpen=100)).astype(int)
    small_sharp = finish_look(edge, LookRecipe(sharpen=100), output=(128, 128)).astype(int)
    assert np.abs(small_sharp - edge).sum() < np.abs(full_sharp - edge).sum()


def test_card_thumbnails_never_evict_preview_and_export_tables():
    recipe = full("portra", 37)
    cl.grade_lut(recipe)
    cl.grade_lut(recipe, cl.PREVIEW_LUT_SIZE)
    full_hits = cl._full_lut.cache_info().currsize
    preview_hits = cl._preview_lut.cache_info().currsize
    for look in LOOKS:  # what opening the gallery does
        apply_to_pil(cl.sample_image(32, 24), full(look), lut_size=cl.THUMBNAIL_LUT_SIZE)
    assert cl._full_lut.cache_info().currsize == full_hits
    assert cl._preview_lut.cache_info().currsize == preview_hits
    hits = cl._full_lut.cache_info().hits
    cl.grade_lut(recipe)
    assert cl._full_lut.cache_info().hits == hits + 1  # still cached
    assert cl._full_lut.cache_info().maxsize <= 4


def test_palette_transparency_is_kept_in_the_preview():
    img = Image.new("P", (8, 8), 0)
    img.putpalette([0, 0, 0, 200, 120, 90] + [0] * 762)
    img.paste(1, (0, 0, 4, 8))
    img.info["transparency"] = 0
    out = apply_to_pil(img, full("portra"))
    assert out.mode == "RGBA"
    alpha = np.asarray(out.getchannel("A"))
    assert (alpha[:, :4] == 255).all() and (alpha[:, 4:] == 0).all()


def test_custom_values_are_cleaned_in_one_place():
    clean = cl.clean_custom_values({"contrast": "12.6", "sharpness": -3, "red": None, "x": 5})
    assert clean["contrast"] == 13 and clean["sharpness"] == 0 and clean["red"] == 0
    assert set(clean) == set(cl.CUSTOM_NAMES)
    assert cl.clean_custom_values("junk") == dict.fromkeys(cl.CUSTOM_NAMES, 0)
    assert custom_recipe({"contrast": "12.6"}).tone.contrast == 13


def test_cli_uses_saved_looks(image_factory, tiny_specs, tmp_path, capsys):
    from pixelift.storage.settings import settings_path

    settings = Settings(camera_look="user:Warm", camera_look_intensity=100)
    settings.camera_look_saved = {"Warm": {"temperature": 60, "contrast": 20}}
    save_settings(settings)
    assert settings_path().exists()
    src = image_factory("photo.png")
    args = [str(src), "-s", "4", "-m", "test-x4", "--device", "cpu"]
    # The look selected in the app is the default…
    assert cli.run_cli([*args, "-o", str(tmp_path / "a")]) == 0
    assert "look Warm 100%" in capsys.readouterr().out
    # …and saved looks can be named explicitly.
    assert cli.run_cli([*args, "-o", str(tmp_path / "b"), "--look", "user:Warm"]) == 0
    assert len(list((tmp_path / "b").glob("photo_4x_look-*.png"))) == 1
    assert cli.run_cli(["--list-looks"]) == 0
    assert "user:Warm" in capsys.readouterr().out
