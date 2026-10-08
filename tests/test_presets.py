"""Presets: one-click recipes made only of existing settings."""

from __future__ import annotations

import dataclasses

import pytest

from pixelift.core import presets
from pixelift.core.restoration import settings as rs
from pixelift.storage.settings import Settings, load_settings, save_settings


@pytest.mark.parametrize("preset", presets.all_presets(), ids=lambda p: p.id)
def test_every_preset_uses_only_valid_existing_settings(preset):
    settings = Settings()
    fields = {f.name for f in dataclasses.fields(Settings)}
    assert set(preset.values) <= fields  # no invented settings
    presets.apply(settings, preset.id)
    before = dataclasses.asdict(settings)
    settings.normalise()  # nothing out of range or unknown
    assert dataclasses.asdict(settings) == before
    assert presets.matches(settings, preset.id)
    assert presets.label(settings) == preset.name


def test_presets_never_change_technical_choices():
    technical = {"scale", "restore_scale", "model", "output_format", "output_dir", "mode"}
    technical |= {"restore_preset"}  # colorize / upscale stay the user's choice
    for preset in presets.all_presets():
        assert not technical & set(preset.values), preset.id


def test_modified_reset_and_custom_labels():
    settings = Settings()
    assert presets.label(settings) == "Original"
    settings.lighting_profile = "vivid"
    assert presets.label(settings) == "Custom"  # changed, but no preset chosen
    presets.apply(settings, "cinematic")
    assert presets.label(settings) == "Cinematic"
    settings.camera_look = "portra"  # the user changes the look afterwards
    assert presets.status(settings) == ("Cinematic", True)
    assert presets.label(settings) == "Cinematic · Modified"
    presets.apply(settings, "cinematic")  # reset
    assert presets.label(settings) == "Cinematic"
    presets.apply(settings, presets.ORIGINAL)
    assert not settings.lighting().active and not settings.camera_look_settings().active


def test_mode_specific_presets_and_old_photo_respects_colorize_choice():
    upscale = [p.id for p in presets.presets_for(presets.UPSCALE)]
    restore = [p.id for p in presets.presets_for(presets.RESTORE)]
    assert "old-photo" not in upscale and "portrait" not in restore
    assert restore == ["original", "old-photo", "natural", "vintage", "monochrome"]
    assert 6 <= len(upscale) <= 9  # a small, curated set
    settings = Settings(mode="restore", restore_preset=rs.PRESET_RESTORE, restore_level=rs.HEAVY)
    presets.apply(settings, "old-photo")
    assert settings.restore_level == rs.STANDARD
    assert settings.restore_preset == rs.PRESET_RESTORE  # never forced to colorize


def test_preset_choice_persists_and_unknown_ids_are_dropped(tmp_path):
    path = tmp_path / "settings.json"
    settings = Settings()
    presets.apply(settings, "film")
    save_settings(settings, path)
    loaded = load_settings(path)
    assert loaded.preset == "film" and presets.label(loaded) == "Film"
    loaded.preset = "ultra-hd-magic"
    assert loaded.normalise().preset == ""
