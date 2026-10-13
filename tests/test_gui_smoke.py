"""Construct GUI widgets without showing them. Skipped without GTK/a display."""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.gui

if not (os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY")):
    pytest.skip("no display", allow_module_level=True)
gi = pytest.importorskip("gi")
try:
    gi.require_version("Gtk", "4.0")
    gi.require_version("Adw", "1")
    from gi.repository import Adw
except (ValueError, ImportError):
    pytest.skip("GTK4/libadwaita not available", allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def _adw():
    Adw.init()


def test_compare_view_zoom_math():
    from pixelift.ui.preview import CompareView

    view = CompareView()
    view.zoom_by(2.0)
    assert view.zoom is None  # nothing on the stage: nothing to zoom
    view.set_images((4000, 3000))
    assert view.zoom is None and view.center == (2000, 3000 / 2)
    view.set_zoom(2.0)
    assert view.effective_zoom() == 2.0
    view.set_zoom(1000.0)
    assert view.effective_zoom() == 16.0  # clamped
    view.set_zoom(None)
    assert view.zoom is None


def test_queue_row_states(tmp_path):
    from conftest import make_image

    from pixelift.core.batch_processor import ItemStatus, QueueItem
    from pixelift.ui.image_queue import QueueRow
    from pixelift.utils.image_utils import make_thumbnail, probe_image

    path = make_image(tmp_path / "a.png")
    row = QueueRow(QueueItem(path), *(lambda _r: None,) * 3)
    row.set_info(probe_image(path), make_thumbnail(path))
    assert row.ready
    assert "40×30" in row.details.get_label() and "160×120" in row.details.get_label()
    row.item.status = ItemStatus.PROCESSING
    row.item.progress = 0.5
    row.refresh()
    assert "50%" in row.status.get_label() and not row.remove_btn.get_sensitive()


def _pump_until(condition, timeout=5.0):
    """Run the GLib main loop until ``condition()`` holds (idle / timeout callbacks)."""
    import time

    from gi.repository import GLib

    context = GLib.MainContext.default()
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting for the main loop"
        context.iteration(False) or time.sleep(0.01)


class _StubApp:
    def __init__(self):
        from pixelift.storage.settings import Settings

        self.settings = Settings()
        self.listeners = []
        self.saves = 0
        self.notifies = 0

    def on_lighting_changed(self, listener):
        self.listeners.append(listener)

    def off_lighting_changed(self, listener):
        self.listeners.remove(listener)

    def lighting_changed(self):
        self.notifies += 1
        self.settings.normalise()
        for listener in list(self.listeners):
            listener()

    def save_settings(self):
        self.saves += 1


def test_lighting_controls_follow_and_update_settings():
    from pixelift.core import lighting
    from pixelift.ui.widgets.lighting import LightingControls

    app = _StubApp()
    controls = LightingControls(app)
    ids = [p.id for p in lighting.all_profiles()]
    # Defaults: Original at 100%, nothing to reset, intensity not applicable.
    assert controls.profile_dd.get_selected() == 0
    assert not controls.reset_btn.get_visible()
    assert not controls.intensity_box.get_sensitive()
    assert not controls.custom_btn.get_visible()

    controls.profile_dd.set_selected(ids.index("golden-hour"))
    assert app.settings.lighting_profile == "golden-hour" and app.saves == 1
    assert controls.intensity_box.get_sensitive() and controls.reset_btn.get_visible()
    # A drag updates the value at once, but notifies and saves only later.
    for value in (60, 50, 40):
        controls.intensity.set_value(value)
    assert app.settings.lighting_intensity == 40
    assert (app.notifies, app.saves) == (1, 1)
    controls.shutdown()  # flushes the pending notify and save, once each
    assert (app.notifies, app.saves) == (2, 2) and controls.sync not in app.listeners


def test_lighting_controls_custom_and_reset():
    from pixelift.core import lighting
    from pixelift.ui.widgets.lighting import LightingControls

    app = _StubApp()
    app.settings.lighting_profile = lighting.CUSTOM
    app.settings.lighting_contrast = 25
    controls = LightingControls(app)
    assert controls.custom_btn.get_visible() and not controls.intensity_box.get_visible()
    assert controls.custom_sliders["contrast"].get_value() == 25
    controls.custom_sliders["saturation"].set_value(-30)
    assert app.settings.lighting_saturation == -30
    controls.reset_custom()
    assert all(s.get_value() == 0 for s in controls.custom_sliders.values())
    assert app.settings.lighting().custom.is_neutral
    controls.reset()
    assert (app.settings.lighting_profile, app.settings.lighting_intensity) == ("original", 100)
    assert controls.profile_dd.get_selected() == 0
    controls.shutdown()


def test_hdr_controls_follow_and_update_settings():
    from pixelift.core import hdr
    from pixelift.ui.widgets.hdr import HdrControls

    app = _StubApp()
    controls = HdrControls(app)
    ids = [p.id for p in hdr.all_profiles()]
    # Defaults: off, Natural HDR at its own intensity; the controls wait for the switch.
    assert not controls.switch.get_active() and not controls.reset_btn.get_visible()
    assert not controls.profile_dd.get_sensitive() and not controls.intensity_box.get_sensitive()
    assert controls.intensity.get_value() == hdr.get_profile("natural").default_intensity

    controls.switch.set_active(True)
    assert app.settings.hdr_enabled and app.settings.hdr().active and app.saves == 1
    assert controls.profile_dd.get_sensitive() and controls.reset_btn.get_visible()
    # A new profile starts at its own default intensity.
    controls.profile_dd.set_selected(ids.index("dramatic"))
    dramatic = hdr.get_profile("dramatic").default_intensity
    assert (app.settings.hdr_profile, app.settings.hdr_intensity) == ("dramatic", dramatic)
    assert controls.intensity.get_value() == dramatic
    # A drag updates the value at once, but notifies and saves only later.
    notifies = app.notifies
    for value in (50, 40, 0):
        controls.intensity.set_value(value)
    assert app.settings.hdr_intensity == 0 and not app.settings.hdr().active
    assert app.notifies == notifies
    controls.advanced_sliders["shadows"].set_value(30)
    assert app.settings.hdr_shadows == 30
    controls.reset_advanced()
    assert app.settings.hdr_shadows == 0 and controls.advanced_sliders["shadows"].get_value() == 0
    controls.reset()
    assert not app.settings.hdr_enabled and not controls.switch.get_active()
    assert app.settings.hdr() == hdr.HdrSettings()
    controls.shutdown()
    assert controls.sync not in app.listeners


def test_stage_preview_transform_applies_hdr_to_crops():
    """The zoomed-in detail of the live preview gets the same HDR as the whole."""
    import numpy as np
    from PIL import Image

    from pixelift.core import camera_looks as cl
    from pixelift.core import hdr

    rgb = np.zeros((90, 120, 3), np.uint8)
    rgb[:45] = 230
    rgb[45:] = (30, 25, 20)
    recipe = hdr.HdrSettings(True, "balanced", 65).recipe()
    plan = hdr.analyse(rgb)
    whole = cl.apply_to_pil(
        Image.fromarray(rgb),
        cl.LookRecipe(),
        between=lambda a: hdr.apply_hdr(a, recipe, out=a, plan=plan),
    )
    assert np.asarray(whole)[60:, :, :].mean() > rgb[60:].mean() + 3


class _RestoreStubApp(_StubApp):
    """Enough of the application for the restoration widgets."""

    def __init__(self, models_dir):
        super().__init__()
        from pixelift.core.model_manager import ModelManager
        from pixelift.ui.downloads import DownloadTracker

        self.settings.mode = "restore"
        self.model_manager = ModelManager(models_dir)
        self.downloads = DownloadTracker(self.model_manager)
        self.settings_listeners = []
        self.changes = 0
        self.shown = []

    def on_settings_changed(self, listener):
        self.settings_listeners.append(listener)

    def off_settings_changed(self, listener):
        self.settings_listeners.remove(listener)

    def settings_changed(self):
        self.changes += 1
        self.settings.normalise()
        for listener in list(self.settings_listeners):
            listener()

    def show_preferences(self, page=None):
        self.shown.append(page)


def test_restoration_dialog_edits_settings(models_dir):
    from pixelift.core.restoration import settings as rs
    from pixelift.ui.widgets.restoration import RestorationDialog, RestorationPanel

    app = _RestoreStubApp(models_dir)
    panel = RestorationPanel(app, lambda: None)
    dialog = RestorationDialog(app)
    assert dialog.face_buttons[rs.FACE_NATURAL].get_active()  # Standard: Natural faces
    assert not dialog.colorize_group.get_sensitive()  # no colorization without consent
    # Moving a damage slider turns the level into Custom, starting from Standard.
    dialog.stage_sliders["dust"].set_value(90)
    assert app.settings.restore_level == rs.CUSTOM and app.settings.restore_dust == 90
    assert app.settings.restore_scratches == rs.LEVEL_STAGES[rs.STANDARD].scratches
    dialog.saver.flush()
    assert panel.level_dd.get_selected() == rs.LEVELS.index(rs.CUSTOM)
    dialog.face_buttons[rs.FACE_OFF].set_active(True)
    assert app.settings.restoration().stages().face == rs.FACE_OFF
    # Choosing a colorize preset in the panel enables the colorization options.
    panel.preset_dd.set_selected(rs.PRESETS.index(rs.PRESET_COLORIZE))
    assert app.settings.restore_preset == rs.PRESET_COLORIZE
    assert dialog.colorize_group.get_sensitive()
    dialog.strength.set_value(25)
    assert "subtle" in dialog.strength_row.get_subtitle()
    # Model status: nothing installed in the test models folder.
    assert all(row.get_subtitle() == "Not installed" for row, _icon in dialog.model_rows)
    dialog._manage_models()
    assert app.shown == ["models"]


def test_black_and_white_banner_and_row(tmp_path):
    from conftest import make_image

    from pixelift.core.batch_processor import QueueItem
    from pixelift.ui.image_queue import QueueRow
    from pixelift.ui.widgets.restoration import BlackAndWhiteBanner
    from pixelift.utils.image_utils import make_thumbnail, probe_image

    choices = []
    banner = BlackAndWhiteBanner(choices.append)
    banner.show_for(2)
    assert banner.get_reveal_child() and "2 Black & White" in banner.title.get_label()
    keep, colorize = list(banner.buttons)
    keep.emit("clicked")
    colorize.emit("clicked")
    assert choices == [False, True]

    path = make_image(tmp_path / "old.png", mode="L")
    row = QueueRow(QueueItem(path), *(lambda _r: None,) * 3)
    row.set_info(probe_image(path), make_thumbnail(path), True)
    row.set_scale(1, show_monochrome=True)
    label = row.details.get_label()
    assert label.startswith("40×30 · ") and label.endswith("→  40×30 · B&W")
    row.set_scale(4, show_monochrome=False)
    assert "B&W" not in row.details.get_label()


def test_camera_look_controls_follow_and_update_settings():
    from pixelift.core import camera_looks as cl
    from pixelift.ui.widgets.camera_looks import CameraLookControls

    app = _StubApp()
    controls = CameraLookControls(app)
    # Defaults: Original at 50%, nothing to reset, intensity and grain not applicable.
    assert controls.look_label.get_label() == "Original"
    assert not controls.reset_btn.get_visible() and not controls.intensity_box.get_sensitive()
    assert not controls.custom_btn.get_visible() and not controls.favorite_btn.get_visible()
    assert len(controls.cards) == len(cl.all_looks())

    controls.select("fujifilm-provia")
    assert app.settings.camera_look == "fujifilm-provia" and app.saves == 1
    assert controls.look_label.get_label() == "Fujifilm Provia"
    assert controls.cards["fujifilm-provia"].has_css_class("selected-look")
    assert controls.intensity_box.get_sensitive() and controls.reset_btn.get_visible()
    # A drag updates the value (and the gallery's slider) at once, but notifies
    # and saves only later.
    notifies = app.notifies
    for value in (60, 70, 80):
        controls.intensity.set_value(value)
    assert app.settings.camera_look_intensity == 80
    assert controls.gallery_intensity.get_value() == 80
    assert app.notifies == notifies
    controls.grain_dd.set_selected(cl.GRAIN_CHOICES.index("medium"))
    assert app.settings.camera_look_grain == "medium"
    controls.reset()
    assert (app.settings.camera_look, app.settings.camera_look_intensity) == ("original", 50)
    assert app.settings.camera_look_grain == cl.GRAIN_AUTO
    controls.shutdown()
    assert controls.sync not in app.listeners


def test_camera_look_favorites_and_categories():
    from pixelift.ui.widgets.camera_looks import CameraLookControls

    app = _StubApp()
    controls = CameraLookControls(app)
    controls.select("portra")
    controls.favorite_btn.set_active(True)
    assert app.settings.camera_look_favorites == ["portra"]
    assert controls.cards["portra"].star.get_active()
    controls.cards["kodak"].star.set_active(True)
    assert app.settings.camera_look_favorites == ["portra", "kodak"]

    def visible():
        return [k for k, card in controls.cards.items() if controls._card_visible(card)]

    controls.filter_buttons["favorites"].set_active(True)
    assert visible() == ["kodak", "portra"]
    controls.filter_buttons["monochrome"].set_active(True)
    assert visible() == ["leica-monochrome", "black-white"]
    controls.filter_buttons["saved"].set_active(True)
    assert visible() == [] and controls.empty_label.get_visible()
    controls.filter_buttons["all"].set_active(True)
    assert len(visible()) == len(controls.cards)
    controls.cards["portra"].star.set_active(False)
    assert app.settings.camera_look_favorites == ["kodak"]
    controls.shutdown()


def test_camera_look_custom_save_and_delete():
    from pixelift.core import camera_looks as cl
    from pixelift.ui.widgets.camera_looks import CameraLookControls, _render_thumbnails

    app = _StubApp()
    app.settings.camera_look = cl.CUSTOM
    app.settings.look_contrast = 25
    controls = CameraLookControls(app)
    assert controls.custom_btn.get_visible() and not controls.intensity_box.get_visible()
    assert controls.custom_sliders["contrast"].get_value() == 25
    controls.custom_sliders["green"].set_value(30)
    assert app.settings.look_green == 30
    controls.save_entry.set_text("Lush")
    controls.save_custom()
    assert app.settings.camera_look == "user:Lush"
    assert app.settings.camera_look_saved["Lush"]["green"] == 30
    assert "user:Lush" in controls.cards and controls.look_label.get_label() == "Lush"
    assert app.settings.camera_look_settings().active
    thumbs = _render_thumbnails(cl.sample_image(), controls._card_recipes())
    assert set(thumbs) == set(controls.cards)
    controls.delete_saved("user:Lush")
    assert app.settings.camera_look == cl.ORIGINAL and "user:Lush" not in controls.cards
    controls.reset_custom()
    assert all(s.get_value() == 0 for s in controls.custom_sliders.values())
    controls.shutdown()


def test_saved_look_opens_at_full_strength_and_refreshes_its_card():
    from pixelift.core import camera_looks as cl
    from pixelift.ui.widgets.camera_looks import CameraLookControls

    app = _StubApp()
    app.settings.camera_look = cl.CUSTOM
    app.settings.look_contrast = 30
    controls = CameraLookControls(app)
    assert app.settings.camera_look_intensity == 50
    designed = app.settings.camera_look_settings().recipe()
    controls.save_entry.set_text("Punchy")
    controls.save_custom()
    # Saving keeps exactly what was designed: no silent halving.
    assert app.settings.camera_look_intensity == 100
    assert app.settings.camera_look_settings().recipe().tone == designed.tone
    # Replacing a saved look re-renders its card.
    controls._thumbs_stale = False
    controls.select(cl.CUSTOM)
    controls.custom_sliders["contrast"].set_value(60)
    controls.save_entry.set_text("Punchy")
    controls.save_custom()
    assert controls._thumbs_stale or controls._thumb_generation > 0
    assert app.settings.camera_look_saved["Punchy"]["contrast"] == 60
    # Choosing a saved card also opens it at full strength.
    controls.select("kodak")
    controls.intensity.set_value(30)
    controls.select("user:Punchy")
    assert app.settings.camera_look_intensity == 100
    controls.shutdown()


def test_startup_logo_timeline():
    from pixelift.ui.startup import INTRO_MS, LOGO_SETTLED_MS, LogoMark

    logo = LogoMark()
    logo.time_ms = 0
    assert logo.entrance() == 0 and logo.glow() == 0 and logo.sweep() is None
    logo.time_ms = LOGO_SETTLED_MS
    assert logo.entrance() == 1  # fully in: the screen may give way
    logo.time_ms = 850
    assert logo.glow() == 1 and 0 < logo.sweep() < 1  # glow peak, sweep crossing
    logo.time_ms = INTRO_MS
    assert logo.sweep() is None  # the sweep runs once
    assert 0.45 < logo.glow() < 0.75  # settled into the slow ambient pulse


def test_startup_screen_states():
    from pixelift.ui.startup import STATUS_STARTING, StartupScreen

    retried, exited = [], []
    screen = StartupScreen(lambda: retried.append(1), lambda: exited.append(1))
    assert screen.status.get_label() == STATUS_STARTING and not screen.error_box.get_visible()
    settled = []
    screen.when_settled(lambda: settled.append(1))
    assert not settled  # not mapped yet: the logo has not come in
    screen._settle()
    screen.when_settled(lambda: settled.append(2))
    assert settled == [1, 2]

    screen.show_error("Traceback: boom")
    assert screen.error_box.get_visible() and not screen.dots.get_visible()
    assert screen.title.get_label() == "Unable to start Pixelift"
    assert "boom" in screen.details.get_label()  # technical details, folded away
    screen.retry_btn.emit("clicked")
    assert retried == [1]
    screen.show_starting()
    assert screen.status.get_visible() and not screen.error_box.get_visible()
    screen.stop()


class _PresetStubApp(_StubApp):
    def settings_changed(self):
        self.lighting_changed()


def test_preset_panel_applies_and_tracks_modified():
    from pixelift.core import camera_looks as cl
    from pixelift.core import presets
    from pixelift.ui.widgets.presets import PresetPanel, _render_thumbnails, preset_recipe

    app = _PresetStubApp()
    panel = PresetPanel(app)
    assert panel.cards["original"].has_css_class("selected-preset")
    assert not panel.cards["old-photo"].get_visible()  # Restore Photos only
    panel.cards["landscape"].activate()  # one click, no confirmation
    assert app.settings.preset == "landscape" and app.settings.camera_look == "nikon-landscape"
    assert panel.cards["landscape"].has_css_class("selected-preset")
    assert not panel.reset_btn.get_visible()
    app.settings.lighting_intensity = 20
    app.lighting_changed()
    assert panel.reset_btn.get_visible() and panel.reset_btn.get_label() == "Reset to Landscape"
    panel.reset_btn.emit("clicked")
    assert presets.label(app.settings) == "Landscape" and not panel.reset_btn.get_visible()

    app.settings.mode = "restore"
    app.lighting_changed()
    assert panel.cards["old-photo"].get_visible() and not panel.cards["portrait"].get_visible()
    # Thumbnails: every preset rendered from the user's photo, settings untouched.
    recipes = {p.id: preset_recipe(app.settings, p.id) for p in presets.all_presets()}
    assert app.settings.preset == "landscape"
    assert set(_render_thumbnails(cl.sample_image(), recipes)) == set(panel.cards)
    panel.shutdown()
    assert panel.sync not in app.listeners


def test_preferences_show_and_clear_the_upscale_cache(models_dir):
    import numpy as np

    from pixelift.core import upscale_cache
    from pixelift.ui.settings import PreferencesDialog

    app = _RestoreStubApp(models_dir)
    app.device_report = None
    app.when_devices_ready = lambda _callback: None
    assert upscale_cache.store("k", np.zeros((64, 64, 3), np.uint8))
    dialog = PreferencesDialog(app)
    assert "KB" in dialog.cache_row.get_subtitle() and dialog.cache_clear.get_sensitive()
    dialog.cache_clear.emit("clicked")
    assert upscale_cache.total_bytes() == 0
    assert dialog.cache_row.get_subtitle().startswith("Empty")
    assert not dialog.cache_clear.get_sensitive()


class _SettingsStubApp(_StubApp):
    """Adds the settings-changed listeners the Export panel's controls use."""

    def __init__(self):
        super().__init__()
        self.settings_listeners = []
        self.changes = 0

    def on_settings_changed(self, listener):
        self.settings_listeners.append(listener)

    def off_settings_changed(self, listener):
        self.settings_listeners.remove(listener)

    def settings_changed(self):
        self.changes += 1
        self.settings.normalise()
        for listener in list(self.settings_listeners):
            listener()


def test_hdr_output_controls_follow_and_update_settings(monkeypatch):
    from pixelift.core.hdr_output import codecs
    from pixelift.ui.widgets.hdr_output import FORMAT_IDS, PRIMARY_IDS, HdrOutputControls

    monkeypatch.setattr(codecs, "avif_available", lambda: False)
    app = _SettingsStubApp()
    controls = HdrOutputControls(app)
    _pump_until(lambda: controls._avif is not None)  # probed off the main thread
    assert controls.format_model.get_string(FORMAT_IDS.index("avif")).endswith("(unavailable)")
    # SDR by default: the HDR options stay hidden.
    assert controls.range_buttons["sdr"].get_active() and not controls.hdr_box.get_visible()
    controls.set_source(False)
    assert controls.status.get_label() == "SDR source → SDR output"

    controls.range_buttons["hdr"].set_active(True)
    assert app.settings.output_range == "hdr" and app.settings.hdr_output().enabled
    assert controls.hdr_box.get_visible()
    assert "SDR source → HDR conversion" in controls.status.get_label()
    assert "no lost detail is recovered" in controls.status.get_label()
    changes = app.changes
    controls.peak.set_value(4000)
    assert app.settings.hdr_peak_nits == 4000
    assert app.changes == changes  # spin steps are debounced: saved once settled
    _pump_until(lambda: app.changes == changes + 1)
    controls.primaries_dd.set_selected(PRIMARY_IDS.index("p3"))
    controls.white.set_value(250)
    controls.expand.set_active(True)
    assert app.settings.hdr_output().tag() == "pq4000-p3-w250-x"
    # A missing encoder is a visible warning, not a silent SDR fallback.
    controls.format_dd.set_selected(FORMAT_IDS.index("avif"))
    assert "libaom" in controls.status.get_label()
    assert controls.status.has_css_class("warning")
    # An HDR source written as SDR warns that its range is discarded.
    controls.range_buttons["sdr"].set_active(True)
    controls.set_source(True)
    assert "discarded" in controls.status.get_label()
    app.settings.mode = "restore"
    app.settings_changed()
    assert not controls.get_sensitive() and "SDR" in controls.status.get_label()


def test_hdr_output_controls_transfer_and_gain_map():
    from pixelift.ui.widgets.hdr_output import FORMAT_IDS, TRANSFER_IDS, HdrOutputControls

    app = _SettingsStubApp()
    controls = HdrOutputControls(app)
    controls.range_buttons["hdr"].set_active(True)
    assert controls.peak.get_sensitive() and controls.primaries_dd.get_sensitive()
    controls.transfer_dd.set_selected(TRANSFER_IDS.index("hlg"))
    assert app.settings.hdr_transfer == "hlg" and app.settings.hdr_output().hlg
    # HLG fixes the peak (1000 cd/m² reference) and the primaries (Rec. 2020).
    assert not controls.peak.get_sensitive() and not controls.primaries_dd.get_sensitive()
    assert "HLG" in controls.status.get_label()
    controls.format_dd.set_selected(FORMAT_IDS.index("jpeg"))
    assert app.settings.hdr_output().gain_map
    assert not controls.transfer_dd.get_visible() and controls.peak.get_sensitive()
    assert "gain-map JPEG" in controls.status.get_label()


def test_hdr_output_controls_shutdown_unsubscribes():
    from pixelift.ui.widgets.hdr_output import HdrOutputControls

    app = _SettingsStubApp()
    controls = HdrOutputControls(app)
    assert controls.sync in app.settings_listeners
    controls.shutdown()
    assert controls.sync not in app.settings_listeners
    controls.set_source(True, gain_map=True)
    assert "its own SDR picture" in controls.status.get_label()
    assert "gain map is not used" in controls.status.get_label()


def test_hdr_output_controls_avif_probe_off_main_thread(monkeypatch):
    import threading

    from pixelift.core.hdr_output import codecs
    from pixelift.ui.widgets.hdr_output import FORMAT_IDS, HdrOutputControls

    release = threading.Event()
    main = threading.current_thread()
    threads = []

    def probe():
        threads.append(threading.current_thread())
        release.wait(5)
        return True

    monkeypatch.setattr(codecs, "avif_available", probe)
    app = _SettingsStubApp()
    app.settings.output_range, app.settings.hdr_output_format = "hdr", "avif"
    controls = HdrOutputControls(app)  # returns at once: the probe is still blocked
    avif = FORMAT_IDS.index("avif")
    assert "unavailable" not in controls.format_model.get_string(avif)  # neutral until known
    assert "libaom" not in controls.status.get_label()  # unknown yet: no warning
    release.set()
    _pump_until(lambda: controls._avif is True)
    assert threads and threads[0] is not main
    assert "unavailable" not in controls.format_model.get_string(avif)
    assert controls.format_dd.get_selected() == avif
    controls.shutdown()


def test_hdr_output_controls_flush_debounced_save_on_shutdown():
    from pixelift.ui.widgets.hdr_output import HdrOutputControls

    app = _SettingsStubApp()
    app.settings.output_range = "hdr"
    controls = HdrOutputControls(app)
    changes = app.changes
    controls.white.set_value(250)
    controls.white.set_value(260)
    assert app.settings.hdr_sdr_white == 260 and app.changes == changes
    controls.shutdown()  # a pending save is not lost
    assert app.changes == changes + 1


def test_add_paths_collects_off_main_thread_and_dedups(tmp_path):
    import time

    from conftest import make_image
    from gi.repository import GLib

    from pixelift.ui import main_window as mw

    folder = tmp_path / "photos"
    folder.mkdir()
    first, second = make_image(folder / "a.png"), make_image(folder / "b.png")
    added, toasts = [], []

    class Fake:
        ready, _closed = True, False
        add_paths = mw.MainWindow.add_paths
        _add_collected = mw.MainWindow._add_collected

        def __init__(self):
            self.rows = [type("R", (), {"item": mw.QueueItem(first)})()]
            self.toasts = type("T", (), {"add_toast": lambda _s, t: toasts.append(t)})()

        def _add_row(self, item):
            added.append(item.path)

        def _update_state(self):
            pass

    window = Fake()
    window.add_paths([folder])
    assert not added  # collected in the background
    _pump_until(lambda: bool(toasts))
    assert added == [second]  # a.png is already queued
    # A window closed meanwhile ignores the late result.
    window.rows, window._closed = [], True
    toasts.clear()
    window.add_paths([folder])
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        GLib.MainContext.default().iteration(False)
    assert added == [second] and not toasts
