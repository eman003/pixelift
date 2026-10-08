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
    row = QueueRow(QueueItem(path), *(lambda _r: None,) * 4)
    row.set_info(probe_image(path), make_thumbnail(path))
    assert row.ready
    assert "40×30" in row.details.get_label() and "160×120" in row.details.get_label()
    row.item.status = ItemStatus.PROCESSING
    row.item.progress = 0.5
    row.refresh()
    assert "50%" in row.status.get_label() and not row.remove_btn.get_sensitive()


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
