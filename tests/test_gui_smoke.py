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
