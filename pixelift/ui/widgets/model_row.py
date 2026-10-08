"""A row showing one AI model with download / progress / remove controls."""

from __future__ import annotations

from collections.abc import Callable

from gi.repository import Adw, Gtk

from pixelift.models import ModelSpec
from pixelift.ui.downloads import DownloadState, DownloadTracker
from pixelift.ui.widgets.dialogs import show_error
from pixelift.utils.image_utils import human_size


class ModelRow(Adw.ActionRow):
    def __init__(
        self, spec: ModelSpec, tracker: DownloadTracker, allow_remove: bool = True
    ) -> None:
        super().__init__(title=spec.name)
        self.spec = spec
        self.tracker = tracker
        self.allow_remove = allow_remove
        self.set_subtitle(
            f"{spec.description}\n{human_size(spec.size_bytes)} · {spec.native_scale}× · "
            f"{spec.license}"
        )
        self.set_subtitle_lines(3)

        self.status_icon = Gtk.Image(icon_name="object-select-symbolic", tooltip_text="Installed")
        self.status_icon.add_css_class("success")
        self.progress = Gtk.ProgressBar(valign=Gtk.Align.CENTER, width_request=90, show_text=True)
        self.download_btn = Gtk.Button(label="Download", valign=Gtk.Align.CENTER)
        self.download_btn.add_css_class("suggested-action")
        self.download_btn.connect("clicked", lambda _b: tracker.start(spec.id))
        self.cancel_btn = Gtk.Button(
            icon_name="process-stop-symbolic",
            valign=Gtk.Align.CENTER,
            tooltip_text="Cancel download",
        )
        self.cancel_btn.add_css_class("flat")
        self.cancel_btn.connect("clicked", lambda _b: tracker.cancel(spec.id))
        self.remove_btn = Gtk.Button(
            icon_name="user-trash-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Remove model"
        )
        self.remove_btn.add_css_class("flat")
        self.remove_btn.connect("clicked", lambda _b: tracker.remove(spec.id))
        for widget in (
            self.status_icon,
            self.progress,
            self.download_btn,
            self.cancel_btn,
            self.remove_btn,
        ):
            self.add_suffix(widget)

        self._unsubscribe: Callable[[], None] = tracker.subscribe(self._on_change)
        self.connect("destroy", lambda _w: self._unsubscribe())
        self.connect("unrealize", lambda _w: self._unsubscribe())
        self.refresh()

    def _on_change(self, model_id: str, state: DownloadState | None) -> None:
        if model_id != self.spec.id:
            return
        self.refresh()
        if state is not None and state.error is not None and not state.active:
            show_error(self, state.error)
            self.tracker.states.pop(model_id, None)

    def refresh(self) -> None:
        installed = self.tracker.manager.is_installed(self.spec)
        state = self.tracker.states.get(self.spec.id)
        downloading = bool(state and state.active)
        self.status_icon.set_visible(installed and not downloading)
        self.remove_btn.set_visible(installed and self.allow_remove and not downloading)
        self.download_btn.set_visible(not installed and not downloading)
        self.progress.set_visible(downloading)
        self.cancel_btn.set_visible(downloading)
        if downloading and state:
            self.progress.set_fraction(state.fraction)
            self.progress.set_text(f"{state.fraction * 100:.0f}%")
