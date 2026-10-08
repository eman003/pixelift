"""Lighting controls (profile, intensity, Custom adjustments) bound to the settings.

Used by the main window's control panel and by the preview window; both stay
in sync through the application's settings-changed listeners.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gi.repository import GLib, Gtk

from pixelift.core import lighting

if TYPE_CHECKING:
    from pixelift.ui.application import UpscalerApplication

ADJUSTMENT_LABELS = {
    "exposure": "Exposure",
    "brightness": "Brightness",
    "contrast": "Contrast",
    "highlights": "Highlights",
    "shadows": "Shadows",
    "temperature": "Temperature",
    "tint": "Tint",
    "saturation": "Saturation",
}
# Slider drags: listeners (live previews) hear about changes after a short
# pause; the settings file is only written once the drag has settled.
NOTIFY_DELAY_MS = 120
SAVE_DELAY_MS = 800


def _slider(low: int, high: int) -> Gtk.Scale:
    scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, low, high, 1)
    scale.set_draw_value(False)
    scale.set_hexpand(True)
    scale.set_size_request(120, -1)
    return scale


class LightingControls(Gtk.Box):
    def __init__(self, app: UpscalerApplication, label_width: int = 6) -> None:
        super().__init__(spacing=8)
        self.app = app
        self._syncing = False
        self._notify_source = 0
        self._save_source = 0
        self._profiles = lighting.all_profiles()

        title = Gtk.Label(label="Lighting", xalign=0, width_chars=label_width)
        title.add_css_class("dim-label")
        self.append(title)

        self.profile_dd = Gtk.DropDown.new_from_strings([p.name for p in self._profiles])
        self.profile_dd.set_tooltip_text("Lighting adjustment applied before upscaling")
        self.profile_dd.connect("notify::selected", self._on_profile_changed)
        self.append(self.profile_dd)

        self.intensity_box = Gtk.Box(spacing=6, hexpand=True)
        self.intensity = _slider(0, 100)
        self.intensity.set_tooltip_text("Intensity: 0% is the original image")
        self.intensity.connect("value-changed", self._on_intensity_changed)
        self.intensity_label = Gtk.Label(width_chars=4, xalign=1)
        self.intensity_label.add_css_class("numeric")
        self.intensity_label.add_css_class("caption")
        self.intensity_box.append(self.intensity)
        self.intensity_box.append(self.intensity_label)
        self.append(self.intensity_box)

        self.custom_btn = Gtk.MenuButton(
            icon_name="document-edit-symbolic",
            tooltip_text="Custom Adjustments",
            popover=self._build_custom_popover(),
        )
        self.append(self.custom_btn)

        self.reset_btn = Gtk.Button(icon_name="edit-undo-symbolic", tooltip_text="Reset Lighting")
        self.reset_btn.connect("clicked", lambda _b: self.reset())
        self.append(self.reset_btn)

        app.on_lighting_changed(self.sync)
        self.connect("destroy", lambda _w: self.shutdown())
        self.sync()

    def _build_custom_popover(self) -> Gtk.Popover:
        grid = Gtk.Grid(
            row_spacing=6,
            column_spacing=12,
            margin_top=6,
            margin_bottom=6,
            margin_start=6,
            margin_end=6,
        )
        self.custom_sliders: dict[str, Gtk.Scale] = {}
        for row, name in enumerate(lighting.ADJUSTMENT_NAMES):
            label = Gtk.Label(label=ADJUSTMENT_LABELS.get(name, name.title()), xalign=0)
            slider = _slider(-100, 100)
            slider.set_size_request(220, -1)
            slider.add_mark(0, Gtk.PositionType.BOTTOM, None)
            slider.connect("value-changed", self._on_custom_changed, name)
            value = Gtk.Label(width_chars=4, xalign=1)
            value.add_css_class("numeric")
            value.add_css_class("caption")
            slider.connect("value-changed", lambda s, v=value: v.set_label(f"{s.get_value():+.0f}"))
            grid.attach(label, 0, row, 1, 1)
            grid.attach(slider, 1, row, 1, 1)
            grid.attach(value, 2, row, 1, 1)
            self.custom_sliders[name] = slider
        reset = Gtk.Button(label="Reset", halign=Gtk.Align.END, margin_top=6)
        reset.connect("clicked", lambda _b: self.reset_custom())
        grid.attach(reset, 0, len(lighting.ADJUSTMENT_NAMES), 3, 1)
        return Gtk.Popover(child=grid)

    # --- settings -> widgets ----------------------------------------------
    def sync(self) -> None:
        settings = self.app.settings
        ids = [p.id for p in self._profiles]
        profile = settings.lighting_profile
        self._syncing = True
        try:
            self.profile_dd.set_selected(ids.index(profile) if profile in ids else 0)
            self.intensity.set_value(settings.lighting_intensity)
            for name, slider in self.custom_sliders.items():
                slider.set_value(getattr(settings, f"lighting_{name}"))
        finally:
            self._syncing = False
        self._update_state()

    def _update_state(self) -> None:
        settings = self.app.settings
        profile = settings.lighting_profile
        custom = profile == lighting.CUSTOM
        self.intensity_box.set_visible(not custom)
        self.intensity_box.set_sensitive(profile != lighting.ORIGINAL)
        self.intensity_label.set_label(f"{settings.lighting_intensity}%")
        self.custom_btn.set_visible(custom)
        if profile in {p.id for p in self._profiles}:
            self.profile_dd.set_tooltip_text(lighting.get_profile(profile).description)
        self.reset_btn.set_visible(
            profile != lighting.ORIGINAL or settings.lighting_intensity != 100
        )

    # --- widgets -> settings ----------------------------------------------
    def _on_profile_changed(self, *_args: object) -> None:
        if self._syncing:
            return
        index = self.profile_dd.get_selected()
        if index >= len(self._profiles):
            return
        self.app.settings.lighting_profile = self._profiles[index].id
        self._update_state()
        self._save(now=True)
        if self.app.settings.lighting_profile == lighting.CUSTOM:
            self.custom_btn.popup()

    def _on_intensity_changed(self, scale: Gtk.Scale) -> None:
        if self._syncing:
            return
        self.app.settings.lighting_intensity = round(scale.get_value())
        self._update_state()
        self._save()

    def _on_custom_changed(self, scale: Gtk.Scale, name: str) -> None:
        if self._syncing:
            return
        setattr(self.app.settings, f"lighting_{name}", round(scale.get_value()))
        self._save()

    def reset(self) -> None:
        """Back to the default: Original at 100%."""
        self.app.settings.lighting_profile = lighting.ORIGINAL
        self.app.settings.lighting_intensity = 100
        self._save(now=True)

    def reset_custom(self) -> None:
        for name in lighting.ADJUSTMENT_NAMES:
            setattr(self.app.settings, f"lighting_{name}", 0)
        self._save(now=True)

    def _save(self, now: bool = False) -> None:
        """Notify lighting listeners and save, at once or debounced."""
        self._cancel_pending()
        if now:
            self.app.lighting_changed()
            self.app.save_settings()
            return

        def notify() -> bool:
            self._notify_source = 0
            self.app.lighting_changed()
            return GLib.SOURCE_REMOVE

        def save() -> bool:
            self._save_source = 0
            self.app.save_settings()
            return GLib.SOURCE_REMOVE

        self._notify_source = GLib.timeout_add(NOTIFY_DELAY_MS, notify)
        self._save_source = GLib.timeout_add(SAVE_DELAY_MS, save)

    def _cancel_pending(self) -> tuple[bool, bool]:
        """Remove pending notify/save timeouts; returns which were pending."""
        pending = (bool(self._notify_source), bool(self._save_source))
        for source in (self._notify_source, self._save_source):
            if source:
                GLib.source_remove(source)
        self._notify_source = self._save_source = 0
        return pending

    def shutdown(self) -> None:
        """Flush pending changes and stop listening (when the window closes)."""
        notify, save = self._cancel_pending()
        if notify:
            self.app.lighting_changed()
        if save:
            self.app.save_settings()
        self.app.off_lighting_changed(self.sync)
