"""HDR Enhancement controls (on/off, profile, intensity, Advanced) bound to the settings.

Used by the main window's control panel; it stays in sync through the
application's settings-changed listeners. Changes notify the lighting
listeners, like the lighting and camera look: the stage preview updates and
results made with other settings are queued again (reusing the cached
upscale, so the AI model does not run again).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gi.repository import Gtk

from pixelift.core import hdr
from pixelift.ui.widgets.lighting import LiveSettingsBox, _slider

if TYPE_CHECKING:
    from pixelift.ui.application import UpscalerApplication

ADVANCED_LABELS = {
    "shadows": "Shadow Recovery",
    "highlights": "Highlight Protection",
    "local_contrast": "Local Contrast",
    "saturation": "Colour",
}


class HdrControls(LiveSettingsBox):
    def __init__(self, app: UpscalerApplication, label_width: int = 6) -> None:
        super().__init__(app, spacing=8)
        self._profiles = hdr.all_profiles()

        title = Gtk.Label(label="HDR", xalign=0, width_chars=label_width)
        title.add_css_class("dim-label")
        self.append(title)

        self.switch = Gtk.Switch(valign=Gtk.Align.CENTER, tooltip_text="HDR Enhancement")
        self.switch.update_property([Gtk.AccessibleProperty.LABEL], ["HDR Enhancement"])
        self.switch.connect("notify::active", self._on_enabled_changed)
        self.append(self.switch)

        self.profile_dd = Gtk.DropDown.new_from_strings([p.name for p in self._profiles])
        self.profile_dd.update_property([Gtk.AccessibleProperty.LABEL], ["HDR profile"])
        self.profile_dd.connect("notify::selected", self._on_profile_changed)
        self.append(self.profile_dd)

        self.intensity_box = Gtk.Box(spacing=6, hexpand=True)
        self.intensity = _slider(0, 100)
        self.intensity.set_tooltip_text("Intensity: 0% is the upscaled image without HDR")
        self.intensity.update_property([Gtk.AccessibleProperty.LABEL], ["HDR intensity"])
        self.intensity.connect("value-changed", self._on_intensity_changed)
        self.intensity_label = Gtk.Label(width_chars=4, xalign=1)
        self.intensity_label.add_css_class("numeric")
        self.intensity_label.add_css_class("caption")
        self.intensity_box.append(self.intensity)
        self.intensity_box.append(self.intensity_label)
        self.append(self.intensity_box)

        self.advanced_btn = Gtk.MenuButton(
            icon_name="emblem-system-symbolic",
            tooltip_text="Advanced HDR Settings",
            popover=self._build_advanced_popover(),
        )
        self.append(self.advanced_btn)

        self.reset_btn = Gtk.Button(icon_name="edit-undo-symbolic", tooltip_text="Reset HDR")
        self.reset_btn.connect("clicked", lambda _b: self.reset())
        self.append(self.reset_btn)

        app.on_lighting_changed(self.sync)
        self.connect("destroy", lambda _w: self.shutdown())
        self.sync()

    def _build_advanced_popover(self) -> Gtk.Popover:
        grid = Gtk.Grid(
            row_spacing=6,
            column_spacing=12,
            margin_top=6,
            margin_bottom=6,
            margin_start=6,
            margin_end=6,
        )
        note = Gtk.Label(
            label="Fine-tune the profile: 0 is the profile as designed.",
            xalign=0,
            wrap=True,
            max_width_chars=40,
        )
        note.add_css_class("dim-label")
        note.add_css_class("caption")
        grid.attach(note, 0, 0, 3, 1)
        self.advanced_sliders: dict[str, Gtk.Scale] = {}
        for row, name in enumerate(hdr.ADVANCED_NAMES, start=1):
            label = Gtk.Label(label=ADVANCED_LABELS[name], xalign=0)
            slider = _slider(-100, 100)
            slider.set_size_request(220, -1)
            slider.add_mark(0, Gtk.PositionType.BOTTOM, None)
            slider.update_property([Gtk.AccessibleProperty.LABEL], [ADVANCED_LABELS[name]])
            slider.connect("value-changed", self._on_advanced_changed, name)
            value = Gtk.Label(width_chars=4, xalign=1)
            value.add_css_class("numeric")
            value.add_css_class("caption")
            slider.connect("value-changed", lambda s, v=value: v.set_label(f"{s.get_value():+.0f}"))
            grid.attach(label, 0, row, 1, 1)
            grid.attach(slider, 1, row, 1, 1)
            grid.attach(value, 2, row, 1, 1)
            self.advanced_sliders[name] = slider
        reset = Gtk.Button(label="Reset", halign=Gtk.Align.END, margin_top=6)
        reset.connect("clicked", lambda _b: self.reset_advanced())
        grid.attach(reset, 0, len(hdr.ADVANCED_NAMES) + 1, 3, 1)
        return Gtk.Popover(child=grid)

    # --- settings -> widgets ----------------------------------------------
    def sync(self) -> None:
        settings = self.app.settings
        ids = [p.id for p in self._profiles]
        self._syncing = True
        try:
            self.switch.set_active(settings.hdr_enabled)
            if settings.hdr_profile in ids:
                self.profile_dd.set_selected(ids.index(settings.hdr_profile))
            self.intensity.set_value(settings.hdr_intensity)
            for name, slider in self.advanced_sliders.items():
                slider.set_value(getattr(settings, f"hdr_{name}"))
        finally:
            self._syncing = False
        self._update_state()

    def _update_state(self) -> None:
        settings = self.app.settings
        enabled = settings.hdr_enabled
        for widget in (self.profile_dd, self.intensity_box, self.advanced_btn):
            widget.set_sensitive(enabled)
        self.intensity_label.set_label(f"{settings.hdr_intensity}%")
        if hdr.has_profile(settings.hdr_profile):
            profile = hdr.get_profile(settings.hdr_profile)
            self.profile_dd.set_tooltip_text(profile.description)
            default = profile.default_intensity
        else:
            default = settings.hdr_intensity
        advanced = any(getattr(settings, f"hdr_{n}") for n in hdr.ADVANCED_NAMES)
        if advanced:
            self.advanced_btn.add_css_class("accent")
        else:
            self.advanced_btn.remove_css_class("accent")
        self.reset_btn.set_visible(enabled or advanced or settings.hdr_intensity != default)

    # --- widgets -> settings ----------------------------------------------
    def _on_enabled_changed(self, *_args: object) -> None:
        if self._syncing:
            return
        self.app.settings.hdr_enabled = self.switch.get_active()
        self._update_state()
        self._save(now=True)

    def _on_profile_changed(self, *_args: object) -> None:
        if self._syncing:
            return
        index = self.profile_dd.get_selected()
        if index >= len(self._profiles):
            return
        profile = self._profiles[index]
        if profile.id == self.app.settings.hdr_profile:
            return
        self.app.settings.hdr_profile = profile.id
        # Each profile starts at its own, tested strength.
        self.app.settings.hdr_intensity = profile.default_intensity
        self._syncing = True
        try:
            self.intensity.set_value(profile.default_intensity)
        finally:
            self._syncing = False
        self._update_state()
        self._save(now=True)

    def _on_intensity_changed(self, scale: Gtk.Scale) -> None:
        if self._syncing:
            return
        self.app.settings.hdr_intensity = round(scale.get_value())
        self._update_state()
        self._save()

    def _on_advanced_changed(self, scale: Gtk.Scale, name: str) -> None:
        if self._syncing:
            return
        setattr(self.app.settings, f"hdr_{name}", round(scale.get_value()))
        self._update_state()
        self._save()

    def reset(self) -> None:
        """Back to the default: off, Natural HDR at its own intensity."""
        settings = self.app.settings
        settings.hdr_enabled = False
        settings.hdr_profile = hdr.DEFAULT_PROFILE
        settings.hdr_intensity = hdr.get_profile(hdr.DEFAULT_PROFILE).default_intensity
        for name in hdr.ADVANCED_NAMES:
            setattr(settings, f"hdr_{name}", 0)
        self._save(now=True)

    def reset_advanced(self) -> None:
        for name in hdr.ADVANCED_NAMES:
            setattr(self.app.settings, f"hdr_{name}", 0)
        self._save(now=True)

    def shutdown(self) -> None:
        """Flush pending changes and stop listening (when the window closes)."""
        self._flush_pending()
        self.app.off_lighting_changed(self.sync)
