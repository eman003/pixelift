"""Photo-restoration controls: the main-window panel, the B&W banner and the options dialog.

Everything reads and writes the application settings (``restore_*`` fields);
the widgets stay in sync through the application's settings listeners, like
the lighting controls.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from gi.repository import Adw, GLib, Gtk

from pixelift.core.restoration import settings as rs
from pixelift.models import COLORIZE, FACE_DETECT, FACE_RESTORE, UPSCALE, specs_of_kind

if TYPE_CHECKING:
    from pixelift.ui.application import UpscalerApplication

PRIVACY_TEXT = (
    "🔒 Restoration runs entirely on this computer. Your photos are never uploaded — "
    "no account, cloud service or internet connection is needed (AI models are "
    "downloaded once, only when you choose to)."
)
STAGE_LABELS = {
    "dust": ("Dust", "Specks, spots and pinholes"),
    "scratches": ("Scratches", "Thin lines and cracks"),
    "noise": ("Noise", "Film grain and scanner noise"),
    "fading": ("Fading", "Recover faded contrast and colour"),
    "sharpness": ("Sharpness", "Bring back fine detail"),
}
COLOR_LABELS = {
    "temperature": "Temperature",
    "tint": "Tint",
    "exposure": "Exposure",
    "contrast": "Contrast",
    "saturation": "Saturation",
}
SAVE_DELAY_MS = 300


def _slider(low: int, high: int, marks: tuple[int, ...] = ()) -> Gtk.Scale:
    scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, low, high, 1)
    scale.set_draw_value(False)
    scale.set_hexpand(True)
    scale.set_valign(Gtk.Align.CENTER)
    scale.set_size_request(180, -1)
    for mark in marks:
        scale.add_mark(mark, Gtk.PositionType.BOTTOM, None)
    return scale


class _Saver:
    """Debounced ``settings_changed`` for slider drags."""

    def __init__(self, app: UpscalerApplication) -> None:
        self.app = app
        self._source = 0

    def now(self) -> None:
        self.cancel()
        self.app.settings_changed()

    def later(self) -> None:
        self.cancel()

        def fire() -> bool:
            self._source = 0
            self.app.settings_changed()
            return GLib.SOURCE_REMOVE

        self._source = GLib.timeout_add(SAVE_DELAY_MS, fire)

    def cancel(self) -> bool:
        pending = bool(self._source)
        if self._source:
            GLib.source_remove(self._source)
            self._source = 0
        return pending

    def flush(self) -> None:
        if self.cancel():
            self.app.settings_changed()


def edit_stage(app: UpscalerApplication, name: str, value: object) -> None:
    """Change one stage; a preset level becomes Custom (starting from its values)."""
    settings = app.settings
    if settings.restore_level != rs.CUSTOM:
        settings.set_restoration_stages(rs.LEVEL_STAGES[settings.restore_level])
        settings.restore_level = rs.CUSTOM
    setattr(settings, f"restore_{name}", value)


def model_summary(app: UpscalerApplication) -> list[tuple[str, bool]]:
    """(label, installed) for each kind of AI model restoration can use."""
    manager = app.model_manager
    upscalers = specs_of_kind(UPSCALE)
    faces = specs_of_kind(FACE_RESTORE) + specs_of_kind(FACE_DETECT)
    colorizers = specs_of_kind(COLORIZE)
    return [
        ("Real-ESRGAN", any(manager.is_installed(s) for s in upscalers)),
        ("Face Restoration", bool(faces) and all(manager.is_installed(s) for s in faces)),
        ("Photo Colorization", any(manager.is_installed(s) for s in colorizers)),
    ]


class RestorationPanel(Gtk.Box):
    """Preset and level pickers plus the Options button (main window, Restore mode)."""

    def __init__(self, app: UpscalerApplication, on_options: Callable[[], None]) -> None:
        super().__init__(spacing=18)
        self.app = app
        self._syncing = False

        self.preset_dd = Gtk.DropDown.new_from_strings([rs.PRESET_LABELS[p] for p in rs.PRESETS])
        self.preset_dd.set_tooltip_text("What to do with each photograph")
        self.preset_dd.connect("notify::selected", self._on_preset)
        self.level_dd = Gtk.DropDown.new_from_strings([rs.LEVEL_LABELS[lv] for lv in rs.LEVELS])
        self.level_dd.connect("notify::selected", self._on_level)
        options = Gtk.Button(label="Options…", tooltip_text="Restoration options")
        options.connect("clicked", lambda _b: on_options())

        self.append(_labeled("Restore", self.preset_dd))
        self.append(_labeled("Level", self.level_dd))
        self.append(options)
        app.on_settings_changed(self.sync)
        self.sync()

    def sync(self) -> None:
        settings = self.app.settings
        self._syncing = True
        try:
            self.preset_dd.set_selected(rs.PRESETS.index(settings.restore_preset))
            self.level_dd.set_selected(rs.LEVELS.index(settings.restore_level))
        finally:
            self._syncing = False
        self.level_dd.set_tooltip_text(rs.LEVEL_DESCRIPTIONS[settings.restore_level])

    def _on_preset(self, *_args: object) -> None:
        index = self.preset_dd.get_selected()
        if self._syncing or index >= len(rs.PRESETS):
            return
        self.app.settings.restore_preset = rs.PRESETS[index]
        self.app.settings_changed()

    def _on_level(self, *_args: object) -> None:
        index = self.level_dd.get_selected()
        if self._syncing or index >= len(rs.LEVELS):
            return
        self.app.settings.restore_level = rs.LEVELS[index]
        self.app.settings_changed()


def _labeled(label: str, widget: Gtk.Widget) -> Gtk.Box:
    box = Gtk.Box(spacing=8)
    title = Gtk.Label(label=label, xalign=0, width_chars=6)
    title.add_css_class("dim-label")
    box.append(title)
    widget.set_hexpand(True)
    box.append(widget)
    return box


class BlackAndWhiteBanner(Gtk.Revealer):
    """Asks before colorizing: shown when black-and-white photos are in the queue."""

    def __init__(self, on_choice: Callable[[bool], None]) -> None:
        super().__init__(transition_type=Gtk.RevealerTransitionType.SLIDE_DOWN)
        box = Gtk.Box(spacing=12, margin_bottom=12)
        box.add_css_class("card")
        box.add_css_class("bw-banner")
        icon = Gtk.Image(icon_name="image-x-generic-symbolic", pixel_size=32)
        text = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
        self.title = Gtk.Label(label="Black & White Photo Detected", xalign=0)
        self.title.add_css_class("heading")
        self.body = Gtk.Label(xalign=0, wrap=True)
        self.body.add_css_class("caption")
        text.append(self.title)
        text.append(self.body)
        buttons = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
        keep = Gtk.Button(label="Restore in B&W")
        keep.connect("clicked", lambda _b: on_choice(False))
        color = Gtk.Button(label="Restore & Colorize")
        color.add_css_class("suggested-action")
        color.connect("clicked", lambda _b: on_choice(True))
        buttons.append(keep)
        buttons.append(color)
        self.buttons = buttons
        for widget in (icon, text, buttons):
            box.append(widget)
        self.set_child(box)

    def show_for(self, count: int) -> None:
        self.title.set_label(
            "Black & White Photo Detected"
            if count == 1
            else f"{count} Black & White Photos Detected"
        )
        self.body.set_label(
            "Keep the original black and white, or add natural-looking colour? "
            "Colour photographs are never colorized."
        )
        self.set_reveal_child(True)


class RestorationDialog(Adw.PreferencesDialog):
    """All restoration options, grouped like the restoration pipeline."""

    def __init__(self, app: UpscalerApplication) -> None:
        super().__init__(title="AI Photo Restoration", search_enabled=False)
        self.app = app
        self.saver = _Saver(app)
        self._syncing = False
        page = Adw.PreferencesPage(title="Restoration", icon_name="image-x-generic-symbolic")
        self.add(page)
        self._build_mode(page)
        self._build_faces(page)
        self._build_damage(page)
        self._build_color(page)
        self._build_colorize(page)
        self._build_finish(page)
        self._build_models(page)
        app.on_settings_changed(self.sync)
        self._unsubscribe = app.downloads.subscribe(lambda *_: self._sync_models())
        self.connect("closed", self._on_closed)
        self.sync()

    # --- construction ------------------------------------------------------
    def _build_mode(self, page: Adw.PreferencesPage) -> None:
        group = Adw.PreferencesGroup(title="Restoration", description=PRIVACY_TEXT)
        page.add(group)
        self.preset_row = Adw.ComboRow(
            title="Processing",
            model=Gtk.StringList.new([rs.PRESET_LABELS[p] for p in rs.PRESETS]),
        )
        self.preset_row.connect("notify::selected", self._on_preset)
        group.add(self.preset_row)
        self.level_row = Adw.ComboRow(
            title="Restoration level",
            model=Gtk.StringList.new([rs.LEVEL_LABELS[lv] for lv in rs.LEVELS]),
        )
        self.level_row.connect("notify::selected", self._on_level)
        group.add(self.level_row)
        self.scale_row = Adw.ComboRow(
            title="Upscale by", model=Gtk.StringList.new(["2×", "4×"]), subtitle="Real-ESRGAN"
        )
        self.scale_row.connect("notify::selected", self._on_scale)
        group.add(self.scale_row)

    def _build_faces(self, page: Adw.PreferencesPage) -> None:
        group = Adw.PreferencesGroup(
            title="Face Restoration",
            description="Natural keeps the person's identity: face shape, age, skin tone, "
            "hair and expression stay as photographed. Faces too damaged to restore "
            "reliably are left close to the original.",
        )
        page.add(group)
        row = Adw.ActionRow(title="Faces")
        box = Gtk.Box(css_classes=["linked"], valign=Gtk.Align.CENTER)
        self.face_buttons: dict[str, Gtk.ToggleButton] = {}
        first: Gtk.ToggleButton | None = None
        for mode in rs.FACE_MODES:
            button = Gtk.ToggleButton(label=rs.FACE_LABELS[mode], group=first)
            first = first or button
            button.connect("toggled", self._on_face, mode)
            box.append(button)
            self.face_buttons[mode] = button
        row.add_suffix(box)
        group.add(row)
        self.fidelity = _slider(0, 100, (rs.DEFAULT_FIDELITY,))
        self.fidelity.connect("value-changed", self._on_fidelity)
        fidelity_row = Adw.ActionRow(
            title="Restoration fidelity",
            subtitle="Original ← → AI enhanced. Lower keeps more of the original pixels.",
        )
        ends = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
        for widget in (Gtk.Label(label="Original"), self.fidelity, Gtk.Label(label="AI")):
            if isinstance(widget, Gtk.Label):
                widget.add_css_class("caption")
                widget.add_css_class("dim-label")
            ends.append(widget)
        fidelity_row.add_suffix(ends)
        group.add(fidelity_row)

    def _build_damage(self, page: Adw.PreferencesPage) -> None:
        group = Adw.PreferencesGroup(
            title="Damage Restoration",
            description="Repaired with conventional image processing — nothing is invented. "
            "Changing a value switches the level to Custom.",
        )
        page.add(group)
        self.stage_sliders: dict[str, Gtk.Scale] = {}
        for name in rs.STAGE_SLIDERS:
            title, subtitle = STAGE_LABELS[name]
            slider = _slider(0, 100)
            slider.connect("value-changed", self._on_stage, name)
            row = Adw.ActionRow(title=title, subtitle=subtitle)
            row.add_suffix(slider)
            group.add(row)
            self.stage_sliders[name] = slider
        self.auto_color_row = Adw.SwitchRow(
            title="Automatic colour correction",
            subtitle="Remove yellowing and colour casts, recover faded colours",
        )
        self.auto_color_row.connect("notify::active", self._on_switch, "auto_color")
        group.add(self.auto_color_row)
        self.detail_row = Adw.SwitchRow(
            title="AI detail reconstruction",
            subtitle="Rebuild lost detail with Real-ESRGAN, even without upscaling",
        )
        self.detail_row.connect("notify::active", self._on_switch, "detail")
        group.add(self.detail_row)

    def _build_color(self, page: Adw.PreferencesPage) -> None:
        group = Adw.PreferencesGroup(
            title="Color Correction",
            description="Manual adjustments, applied after the automatic correction. They "
            "use the same engine as the Lighting profiles, which apply too.",
        )
        reset = Gtk.Button(
            icon_name="edit-undo-symbolic", tooltip_text="Reset", valign=Gtk.Align.CENTER
        )
        reset.add_css_class("flat")
        reset.connect("clicked", lambda _b: self._reset_color())
        group.set_header_suffix(reset)
        page.add(group)
        self.color_sliders: dict[str, Gtk.Scale] = {}
        for name in rs.COLOR_SLIDERS:
            slider = _slider(-100, 100, (0,))
            slider.connect("value-changed", self._on_color, name)
            row = Adw.ActionRow(title=COLOR_LABELS[name])
            row.add_suffix(slider)
            group.add(row)
            self.color_sliders[name] = slider

    def _build_colorize(self, page: Adw.PreferencesPage) -> None:
        self.colorize_group = Adw.PreferencesGroup(
            title="Colorization",
            description="Only black-and-white photographs are colorized, and only with "
            "“Restore + Colorize” or “Full Restoration”.",
        )
        page.add(self.colorize_group)
        self.vivid = _slider(0, 100)
        self.vivid.connect("value-changed", self._on_colorize_value, "colorize_vivid")
        row = Adw.ActionRow(title="Style", subtitle="Natural historical colour ← → vivid")
        row.add_suffix(self.vivid)
        self.colorize_group.add(row)
        self.strength = _slider(0, 100, (50,))
        self.strength.connect("value-changed", self._on_colorize_value, "colorize_strength")
        self.strength_row = Adw.ActionRow(title="Color strength")
        self.strength_row.add_suffix(self.strength)
        self.colorize_group.add(self.strength_row)
        self.tones_row = Adw.SwitchRow(
            title="Preserve original tones",
            subtitle="Keep every brightness value of the photo; only add colour",
        )
        self.tones_row.connect("notify::active", self._on_tones)
        self.colorize_group.add(self.tones_row)

    def _build_finish(self, page: Adw.PreferencesPage) -> None:
        group = Adw.PreferencesGroup(
            title="Modern Finish",
            description="A cleaner, clearer photograph with balanced colour and recovered "
            "shadows — without changing its historical appearance.",
        )
        page.add(group)
        self.modern_row = Adw.ComboRow(
            title="Finish",
            model=Gtk.StringList.new([rs.MODERN_LABELS[m] for m in rs.MODERN_MODES]),
        )
        self.modern_row.connect("notify::selected", self._on_modern)
        group.add(self.modern_row)

    def _build_models(self, page: Adw.PreferencesPage) -> None:
        group = Adw.PreferencesGroup(title="AI Models")
        manage = Gtk.Button(label="Manage Models", valign=Gtk.Align.CENTER)
        manage.connect("clicked", lambda _b: self._manage_models())
        group.set_header_suffix(manage)
        page.add(group)
        self.model_rows: list[tuple[Adw.ActionRow, Gtk.Image]] = []
        for label, _installed in model_summary(self.app):
            row = Adw.ActionRow(title=label)
            icon = Gtk.Image()
            row.add_prefix(icon)
            group.add(row)
            self.model_rows.append((row, icon))

    # --- settings -> widgets ----------------------------------------------
    def sync(self) -> None:
        settings = self.app.settings
        restoration = settings.restoration()
        stages = restoration.stages()
        self._syncing = True
        try:
            self.preset_row.set_selected(rs.PRESETS.index(settings.restore_preset))
            self.level_row.set_selected(rs.LEVELS.index(settings.restore_level))
            self.level_row.set_subtitle(rs.LEVEL_DESCRIPTIONS[settings.restore_level])
            self.scale_row.set_selected(0 if settings.restore_scale == 2 else 1)
            self.face_buttons[stages.face].set_active(True)
            self.fidelity.set_value(settings.restore_fidelity)
            for name, slider in self.stage_sliders.items():
                slider.set_value(getattr(stages, name))
            self.auto_color_row.set_active(stages.auto_color)
            self.detail_row.set_active(stages.detail)
            for name, slider in self.color_sliders.items():
                slider.set_value(getattr(settings, f"restore_{name}"))
            self.vivid.set_value(settings.restore_colorize_vivid)
            self.strength.set_value(settings.restore_colorize_strength)
            self.tones_row.set_active(settings.restore_preserve_tones)
            self.modern_row.set_selected(rs.MODERN_MODES.index(settings.restore_modern))
        finally:
            self._syncing = False
        colorize, upscale = rs.preset_flags(settings.restore_preset)
        self.scale_row.set_sensitive(upscale)
        self.colorize_group.set_sensitive(colorize)
        self.strength_row.set_subtitle(_strength_text(settings.restore_colorize_strength))
        self._sync_models()

    def _sync_models(self) -> None:
        for (row, icon), (_label, installed) in zip(
            self.model_rows, model_summary(self.app), strict=True
        ):
            icon.set_from_icon_name(
                "object-select-symbolic" if installed else "folder-download-symbolic"
            )
            for css in ("success", "dim-label"):
                icon.remove_css_class(css)
            icon.add_css_class("success" if installed else "dim-label")
            row.set_subtitle("Installed" if installed else "Not installed")

    # --- widgets -> settings ----------------------------------------------
    def _on_preset(self, row: Adw.ComboRow, _p: object) -> None:
        if not self._syncing:
            self.app.settings.restore_preset = rs.PRESETS[row.get_selected()]
            self.saver.now()

    def _on_level(self, row: Adw.ComboRow, _p: object) -> None:
        if not self._syncing:
            self.app.settings.restore_level = rs.LEVELS[row.get_selected()]
            self.saver.now()

    def _on_scale(self, row: Adw.ComboRow, _p: object) -> None:
        if not self._syncing:
            self.app.settings.restore_scale = (2, 4)[row.get_selected()]
            self.saver.now()

    def _on_face(self, button: Gtk.ToggleButton, mode: str) -> None:
        if self._syncing or not button.get_active():
            return
        edit_stage(self.app, "face", mode)
        self.saver.now()

    def _on_fidelity(self, scale: Gtk.Scale) -> None:
        if not self._syncing:
            self.app.settings.restore_fidelity = round(scale.get_value())
            self.saver.later()

    def _on_stage(self, scale: Gtk.Scale, name: str) -> None:
        if self._syncing:
            return
        edit_stage(self.app, name, round(scale.get_value()))
        if self.level_row.get_selected() != rs.LEVELS.index(rs.CUSTOM):
            self._syncing = True
            self.level_row.set_selected(rs.LEVELS.index(rs.CUSTOM))
            self._syncing = False
        self.saver.later()

    def _on_switch(self, row: Adw.SwitchRow, _p: object, name: str) -> None:
        if not self._syncing:
            edit_stage(self.app, name, row.get_active())
            self.saver.now()

    def _on_color(self, scale: Gtk.Scale, name: str) -> None:
        if not self._syncing:
            setattr(self.app.settings, f"restore_{name}", round(scale.get_value()))
            self.saver.later()

    def _reset_color(self) -> None:
        for name in rs.COLOR_SLIDERS:
            setattr(self.app.settings, f"restore_{name}", 0)
        self.saver.now()

    def _on_colorize_value(self, scale: Gtk.Scale, key: str) -> None:
        if self._syncing:
            return
        value = round(scale.get_value())
        setattr(self.app.settings, f"restore_{key}", value)
        if key == "colorize_strength":
            self.strength_row.set_subtitle(_strength_text(value))
        self.saver.later()

    def _on_tones(self, row: Adw.SwitchRow, _p: object) -> None:
        if not self._syncing:
            self.app.settings.restore_preserve_tones = row.get_active()
            self.saver.now()

    def _on_modern(self, row: Adw.ComboRow, _p: object) -> None:
        if not self._syncing:
            self.app.settings.restore_modern = rs.MODERN_MODES[row.get_selected()]
            self.saver.now()

    def _manage_models(self) -> None:
        self.close()
        self.app.show_preferences(page="models")

    def _on_closed(self, _dialog: Adw.Dialog) -> None:
        self.saver.flush()
        self.app.off_settings_changed(self.sync)
        self._unsubscribe()


def _strength_text(value: int) -> str:
    if value == 0:
        return "0% — original grey"
    for limit, text in ((30, "subtle"), (60, "natural"), (85, "stronger")):
        if value <= limit:
            return f"{value}% — {text} colour"
    return f"{value}% — full colorization"
