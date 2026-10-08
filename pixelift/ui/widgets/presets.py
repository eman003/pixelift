"""The preset browser: one card per preset, previewed on the user's own photo.

Choosing a card writes the preset's values into the settings (see
pixelift.core.presets) — one click, no confirmation; the workspace previews the
result live and every value stays editable in the Light, Look and Enhance tabs.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from gi.repository import Gtk, Pango
from PIL import Image, ImageOps

from pixelift.core import camera_looks as cl
from pixelift.core import presets
from pixelift.ui.async_utils import run_in_thread
from pixelift.ui.widgets.textures import texture_from_pil

if TYPE_CHECKING:
    from pixelift.storage.settings import Settings
    from pixelift.ui.application import UpscalerApplication

THUMB_SIZE = (100, 64)


def preset_recipe(settings: Settings, preset_id: str) -> tuple:
    """(lighting adjustments, look recipe) the preset gives, settings untouched."""
    trial = dataclasses.replace(settings)
    presets.apply(trial, preset_id)
    trial.normalise()
    return trial.lighting().adjustments(), trial.camera_look_settings().recipe()


def _render_thumbnails(sample: Image.Image, recipes: dict[str, tuple]) -> dict[str, Image.Image]:
    """Every preset applied to a small copy of ``sample`` (background thread)."""
    # Exactly the card's size: a picture's natural size is its image's, so a
    # larger thumbnail would widen every card.
    base = ImageOps.fit(sample.convert("RGB"), THUMB_SIZE, Image.Resampling.BILINEAR)
    return {
        preset_id: cl.apply_to_pil(base, recipe, adjustments, lut_size=cl.THUMBNAIL_LUT_SIZE)
        for preset_id, (adjustments, recipe) in recipes.items()
    }


class _Card(Gtk.FlowBoxChild):
    def __init__(self, preset: presets.Preset) -> None:
        super().__init__()
        self.preset = preset
        self.add_css_class("preset-card")
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self.picture = Gtk.Picture(can_shrink=True, content_fit=Gtk.ContentFit.COVER)
        self.picture.set_size_request(*THUMB_SIZE)
        self.picture.add_css_class("preset-thumb")
        box.append(self.picture)
        name = Gtk.Label(label=preset.name, xalign=0, margin_top=4)
        name.add_css_class("heading")
        box.append(name)
        description = Gtk.Label(
            label=preset.description,
            xalign=0,
            wrap=True,
            lines=2,
            max_width_chars=14,
            ellipsize=Pango.EllipsizeMode.END,
            valign=Gtk.Align.START,
        )
        description.add_css_class("caption")
        description.add_css_class("dim-label")
        box.append(description)
        box.set_size_request(THUMB_SIZE[0], -1)
        self.set_child(box)
        self.set_tooltip_text(f"{preset.name} — {preset.description}")


class PresetPanel(Gtk.Box):
    """Preset cards for the current mode, plus Reset when the preset was changed."""

    def __init__(self, app: UpscalerApplication) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.app = app
        self._sample: Image.Image = cl.sample_image()
        self._thumbs_stale = True
        self._thumb_generation = 0

        self.flow = Gtk.FlowBox(
            selection_mode=Gtk.SelectionMode.NONE,
            activate_on_single_click=True,
            homogeneous=True,
            min_children_per_line=2,
            row_spacing=6,
            column_spacing=6,
            valign=Gtk.Align.START,
            halign=Gtk.Align.CENTER,  # natural-size cards in one row; wraps when narrow
        )
        self.flow.connect("child-activated", lambda _f, card: self.choose(card.preset.id))
        self.cards: dict[str, _Card] = {}
        for preset in presets.all_presets():
            card = _Card(preset)
            self.cards[preset.id] = card
            self.flow.append(card)
        self.append(self.flow)

        footer = Gtk.Box(spacing=8)
        hint = Gtk.Label(
            label="A preset is a starting point — fine-tune it in Light and Look.",
            xalign=0,
            hexpand=True,
            ellipsize=Pango.EllipsizeMode.END,
        )
        hint.add_css_class("caption")
        hint.add_css_class("dim-label")
        footer.append(hint)
        self.reset_btn = Gtk.Button(valign=Gtk.Align.CENTER)
        self.reset_btn.add_css_class("flat")
        self.reset_btn.connect("clicked", lambda _b: self.reset())
        footer.append(self.reset_btn)
        self.append(footer)

        app.on_lighting_changed(self.sync)  # also hears every settings change
        self.connect("map", lambda _w: self._refresh_thumbnails())
        self.connect("destroy", lambda _w: self.shutdown())
        self.sync()

    # --- settings -> widgets -------------------------------------------------
    def sync(self) -> None:
        settings = self.app.settings
        offered = {p.id for p in presets.presets_for(settings.mode)}
        self.flow.set_max_children_per_line(len(offered))
        current, modified = presets.status(settings)
        chosen = settings.preset or (presets.ORIGINAL if current == "Original" else "")
        for preset_id, card in self.cards.items():
            card.set_visible(preset_id in offered)
            if preset_id == chosen:
                card.add_css_class("selected-preset")
            else:
                card.remove_css_class("selected-preset")
        self.reset_btn.set_visible(modified)
        if modified:
            self.reset_btn.set_label(f"Reset to {current}")
            self.reset_btn.set_tooltip_text(f"Back to the {current} preset as it comes")

    # --- widgets -> settings -------------------------------------------------
    def choose(self, preset_id: str) -> None:
        """Apply a preset at once: the workspace previews it."""
        presets.apply(self.app.settings, preset_id)
        self.app.settings_changed()

    def reset(self) -> None:
        if self.app.settings.preset:
            self.choose(self.app.settings.preset)

    # --- thumbnails ------------------------------------------------------------
    def set_sample(self, image: Image.Image | None) -> None:
        """The photo the cards show (None: the synthetic sample)."""
        sample = image if image is not None else cl.sample_image()
        if sample is self._sample:
            return
        self._sample = sample
        self._thumbs_stale = True
        if self.get_mapped():
            self._refresh_thumbnails()

    def _refresh_thumbnails(self) -> None:
        if not self._thumbs_stale:
            return
        self._thumbs_stale = False
        self._thumb_generation += 1
        generation = self._thumb_generation
        settings = self.app.settings
        recipes = {p.id: preset_recipe(settings, p.id) for p in presets.all_presets()}

        def done(thumbs: dict[str, Image.Image]) -> None:
            if generation != self._thumb_generation:
                return
            for preset_id, img in thumbs.items():
                self.cards[preset_id].picture.set_paintable(texture_from_pil(img))

        def failed(_exc: BaseException) -> None:
            if generation == self._thumb_generation:
                self._thumbs_stale = True

        run_in_thread(
            _render_thumbnails,
            self._sample,
            recipes,
            on_done=done,
            on_error=failed,
            name="preset-thumbnails",
        )

    def shutdown(self) -> None:
        self._thumb_generation += 1
        self.app.off_lighting_changed(self.sync)
