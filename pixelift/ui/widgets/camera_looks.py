"""Camera look controls: a compact row plus a gallery of look cards.

The row (look button, intensity, grain, favourite, reset) sits under the
lighting controls in the main window's Look tab; it stays in sync through the
application's lighting listeners. The gallery shows every
look as a card with a thumbnail rendered from the user's own photo (a
synthetic sample until one is added), filtered by category or favourites.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from gi.repository import Gtk, Pango
from PIL import Image

from pixelift.core import camera_looks as cl
from pixelift.ui.async_utils import run_in_thread
from pixelift.ui.widgets.lighting import LiveSettingsBox, _slider
from pixelift.ui.widgets.textures import texture_from_pil

if TYPE_CHECKING:
    from pixelift.ui.application import UpscalerApplication

CUSTOM_LABELS = {
    "exposure": "Exposure",
    "contrast": "Contrast",
    "highlights": "Highlights",
    "shadows": "Shadows",
    "temperature": "Temperature",
    "tint": "Tint",
    "saturation": "Saturation",
    "vibrance": "Vibrance",
    "red": "Red",
    "orange": "Orange",
    "yellow": "Yellow",
    "green": "Green",
    "aqua": "Aqua",
    "blue": "Blue",
    "purple": "Purple",
    "magenta": "Magenta",
    "sharpness": "Sharpness",
}
GRAIN_LABELS = {
    cl.GRAIN_AUTO: "Grain: Look Default",
    "off": "Grain: Off",
    "low": "Grain: Low",
    "medium": "Grain: Medium",
    "high": "Grain: High",
}
FAVORITES = "favorites"
SAVED = "saved"
FILTERS: tuple[tuple[str, str], ...] = (
    ("all", "All"),
    (FAVORITES, "★ Favorites"),
    *((c.id, c.name) for c in cl.CATEGORIES),
    (SAVED, "My Looks"),
)
THUMB_SIZE = (132, 92)


def _render_thumbnails(
    sample: Image.Image, recipes: list[tuple[str, cl.LookRecipe]]
) -> dict[str, Image.Image]:
    """Every look applied to a small copy of ``sample`` (background thread)."""
    base = sample.convert("RGB")
    base.thumbnail((THUMB_SIZE[0] * 2, THUMB_SIZE[1] * 2), Image.Resampling.BILINEAR)
    return {
        look_id: cl.apply_to_pil(base, recipe, lut_size=cl.THUMBNAIL_LUT_SIZE)
        for look_id, recipe in recipes
    }


class _Card(Gtk.FlowBoxChild):
    """One look in the gallery: thumbnail, name, category and description."""

    def __init__(
        self,
        look_id: str,
        name: str,
        label: str,
        description: str,
        on_favorite: Callable[[str, bool], None],
        on_delete: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self.look_id = look_id
        self.look = cl.get_look(look_id) if cl.has_look(look_id) else None
        self.add_css_class("look-card")
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self.picture = Gtk.Picture(can_shrink=True, content_fit=Gtk.ContentFit.COVER)
        self.picture.set_size_request(*THUMB_SIZE)
        self.picture.add_css_class("look-thumb")
        overlay = Gtk.Overlay(child=self.picture)
        self.star: Gtk.ToggleButton | None = None
        if look_id != cl.ORIGINAL:
            self.star = Gtk.ToggleButton(
                icon_name="non-starred-symbolic",
                tooltip_text="Favorite",
                halign=Gtk.Align.END,
                valign=Gtk.Align.START,
                margin_top=4,
                margin_end=4,
            )
            self.star.add_css_class("circular")
            self.star.add_css_class("osd")
            self.star.connect("toggled", lambda b: self._on_star(b, on_favorite))
            overlay.add_overlay(self.star)
        if on_delete is not None:
            delete = Gtk.Button(
                icon_name="user-trash-symbolic",
                tooltip_text="Delete This Look",
                halign=Gtk.Align.START,
                valign=Gtk.Align.START,
                margin_top=4,
                margin_start=4,
            )
            delete.add_css_class("circular")
            delete.add_css_class("osd")
            delete.connect("clicked", lambda _b: on_delete(look_id))
            overlay.add_overlay(delete)
        box.append(overlay)
        title = Gtk.Label(label=name, xalign=0, ellipsize=Pango.EllipsizeMode.END)
        title.add_css_class("heading")
        box.append(title)
        if label:
            category = Gtk.Label(label=label, xalign=0, ellipsize=Pango.EllipsizeMode.END)
            category.add_css_class("caption")
            category.add_css_class("accent")
            box.append(category)
        details = Gtk.Label(
            label=description,
            xalign=0,
            wrap=True,
            lines=2,
            max_width_chars=18,
            ellipsize=Pango.EllipsizeMode.END,
            valign=Gtk.Align.START,
        )
        details.add_css_class("caption")
        details.add_css_class("dim-label")
        box.append(details)
        box.set_size_request(THUMB_SIZE[0], -1)
        self.set_child(box)
        self.set_tooltip_text(f"{name}\n{description}")
        self._syncing = False

    def _on_star(self, button: Gtk.ToggleButton, on_favorite: Callable[[str, bool], None]) -> None:
        button.set_icon_name("starred-symbolic" if button.get_active() else "non-starred-symbolic")
        if not self._syncing:
            on_favorite(self.look_id, button.get_active())

    def set_favorite(self, favorite: bool) -> None:
        if self.star is not None and self.star.get_active() != favorite:
            self._syncing = True
            try:
                self.star.set_active(favorite)
            finally:
                self._syncing = False

    def matches(self, category: str, favorites: set[str]) -> bool:
        if category == "all":
            return True
        if category == FAVORITES:
            return self.look_id in favorites
        if category == SAVED:
            return self.look_id.startswith(cl.USER_PREFIX)
        return self.look is not None and self.look.in_category(category)


class CameraLookControls(LiveSettingsBox):
    def __init__(self, app: UpscalerApplication, label_width: int = 6) -> None:
        super().__init__(app, spacing=8)
        self._sample: Image.Image = cl.sample_image()
        self._thumbs_stale = True
        self._thumb_generation = 0
        self._category = "all"
        self.cards: dict[str, _Card] = {}

        title = Gtk.Label(label="Look", xalign=0, width_chars=label_width)
        title.add_css_class("dim-label")
        title.set_tooltip_text("Camera-inspired look")
        self.append(title)

        self.look_label = Gtk.Label(
            ellipsize=Pango.EllipsizeMode.END, xalign=0, width_chars=12, max_width_chars=24
        )
        self.look_btn = Gtk.MenuButton(
            popover=self._build_gallery(), tooltip_text="Choose a camera-inspired look"
        )
        content = Gtk.Box(spacing=6)
        content.append(Gtk.Image(icon_name="camera-photo-symbolic"))
        content.append(self.look_label)
        self.look_btn.set_child(content)
        self.look_btn.set_always_show_arrow(True)
        self.append(self.look_btn)

        self.intensity_box = Gtk.Box(spacing=6, hexpand=True)
        self.intensity = _slider(0, 100)
        self.intensity.set_tooltip_text(
            "Look intensity: 0% is the original image, 100% the full look"
        )
        for mark in (25, 50, 75):
            self.intensity.add_mark(mark, Gtk.PositionType.BOTTOM, None)
        self.intensity.connect("value-changed", self._on_intensity_changed)
        self.intensity_label = Gtk.Label(width_chars=4, xalign=1)
        self.intensity_label.add_css_class("numeric")
        self.intensity_label.add_css_class("caption")
        self.intensity_box.append(self.intensity)
        self.intensity_box.append(self.intensity_label)
        self.append(self.intensity_box)

        self.grain_dd = Gtk.DropDown.new_from_strings([GRAIN_LABELS[g] for g in cl.GRAIN_CHOICES])
        self.grain_dd.set_tooltip_text("Film grain, added to the final image")
        self.grain_dd.connect("notify::selected", self._on_grain_changed)
        self.append(self.grain_dd)

        self.custom_btn = Gtk.MenuButton(
            icon_name="document-edit-symbolic",
            tooltip_text="Custom Look",
            popover=self._build_custom_popover(),
        )
        self.append(self.custom_btn)

        self.favorite_btn = Gtk.ToggleButton(icon_name="non-starred-symbolic")
        self.favorite_btn.connect("toggled", self._on_favorite_toggled)
        self.append(self.favorite_btn)

        self.reset_btn = Gtk.Button(icon_name="edit-undo-symbolic", tooltip_text="Reset Look")
        self.reset_btn.connect("clicked", lambda _b: self.reset())
        self.append(self.reset_btn)

        app.on_lighting_changed(self.sync)
        self.connect("destroy", lambda _w: self.shutdown())
        self.sync()

    # --- construction ------------------------------------------------------
    def _build_gallery(self) -> Gtk.Popover:
        outer = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=8,
            margin_top=6,
            margin_bottom=6,
            margin_start=6,
            margin_end=6,
        )
        filters = Gtk.Box(css_classes=["linked"])
        self.filter_buttons: dict[str, Gtk.ToggleButton] = {}
        group: Gtk.ToggleButton | None = None
        for key, label in FILTERS:
            btn = Gtk.ToggleButton(label=label, group=group)
            group = group or btn
            btn.connect("toggled", self._on_filter_toggled, key)
            filters.append(btn)
            self.filter_buttons[key] = btn
        self.filter_buttons["all"].set_active(True)
        outer.append(
            Gtk.ScrolledWindow(
                child=filters,
                vscrollbar_policy=Gtk.PolicyType.NEVER,
                propagate_natural_height=True,
            )
        )

        self.flow = Gtk.FlowBox(
            selection_mode=Gtk.SelectionMode.NONE,
            activate_on_single_click=True,
            homogeneous=True,
            min_children_per_line=2,
            max_children_per_line=4,
            row_spacing=6,
            column_spacing=6,
            valign=Gtk.Align.START,
        )
        self.flow.set_filter_func(self._card_visible)
        self.flow.connect("child-activated", self._on_card_activated)
        self.empty_label = Gtk.Label(
            label="No looks here yet — star a look to add it to your favorites.",
            wrap=True,
            margin_top=24,
            margin_bottom=24,
        )
        self.empty_label.add_css_class("dim-label")
        cards = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        cards.append(self.flow)
        cards.append(self.empty_label)
        outer.append(
            Gtk.ScrolledWindow(
                child=cards,
                hscrollbar_policy=Gtk.PolicyType.NEVER,
                min_content_height=360,
                min_content_width=600,
                max_content_height=520,
                propagate_natural_height=True,
            )
        )

        line = Gtk.Box(spacing=8)
        caption = Gtk.Label(label="Intensity", xalign=0)
        caption.add_css_class("dim-label")
        self.gallery_intensity = _slider(0, 100)
        self.gallery_intensity.connect("value-changed", self._on_intensity_changed)
        self.gallery_intensity_label = Gtk.Label(width_chars=4, xalign=1)
        self.gallery_intensity_label.add_css_class("numeric")
        line.append(caption)
        line.append(self.gallery_intensity)
        line.append(self.gallery_intensity_label)
        outer.append(line)

        note = Gtk.Label(label=cl.DISCLAIMER, wrap=True, xalign=0, max_width_chars=80)
        note.add_css_class("caption")
        note.add_css_class("dim-label")
        outer.append(note)

        popover = Gtk.Popover(child=outer)
        popover.connect("show", lambda _p: self._refresh_thumbnails())
        self.gallery = popover
        return popover

    def _build_custom_popover(self) -> Gtk.Popover:
        grid = Gtk.Grid(
            row_spacing=4,
            column_spacing=12,
            margin_top=6,
            margin_bottom=6,
            margin_start=6,
            margin_end=6,
        )
        self.custom_sliders: dict[str, Gtk.Scale] = {}
        for index, name in enumerate(cl.CUSTOM_NAMES):
            # Two columns: tone on the left, colour bands on the right.
            column, row = (0, index) if index < 8 else (3, index - 8)
            label = Gtk.Label(label=CUSTOM_LABELS.get(name, name.title()), xalign=0)
            low, high = cl.custom_range(name)
            slider = _slider(low, high)
            slider.set_size_request(170, -1)
            if low < 0:
                slider.add_mark(0, Gtk.PositionType.BOTTOM, None)
            if name in cl.BANDS:
                slider.set_tooltip_text(f"Saturation of {name} colours")
            slider.connect("value-changed", self._on_custom_changed, name)
            value = Gtk.Label(width_chars=4, xalign=1)
            value.add_css_class("numeric")
            value.add_css_class("caption")
            slider.connect(
                "value-changed",
                lambda s, v=value, signed=low < 0: v.set_label(
                    f"{s.get_value():+.0f}" if signed else f"{s.get_value():.0f}"
                ),
            )
            grid.attach(label, column, row, 1, 1)
            grid.attach(slider, column + 1, row, 1, 1)
            grid.attach(value, column + 2, row, 1, 1)
            self.custom_sliders[name] = slider

        save_row = Gtk.Box(spacing=6, margin_top=8)
        self.save_entry = Gtk.Entry(placeholder_text="Name", hexpand=True, max_length=60)
        self.save_entry.connect("activate", lambda _e: self.save_custom())
        self.save_entry.connect("changed", lambda _e: self._update_save_button())
        self.save_btn = Gtk.Button(label="Save as My Look")
        self.save_btn.connect("clicked", lambda _b: self.save_custom())
        reset = Gtk.Button(label="Reset")
        reset.connect("clicked", lambda _b: self.reset_custom())
        save_row.append(self.save_entry)
        save_row.append(self.save_btn)
        save_row.append(reset)
        grid.attach(save_row, 0, 10, 6, 1)
        return Gtk.Popover(child=grid)

    def _rebuild_cards(self) -> None:
        """(Re)create the cards when the set of looks changed (saved looks)."""
        settings = self.app.settings
        wanted = [look.id for look in cl.all_looks()] + [
            cl.USER_PREFIX + name for name in sorted(settings.camera_look_saved)
        ]
        if list(self.cards) == wanted:
            return
        while (child := self.flow.get_first_child()) is not None:
            self.flow.remove(child)
        self.cards = {}
        for look_id in wanted:
            if look_id.startswith(cl.USER_PREFIX):
                card = _Card(
                    look_id,
                    look_id[len(cl.USER_PREFIX) :],
                    "Your look",
                    "Saved from Custom",
                    self._set_favorite,
                    self.delete_saved,
                )
            else:
                look = cl.get_look(look_id)
                card = _Card(
                    look_id, look.name, look.category_label, look.description, self._set_favorite
                )
            self.cards[look_id] = card
            self.flow.append(card)
        self._thumbs_stale = True
        if self.gallery.get_visible():
            self._refresh_thumbnails()

    # --- thumbnails ---------------------------------------------------------
    def set_sample(self, image: Image.Image | None) -> None:
        """The photo the card thumbnails show (None: the synthetic sample)."""
        sample = image if image is not None else cl.sample_image()
        if sample is self._sample:
            return
        self._sample = sample
        self._thumbs_stale = True
        if self.gallery.get_visible():
            self._refresh_thumbnails()

    def _card_recipes(self) -> list[tuple[str, cl.LookRecipe]]:
        settings = self.app.settings
        recipes = []
        for look_id in self.cards:
            if look_id.startswith(cl.USER_PREFIX):
                values = settings.camera_look_saved.get(look_id[len(cl.USER_PREFIX) :], {})
                recipes.append((look_id, cl.custom_recipe(values)))
            elif look_id == cl.CUSTOM:
                recipes.append((look_id, cl.custom_recipe(settings.look_values())))
            else:
                # Cards show each look at full strength: its character at a glance.
                recipes.append((look_id, cl.get_look(look_id).recipe))
        return recipes

    def _refresh_thumbnails(self) -> None:
        if not self._thumbs_stale:
            return
        self._thumbs_stale = False
        self._thumb_generation += 1
        generation = self._thumb_generation

        def done(thumbs: dict[str, Image.Image]) -> None:
            if generation != self._thumb_generation:
                return
            for look_id, img in thumbs.items():
                card = self.cards.get(look_id)
                if card is not None:
                    card.picture.set_paintable(texture_from_pil(img))

        def failed(_exc: BaseException) -> None:
            if generation == self._thumb_generation:
                self._thumbs_stale = True

        run_in_thread(
            _render_thumbnails,
            self._sample,
            self._card_recipes(),
            on_done=done,
            on_error=failed,
            name="look-thumbnails",
        )

    # --- settings -> widgets ----------------------------------------------
    def sync(self) -> None:
        settings = self.app.settings
        self._rebuild_cards()
        self._syncing = True
        try:
            self.intensity.set_value(settings.camera_look_intensity)
            self.gallery_intensity.set_value(settings.camera_look_intensity)
            self.grain_dd.set_selected(cl.GRAIN_CHOICES.index(settings.camera_look_grain))
            for name, slider in self.custom_sliders.items():
                slider.set_value(getattr(settings, f"look_{name}"))
        finally:
            self._syncing = False
        favorites = set(settings.camera_look_favorites)
        for look_id, card in self.cards.items():
            card.set_favorite(look_id in favorites)
        self._update_state()

    def _update_state(self) -> None:
        settings = self.app.settings
        look = settings.camera_look
        custom = look == cl.CUSTOM
        original = look == cl.ORIGINAL
        look_settings = settings.camera_look_settings()
        self.look_label.set_label(look_settings.name())
        description = cl.get_look(look).description if cl.has_look(look) else "Saved from Custom"
        label = cl.get_look(look).category_label if cl.has_look(look) else "Your look"
        self.look_btn.set_tooltip_text(f"{label}\n{description}" if label else description)
        self.intensity_box.set_visible(not custom)
        self.intensity_box.set_sensitive(not original)
        self.gallery_intensity.set_sensitive(not original and not custom)
        percent = f"{settings.camera_look_intensity}%"
        self.intensity_label.set_label(percent)
        self.gallery_intensity_label.set_label(percent)
        self.grain_dd.set_sensitive(not original)
        self.custom_btn.set_visible(custom)
        self.favorite_btn.set_visible(not original)
        favorite = look in settings.camera_look_favorites
        self._syncing = True
        try:
            self.favorite_btn.set_active(favorite)
        finally:
            self._syncing = False
        self.favorite_btn.set_icon_name("starred-symbolic" if favorite else "non-starred-symbolic")
        self.favorite_btn.set_tooltip_text(
            "Remove from Favorites" if favorite else "Add to Favorites"
        )
        self.reset_btn.set_visible(
            not original
            or settings.camera_look_intensity != cl.DEFAULT_INTENSITY
            or settings.camera_look_grain != cl.GRAIN_AUTO
        )
        for look_id, card in self.cards.items():
            if look_id == look:
                card.add_css_class("selected-look")
            else:
                card.remove_css_class("selected-look")
        self._update_save_button()
        self._update_empty()

    def _update_save_button(self) -> None:
        name = self.save_entry.get_text().strip()
        self.save_btn.set_sensitive(bool(name))
        exists = name in self.app.settings.camera_look_saved
        self.save_btn.set_label("Replace My Look" if exists else "Save as My Look")

    def _card_visible(self, child: Gtk.FlowBoxChild) -> bool:
        assert isinstance(child, _Card)
        return child.matches(self._category, set(self.app.settings.camera_look_favorites))

    def _update_empty(self) -> None:
        favorites = set(self.app.settings.camera_look_favorites)
        visible = any(card.matches(self._category, favorites) for card in self.cards.values())
        self.empty_label.set_visible(not visible)
        self.empty_label.set_label(
            "No saved looks yet — adjust the Custom look and save it."
            if self._category == SAVED
            else "No looks here yet — star a look to add it to your favorites."
        )
        self.flow.invalidate_filter()

    # --- widgets -> settings ----------------------------------------------
    def select(self, look_id: str) -> None:
        settings = self.app.settings
        if look_id == settings.camera_look:
            return
        settings.camera_look = look_id
        if look_id.startswith(cl.USER_PREFIX):
            # A saved look opens as it was designed; the slider tones it down.
            settings.camera_look_intensity = 100
        self._update_state()
        self._save(now=True)

    def _on_card_activated(self, _flow: Gtk.FlowBox, child: Gtk.FlowBoxChild) -> None:
        assert isinstance(child, _Card)
        self.select(child.look_id)
        if child.look_id == cl.CUSTOM:
            self.gallery.popdown()
            self.custom_btn.popup()

    def _on_filter_toggled(self, button: Gtk.ToggleButton, key: str) -> None:
        if button.get_active() and key != self._category:
            self._category = key
            self._update_empty()

    def _on_intensity_changed(self, scale: Gtk.Scale) -> None:
        if self._syncing:
            return
        value = round(scale.get_value())
        self.app.settings.camera_look_intensity = value
        self._syncing = True
        try:
            for other in (self.intensity, self.gallery_intensity):
                if other is not scale:
                    other.set_value(value)
        finally:
            self._syncing = False
        self._update_state()
        self._save()

    def _on_grain_changed(self, *_args: object) -> None:
        if self._syncing:
            return
        index = self.grain_dd.get_selected()
        if index < len(cl.GRAIN_CHOICES):
            self.app.settings.camera_look_grain = cl.GRAIN_CHOICES[index]
            self._update_state()
            self._save(now=True)

    def _on_custom_changed(self, scale: Gtk.Scale, name: str) -> None:
        if self._syncing:
            return
        setattr(self.app.settings, f"look_{name}", round(scale.get_value()))
        if cl.CUSTOM in self.cards:
            self._thumbs_stale = True
        self._save()

    def _on_favorite_toggled(self, button: Gtk.ToggleButton) -> None:
        if not self._syncing:
            self._set_favorite(self.app.settings.camera_look, button.get_active())

    def _set_favorite(self, look_id: str, favorite: bool) -> None:
        favorites = self.app.settings.camera_look_favorites
        if favorite and look_id not in favorites:
            favorites.append(look_id)
        elif not favorite and look_id in favorites:
            favorites.remove(look_id)
        else:
            return
        # Favourites do not change the image: save without re-rendering previews.
        self.app.save_settings()
        if look_id in self.cards:
            self.cards[look_id].set_favorite(favorite)
        self._update_state()

    def save_custom(self) -> None:
        """Save the Custom values as a named look and select it."""
        name = self.save_entry.get_text().strip()[:60]
        if not name:
            return
        settings = self.app.settings
        settings.camera_look_saved[name] = settings.look_values()
        settings.camera_look = cl.USER_PREFIX + name
        # Exactly as designed (Custom always renders at full strength).
        settings.camera_look_intensity = 100
        self.save_entry.set_text("")
        self.custom_btn.get_popover().popdown()
        self._thumbs_stale = True  # also when an existing look was replaced
        self._rebuild_cards()
        self._save(now=True)

    def delete_saved(self, look_id: str) -> None:
        settings = self.app.settings
        settings.camera_look_saved.pop(look_id[len(cl.USER_PREFIX) :], None)
        if look_id in settings.camera_look_favorites:
            settings.camera_look_favorites.remove(look_id)
        if settings.camera_look == look_id:
            settings.camera_look = cl.ORIGINAL
        self._rebuild_cards()
        self._save(now=True)

    def reset(self) -> None:
        """Back to the default: Original, default intensity, the look's own grain."""
        settings = self.app.settings
        settings.camera_look = cl.ORIGINAL
        settings.camera_look_intensity = cl.DEFAULT_INTENSITY
        settings.camera_look_grain = cl.GRAIN_AUTO
        self._save(now=True)

    def reset_custom(self) -> None:
        for name in cl.CUSTOM_NAMES:
            setattr(self.app.settings, f"look_{name}", 0)
        self._thumbs_stale = True
        self._save(now=True)

    def shutdown(self) -> None:
        """Flush pending changes and stop listening (when the window closes)."""
        self._thumb_generation += 1
        self._flush_pending()
        self.app.off_lighting_changed(self.sync)
