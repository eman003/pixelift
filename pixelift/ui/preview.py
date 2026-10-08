"""The stage canvas: the photo, with a before / after divider, zoom and pan.

The main window's stage draws the selected photo here. The "after" side is
whatever the photo will become — the lighting and camera look previewed live,
a restoration preview, or the processed result — and the "before" side the
original. With ``comparing`` on, a divider splits the two; dragging it (or
←/→, Home/End) reveals more of either.

Zoom: scroll, pinch, double-click (fit ↔ 100 %) or +/−/0/1; drag to pan.

Memory: both images are loaded as previews no larger than the stage's preview
size. When the user zooms in beyond the preview's resolution, only the visible
region is decoded at full resolution (in a background thread) and drawn on top
— the full-resolution image is never kept in memory by the UI. A side's
``transform`` (the look) is applied to that region too.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from gi.repository import Gdk, GLib, GObject, Graphene, Gsk, Gtk, Pango
from PIL import Image

from pixelift.ui.async_utils import run_in_thread
from pixelift.ui.widgets.textures import texture_from_pil
from pixelift.utils.image_utils import load_region

log = logging.getLogger(__name__)

MAX_ZOOM = 16.0
HANDLE_RADIUS = 14
LABEL_INSET = 12
KEY_SPLIT_STEP = 0.05


def _rect(x: float, y: float, w: float, h: float) -> Graphene.Rect:
    return Graphene.Rect().init(x, y, w, h)


def _rgba(spec: str) -> Gdk.RGBA:
    color = Gdk.RGBA()
    color.parse(spec)
    return color


class _Side:
    """One image (before or after): preview texture + optional detail crop."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.texture: Gdk.Texture | None = None
        self.source_size: tuple[int, int] = (0, 0)  # full-resolution size of the file
        self.detail: tuple[Gdk.Texture, tuple[float, float, float, float]] | None = None
        self.generation = 0
        # Applied (off the main thread) to detail crops decoded from ``path``;
        # gets the crop and its scale relative to the full-resolution file.
        self.transform: Callable[[Image.Image, float], Image.Image] | None = None

    def clear(self) -> None:
        self.path = None
        self.texture = None
        self.detail = None
        self.transform = None
        self.generation += 1


class CompareView(Gtk.Widget):
    __gtype_name__ = "UpscalerCompareView"
    __gsignals__: ClassVar = {"view-changed": (GObject.SignalFlags.RUN_FIRST, None, ())}

    comparing = GObject.Property(type=bool, default=False)  # show the divider

    def __init__(self, padding: tuple[int, int, int, int] = (0, 0, 0, 0)) -> None:
        super().__init__(
            hexpand=True, vexpand=True, focusable=True, accessible_role=Gtk.AccessibleRole.IMG
        )
        self.add_css_class("compare-view")
        self.set_overflow(Gtk.Overflow.HIDDEN)
        self.update_property(
            [Gtk.AccessibleProperty.DESCRIPTION],
            [
                "C compares before and after; left and right arrows move the divider. "
                "Plus and minus zoom, 1 shows actual pixels, 0 fits the window."
            ],
        )
        self.padding = padding  # (top, right, bottom, left): kept clear when fitted
        self.before = _Side()
        self.after = _Side()
        self.image_size: tuple[int, int] = (0, 0)  # logical size = the after side's size
        self.before_label = "Original"
        self.after_label = "After"
        self.split = 0.5  # divider position across the visible part of the image
        self.zoom: float | None = None  # None = fit to window
        self.center = (0.0, 0.0)
        self._drag_kind = ""
        self._drag_center = (0.0, 0.0)
        self._pinch_zoom = 1.0
        self._pointer = (0.0, 0.0)
        self._detail_source = 0

        drag = Gtk.GestureDrag()
        drag.connect("drag-begin", self._on_drag_begin)
        drag.connect("drag-update", self._on_drag_update)
        drag.connect("drag-end", self._on_drag_end)
        self.add_controller(drag)
        click = Gtk.GestureClick()
        click.connect("pressed", self._on_click)
        self.add_controller(click)
        scroll = Gtk.EventControllerScroll.new(Gtk.EventControllerScrollFlags.VERTICAL)
        scroll.connect("scroll", self._on_scroll)
        self.add_controller(scroll)
        motion = Gtk.EventControllerMotion()
        motion.connect("motion", self._on_motion)
        self.add_controller(motion)
        pinch = Gtk.GestureZoom()
        pinch.connect("begin", lambda *_: setattr(self, "_pinch_zoom", self.effective_zoom()))
        pinch.connect("scale-changed", self._on_pinch)
        self.add_controller(pinch)
        keys = Gtk.EventControllerKey()
        keys.connect("key-pressed", self._on_key)
        self.add_controller(keys)
        self.connect("notify::comparing", self._on_comparing_changed)

    # --- state -------------------------------------------------------------
    @property
    def can_compare(self) -> bool:
        return self.before.texture is not None and self.after.texture is not None

    @property
    def showing_split(self) -> bool:
        return self.comparing and self.can_compare

    def set_images(self, image_size: tuple[int, int]) -> None:
        """A new photo: fit it to the window."""
        self.image_size = image_size
        self.zoom = None
        self.center = (image_size[0] / 2, image_size[1] / 2)
        self._view_changed()

    def clear(self) -> None:
        self.before.clear()
        self.after.clear()
        self.image_size = (0, 0)
        self.zoom = None
        self._view_changed()

    def _on_comparing_changed(self, *_args: object) -> None:
        self._update_cursor()
        self.queue_draw()

    def _view_center(self) -> tuple[float, float]:
        """Where the image's ``center`` is drawn: the middle of the padded area."""
        top, right, bottom, left = self.padding
        return (
            left + (self.get_width() - left - right) / 2,
            top + (self.get_height() - top - bottom) / 2,
        )

    def fit_zoom(self) -> float:
        top, right, bottom, left = self.padding
        w = self.get_width() - left - right
        h = self.get_height() - top - bottom
        iw, ih = self.image_size
        if not iw or not ih or w <= 0 or h <= 0:
            return 1.0
        return min(w / iw, h / ih)

    def effective_zoom(self) -> float:
        return self.fit_zoom() if self.zoom is None else self.zoom

    def set_zoom(self, zoom: float | None, anchor: tuple[float, float] | None = None) -> None:
        """Zoom keeping the image point under ``anchor`` in place.

        Zoom ranges from the smaller of fit and 100 % (actual pixels) to
        ``MAX_ZOOM``: a large photo zoomed out past fit fits again, a small one
        can still be seen at 100 %.
        """
        if not self.image_size[0]:
            return  # nothing on the stage
        fit = self.fit_zoom()
        if zoom is not None:
            zoom = min(max(MAX_ZOOM, fit), max(zoom, min(fit, self._actual_zoom())))
        if zoom is None or abs(zoom - fit) <= fit * 0.001:
            self.zoom = None
            self.center = (self.image_size[0] / 2, self.image_size[1] / 2)
        else:
            old = self.effective_zoom()
            if anchor is not None:
                vx, vy = self._view_center()
                ax, ay = anchor[0] - vx, anchor[1] - vy
                px = self.center[0] + ax / old
                py = self.center[1] + ay / old
                self.center = (px - ax / zoom, py - ay / zoom)
            self.zoom = zoom
            self._clamp_center()
        self._update_cursor()
        self._view_changed()

    def _actual_zoom(self) -> float:
        """1 image pixel = 1 physical screen pixel."""
        return 1.0 / max(1, self.get_scale_factor())

    def actual_pixels(self, anchor: tuple[float, float] | None = None) -> None:
        self.set_zoom(self._actual_zoom(), anchor)

    def zoom_by(self, factor: float) -> None:
        self.set_zoom(self.effective_zoom() * factor, self._view_center())

    def _clamp_center(self) -> None:
        iw, ih = self.image_size
        self.center = (min(max(self.center[0], 0), iw), min(max(self.center[1], 0), ih))

    def _view_changed(self) -> None:
        self.queue_draw()
        self.emit("view-changed")
        if self._detail_source:
            GLib.source_remove(self._detail_source)
        self._detail_source = GLib.timeout_add(180, self._request_details)

    def invalidate_detail(self, side: _Side) -> None:
        """Drop ``side``'s detail crop (its content changed) and load it again."""
        side.generation += 1
        side.detail = None
        self.queue_draw()
        self._request_detail(side)

    # --- geometry ----------------------------------------------------------
    def _origin(self, zoom: float) -> tuple[float, float]:
        vx, vy = self._view_center()
        return vx - self.center[0] * zoom, vy - self.center[1] * zoom

    def _visible_rect(self) -> tuple[float, float, float, float]:
        """The part of the drawn image inside the widget: (x0, y0, x1, y1)."""
        zoom = self.effective_zoom()
        ox, oy = self._origin(zoom)
        iw, ih = self.image_size
        return (
            max(0.0, ox),
            max(0.0, oy),
            min(float(self.get_width()), ox + iw * zoom),
            min(float(self.get_height()), oy + ih * zoom),
        )

    def _split_x(self) -> float:
        x0, _y0, x1, _y1 = self._visible_rect()
        return round(x0 + self.split * max(0.0, x1 - x0))

    def _set_split_at(self, x: float) -> None:
        x0, _y0, x1, _y1 = self._visible_rect()
        self.split = min(1.0, max(0.0, (x - x0) / max(1.0, x1 - x0)))
        self.queue_draw()

    def _near_divider(self, x: float) -> bool:
        return self.showing_split and abs(x - self._split_x()) <= HANDLE_RADIUS + 6

    # --- detail loading ----------------------------------------------------
    def _request_details(self) -> bool:
        self._detail_source = 0
        for side in (self.before, self.after):
            self._request_detail(side)
        return GLib.SOURCE_REMOVE

    def _request_detail(self, side: _Side) -> None:
        iw, ih = self.image_size
        if side.path is None or side.texture is None or not iw:
            return
        sf = self.get_scale_factor()
        zoom = self.effective_zoom()
        texels_per_px = side.texture.get_width() / iw
        sw, sh = side.source_size
        if texels_per_px >= zoom * sf * 0.95 or sw <= side.texture.get_width():
            if side.detail:
                side.detail = None
                self.queue_draw()
            return
        vx, vy = self._view_center()
        x0 = max(0.0, self.center[0] - vx / zoom)
        y0 = max(0.0, self.center[1] - vy / zoom)
        x1 = min(float(iw), self.center[0] + (self.get_width() - vx) / zoom)
        y1 = min(float(ih), self.center[1] + (self.get_height() - vy) / zoom)
        if x1 <= x0 or y1 <= y0:
            return
        fx, fy = sw / iw, sh / ih
        box = (
            int(x0 * fx),
            int(y0 * fy),
            max(int(x0 * fx) + 1, int(x1 * fx + 0.999)),
            max(int(y0 * fy) + 1, int(y1 * fy + 0.999)),
        )
        out_w = max(1, min(box[2] - box[0], int((x1 - x0) * zoom * sf)))
        out_h = max(1, min(box[3] - box[1], int((y1 - y0) * zoom * sf)))
        logical = (box[0] / fx, box[1] / fy, box[2] / fx, box[3] / fy)
        side.generation += 1
        generation = side.generation
        path = side.path
        transform = side.transform

        def load() -> Image.Image:
            region = load_region(path, box, (out_w, out_h))
            return transform(region, out_w / (box[2] - box[0])) if transform else region

        def done(img: Image.Image) -> None:
            if generation == side.generation:
                side.detail = (texture_from_pil(img), logical)
                self.queue_draw()

        run_in_thread(
            load,
            on_done=done,
            on_error=lambda e: log.warning("Detail load failed: %s", e),
            name="preview-detail",
        )

    # --- input -------------------------------------------------------------
    def _update_cursor(self) -> None:
        x = self._pointer[0]
        if self._drag_kind == "pan":
            name = "grabbing"
        elif self._near_divider(x) or (self.showing_split and self.zoom is None):
            name = "col-resize"  # fitted: dragging anywhere moves the divider
        elif self.zoom is not None:
            name = "grab"
        else:
            name = None
        self.set_cursor_from_name(name)

    def _on_motion(self, _ctrl: Gtk.EventControllerMotion, x: float, y: float) -> None:
        self._pointer = (x, y)
        if not self._drag_kind:
            self._update_cursor()

    def _on_drag_begin(self, _gesture: Gtk.GestureDrag, x: float, y: float) -> None:
        self.grab_focus()
        self._pointer = (x, y)
        if self._near_divider(x):
            self._drag_kind = "split"
            self._set_split_at(x)
        elif self.showing_split and self.zoom is None:
            self._drag_kind = "split-anywhere"  # moves once the pointer does
        elif self.zoom is not None:
            self._drag_kind = "pan"
            self._drag_center = self.center
        else:
            self._drag_kind = ""
        self._update_cursor()

    def _on_drag_update(self, gesture: Gtk.GestureDrag, dx: float, dy: float) -> None:
        if self._drag_kind.startswith("split"):
            _ok, sx, _sy = gesture.get_start_point()
            self._set_split_at(sx + dx)
        elif self._drag_kind == "pan":
            zoom = self.effective_zoom()
            self.center = (self._drag_center[0] - dx / zoom, self._drag_center[1] - dy / zoom)
            self._clamp_center()
            self._view_changed()

    def _on_drag_end(self, *_args: object) -> None:
        self._drag_kind = ""
        self._update_cursor()

    def _on_click(self, _gesture: Gtk.GestureClick, n_press: int, x: float, y: float) -> None:
        if n_press == 2 and self.image_size[0]:  # fit ↔ actual pixels, at the pointer
            if self.zoom is None:
                self.actual_pixels((x, y))
            else:
                self.set_zoom(None)

    def _on_scroll(self, _ctrl: Gtk.EventControllerScroll, _dx: float, dy: float) -> bool:
        if not self.image_size[0]:
            return False  # nothing to zoom: let the stage scroll on
        self.set_zoom(self.effective_zoom() * (1.15**-dy), self._pointer)
        return True

    def _on_pinch(self, gesture: Gtk.GestureZoom, scale: float) -> None:
        ok, x, y = gesture.get_bounding_box_center()
        self.set_zoom(self._pinch_zoom * scale, (x, y) if ok else None)

    def _on_key(
        self, _ctrl: Gtk.EventControllerKey, keyval: int, _code: int, state: Gdk.ModifierType
    ) -> bool:
        if state & (Gdk.ModifierType.CONTROL_MASK | Gdk.ModifierType.ALT_MASK):
            return False
        name = Gdk.keyval_name(keyval) or ""
        if name in ("Left", "Right", "Home", "End") and self.showing_split:
            step = {"Left": -KEY_SPLIT_STEP, "Right": KEY_SPLIT_STEP, "Home": -1, "End": 1}[name]
            self.split = min(1.0, max(0.0, self.split + step))
            self.queue_draw()
        elif name in ("plus", "equal", "KP_Add"):
            self.zoom_by(1.25)
        elif name in ("minus", "KP_Subtract"):
            self.zoom_by(1 / 1.25)
        elif name in ("1", "KP_1"):
            self.actual_pixels()
        elif name in ("0", "KP_0"):
            self.set_zoom(None)
        else:
            return False
        return True

    # --- drawing -----------------------------------------------------------
    def do_size_allocate(self, width: int, height: int, baseline: int) -> None:
        if self.zoom is None:
            self._view_changed()

    def _draw_side(
        self,
        snapshot: Gtk.Snapshot,
        side: _Side,
        origin: tuple[float, float],
        zoom: float,
        filt: Gsk.ScalingFilter,
    ) -> None:
        if side.texture is None:
            return
        iw, ih = self.image_size
        ox, oy = origin
        snapshot.append_scaled_texture(side.texture, filt, _rect(ox, oy, iw * zoom, ih * zoom))
        if side.detail:
            texture, (x0, y0, x1, y1) = side.detail
            snapshot.append_scaled_texture(
                texture,
                filt,
                _rect(ox + x0 * zoom, oy + y0 * zoom, (x1 - x0) * zoom, (y1 - y0) * zoom),
            )

    def _label(
        self, snapshot: Gtk.Snapshot, text: str, x: float, y: float, align_right: bool = False
    ) -> float:
        """Draw a stage-chip-style label; returns its width."""
        layout = self.create_pango_layout(text)
        layout.set_attributes(
            Pango.AttrList.from_string("0 -1 weight 800, 0 -1 scale 0.78, 0 -1 letter-spacing 600")
        )
        _ink, logical = layout.get_pixel_extents()
        pad_x, pad_y = 12, 3
        w, h = logical.width + 2 * pad_x, logical.height + 2 * pad_y
        if align_right:
            x -= w
        bubble = Gsk.RoundedRect()
        bubble.init_from_rect(_rect(x, y, w, h), h / 2)
        snapshot.push_rounded_clip(bubble)
        snapshot.append_color(_rgba("rgba(0,0,0,0.55)"), _rect(x, y, w, h))
        snapshot.pop()
        snapshot.save()
        snapshot.translate(Graphene.Point().init(x + pad_x, y + pad_y))
        snapshot.append_layout(layout, _rgba("white"))
        snapshot.restore()
        return w

    def do_snapshot(self, snapshot: Gtk.Snapshot) -> None:
        w, h = self.get_width(), self.get_height()
        if not self.image_size[0] or (self.before.texture is None and self.after.texture is None):
            return
        zoom = self.effective_zoom()
        origin = self._origin(zoom)
        device_zoom = zoom * self.get_scale_factor()
        if device_zoom >= 3:
            filt = Gsk.ScalingFilter.NEAREST  # pixel peeping: show real pixels
        elif device_zoom < 1:
            filt = Gsk.ScalingFilter.TRILINEAR
        else:
            filt = Gsk.ScalingFilter.LINEAR

        if not self.showing_split:
            side = self.after if self.after.texture is not None else self.before
            self._draw_side(snapshot, side, origin, zoom, filt)
            return

        x0, y0, x1, y1 = self._visible_rect()
        sx = self._split_x()
        snapshot.push_clip(_rect(0, 0, sx, h))
        self._draw_side(snapshot, self.before, origin, zoom, filt)
        snapshot.pop()
        snapshot.push_clip(_rect(sx, 0, w - sx, h))
        self._draw_side(snapshot, self.after, origin, zoom, filt)
        snapshot.pop()

        # The divider spans the image, not the empty stage around it.
        snapshot.append_color(_rgba("rgba(255,255,255,0.9)"), _rect(sx - 1, y0, 2, y1 - y0))
        cy = (y0 + y1) / 2
        knob = _rect(sx - HANDLE_RADIUS, cy - HANDLE_RADIUS, 2 * HANDLE_RADIUS, 2 * HANDLE_RADIUS)
        handle = Gsk.RoundedRect()
        handle.init_from_rect(knob, HANDLE_RADIUS)
        snapshot.append_outset_shadow(handle, _rgba("rgba(0,0,0,0.35)"), 0, 1, 0, 4)
        snapshot.push_rounded_clip(handle)
        snapshot.append_color(_rgba("white"), knob)
        snapshot.pop()
        arrows = self.create_pango_layout("◀ ▶")
        arrows.set_attributes(Pango.AttrList.from_string("0 -1 scale 0.55"))
        _ink, logical = arrows.get_pixel_extents()
        snapshot.save()
        snapshot.translate(Graphene.Point().init(sx - logical.width / 2, cy - logical.height / 2))
        snapshot.append_layout(arrows, _rgba("#333333"))
        snapshot.restore()

        # Each side's label, while there is room for it.
        top = y0 + LABEL_INSET
        snapshot.push_clip(_rect(x0, y0, sx - x0, y1 - y0))
        self._label(snapshot, self.before_label, x0 + LABEL_INSET, top)
        snapshot.pop()
        snapshot.push_clip(_rect(sx, y0, x1 - sx, y1 - y0))
        self._label(snapshot, self.after_label, x1 - LABEL_INSET, top, align_right=True)
        snapshot.pop()
