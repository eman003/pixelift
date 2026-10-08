"""Before / after comparison window with slider, zoom and pan.

Memory: both images are loaded as previews no larger than ``PREVIEW_MAX``
pixels per side. When the user zooms in beyond the preview's resolution, only
the visible region is decoded at full resolution (in a background thread) and
drawn on top — the full-resolution image is never kept in memory by the UI.

Before an image is upscaled, the "after" side previews the lighting profile
and the camera look: they are applied to a downscaled copy of the original
(never the AI model), and to the decoded region when zoomed in, so slider
changes update in a moment.

In Restore Photos mode, "Preview Restoration" runs the full restoration
(without upscaling) on that downscaled copy, so settings can be judged before
the whole photo is processed. Restored results compare as
"Original Scan" / "Restored".
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from gi.repository import Adw, Gdk, GLib, GObject, Graphene, Gsk, Gtk, Pango
from PIL import Image

from pixelift.core import camera_looks as cl
from pixelift.core.batch_processor import ItemStatus, QueueItem
from pixelift.core.control import JobControl
from pixelift.core.errors import CancelledError, UpscalerError, friendly_error
from pixelift.core.lighting import Adjustments
from pixelift.ui.async_utils import run_in_thread
from pixelift.ui.widgets.camera_looks import CameraLookControls
from pixelift.ui.widgets.dialogs import show_error
from pixelift.ui.widgets.lighting import LightingControls
from pixelift.ui.widgets.textures import texture_from_pil
from pixelift.utils.image_utils import ImageInfo, load_preview, load_region

log = logging.getLogger(__name__)

PREVIEW_MAX = 4096
LIGHTING_PREVIEW_MAX = 1600  # live lighting preview; zooming in loads full detail
MAX_ZOOM = 16.0
HANDLE_RADIUS = 14


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

    def __init__(self) -> None:
        super().__init__(hexpand=True, vexpand=True, focusable=True)
        self.add_css_class("compare-view")
        self.set_overflow(Gtk.Overflow.HIDDEN)
        self.before = _Side()
        self.after = _Side()
        self.image_size: tuple[int, int] = (0, 0)  # logical size = upscaled size
        self.mode = "slider"
        self.before_label = "Before"
        self.after_label = "After"
        self.split = 0.5
        self.zoom: float | None = None  # None = fit to window
        self.center = (0.0, 0.0)
        self._drag_kind = ""
        self._drag_start = (0.0, 0.0)
        self._drag_center = (0.0, 0.0)
        self._pinch_zoom = 1.0
        self._pointer = (0.0, 0.0)
        self._detail_source = 0

        drag = Gtk.GestureDrag()
        drag.connect("drag-begin", self._on_drag_begin)
        drag.connect("drag-update", self._on_drag_update)
        drag.connect("drag-end", lambda *_: self.set_cursor_from_name(None))
        self.add_controller(drag)
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

    # --- state -------------------------------------------------------------
    def set_images(self, image_size: tuple[int, int]) -> None:
        self.image_size = image_size
        self.center = (image_size[0] / 2, image_size[1] / 2)
        self.zoom = None
        self.queue_draw()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.queue_draw()

    def fit_zoom(self) -> float:
        w, h = self.get_width(), self.get_height()
        iw, ih = self.image_size
        if not iw or not ih or not w or not h:
            return 1.0
        return min(w / iw, h / ih)

    def effective_zoom(self) -> float:
        return self.fit_zoom() if self.zoom is None else self.zoom

    def set_zoom(self, zoom: float | None, anchor: tuple[float, float] | None = None) -> None:
        if zoom is None:
            self.zoom = None
            self.center = (self.image_size[0] / 2, self.image_size[1] / 2)
        else:
            old = self.effective_zoom()
            low = min(self.fit_zoom(), 1.0) * 0.5
            zoom = max(low, min(MAX_ZOOM, zoom))
            if anchor is not None:
                ax, ay = anchor
                wc = (self.get_width() / 2, self.get_height() / 2)
                px = self.center[0] + (ax - wc[0]) / old
                py = self.center[1] + (ay - wc[1]) / old
                self.center = (px - (ax - wc[0]) / zoom, py - (ay - wc[1]) / zoom)
            self.zoom = zoom
            self._clamp_center()
        self._view_changed()

    def actual_pixels(self) -> None:
        """1 image pixel = 1 physical screen pixel."""
        self.set_zoom(1.0 / max(1, self.get_scale_factor()))

    def zoom_by(self, factor: float) -> None:
        self.set_zoom(self.effective_zoom() * factor, (self.get_width() / 2, self.get_height() / 2))

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
        w, h = self.get_width(), self.get_height()
        x0 = max(0.0, self.center[0] - w / 2 / zoom)
        y0 = max(0.0, self.center[1] - h / 2 / zoom)
        x1 = min(float(iw), self.center[0] + w / 2 / zoom)
        y1 = min(float(ih), self.center[1] + h / 2 / zoom)
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
    def _near_divider(self, x: float) -> bool:
        return (
            self.mode == "slider"
            and self.before.texture is not None
            and self.after.texture is not None
            and abs(x - self.split * self.get_width()) <= HANDLE_RADIUS + 6
        )

    def _on_motion(self, _ctrl: Gtk.EventControllerMotion, x: float, y: float) -> None:
        self._pointer = (x, y)
        self.set_cursor_from_name("col-resize" if self._near_divider(x) else "grab")

    def _on_drag_begin(self, _gesture: Gtk.GestureDrag, x: float, y: float) -> None:
        self.grab_focus()
        self._drag_start = (x, y)
        self._drag_kind = "split" if self._near_divider(x) else "pan"
        if self._drag_kind == "pan":
            self.zoom = self.effective_zoom()
            self._drag_center = self.center
            self.set_cursor_from_name("grabbing")

    def _on_drag_update(self, _gesture: Gtk.GestureDrag, dx: float, dy: float) -> None:
        if self._drag_kind == "split":
            width = max(1, self.get_width())
            self.split = min(1.0, max(0.0, (self._drag_start[0] + dx) / width))
            self.queue_draw()
        else:
            zoom = self.effective_zoom()
            self.center = (self._drag_center[0] - dx / zoom, self._drag_center[1] - dy / zoom)
            self._clamp_center()
            self._view_changed()

    def _on_scroll(self, _ctrl: Gtk.EventControllerScroll, _dx: float, dy: float) -> bool:
        self.set_zoom(self.effective_zoom() * (1.15**-dy), self._pointer)
        return True

    def _on_pinch(self, gesture: Gtk.GestureZoom, scale: float) -> None:
        ok, x, y = gesture.get_bounding_box_center()
        self.set_zoom(self._pinch_zoom * scale, (x, y) if ok else None)

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
    ) -> None:
        layout = self.create_pango_layout(text)
        layout.set_attributes(Pango.AttrList.from_string("0 -1 weight bold"))
        _ink, logical = layout.get_pixel_extents()
        pad_x, pad_y = 10, 4
        w, h = logical.width + 2 * pad_x, logical.height + 2 * pad_y
        if align_right:
            x -= w
        bubble = Gsk.RoundedRect()
        bubble.init_from_rect(_rect(x, y, w, h), h / 2)
        snapshot.push_rounded_clip(bubble)
        snapshot.append_color(_rgba("rgba(0,0,0,0.6)"), _rect(x, y, w, h))
        snapshot.pop()
        snapshot.save()
        snapshot.translate(Graphene.Point().init(x + pad_x, y + pad_y))
        snapshot.append_layout(layout, _rgba("white"))
        snapshot.restore()

    def do_snapshot(self, snapshot: Gtk.Snapshot) -> None:
        w, h = self.get_width(), self.get_height()
        iw = self.image_size[0]
        if not iw or (self.before.texture is None and self.after.texture is None):
            return
        zoom = self.effective_zoom()
        origin = (w / 2 - self.center[0] * zoom, h / 2 - self.center[1] * zoom)
        device_zoom = zoom * self.get_scale_factor()
        if device_zoom >= 3:
            filt = Gsk.ScalingFilter.NEAREST  # pixel peeping: show real pixels
        elif device_zoom < 1:
            filt = Gsk.ScalingFilter.TRILINEAR
        else:
            filt = Gsk.ScalingFilter.LINEAR

        has_both = self.before.texture is not None and self.after.texture is not None
        mode = self.mode if has_both else ("after" if self.after.texture else "before")
        if mode == "before":
            self._draw_side(snapshot, self.before, origin, zoom, filt)
            self._label(snapshot, self.before_label, 12, 12)
        elif mode == "after":
            self._draw_side(snapshot, self.after, origin, zoom, filt)
            self._label(snapshot, self.after_label if has_both else "Upscaled", 12, 12)
        else:
            sx = round(self.split * w)
            snapshot.push_clip(_rect(0, 0, sx, h))
            self._draw_side(snapshot, self.before, origin, zoom, filt)
            snapshot.pop()
            snapshot.push_clip(_rect(sx, 0, w - sx, h))
            self._draw_side(snapshot, self.after, origin, zoom, filt)
            snapshot.pop()
            snapshot.append_color(_rgba("rgba(255,255,255,0.9)"), _rect(sx - 1, 0, 2, h))
            handle = Gsk.RoundedRect()
            cy = h / 2
            handle.init_from_rect(
                _rect(sx - HANDLE_RADIUS, cy - HANDLE_RADIUS, 2 * HANDLE_RADIUS, 2 * HANDLE_RADIUS),
                HANDLE_RADIUS,
            )
            snapshot.push_rounded_clip(handle)
            snapshot.append_color(
                _rgba("white"),
                _rect(sx - HANDLE_RADIUS, cy - HANDLE_RADIUS, 2 * HANDLE_RADIUS, 2 * HANDLE_RADIUS),
            )
            snapshot.pop()
            arrows = self.create_pango_layout("◀ ▶")
            arrows.set_attributes(Pango.AttrList.from_string("0 -1 scale 0.55"))
            _ink, logical = arrows.get_pixel_extents()
            snapshot.save()
            snapshot.translate(
                Graphene.Point().init(sx - logical.width / 2, cy - logical.height / 2)
            )
            snapshot.append_layout(arrows, _rgba("#333333"))
            snapshot.restore()
            if sx > 90:
                self._label(snapshot, self.before_label, 12, 12)
            if w - sx > 90:
                self._label(snapshot, self.after_label, w - 12, 12, align_right=True)


class PreviewWindow(Adw.Window):
    def __init__(self, parent: Gtk.Window, item: QueueItem, info: ImageInfo) -> None:
        super().__init__(transient_for=parent, modal=False, title=item.path.name)
        self.set_default_size(1100, 760)
        self.app = parent.get_application()
        self.item = item
        self.info = info
        self._lighting_base: Image.Image | None = None  # downscaled original
        # (lighting, camera look, output scale) shown on the after side; None: unknown.
        self._lighting_shown: tuple[Adjustments, cl.LookRecipe, int] | None = (
            Adjustments(),
            cl.LookRecipe(),
            0,
        )
        self.lighting: LightingControls | None = None
        self.camera_look: CameraLookControls | None = None
        self._lighting_busy = False
        self._lighting_generation = 0
        output = (
            item.result.output
            if item.result and item.status in (ItemStatus.DONE, ItemStatus.SKIPPED)
            else None
        )
        self.output = output if output and output.exists() else None
        self.restored = bool(self.output and item.result and item.result.restored)
        self._restore_busy = False
        self._restore_control: JobControl | None = None  # the running restoration preview
        self._closed = False
        self._restore_shown = False  # the after side shows a restoration preview

        self.view = CompareView()
        if self.restored:
            self.view.before_label = "Original Scan"
            self.view.after_label = "Restored"
        self.view.before.path = item.path
        self.view.after.path = self.output
        self.view.connect("view-changed", lambda _v: self._update_zoom_label())

        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()
        subtitle = f"{info.width}×{info.height}"
        if item.result and self.output:
            ow, oh = item.result.output_size
            subtitle += f"  →  {ow}×{oh}"
        elif getattr(parent, "restoring", False):
            subtitle += " · not restored yet"
        else:
            subtitle += " · not upscaled yet"
        header.set_title_widget(Adw.WindowTitle(title=item.path.name, subtitle=subtitle))

        modes = Gtk.Box(css_classes=["linked"])
        self.mode_buttons: dict[str, Gtk.ToggleButton] = {}
        group: Gtk.ToggleButton | None = None
        for mode, label, tip in (
            ("slider", "Compare", "Before/after slider (S)"),
            ("before", "Before", "Original (B)"),
            ("after", "After", "Upscaled (A)"),
        ):
            btn = Gtk.ToggleButton(label=label, tooltip_text=tip, group=group)
            group = group or btn
            btn.connect("toggled", self._on_mode_toggled, mode)
            modes.append(btn)
            self.mode_buttons[mode] = btn
        self.modes = modes
        modes.set_sensitive(self.output is not None)
        header.pack_start(modes)
        if self.restored:
            self.mode_buttons["before"].set_label("Original")
            self.mode_buttons["after"].set_label("Restored")
        self.parent_window = parent
        self.restore_btn: Gtk.Button | None = None
        if self.output is None and getattr(parent, "restoring", False):
            self.restore_btn = Gtk.Button(
                label="Preview Restoration",
                tooltip_text="Restore a reduced-size copy with the current settings",
            )
            self.restore_btn.add_css_class("suggested-action")
            self.restore_btn.connect("clicked", lambda _b: self._preview_restoration())
            header.pack_start(self.restore_btn)

        zoom_box = Gtk.Box(css_classes=["linked"])
        zoom_out = Gtk.Button(icon_name="zoom-out-symbolic", tooltip_text="Zoom Out (−)")
        zoom_out.connect("clicked", lambda _b: self.view.zoom_by(1 / 1.25))
        self.zoom_label = Gtk.Button(label="Fit", tooltip_text="Fit to Window (0)")
        self.zoom_label.connect("clicked", lambda _b: self.view.set_zoom(None))
        self.zoom_label.add_css_class("numeric")
        zoom_in = Gtk.Button(icon_name="zoom-in-symbolic", tooltip_text="Zoom In (+)")
        zoom_in.connect("clicked", lambda _b: self.view.zoom_by(1.25))
        for w in (zoom_out, self.zoom_label, zoom_in):
            zoom_box.append(w)
        actual = Gtk.Button(
            icon_name="zoom-original-symbolic", tooltip_text="Actual Pixels / 100% (1)"
        )
        actual.connect("clicked", lambda _b: self.view.actual_pixels())
        fit = Gtk.Button(icon_name="zoom-fit-best-symbolic", tooltip_text="Fit to Window (0)")
        fit.connect("clicked", lambda _b: self.view.set_zoom(None))
        header.pack_end(fit)
        header.pack_end(actual)
        header.pack_end(zoom_box)
        toolbar.add_top_bar(header)

        overlay = Gtk.Overlay(child=self.view)
        self.spinner = Gtk.Spinner(
            spinning=True,
            halign=Gtk.Align.CENTER,
            valign=Gtk.Align.CENTER,
            width_request=32,
            height_request=32,
        )
        overlay.add_overlay(self.spinner)
        toolbar.set_content(overlay)

        hint = Gtk.Label(
            label="Drag the divider to compare · Scroll to zoom · Drag to pan · "
            "Space toggles before/after",
            ellipsize=Pango.EllipsizeMode.END,
            margin_top=6,
            margin_bottom=6,
        )
        hint.add_css_class("dim-label")
        hint.add_css_class("caption")
        bottom = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        if self.output is None:
            # Not upscaled yet: tune the lighting and camera look here and see
            # them on the right. (An upscaled result already has them baked in.)
            self.lighting = LightingControls(self.app)
            self.camera_look = CameraLookControls(self.app)
            looks = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
            looks.append(self.lighting)
            looks.append(self.camera_look)
            looks.set_margin_top(6)
            looks.set_margin_start(12)
            looks.set_margin_end(12)
            bottom.append(Adw.Clamp(maximum_size=820, child=looks))
            # A running batch has already taken its lighting (main window too).
            self.set_lighting_editable(not getattr(parent, "running", False))
            self.app.on_lighting_changed(self._update_lighting)
            self.connect("close-request", self._on_close_request)
        bottom.append(hint)
        toolbar.add_bottom_bar(bottom)
        self.set_content(toolbar)

        keys = Gtk.EventControllerKey()
        keys.connect("key-pressed", self._on_key)
        self.add_controller(keys)

        self.mode_buttons["slider" if self.output else "before"].set_active(True)
        run_in_thread(
            self._load, on_done=self._on_loaded, on_error=self._on_load_error, name="preview-load"
        )

    def set_lighting_editable(self, editable: bool) -> None:
        for controls in (self.lighting, self.camera_look):
            if controls is not None:
                controls.set_sensitive(editable)

    def cancel_restoration(self) -> None:
        """Stop a running restoration preview (window closed or a batch started)."""
        if self._restore_control is not None:
            self._restore_control.cancel()

    def _load(self) -> tuple:
        before, before_size = load_preview(self.item.path, PREVIEW_MAX)
        after = after_size = base = None
        if self.output is not None:
            after, after_size = load_preview(self.output, PREVIEW_MAX)
        else:
            # Downscaled base for the live lighting preview, made here rather
            # than on the UI thread.
            base = before.copy()
            base.thumbnail((LIGHTING_PREVIEW_MAX, LIGHTING_PREVIEW_MAX), Image.Resampling.BILINEAR)
        return before, before_size, after, after_size, base

    def _on_loaded(self, result: tuple) -> None:
        before, before_size, after, after_size, base = result
        self.spinner.set_visible(False)
        self.view.before.texture = texture_from_pil(before)
        self.view.before.source_size = before_size
        if after is not None:
            self.view.after.texture = texture_from_pil(after)
            self.view.after.source_size = after_size
            self.view.set_images(after_size)
        else:
            self.view.set_images(before_size)
            self._lighting_base = base
            if self.camera_look is not None:
                self.camera_look.set_sample(base)
            self._update_lighting()
        self._update_zoom_label()

    # --- lighting preview -------------------------------------------------
    # --- restoration preview ---------------------------------------------
    def _preview_restoration(self) -> None:
        if self._lighting_base is None or self._restore_busy:
            return
        parent = self.parent_window
        if getattr(parent, "running", False):
            show_error(
                self,
                UpscalerError(
                    "Photos are being processed right now.",
                    ["Preview again when the batch has finished"],
                    title="Preview unavailable.",
                ),
            )
            return
        try:
            restorer, settings, options = parent.preview_restorer()
        except UpscalerError as err:
            show_error(self, err)
            return
        import dataclasses

        import numpy as np

        base = np.array(self._lighting_base.convert("RGB"))
        quick = dataclasses.replace(settings, scale=1)  # preview without upscaling
        control = JobControl()
        self._restore_control = control
        self._restore_busy = True
        self._lighting_generation += 1  # a running lighting render must not replace it
        self.spinner.set_visible(True)
        self.restore_btn.set_sensitive(False)
        self.restore_btn.set_label("Restoring…")
        lighting = options.lighting.adjustments()
        look = options.camera_look.recipe()
        look_output = self._output_size(options.output_scale)

        def work() -> Image.Image:
            from pixelift.core.restoration.analysis import detect_monochrome_file

            mono = detect_monochrome_file(self.item.path)
            control.check()
            result = restorer.restore(
                base,
                quick,
                model=options.model,
                lighting=lighting,
                look=look,
                look_output=look_output,
                mono=mono,
                control=control,
            )
            return Image.fromarray(result.rgb).convert("RGBA")

        def finished() -> bool:
            """Clean up; False when the window was closed meanwhile."""
            if self._restore_control is control:
                self._restore_control = None
            self._restore_busy = False
            return not self._closed

        def done(img: Image.Image) -> None:
            if not finished():
                return
            if control.cancelled:  # a batch started meanwhile
                failed(CancelledError())
                return
            self.spinner.set_visible(False)
            self.restore_btn.set_sensitive(True)
            self.restore_btn.set_label("Update Preview")
            after = self.view.after
            after.texture = texture_from_pil(img)
            after.path = None  # no full-resolution detail: it is a preview
            after.detail = None
            after.transform = None
            after.source_size = self.view.before.source_size
            self._restore_shown = True
            self.view.before_label = "Original Scan"
            self.view.after_label = "Restored (preview)"
            self.modes.set_sensitive(True)
            self.mode_buttons["before"].set_label("Original")
            self.mode_buttons["after"].set_label("Restored")
            self.mode_buttons["slider"].set_active(True)
            self.view.queue_draw()

        def failed(exc: BaseException) -> None:
            if not finished():
                return
            self.spinner.set_visible(False)
            self.restore_btn.set_sensitive(True)
            self.restore_btn.set_label("Preview Restoration")
            if not isinstance(exc, CancelledError):
                show_error(self, friendly_error(exc))

        run_in_thread(work, on_done=done, on_error=failed, name="preview-restore")

    def _wanted_look(self) -> tuple[Adjustments, cl.LookRecipe, int]:
        settings = self.app.settings
        scale = settings.processing_options().output_scale
        return settings.lighting().adjustments(), settings.camera_look_settings().recipe(), scale

    def _output_size(self, scale: int) -> tuple[int, int]:
        """The export's size: the look's grain and sharpening are previewed as
        they will look there."""
        sw, sh = self.view.before.source_size
        return sw * scale, sh * scale

    def _update_lighting(self) -> None:
        """Re-render the after side if the lighting or camera look changed."""
        if self._lighting_base is None:
            return
        if self._restore_shown or self._restore_busy:
            # Settings changed after a restoration preview: offer to refresh it.
            if self.restore_btn is not None and not self._restore_busy:
                self.restore_btn.set_label("Update Preview")
            return
        wanted = self._wanted_look()
        if wanted == self._lighting_shown:
            return
        if self._lighting_busy:
            return  # picked up when the running render finishes
        self._lighting_shown = wanted
        after = self.view.after
        adjustments, recipe, scale = wanted
        output = self._output_size(scale)
        if adjustments.is_neutral and recipe.is_neutral:
            after.clear()
            self._set_lighting_mode(False)
            return
        self._lighting_busy = True
        self._lighting_generation += 1
        generation = self._lighting_generation

        def done(img: Image.Image) -> None:
            self._lighting_busy = False
            if generation != self._lighting_generation:
                return
            after.texture = texture_from_pil(img)
            after.path = self.item.path
            after.source_size = self.view.before.source_size
            sw, sh = self.view.before.source_size
            after.transform = lambda region, crop_scale: cl.apply_to_pil(
                region,
                recipe,
                adjustments,
                frame=(round(sw * crop_scale), round(sh * crop_scale)),
                output=output,
            )
            self._set_lighting_mode(True, adjustments, recipe)
            self.view.invalidate_detail(after)
            self._update_lighting()  # settings may have moved on meanwhile

        def failed(exc: BaseException) -> None:
            self._lighting_busy = False
            if generation != self._lighting_generation:
                return
            log.warning("Lighting preview failed: %s", exc)
            # Nothing valid is shown now: any later change renders again, and
            # one made during this render is picked up straight away.
            self._lighting_shown = None
            if self._wanted_look() != wanted:
                self._update_lighting()

        run_in_thread(
            lambda: cl.apply_to_pil(
                self._lighting_base, recipe, adjustments, quick=True, output=output
            ),
            on_done=done,
            on_error=failed,
            name="preview-lighting",
        )

    def _set_lighting_mode(
        self,
        on: bool,
        adjustments: Adjustments | None = None,
        recipe: cl.LookRecipe | None = None,
    ) -> None:
        was_on = self.modes.get_sensitive()
        self.modes.set_sensitive(on)
        lit = adjustments is not None and not adjustments.is_neutral
        graded = recipe is not None and not recipe.is_neutral
        label = "Adjusted" if lit and graded else "Camera Look" if graded else "Lighting"
        self.view.after_label = label
        self.mode_buttons["after"].set_label(label if on else "After")
        self.mode_buttons["after"].set_tooltip_text(f"With {label.lower()} (A)")
        if on and not was_on:
            self.mode_buttons["slider"].set_active(True)
        elif not on:
            self.mode_buttons["before"].set_active(True)
        self.view.queue_draw()

    def _on_close_request(self, _window: Gtk.Window) -> bool:
        self._closed = True
        self.cancel_restoration()
        self._lighting_generation += 1
        self.app.off_lighting_changed(self._update_lighting)
        for controls in (self.lighting, self.camera_look):
            if controls is not None:
                controls.shutdown()
        return False

    def _on_load_error(self, exc: BaseException) -> None:
        self.spinner.set_visible(False)
        log.error("Preview failed: %s", exc)
        self.set_content(
            Adw.StatusPage(
                icon_name="dialog-error-symbolic",
                title="Unable to show preview",
                description="The image could not be loaded.",
            )
        )

    def _on_mode_toggled(self, button: Gtk.ToggleButton, mode: str) -> None:
        if button.get_active():
            self.view.set_mode(mode)

    def _update_zoom_label(self) -> None:
        zoom = self.view.effective_zoom() * self.view.get_scale_factor()
        prefix = "Fit · " if self.view.zoom is None else ""
        self.zoom_label.set_label(f"{prefix}{zoom * 100:.0f}%")

    def _on_key(
        self, _ctrl: Gtk.EventControllerKey, keyval: int, _code: int, _state: Gdk.ModifierType
    ) -> bool:
        name = Gdk.keyval_name(keyval) or ""
        if name == "Escape":
            self.close()
        elif name == "space" and self.modes.get_sensitive():
            target = "after" if self.view.mode == "before" else "before"
            self.mode_buttons[target].set_active(True)
        elif name in ("s", "b", "a") and self.modes.get_sensitive():
            self.mode_buttons[{"s": "slider", "b": "before", "a": "after"}[name]].set_active(True)
        elif name in ("plus", "equal", "KP_Add"):
            self.view.zoom_by(1.25)
        elif name in ("minus", "KP_Subtract"):
            self.view.zoom_by(1 / 1.25)
        elif name in ("1", "KP_1"):
            self.view.actual_pixels()
        elif name in ("0", "KP_0", "f"):
            self.view.set_zoom(None)
        else:
            return False
        return True
