"""Queue rows and background thumbnail/metadata loading."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gi.repository import Gio, Gtk, Pango
from PIL import Image, ImageOps

from pixelift.core.batch_processor import ItemStatus, QueueItem
from pixelift.core.errors import UpscalerError, friendly_error
from pixelift.core.restoration.analysis import ANALYSIS_SIDE, detect_monochrome_file
from pixelift.ui.async_utils import idle
from pixelift.ui.widgets.textures import texture_from_pil
from pixelift.utils.image_utils import ImageInfo, human_size, make_thumbnail, probe_image

log = logging.getLogger(__name__)

THUMB_SIZE = 64


class ThumbnailLoader:
    """Probes images and builds thumbnails on a small thread pool."""

    def __init__(self, workers: int = 2) -> None:
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="thumbs")

    def request(
        self,
        path: Path,
        on_ready: Callable[[ImageInfo, Image.Image, bool], None],
        on_error: Callable[[UpscalerError], None],
    ) -> None:
        def job() -> None:
            try:
                info = probe_image(path)
                # One reduced decode for both the thumbnail and B&W detection.
                decoded = make_thumbnail(path, ANALYSIS_SIDE)
                thumb = ImageOps.fit(decoded, (THUMB_SIZE * 2, THUMB_SIZE * 2))  # square crop
            except Exception as exc:  # noqa: BLE001
                idle(on_error, friendly_error(exc))
                return
            try:
                # Cached: restoration's output naming asks again for free.
                mono = detect_monochrome_file(path, decoded).monochrome
            except Exception:
                log.debug("B&W detection failed for %s", path, exc_info=True)
                mono = False
            idle(on_ready, info, thumb, mono)

        self._pool.submit(job)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


_STATUS_CLASSES = {
    ItemStatus.DONE: "status-done",
    ItemStatus.FAILED: "status-failed",
    ItemStatus.SKIPPED: "status-skipped",
    ItemStatus.CANCELLED: "status-skipped",
    ItemStatus.PROCESSING: "status-processing",
}


class QueueRow(Gtk.ListBoxRow):
    """One image in the queue."""

    def __init__(
        self,
        item: QueueItem,
        on_remove: Callable[[QueueRow], None],
        on_retry: Callable[[QueueRow], None],
        on_show_error: Callable[[QueueRow], None],
    ) -> None:
        super().__init__()
        self.item = item
        self.info: ImageInfo | None = None
        self.load_error: UpscalerError | None = None
        self.monochrome = False  # looks like a black-and-white photograph
        self.show_monochrome = False
        self.scale = 4  # how much larger the output will be
        self.add_css_class("queue-row")
        self.set_activatable(True)

        box = Gtk.Box(spacing=12)
        self.set_child(box)

        self.thumb = Gtk.Image(
            pixel_size=THUMB_SIZE, icon_name="image-x-generic-symbolic", valign=Gtk.Align.CENTER
        )
        self.thumb.add_css_class("thumbnail")
        self.thumb.set_overflow(Gtk.Overflow.HIDDEN)
        box.append(self.thumb)

        text = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3, hexpand=True)
        text.set_valign(Gtk.Align.CENTER)
        self.title = Gtk.Label(label=item.path.name, xalign=0, ellipsize=Pango.EllipsizeMode.MIDDLE)
        self.title.add_css_class("heading")
        self.title.set_tooltip_text(str(item.path))
        self.details = Gtk.Label(label="Reading…", xalign=0, ellipsize=Pango.EllipsizeMode.END)
        self.details.add_css_class("dim-label")
        self.details.add_css_class("caption")
        self.details.add_css_class("numeric")
        # Status above its progress bar: both stay readable in a narrow sidebar.
        status_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.status = Gtk.Label(xalign=0, ellipsize=Pango.EllipsizeMode.END)
        self.status.add_css_class("caption")
        self.progress = Gtk.ProgressBar(hexpand=True, visible=False)
        status_box.append(self.status)
        status_box.append(self.progress)
        text.append(self.title)
        text.append(self.details)
        text.append(status_box)
        box.append(text)

        buttons = Gtk.Box(spacing=4, valign=Gtk.Align.CENTER)
        self.error_btn = self._icon_button("dialog-warning-symbolic", "Show error", on_show_error)
        self.retry_btn = self._icon_button("view-refresh-symbolic", "Retry", on_retry)
        self.open_btn = self._icon_button(
            "folder-open-symbolic", "Show in folder", lambda _r: self.show_in_folder()
        )
        self.remove_btn = self._icon_button("window-close-symbolic", "Remove from queue", on_remove)
        for btn in (
            self.error_btn,
            self.retry_btn,
            self.open_btn,
            self.remove_btn,
        ):
            buttons.append(btn)
        box.append(buttons)
        self.refresh()

    def _icon_button(
        self, icon: str, tooltip: str, callback: Callable[[QueueRow], None]
    ) -> Gtk.Button:
        button = Gtk.Button(icon_name=icon, tooltip_text=tooltip, valign=Gtk.Align.CENTER)
        button.add_css_class("flat")
        button.add_css_class("circular")
        button.update_property([Gtk.AccessibleProperty.LABEL], [tooltip])
        button.connect("clicked", lambda _b: callback(self))
        return button

    # --- data --------------------------------------------------------------
    def set_info(self, info: ImageInfo, thumb: Image.Image, monochrome: bool = False) -> None:
        self.info = info
        self.monochrome = monochrome
        self.thumb.set_from_paintable(texture_from_pil(thumb))
        self.refresh()

    def set_load_error(self, error: UpscalerError) -> None:
        self.load_error = error
        self.item.status = ItemStatus.FAILED
        self.item.error = error
        self.thumb.set_from_icon_name("image-missing-symbolic")
        self.refresh()

    @property
    def ready(self) -> bool:
        return self.info is not None and self.load_error is None

    def set_scale(self, scale: int, show_monochrome: bool = False) -> None:
        """Output size factor; ``show_monochrome`` tags B&W photos (Restore mode)."""
        self.scale = scale
        self.show_monochrome = show_monochrome
        self.refresh()

    # --- presentation ------------------------------------------------------
    def refresh(self) -> None:
        item = self.item
        if self.load_error:
            self.details.set_label(
                human_size(item.path.stat().st_size) if item.path.exists() else "Missing file"
            )
        elif self.info:
            info = self.info
            if item.result and item.status in (ItemStatus.DONE, ItemStatus.SKIPPED):
                out_w, out_h = item.result.output_size
            else:
                out_w, out_h = info.width * self.scale, info.height * self.scale
            text = f"{info.width}×{info.height} · {human_size(info.file_size)}  →  {out_w}×{out_h}"
            if self.monochrome and self.show_monochrome:
                text += " · B&W"
            self.details.set_label(text)

        status = item.status
        text = status.value
        if status is ItemStatus.PROCESSING:
            text = f"{item.stage or 'Processing'} · {item.progress * 100:.0f}%"
        elif status is ItemStatus.DONE and item.result:
            text = f"Done in {item.result.seconds:.1f} s"
            if item.result.note:
                text += f" · {item.result.note}"
        elif status is ItemStatus.SKIPPED and item.result:
            text = f"Skipped · {item.result.note}"
        elif status is ItemStatus.FAILED and item.error:
            text = f"Failed · {item.error.reason}"
        self.status.set_label(text)
        self.status.set_tooltip_text(text)
        for css in set(_STATUS_CLASSES.values()):
            self.status.remove_css_class(css)
        if status in _STATUS_CLASSES:
            self.status.add_css_class(_STATUS_CLASSES[status])

        busy = status in (ItemStatus.PROCESSING, ItemStatus.QUEUED)
        self.progress.set_visible(status is ItemStatus.PROCESSING)
        self.progress.set_fraction(item.progress)
        has_output = bool(item.result and item.result.output.exists())
        self.open_btn.set_visible(has_output)
        self.retry_btn.set_visible(
            status in (ItemStatus.FAILED, ItemStatus.CANCELLED) and self.load_error is None
        )
        self.error_btn.set_visible(status is ItemStatus.FAILED and item.error is not None)
        self.remove_btn.set_sensitive(not busy)

    def show_in_folder(self) -> None:
        if self.item.result:
            launcher = Gtk.FileLauncher.new(Gio.File.new_for_path(str(self.item.result.output)))
            launcher.open_containing_folder(self.get_root(), None, None)
