"""Main window: drop zone, image queue, output and lighting options, batch controls."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from gi.repository import Adw, Gdk, Gio, GLib, Gtk, Pango

from pixelift import APP_ID, APP_NAME
from pixelift.core import device_manager as dm
from pixelift.core.batch_processor import (
    BatchEvent,
    BatchProcessor,
    EventKind,
    ItemStatus,
    QueueItem,
)
from pixelift.core.errors import ModelNotInstalledError, UpscalerError
from pixelift.core.upscaler import TorchUpscaler
from pixelift.models import all_families
from pixelift.ui.async_utils import idle
from pixelift.ui.image_queue import QueueRow, ThumbnailLoader
from pixelift.ui.widgets.dialogs import show_error
from pixelift.ui.widgets.lighting import LightingControls
from pixelift.utils.image_utils import SUPPORTED_EXTENSIONS, collect_images

if TYPE_CHECKING:
    from pixelift.ui.application import UpscalerApplication

log = logging.getLogger(__name__)

SCALES = (2, 4)
FORMATS = (("png", "PNG"), ("jpeg", "JPEG"), ("webp", "WebP"))


class MainWindow(Adw.ApplicationWindow):
    def __init__(self, app: UpscalerApplication) -> None:
        super().__init__(application=app, title=APP_NAME)
        self.app = app
        self.set_default_size(920, 720)
        self.set_size_request(360, 480)
        self.rows: list[QueueRow] = []
        self.thumbnails = ThumbnailLoader()
        self.processor: BatchProcessor | None = None
        self._upscaler: TorchUpscaler | None = None
        self._upscaler_key: tuple[object, ...] | None = None
        self._inhibit_cookie = 0
        self._syncing = False

        toolbar = Adw.ToolbarView()
        toolbar.add_top_bar(self._build_header())
        self.toasts = Adw.ToastOverlay()
        self.stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE, vexpand=True)
        self.stack.add_named(self._build_empty_page(), "empty")
        self.stack.add_named(self._build_queue_page(), "queue")
        self.toasts.set_child(self.stack)
        self.toasts.add_css_class("drop-zone")
        toolbar.set_content(self.toasts)
        toolbar.add_bottom_bar(self._build_controls())
        toolbar.set_bottom_bar_style(Adw.ToolbarStyle.RAISED)
        self.set_content(toolbar)

        drop = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        drop.connect("drop", self._on_drop)
        self.toasts.add_controller(drop)

        narrow = Adw.Breakpoint.new(Adw.BreakpointCondition.parse("max-width: 640sp"))
        narrow.add_setter(self.options_box, "orientation", Gtk.Orientation.VERTICAL)
        narrow.add_setter(self.lighting, "orientation", Gtk.Orientation.VERTICAL)
        narrow.add_setter(self.buttons_box, "orientation", Gtk.Orientation.VERTICAL)
        self.add_breakpoint(narrow)

        self.connect("close-request", self._on_close_request)
        app.on_settings_changed(self._sync_from_settings)
        app.on_lighting_changed(self._on_lighting_changed)
        app.downloads.subscribe(lambda *_: self._refresh_model_list())
        app.when_devices_ready(self._on_devices)
        self._sync_from_settings()
        self._update_state()

    # --- construction ------------------------------------------------------
    def _build_header(self) -> Adw.HeaderBar:
        header = Adw.HeaderBar()
        self.window_title = Adw.WindowTitle(title=APP_NAME, subtitle="Detecting processing device…")
        header.set_title_widget(self.window_title)

        add = Gtk.Button(icon_name="list-add-symbolic", tooltip_text="Add Images (Ctrl+O)")
        add.connect("clicked", lambda _b: self.open_file_dialog())
        header.pack_start(add)
        self.clear_btn = Gtk.Button(
            icon_name="edit-clear-all-symbolic", tooltip_text="Clear Finished Images"
        )
        self.clear_btn.connect("clicked", lambda _b: self.clear_finished())
        header.pack_start(self.clear_btn)

        menu = Gio.Menu()
        section = Gio.Menu()
        section.append("Preferences", "app.preferences")
        section.append("AI Models", "app.models")
        menu.append_section(None, section)
        section = Gio.Menu()
        section.append("Setup Assistant", "app.first-run")
        section.append("Open Log Folder", "app.open-log")
        section.append("About Pixelift", "app.about")
        menu.append_section(None, section)
        menu_btn = Gtk.MenuButton(
            icon_name="open-menu-symbolic", menu_model=menu, tooltip_text="Main Menu", primary=True
        )
        header.pack_end(menu_btn)
        settings_btn = Gtk.Button(icon_name="emblem-system-symbolic", tooltip_text="Settings")
        settings_btn.set_action_name("app.preferences")
        header.pack_end(settings_btn)

        open_action = Gio.SimpleAction.new("open", None)
        open_action.connect("activate", lambda *_: self.open_file_dialog())
        self.add_action(open_action)
        self.app.set_accels_for_action("win.open", ["<Control>o"])
        return header

    def _build_empty_page(self) -> Gtk.Widget:
        page = Adw.StatusPage(
            icon_name=APP_ID,
            title="Drop images here",
            description="PNG, JPEG, WebP, TIFF or BMP — single images or whole folders.",
        )
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=18, halign=Gtk.Align.CENTER)
        select = Gtk.Button(label="Select Images", halign=Gtk.Align.CENTER)
        select.add_css_class("pill")
        select.add_css_class("suggested-action")
        select.connect("clicked", lambda _b: self.open_file_dialog())
        box.append(select)
        privacy = Gtk.Label(label="🔒 Your images stay on your computer.")
        privacy.add_css_class("dim-label")
        privacy.add_css_class("privacy-note")
        box.append(privacy)
        page.set_child(box)
        return page

    def _build_queue_page(self) -> Gtk.Widget:
        self.listbox = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        self.listbox.add_css_class("boxed-list")
        self.listbox.connect("row-activated", lambda _l, row: self.open_preview(row))
        self.queue_title = Gtk.Label(xalign=0)
        self.queue_title.add_css_class("heading")
        header = Gtk.Box(spacing=6, margin_bottom=8)
        header.append(self.queue_title)
        content = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            margin_top=12,
            margin_bottom=12,
            margin_start=12,
            margin_end=12,
        )
        content.append(header)
        content.append(self.listbox)
        clamp = Adw.Clamp(maximum_size=960, child=content)
        return Gtk.ScrolledWindow(child=clamp, hscrollbar_policy=Gtk.PolicyType.NEVER, vexpand=True)

    def _labeled(self, label: str, widget: Gtk.Widget) -> Gtk.Box:
        box = Gtk.Box(spacing=8)
        title = Gtk.Label(label=label, xalign=0, width_chars=6)
        title.add_css_class("dim-label")
        box.append(title)
        widget.set_hexpand(True)
        box.append(widget)
        return box

    def _build_controls(self) -> Gtk.Widget:
        panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        panel.add_css_class("control-panel")

        self.options_box = Gtk.Box(spacing=18, homogeneous=False)
        self.scale_dd = Gtk.DropDown.new_from_strings([f"{s}×" for s in SCALES])
        self.scale_dd.connect("notify::selected", self._on_option_changed)
        self.model_list = Gtk.StringList()
        self.model_dd = Gtk.DropDown(model=self.model_list)
        self.model_dd.connect("notify::selected", self._on_option_changed)
        self.format_dd = Gtk.DropDown.new_from_strings([name for _, name in FORMATS])
        self.format_dd.connect("notify::selected", self._on_option_changed)
        self.output_btn = Gtk.Button()
        self.output_label = Gtk.Label(ellipsize=Pango.EllipsizeMode.END, xalign=0)
        out_content = Gtk.Box(spacing=6)
        out_content.append(Gtk.Image(icon_name="folder-symbolic"))
        out_content.append(self.output_label)
        self.output_btn.set_child(out_content)
        self.output_btn.connect("clicked", lambda _b: self.choose_output_folder())
        self.output_reset = Gtk.Button(
            icon_name="edit-undo-symbolic", tooltip_text="Save next to the originals"
        )
        self.output_reset.connect("clicked", lambda _b: self._set_output_dir(""))
        output_row = Gtk.Box(css_classes=["linked"])
        output_row.append(self.output_btn)
        output_row.append(self.output_reset)
        self.output_btn.set_hexpand(True)

        self.options_box.append(self._labeled("Scale", self.scale_dd))
        self.options_box.append(self._labeled("Model", self.model_dd))
        self.options_box.append(self._labeled("Format", self.format_dd))
        self.options_box.append(self._labeled("Output", output_row))
        panel.append(self.options_box)
        self.lighting = LightingControls(self.app)
        panel.append(self.lighting)

        # Progress area (only while a batch is running / just finished)
        self.progress_revealer = Gtk.Revealer(transition_type=Gtk.RevealerTransitionType.SLIDE_UP)
        progress_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.overall_bar = Gtk.ProgressBar(show_text=True)
        line = Gtk.Box(spacing=12)
        self.current_label = Gtk.Label(xalign=0, hexpand=True, ellipsize=Pango.EllipsizeMode.MIDDLE)
        self.current_label.add_css_class("caption")
        self.overall_label = Gtk.Label(xalign=1)
        self.overall_label.add_css_class("caption")
        self.overall_label.add_css_class("numeric")
        line.append(self.current_label)
        line.append(self.overall_label)
        progress_box.append(self.overall_bar)
        progress_box.append(line)
        self.progress_revealer.set_child(progress_box)
        panel.append(self.progress_revealer)

        self.buttons_box = Gtk.Box(spacing=8, halign=Gtk.Align.CENTER)
        self.start_btn = Gtk.Button(label="Start Upscaling")
        self.start_btn.add_css_class("pill")
        self.start_btn.add_css_class("suggested-action")
        self.start_btn.connect("clicked", lambda _b: self.start())
        self.pause_btn = Gtk.Button(label="Pause")
        self.pause_btn.add_css_class("pill")
        self.pause_btn.connect("clicked", lambda _b: self.toggle_pause())
        self.cancel_btn = Gtk.Button(label="Cancel")
        self.cancel_btn.add_css_class("pill")
        self.cancel_btn.add_css_class("destructive-action")
        self.cancel_btn.connect("clicked", lambda _b: self.cancel())
        self.retry_btn = Gtk.Button(label="Retry Failed")
        self.retry_btn.add_css_class("pill")
        self.retry_btn.connect("clicked", lambda _b: self.retry_failed())
        for btn in (self.start_btn, self.pause_btn, self.cancel_btn, self.retry_btn):
            self.buttons_box.append(btn)
        panel.append(self.buttons_box)
        return panel

    # --- settings <-> controls ---------------------------------------------
    def _families(self) -> list[tuple[str, str]]:
        return [(f.id, f.name) for f in all_families()]

    def _refresh_model_list(self) -> None:
        self._syncing = True
        try:
            manager = self.app.model_manager
            installed = {s.id for s in manager.installed()}
            names = []
            for family in all_families():
                ok = any(spec_id in installed for spec_id in family.variants.values())
                names.append(family.name if ok else f"{family.name} (not installed)")
            self.model_list.splice(0, self.model_list.get_n_items(), names)
            ids = [fid for fid, _ in self._families()]
            if self.app.settings.model in ids:
                self.model_dd.set_selected(ids.index(self.app.settings.model))
        finally:
            self._syncing = False

    def _sync_from_settings(self) -> None:
        settings = self.app.settings
        self._refresh_model_list()
        self._syncing = True
        try:
            self.scale_dd.set_selected(
                SCALES.index(settings.scale) if settings.scale in SCALES else 1
            )
            fmt_ids = [f for f, _ in FORMATS]
            self.format_dd.set_selected(fmt_ids.index(settings.output_format))
            if settings.output_dir:
                self.output_label.set_label(Path(settings.output_dir).name or settings.output_dir)
                self.output_btn.set_tooltip_text(settings.output_dir)
            else:
                self.output_label.set_label("Next to originals (upscaled/)")
                self.output_btn.set_tooltip_text("<original folder>/upscaled/")
            self.output_reset.set_visible(bool(settings.output_dir))
        finally:
            self._syncing = False
        for row in self.rows:
            row.set_scale(settings.scale)
        if self.app.device_report:
            self._on_devices(self.app.device_report)

    def _on_lighting_changed(self) -> None:
        """Results made with other lighting are out of date: queue them again.

        Their files stay on disk (the lighting is part of the file name), so
        switching back to that lighting just finds and skips them.
        """
        if self.running:
            return
        tag = self.app.settings.lighting().tag()
        stale = [
            r
            for r in self.rows
            if r.item.status in (ItemStatus.DONE, ItemStatus.SKIPPED)
            and r.item.result is not None
            and r.item.result.lighting != tag
        ]
        for row in stale:
            row.item.reset()
            row.refresh()
        if stale:
            self._update_state()

    def _on_option_changed(self, *_args: object) -> None:
        if self._syncing:
            return
        settings = self.app.settings
        settings.scale = SCALES[self.scale_dd.get_selected()]
        families = self._families()
        if self.model_dd.get_selected() < len(families):
            settings.model = families[self.model_dd.get_selected()][0]
        settings.output_format = FORMATS[self.format_dd.get_selected()][0]
        self.app.settings_changed()

    def choose_output_folder(self) -> None:
        dialog = Gtk.FileDialog(title="Choose Output Folder", modal=True)
        if self.app.settings.output_dir:
            dialog.set_initial_folder(Gio.File.new_for_path(self.app.settings.output_dir))

        def done(dlg: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
            try:
                folder = dlg.select_folder_finish(result)
            except GLib.Error:
                return  # dismissed
            if folder and folder.get_path():
                self._set_output_dir(folder.get_path())

        dialog.select_folder(self, None, done)

    def _set_output_dir(self, folder: str) -> None:
        self.app.settings.output_dir = folder
        self.app.settings_changed()

    def _on_devices(self, _report: dm.DeviceReport) -> None:
        if self.running and self._upscaler is not None:
            device = self._upscaler.device
        else:
            device = self.app.selected_device()
        self.window_title.set_subtitle(f"Processing device: {device.label()}")

    # --- adding / removing images -----------------------------------------
    def open_file_dialog(self) -> None:
        image_filter = Gtk.FileFilter(name="Images")
        for ext in sorted(SUPPORTED_EXTENSIONS):
            image_filter.add_suffix(ext.lstrip("."))
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(image_filter)
        dialog = Gtk.FileDialog(
            title="Select Images", modal=True, filters=filters, default_filter=image_filter
        )
        if self.app.settings.last_open_dir:
            dialog.set_initial_folder(Gio.File.new_for_path(self.app.settings.last_open_dir))

        def done(dlg: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
            try:
                files = dlg.open_multiple_finish(result)
            except GLib.Error:
                return
            paths = [Path(f.get_path()) for f in files if f.get_path()]
            if paths:
                self.app.settings.last_open_dir = str(paths[0].parent)
                self.app.settings_changed()
                self.add_paths(paths)

        dialog.open_multiple(self, None, done)

    def _on_drop(self, _target: Gtk.DropTarget, value: Gdk.FileList, _x: float, _y: float) -> bool:
        paths = [Path(f.get_path()) for f in value.get_files() if f.get_path()]
        self.add_paths(paths)
        return bool(paths)

    def add_paths(self, paths: list[Path]) -> None:
        existing = {row.item.path.resolve() for row in self.rows}
        images = [p for p in collect_images(paths) if p.resolve() not in existing]
        for path in images:
            self._add_row(QueueItem(path))
        if images:
            self.toasts.add_toast(
                Adw.Toast(
                    title=f"Added {len(images)} image{'s' if len(images) != 1 else ''}", timeout=2
                )
            )
        elif paths:
            self.toasts.add_toast(Adw.Toast(title="No new supported images found"))
        self._update_state()

    def _add_row(self, item: QueueItem) -> None:
        row = QueueRow(
            item, self.remove_row, self.retry_row, self.open_preview, self.show_row_error
        )
        row.set_scale(self.app.settings.scale)
        self.rows.append(row)
        self.listbox.append(row)

        def ready(info: object, thumb: object, r: QueueRow = row) -> None:
            r.set_info(info, thumb)
            self._update_state()

        def failed(err: UpscalerError, r: QueueRow = row) -> None:
            r.set_load_error(err)
            self._update_state()

        self.thumbnails.request(item.path, ready, failed)

    def remove_row(self, row: QueueRow) -> None:
        if row.item.status in (ItemStatus.PROCESSING, ItemStatus.QUEUED):
            return
        self.rows.remove(row)
        self.listbox.remove(row)
        self._update_state()

    def clear_finished(self) -> None:
        for row in list(self.rows):
            if row.item.status in (ItemStatus.DONE, ItemStatus.SKIPPED):
                self.remove_row(row)

    def show_row_error(self, row: QueueRow) -> None:
        if row.item.error:
            show_error(self, row.item.error)

    def open_preview(self, row: QueueRow) -> None:
        if not row.ready:
            if row.item.error:
                show_error(self, row.item.error)
            return
        from pixelift.ui.preview import PreviewWindow

        PreviewWindow(self, row.item, row.info).present()

    # --- processing --------------------------------------------------------
    @property
    def running(self) -> bool:
        return self.processor is not None and self.processor.running

    def _get_upscaler(self) -> TorchUpscaler:
        settings = self.app.settings
        device = self.app.selected_device()
        key = (device.id, settings.tile_size, settings.gpu_memory_limit_mb, settings.cpu_threads)
        if self._upscaler is None or key != self._upscaler_key:
            if self._upscaler is not None:
                self._upscaler.release()
            self._upscaler = TorchUpscaler(
                self.app.model_manager,
                device,
                tile_size=settings.tile_size,
                memory_limit_mb=settings.gpu_memory_limit_mb,
                cpu_threads=settings.cpu_threads,
                on_device_change=lambda dev, why: idle(self._on_device_fallback, dev, why),
            )
            self._upscaler_key = key
        return self._upscaler

    def _on_device_fallback(self, device: dm.DeviceInfo, reason: str) -> None:
        self.window_title.set_subtitle(f"Processing device: {device.label()}")
        self.toasts.add_toast(Adw.Toast(title=f"{reason} — continuing on the CPU", timeout=6))
        self._upscaler_key = None  # rebuild with the preferred device next time

    def start(self, rows: list[QueueRow] | None = None) -> None:
        if self.running:
            return
        if self.app.device_report is None:
            self.toasts.add_toast(Adw.Toast(title="Still detecting the processing device…"))
            self.app.when_devices_ready(lambda _r: self.start(rows))
            return
        settings = self.app.settings
        options = settings.processing_options()
        try:
            options.validate()
            self.app.model_manager.resolve(options.model, options.scale)
        except ModelNotInstalledError as err:
            self._offer_model_download(err)
            return
        except (ValueError, UpscalerError) as err:
            show_error(self, err if isinstance(err, UpscalerError) else UpscalerError(str(err)))
            return

        candidates = rows if rows is not None else self.rows
        todo = [
            r
            for r in candidates
            if r.ready and r.item.status not in (ItemStatus.DONE, ItemStatus.SKIPPED)
        ]
        if not todo:
            self.toasts.add_toast(
                Adw.Toast(
                    title="Nothing to upscale — add some images first"
                    if not self.rows
                    else "All images are already done"
                )
            )
            return

        try:
            upscaler = self._get_upscaler()
        except Exception as exc:
            log.exception("Could not create upscaler")
            show_error(
                self,
                UpscalerError(
                    f"The AI engine could not start: {exc}",
                    ["Switch the processing device to CPU in Settings"],
                ),
            )
            return
        workers = settings.concurrent_jobs or dm.recommended_concurrency(upscaler.device)
        self.processor = BatchProcessor(
            upscaler, options, lambda ev: idle(self._on_batch_event, ev), workers=workers
        )
        self.window_title.set_subtitle(f"Processing device: {upscaler.device.label()}")
        self._inhibit_cookie = self.app.inhibit(
            self, Gtk.ApplicationInhibitFlags.SUSPEND, "Upscaling images"
        )
        self.processor.start([r.item for r in todo], others=[r.item for r in self.rows])
        self._update_state()

    def _offer_model_download(self, err: ModelNotInstalledError) -> None:
        dialog = Adw.AlertDialog(heading=err.title, body=err.reason)
        dialog.add_response("cancel", "Cancel")
        dialog.add_response("models", "Open AI Models")
        dialog.set_response_appearance("models", Adw.ResponseAppearance.SUGGESTED)
        dialog.set_default_response("models")
        dialog.connect(
            "response",
            lambda _d, resp: resp == "models" and self.app.show_preferences(page="models"),
        )
        dialog.present(self)

    def toggle_pause(self) -> None:
        if not self.processor:
            return
        if self.processor.paused:
            self.processor.resume()
        else:
            self.processor.pause()
        self._update_state()

    def cancel(self) -> None:
        if self.processor:
            self.processor.cancel()
            self.current_label.set_label("Cancelling…")
        self._update_state()

    def retry_failed(self) -> None:
        rows = [
            r
            for r in self.rows
            if r.ready and r.item.status in (ItemStatus.FAILED, ItemStatus.CANCELLED)
        ]
        for row in rows:
            row.item.reset()
            row.refresh()
        self.start(rows)

    def retry_row(self, row: QueueRow) -> None:
        if self.running:
            return
        row.item.reset()
        row.refresh()
        self.start([row])

    def _row_for(self, item: QueueItem | None) -> QueueRow | None:
        if item is None:
            return None
        for row in self.rows:
            if row.item is item:
                return row
        return None

    def _on_batch_event(self, ev: BatchEvent) -> None:
        row = self._row_for(ev.item)
        if row is not None:
            row.refresh()
        self.overall_bar.set_fraction(ev.fraction)
        self.overall_bar.set_text(f"{ev.fraction * 100:.0f}%")
        self.overall_label.set_label(f"Overall: {ev.completed} / {ev.total}")
        if ev.kind in (EventKind.ITEM_STARTED, EventKind.ITEM_PROGRESS) and ev.item:
            paused = " (paused)" if self.processor and self.processor.paused else ""
            self.current_label.set_label(f"Current: {ev.item.path.name}{paused}")
        elif ev.kind is EventKind.PAUSED:
            self.current_label.set_label(
                "Paused — the current image will continue where it stopped"
            )
        elif ev.kind is EventKind.BATCH_FINISHED:
            self._on_batch_finished()
        self._update_state()

    def _on_batch_finished(self) -> None:
        if self._inhibit_cookie:
            self.app.uninhibit(self._inhibit_cookie)
            self._inhibit_cookie = 0
        summary = self.processor.summary if self.processor else None
        if summary is None:
            return
        parts = [f"{summary.done} upscaled"]
        if summary.skipped:
            parts.append(f"{summary.skipped} skipped")
        if summary.failed:
            parts.append(f"{summary.failed} failed")
        if summary.cancelled:
            parts.append(f"{summary.cancelled} cancelled")
        text = ", ".join(parts)
        self.current_label.set_label(f"Finished in {summary.seconds:.0f} s — {text}")
        toast = Adw.Toast(title=text.capitalize(), timeout=6)
        done_rows = [
            r
            for r in self.rows
            if r.item.result and not r.item.result.skipped and r.item.status is ItemStatus.DONE
        ]
        if done_rows:
            toast.set_button_label("Show Folder")
            toast.connect("button-clicked", lambda _t: done_rows[-1].show_in_folder())
        self.toasts.add_toast(toast)
        if not self.is_active():
            notification = Gio.Notification.new("Upscaling finished")
            notification.set_body(text.capitalize())
            self.app.send_notification("batch-finished", notification)

    def _update_state(self) -> None:
        count = len(self.rows)
        self.stack.set_visible_child_name("queue" if count else "empty")
        running = self.running
        done = sum(1 for r in self.rows if r.item.status in (ItemStatus.DONE, ItemStatus.SKIPPED))
        failed = sum(
            1
            for r in self.rows
            if r.ready and r.item.status in (ItemStatus.FAILED, ItemStatus.CANCELLED)
        )
        pending = sum(
            1
            for r in self.rows
            if r.ready and r.item.status not in (ItemStatus.DONE, ItemStatus.SKIPPED)
        )
        self.queue_title.set_label(
            f"{count} image{'s' if count != 1 else ''}" + (f" · {done} done" if done else "")
        )
        self.start_btn.set_visible(not running)
        self.start_btn.set_sensitive(pending > 0)
        self.start_btn.set_label(
            f"Start Upscaling ({pending})" if pending and count > 1 else "Start Upscaling"
        )
        self.pause_btn.set_visible(running)
        self.pause_btn.set_label("Resume" if self.processor and self.processor.paused else "Pause")
        self.cancel_btn.set_visible(running)
        self.cancel_btn.set_sensitive(running and not self.processor.control.cancelled)
        self.retry_btn.set_visible(not running and failed > 0)
        self.clear_btn.set_sensitive(count > 0 and not running)
        self.options_box.set_sensitive(not running)
        self.lighting.set_sensitive(not running)
        from pixelift.ui.preview import PreviewWindow

        for window in self.app.get_windows():
            if isinstance(window, PreviewWindow):
                window.set_lighting_editable(not running)
        self.progress_revealer.set_reveal_child(self.processor is not None)

    def _on_close_request(self, _window: Gtk.Window) -> bool:
        if not self.running:
            self.thumbnails.shutdown()
            return False
        dialog = Adw.AlertDialog(
            heading="Stop upscaling?",
            body="Images are still being processed. Unfinished images will not be saved.",
        )
        dialog.add_response("keep", "Keep Working")
        dialog.add_response("quit", "Stop and Quit")
        dialog.set_response_appearance("quit", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.set_default_response("keep")

        def on_response(_d: Adw.AlertDialog, response: str) -> None:
            if response == "quit" and self.processor:
                self.processor.cancel()
                self.processor.wait(5)
                self.thumbnails.shutdown()
                self.destroy()

        dialog.connect("response", on_response)
        dialog.present(self)
        return True
