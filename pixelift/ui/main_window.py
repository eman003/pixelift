"""Main window: a photo workspace with a contextual control dock.

Three zones, the image first:

* the header: Upscale / Restore Photos mode, adding photos, the menu;
* the workspace: the selected photo on a neutral stage (the lighting and camera
  look previewed live on it, the result once processed), with the photo list
  in a collapsible sidebar;
* the dock: one tab per task (Enhance, Light, Look, Export), each showing its
  current setting and opening its controls on demand, and the primary action.

Both modes share the queue, the worker and the output options.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from gi.repository import Adw, Gdk, Gio, GLib, GObject, Gtk, Pango
from PIL import Image

from pixelift import APP_ID, APP_NAME
from pixelift.core import camera_looks as cl
from pixelift.core import device_manager as dm
from pixelift.core import lighting
from pixelift.core.batch_processor import (
    BatchEvent,
    BatchProcessor,
    EventKind,
    ItemStatus,
    QueueItem,
)
from pixelift.core.errors import ModelNotInstalledError, UpscalerError
from pixelift.core.image_processor import restoration_tag
from pixelift.core.restoration import settings as rs
from pixelift.core.upscaler import TorchUpscaler
from pixelift.models import all_families
from pixelift.ui.async_utils import idle, run_in_thread
from pixelift.ui.image_queue import QueueRow, ThumbnailLoader
from pixelift.ui.startup import STATUS_DEVICE, STATUS_READY, StartupScreen
from pixelift.ui.widgets.camera_looks import GRAIN_LABELS, CameraLookControls
from pixelift.ui.widgets.dialogs import show_error
from pixelift.ui.widgets.lighting import LightingControls
from pixelift.ui.widgets.restoration import (
    PRIVACY_TEXT,
    BlackAndWhiteBanner,
    RestorationDialog,
    RestorationPanel,
)
from pixelift.ui.widgets.textures import texture_from_pil
from pixelift.utils.image_utils import SUPPORTED_EXTENSIONS, collect_images, load_preview

if TYPE_CHECKING:
    from pixelift.ui.application import UpscalerApplication

log = logging.getLogger(__name__)

SCALES = (2, 4)
LOOK_SAMPLE_SIZE = 400  # the camera look cards' sample photo (px, longest side)
STARTUP_FADE_MS = 400  # the startup screen dissolving into the workspace
MAX_DEVICE_WAIT_MS = 1500  # after which the workspace shows; detection finishes later
STAGE_SIZE = 1600  # the workspace preview (px, longest side); Compare shows full detail
FORMATS = (("png", "PNG"), ("jpeg", "JPEG"), ("webp", "WebP"))
TABS = (
    ("enhance", "Enhance", "Scale, AI model and restoration"),
    ("light", "Light", "Lighting profile and intensity"),
    ("look", "Look", "Camera look, intensity and grain"),
    ("export", "Export", "File format and output folder"),
)


class MainWindow(Adw.ApplicationWindow):
    def __init__(self, app: UpscalerApplication) -> None:
        super().__init__(application=app, title=APP_NAME)
        self.app = app
        self.set_default_size(1180, 800)
        self.set_size_request(380, 520)
        self.rows: list[QueueRow] = []
        self.thumbnails = ThumbnailLoader()
        self.processor: BatchProcessor | None = None
        self._upscaler: TorchUpscaler | None = None
        self._upscaler_key: tuple[object, ...] | None = None
        self._inhibit_cookie = 0
        self._syncing = False
        self._bw_answered = False  # asked "Restore in B&W or colorize?" this session
        self._look_sample_path: Path | None = None  # shown on the camera look cards
        # The stage: the selected photo (or its result), with the look previewed live.
        self._stage_row: QueueRow | None = None
        self._stage_source: Path | None = None  # the file the stage shows
        self._stage_base: Image.Image | None = None
        self._stage_is_output = False
        self._stage_base_texture: Gdk.Texture | None = None
        self._stage_look_texture: Gdk.Texture | None = None
        self._stage_shown_look: tuple | None = None  # what the look texture renders
        self._stage_busy = False
        self._stage_generation = 0
        # Startup: the startup screen shows at once; the workspace is built
        # right after its first frame, then the screen dissolves into it.
        self.ready = False
        self._built = False
        self._building: Iterator[None] | None = None
        self._devices_known = False
        self._device_wait_over = False
        self._pending_paths: list[Path] = []
        self._ready_callbacks: list[Callable[[], None]] = []

        self.startup = StartupScreen(on_retry=self._retry_startup, on_exit=self._exit_startup)
        self.root = Gtk.Stack(
            transition_type=Gtk.StackTransitionType.CROSSFADE,
            transition_duration=STARTUP_FADE_MS,
            hhomogeneous=False,  # the hidden workspace is not measured meanwhile
            vhomogeneous=False,
        )
        self.root.add_named(self.startup, "startup")
        self.set_content(self.root)
        self.connect("close-request", self._on_close_request)
        self.startup.connect("map", lambda _w: self._after_next_frame(self._build_ui))
        app.when_devices_ready(self._on_startup_devices)
        self.startup.when_settled(self._maybe_finish_startup)

    # --- startup -----------------------------------------------------------
    def _after_next_frame(self, callback: Callable[[], None]) -> None:
        """Run ``callback`` once the window has painted its next frame."""
        clock = self.get_frame_clock()
        if clock is None:
            GLib.idle_add(lambda: callback() and False)
            return
        handler = 0

        def painted(_clock: Gdk.FrameClock) -> None:
            clock.disconnect(handler)
            GLib.idle_add(lambda: callback() and False)

        handler = clock.connect("after-paint", painted)
        self.startup.queue_draw()

    def _build_ui(self) -> None:
        if self._built or self._building is not None:
            return
        self._building = self._workspace_steps()
        self._build_step()

    def _build_step(self) -> None:
        steps = self._building
        if steps is None:
            return
        try:
            next(steps)
        except StopIteration:
            self._building = None
            self._on_built()
        except Exception:  # shown on the startup screen, logged in full
            log.exception("Could not build the main window")
            self._building = None
            self.startup.show_error(traceback.format_exc())
        else:
            self._after_next_frame(self._build_step)

    def _on_built(self) -> None:
        self._built = True
        if not self._devices_known:
            self.startup.set_status(STATUS_DEVICE)
            # Device detection runs in the background; never hold the window for it long.
            GLib.timeout_add(MAX_DEVICE_WAIT_MS, self._device_wait_elapsed)
        self._maybe_finish_startup()

    def _on_startup_devices(self, _report: dm.DeviceReport) -> None:
        self._devices_known = True
        self._maybe_finish_startup()

    def _device_wait_elapsed(self) -> bool:
        self._device_wait_over = True
        self._maybe_finish_startup()
        return GLib.SOURCE_REMOVE

    def _maybe_finish_startup(self) -> None:
        """Show the workspace once it is built, the device is known (or the wait
        is over) and the logo has settled — whichever comes last."""
        if self.ready or not self._built or not self.startup.logo_settled:
            return
        if not (self._devices_known or self._device_wait_over):
            return
        self.ready = True
        self.startup.set_status(STATUS_READY)
        self.root.set_visible_child_name("app")
        self.set_focus(None)

        def finished() -> bool:
            self.startup.stop()
            self.root.remove(self.startup)
            return GLib.SOURCE_REMOVE

        GLib.timeout_add(STARTUP_FADE_MS + 100, finished)
        callbacks, self._ready_callbacks = self._ready_callbacks, []
        for callback in callbacks:
            callback()
        paths, self._pending_paths = self._pending_paths, []
        if paths:
            self.add_paths(paths)

    def when_ready(self, callback: Callable[[], None]) -> None:
        """Run ``callback`` once the workspace is shown (e.g. the setup assistant)."""
        if self.ready:
            callback()
        else:
            self._ready_callbacks.append(callback)

    def _retry_startup(self) -> None:
        self.startup.show_starting()
        self._after_next_frame(self._build_ui)

    def _exit_startup(self) -> None:
        self.app.quit()

    def _workspace_steps(self) -> Iterator[None]:
        """Build the workspace in steps; a frame is drawn between them, so the
        startup animation keeps moving (the look gallery alone takes ~0.1 s)."""
        app = self.app
        toolbar = Adw.ToolbarView()
        toolbar.add_top_bar(self._build_header())
        self.toasts = Adw.ToastOverlay()
        self.stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE, vexpand=True)
        self.stack.add_named(self._build_empty_page(), "empty")
        self.stack.add_named(self._build_workspace(), "queue")
        self.toasts.set_child(self.stack)
        self.toasts.add_css_class("drop-zone")
        toolbar.set_content(self.toasts)
        yield
        self.lighting = LightingControls(app)
        yield
        self.camera_look = CameraLookControls(app)
        yield
        toolbar.add_bottom_bar(self._build_controls())
        toolbar.set_bottom_bar_style(Adw.ToolbarStyle.RAISED)
        if self.root.get_child_by_name("app") is not None:
            self.root.remove(self.root.get_child_by_name("app"))  # a failed attempt
        self.root.add_named(toolbar, "app")

        drop = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        drop.connect("drop", self._on_drop)
        self.toasts.add_controller(drop)

        # The sidebar floats over the stage on smaller windows; on narrow ones
        # the controls stack vertically too. (Only the last matching applies.)
        medium = Adw.Breakpoint.new(Adw.BreakpointCondition.parse("max-width: 860sp"))
        narrow = Adw.Breakpoint.new(Adw.BreakpointCondition.parse("max-width: 640sp"))
        for breakpoint in (medium, narrow):
            breakpoint.add_setter(self.split_view, "collapsed", True)
            breakpoint.add_setter(self.split_view, "show-sidebar", False)
        for box in (
            self.enhance_box,
            self.export_box,
            self.restore_panel,
            self.lighting,
            self.camera_look,
            self.dock_bar,
        ):
            narrow.add_setter(box, "orientation", Gtk.Orientation.VERTICAL)
        narrow.add_setter(self.tabs_box, "halign", Gtk.Align.FILL)
        narrow.add_setter(self.device_btn, "visible", False)
        narrow.add_setter(self.header_add_btn, "visible", False)
        narrow.add_setter(self.mode_buttons["restore"], "label", "Restore")
        narrow.add_setter(self.bw_banner.get_child(), "orientation", Gtk.Orientation.VERTICAL)
        narrow.add_setter(self.buttons_box, "halign", Gtk.Align.FILL)
        self.add_breakpoint(medium)
        self.add_breakpoint(narrow)

        app.on_settings_changed(self._sync_from_settings)
        app.on_lighting_changed(self._reset_stale)
        app.on_lighting_changed(self._on_look_changed)
        app.downloads.subscribe(lambda *_: self._refresh_model_list())
        app.when_devices_ready(self._on_devices)
        self._sync_from_settings()
        self._on_look_changed()
        self._update_state()

    # --- construction ------------------------------------------------------
    def _build_header(self) -> Adw.HeaderBar:
        header = Adw.HeaderBar()

        self.sidebar_btn = Gtk.ToggleButton(
            icon_name="sidebar-show-symbolic", tooltip_text="Photos (F9)", active=True
        )
        header.pack_start(self.sidebar_btn)
        self.header_add_btn = Gtk.Button(
            icon_name="list-add-symbolic", tooltip_text="Add Images (Ctrl+O)"
        )
        self.header_add_btn.connect("clicked", lambda _b: self.open_file_dialog())
        header.pack_start(self.header_add_btn)

        self.mode_box = Gtk.Box(css_classes=["mode-switcher"], valign=Gtk.Align.CENTER)
        self.mode_buttons: dict[str, Gtk.ToggleButton] = {}
        first: Gtk.ToggleButton | None = None
        for mode, label, tip in (
            ("upscale", "Upscale", "AI upscaling"),
            ("restore", "Restore Photos", "AI photo restoration for old photographs"),
        ):
            button = Gtk.ToggleButton(label=label, tooltip_text=tip, group=first)
            first = first or button
            button.connect("toggled", self._on_mode_toggled, mode)
            self.mode_box.append(button)
            self.mode_buttons[mode] = button
        header.set_title_widget(self.mode_box)

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

        # The processing device, quietly: a chip that opens the settings.
        self.device_label = Gtk.Label(
            label="Detecting device…", ellipsize=Pango.EllipsizeMode.END, max_width_chars=22
        )
        chip = Gtk.Box(spacing=6)
        chip.append(Gtk.Image(icon_name="computer-symbolic"))
        chip.append(self.device_label)
        self.device_btn = Gtk.Button(child=chip, valign=Gtk.Align.CENTER)
        self.device_btn.add_css_class("device-chip")
        self.device_btn.add_css_class("flat")
        self.device_btn.set_action_name("app.preferences")
        header.pack_end(self.device_btn)

        open_action = Gio.SimpleAction.new("open", None)
        open_action.connect("activate", lambda *_: self.open_file_dialog())
        self.add_action(open_action)
        self.app.set_accels_for_action("win.open", ["<Control>o"])
        sidebar_action = Gio.SimpleAction.new("toggle-sidebar", None)
        sidebar_action.connect(
            "activate", lambda *_: self.sidebar_btn.set_active(not self.sidebar_btn.get_active())
        )
        self.add_action(sidebar_action)
        self.app.set_accels_for_action("win.toggle-sidebar", ["F9"])
        return header

    def _build_empty_page(self) -> Gtk.Widget:
        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        card.add_css_class("drop-card")
        icon = Gtk.Image(icon_name=APP_ID, pixel_size=96, margin_bottom=12)
        card.append(icon)
        self.empty_title = Gtk.Label(wrap=True, justify=Gtk.Justification.CENTER)
        self.empty_title.add_css_class("title-1")
        card.append(self.empty_title)
        self.empty_description = Gtk.Label(wrap=True, justify=Gtk.Justification.CENTER)
        self.empty_description.set_max_width_chars(46)
        self.empty_description.add_css_class("dim-label")
        card.append(self.empty_description)
        select = Gtk.Button(label="Select Images", halign=Gtk.Align.CENTER, margin_top=18)
        select.add_css_class("pill")
        select.add_css_class("suggested-action")
        select.connect("clicked", lambda _b: self.open_file_dialog())
        card.append(select)
        hint = Gtk.Label(label="or drop files and folders anywhere · Ctrl+O")
        hint.add_css_class("caption")
        hint.add_css_class("dim-label")
        card.append(hint)
        self.privacy_label = Gtk.Label(wrap=True, justify=Gtk.Justification.CENTER, margin_top=18)
        self.privacy_label.set_max_width_chars(60)
        self.privacy_label.add_css_class("dim-label")
        self.privacy_label.add_css_class("privacy-note")
        card.append(self.privacy_label)
        return Adw.Clamp(
            maximum_size=600,
            child=card,
            valign=Gtk.Align.CENTER,
            margin_top=24,
            margin_bottom=24,
            margin_start=24,
            margin_end=24,
        )

    def _build_workspace(self) -> Gtk.Widget:
        self.split_view = Adw.OverlaySplitView(
            sidebar=self._build_sidebar(),
            content=self._build_stage(),
            min_sidebar_width=280,
            max_sidebar_width=360,
            sidebar_width_fraction=0.27,
        )
        self.split_view.bind_property(
            "show-sidebar",
            self.sidebar_btn,
            "active",
            GObject.BindingFlags.SYNC_CREATE | GObject.BindingFlags.BIDIRECTIONAL,
        )
        return self.split_view

    def _build_sidebar(self) -> Gtk.Widget:
        self.listbox = Gtk.ListBox(
            selection_mode=Gtk.SelectionMode.SINGLE, activate_on_single_click=False
        )
        self.listbox.add_css_class("navigation-sidebar")
        self.listbox.connect("row-activated", lambda _l, row: self.open_preview(row))
        self.listbox.connect("row-selected", self._on_row_selected)

        header = Gtk.Box(spacing=6)
        header.add_css_class("sidebar-header")
        titles = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, hexpand=True)
        self.queue_title = Gtk.Label(label="Photos", xalign=0)
        self.queue_title.add_css_class("heading")
        self.queue_subtitle = Gtk.Label(xalign=0)
        self.queue_subtitle.add_css_class("caption")
        self.queue_subtitle.add_css_class("dim-label")
        titles.append(self.queue_title)
        titles.append(self.queue_subtitle)
        header.append(titles)
        self.clear_btn = Gtk.Button(
            icon_name="edit-clear-all-symbolic",
            tooltip_text="Clear Finished Images",
            valign=Gtk.Align.CENTER,
        )
        self.clear_btn.add_css_class("flat")
        self.clear_btn.connect("clicked", lambda _b: self.clear_finished())
        header.append(self.clear_btn)
        add = Gtk.Button(
            icon_name="list-add-symbolic", tooltip_text="Add Images", valign=Gtk.Align.CENTER
        )
        add.add_css_class("flat")
        add.connect("clicked", lambda _b: self.open_file_dialog())
        header.append(add)

        sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        sidebar.add_css_class("photo-sidebar")
        sidebar.append(header)
        sidebar.append(
            Gtk.ScrolledWindow(
                child=self.listbox, hscrollbar_policy=Gtk.PolicyType.NEVER, vexpand=True
            )
        )
        return sidebar

    def _build_stage(self) -> Gtk.Widget:
        self.stage_picture = Gtk.Picture(
            content_fit=Gtk.ContentFit.CONTAIN, can_shrink=True, hexpand=True, vexpand=True
        )
        self.stage_picture.add_css_class("stage-picture")
        overlay = Gtk.Overlay(child=self.stage_picture)

        self.stage_spinner = Gtk.Spinner(
            halign=Gtk.Align.CENTER, valign=Gtk.Align.CENTER, width_request=32, height_request=32
        )
        overlay.add_overlay(self.stage_spinner)

        self.stage_badge = Gtk.Label(halign=Gtk.Align.START, valign=Gtk.Align.START)
        self.stage_badge.add_css_class("stage-chip")
        self.stage_badge.add_css_class("stage-badge")
        overlay.add_overlay(self.stage_badge)

        # Bottom row: what is shown (left) and what can be done with it (right).
        bottom = Gtk.Box(spacing=8, valign=Gtk.Align.END)
        bottom.add_css_class("stage-bottom")
        self.stage_info = Gtk.Label(
            halign=Gtk.Align.START, hexpand=True, ellipsize=Pango.EllipsizeMode.MIDDLE
        )
        self.stage_info.add_css_class("stage-chip")
        self.stage_info.add_css_class("stage-info")
        self.stage_info.add_css_class("numeric")
        bottom.append(self.stage_info)

        tools = Gtk.Box(spacing=2, valign=Gtk.Align.CENTER)
        tools.add_css_class("stage-chip")
        tools.add_css_class("stage-tools")
        self.original_btn = Gtk.ToggleButton(
            label="Original", tooltip_text="Show the original without the look"
        )
        self.original_btn.connect("toggled", lambda _b: self._show_stage_texture())
        self.stage_compare_btn = Gtk.Button(
            icon_name="view-dual-symbolic", tooltip_text="Compare and Zoom"
        )
        self.stage_compare_btn.connect(
            "clicked", lambda _b: self._stage_row and self.open_preview(self._stage_row)
        )
        self.stage_folder_btn = Gtk.Button(
            icon_name="folder-open-symbolic", tooltip_text="Show in Folder"
        )
        self.stage_folder_btn.connect(
            "clicked", lambda _b: self._stage_row and self._stage_row.show_in_folder()
        )
        for button in (self.original_btn, self.stage_compare_btn, self.stage_folder_btn):
            button.add_css_class("flat")
            tools.append(button)
        self.stage_tools = tools
        bottom.append(tools)
        overlay.add_overlay(bottom)
        overlay.add_css_class("stage")

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.bw_banner = BlackAndWhiteBanner(self._on_bw_choice)
        self.bw_banner.set_margin_top(12)
        self.bw_banner.set_margin_start(12)
        self.bw_banner.set_margin_end(12)
        content.append(self.bw_banner)
        content.append(overlay)
        return content

    def _labeled(self, label: str, widget: Gtk.Widget, expand: bool = True) -> Gtk.Box:
        box = Gtk.Box(spacing=10)
        title = Gtk.Label(label=label, xalign=0, width_chars=6)
        title.add_css_class("dim-label")
        box.append(title)
        widget.set_hexpand(expand)
        box.append(widget)
        return box

    def _segmented(
        self, labels: list[str], on_selected: Callable[[int], None]
    ) -> list[Gtk.ToggleButton]:
        """Grouped toggle buttons; ``on_selected(index)`` when the user picks one."""

        def toggled(button: Gtk.ToggleButton, index: int) -> None:
            if button.get_active() and not self._syncing:
                on_selected(index)

        buttons: list[Gtk.ToggleButton] = []
        group: Gtk.ToggleButton | None = None
        for index, label in enumerate(labels):
            button = Gtk.ToggleButton(label=label, group=group)
            group = group or button
            button.connect("toggled", toggled, index)
            buttons.append(button)
        return buttons

    def _segment_box(self, buttons: list[Gtk.ToggleButton]) -> Gtk.Box:
        box = Gtk.Box(css_classes=["linked", "segmented"], halign=Gtk.Align.START)
        for button in buttons:
            box.append(button)
        return box

    def _build_controls(self) -> Gtk.Widget:
        dock = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        dock.add_css_class("dock")

        # Enhance: what happens to the photo (restoration, scale, AI model).
        self.restore_panel = RestorationPanel(self.app, self.show_restoration_options)
        self.scale_buttons = self._segmented([f"{s}×" for s in SCALES], self._on_scale_selected)
        self.model_list = Gtk.StringList()
        self.model_dd = Gtk.DropDown(model=self.model_list)
        self.model_dd.connect("notify::selected", self._on_model_changed)
        self.enhance_box = Gtk.Box(spacing=24)
        self.scale_box = self._labeled("Scale", self._segment_box(self.scale_buttons), False)
        self.enhance_box.append(self.scale_box)
        self.enhance_box.append(self._labeled("Model", self.model_dd))
        enhance = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        enhance.append(self.restore_panel)
        enhance.append(self.enhance_box)

        # Export: file format and destination.
        self.format_buttons = self._segmented(
            [name for _, name in FORMATS], self._on_format_selected
        )
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
        self.export_box = Gtk.Box(spacing=24)
        self.export_box.append(
            self._labeled("Format", self._segment_box(self.format_buttons), False)
        )
        self.export_box.append(self._labeled("Output", output_row))

        self.panel_stack = Gtk.Stack(
            transition_type=Gtk.StackTransitionType.CROSSFADE,
            vhomogeneous=False,
            interpolate_size=True,
        )
        for key, widget in (
            ("enhance", enhance),
            ("light", self.lighting),
            ("look", self.camera_look),
            ("export", self.export_box),
        ):
            self.panel_stack.add_named(Adw.Clamp(maximum_size=1000, child=widget), key)
        self.panel_stack.add_css_class("dock-panel")
        # Tall panels (narrow windows) scroll rather than squeezing the photo.
        panel_scroll = Gtk.ScrolledWindow(
            child=self.panel_stack,
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            propagate_natural_height=True,
            max_content_height=320,
        )
        self.panel_revealer = Gtk.Revealer(
            transition_type=Gtk.RevealerTransitionType.SLIDE_UP, child=panel_scroll
        )
        dock.append(self.panel_revealer)

        # Progress (only while a batch is running / just finished)
        self.progress_revealer = Gtk.Revealer(transition_type=Gtk.RevealerTransitionType.SLIDE_UP)
        progress_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        progress_box.add_css_class("dock-progress")
        self.overall_bar = Gtk.ProgressBar()
        line = Gtk.Box(spacing=12)
        self.current_label = Gtk.Label(xalign=0, hexpand=True, ellipsize=Pango.EllipsizeMode.MIDDLE)
        self.current_label.add_css_class("caption")
        self.overall_label = Gtk.Label(xalign=1)
        self.overall_label.add_css_class("caption")
        self.overall_label.add_css_class("numeric")
        line.append(self.current_label)
        line.append(self.overall_label)
        progress_box.append(line)
        progress_box.append(self.overall_bar)
        self.progress_revealer.set_child(progress_box)
        dock.append(self.progress_revealer)

        # The bar: task tabs (each showing its current setting) and the action.
        self.dock_bar = Gtk.Box(spacing=16)
        self.dock_bar.add_css_class("dock-bar")
        self.tabs_box = Gtk.Box(spacing=4, homogeneous=True, hexpand=True, halign=Gtk.Align.START)
        self.tab_buttons: dict[str, Gtk.ToggleButton] = {}
        self.tab_values: dict[str, Gtk.Label] = {}
        for key, title, tip in TABS:
            content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=1)
            heading = Gtk.Label(label=title.upper(), xalign=0)
            heading.add_css_class("dock-tab-title")
            value = Gtk.Label(xalign=0, ellipsize=Pango.EllipsizeMode.END, max_width_chars=22)
            value.add_css_class("dock-tab-value")
            content.append(heading)
            content.append(value)
            button = Gtk.ToggleButton(child=content, tooltip_text=tip)
            button.add_css_class("flat")
            button.add_css_class("dock-tab")
            button.connect("toggled", self._on_tab_toggled, key)
            self.tabs_box.append(button)
            self.tab_buttons[key] = button
            self.tab_values[key] = value
        self.dock_bar.append(self.tabs_box)

        self.buttons_box = Gtk.Box(spacing=8, halign=Gtk.Align.END, valign=Gtk.Align.CENTER)
        self.start_btn = Gtk.Button(label="Upscale", hexpand=True)
        self.start_btn.add_css_class("pill")
        self.start_btn.add_css_class("suggested-action")
        self.start_btn.add_css_class("primary-action")
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
        for btn in (self.retry_btn, self.pause_btn, self.cancel_btn, self.start_btn):
            self.buttons_box.append(btn)
        self.dock_bar.append(self.buttons_box)
        dock.append(self.dock_bar)

        escape = Gtk.ShortcutController(scope=Gtk.ShortcutScope.MANAGED)
        escape.add_shortcut(
            Gtk.Shortcut(
                trigger=Gtk.ShortcutTrigger.parse_string("Escape"),
                action=Gtk.CallbackAction.new(lambda *_: self.show_panel(None) or True),
            )
        )
        dock.add_controller(escape)
        return dock

    # --- the dock ----------------------------------------------------------
    def show_panel(self, key: str | None) -> None:
        """Open one task's controls in the dock (None: collapse them)."""
        self._syncing = True
        try:
            for name, button in self.tab_buttons.items():
                button.set_active(name == key)
        finally:
            self._syncing = False
        if key is not None:
            self.panel_stack.set_visible_child_name(key)
        self.panel_revealer.set_reveal_child(key is not None)

    def _on_tab_toggled(self, button: Gtk.ToggleButton, key: str) -> None:
        if self._syncing:
            return
        if button.get_active():
            self.show_panel(key)
        elif self.panel_stack.get_visible_child_name() == key:
            self.show_panel(None)  # clicking the open tab again folds it away

    def _update_tab_values(self) -> None:
        """Each tab shows its current setting: the whole recipe at a glance."""
        settings = self.app.settings
        if self.restoring:
            _colorize, upscale = rs.preset_flags(settings.restore_preset)
            enhance = (
                f"{rs.PRESET_LABELS[settings.restore_preset]} · "
                f"{rs.LEVEL_LABELS[settings.restore_level]}"
            )
            if upscale:
                enhance += f" · {settings.restore_scale}×"
        else:
            names = dict(self._families())
            enhance = f"{settings.scale}× · {names.get(settings.model, settings.model)}"
        light = settings.lighting()
        profiles = {p.id: p.name for p in lighting.all_profiles()}
        light_text = profiles.get(light.profile, "Original")
        if light.profile not in (lighting.ORIGINAL, lighting.CUSTOM):
            light_text += f" · {settings.lighting_intensity}%"
        look = settings.camera_look_settings()
        look_text = look.name()
        if look.look not in (cl.ORIGINAL, cl.CUSTOM):
            look_text += f" · {settings.camera_look_intensity}%"
        if look.look != cl.ORIGINAL and settings.camera_look_grain != cl.GRAIN_AUTO:
            look_text += " · " + GRAIN_LABELS[settings.camera_look_grain]
        fmt = dict(FORMATS)[settings.output_format]
        values = {
            "enhance": (enhance, False),
            "light": (light_text, light.active),
            "look": (look_text, look.active),
            "export": (f"{fmt} · {self.output_label.get_label()}", False),
        }
        for key, (text, modified) in values.items():
            self.tab_values[key].set_label(text)
            self.tab_values[key].set_tooltip_text(text)
            if modified:
                self.tab_buttons[key].add_css_class("modified")
            else:
                self.tab_buttons[key].remove_css_class("modified")

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

    @property
    def restoring(self) -> bool:
        return self.app.settings.mode == "restore"

    def _sync_from_settings(self) -> None:
        settings = self.app.settings
        self._refresh_model_list()
        restoring = self.restoring
        scale = settings.restore_scale if restoring else settings.scale
        self._syncing = True
        try:
            self.mode_buttons[settings.mode].set_active(True)
            self.scale_buttons[SCALES.index(scale) if scale in SCALES else 1].set_active(True)
            fmt_ids = [f for f, _ in FORMATS]
            self.format_buttons[fmt_ids.index(settings.output_format)].set_active(True)
            if settings.output_dir:
                self.output_label.set_label(Path(settings.output_dir).name or settings.output_dir)
                self.output_btn.set_tooltip_text(settings.output_dir)
            else:
                folder = "restored" if restoring else "upscaled"
                self.output_label.set_label(f"Next to originals ({folder}/)")
                self.output_btn.set_tooltip_text(f"<original folder>/{folder}/")
            self.output_reset.set_visible(bool(settings.output_dir))
        finally:
            self._syncing = False
        self._sync_mode()
        output_scale = settings.processing_options().output_scale
        for row in self.rows:
            row.set_scale(output_scale, show_monochrome=restoring)
        if self.app.device_report:
            self._on_devices(self.app.device_report)
        # The tab values and the stage follow in _on_look_changed: settings
        # changes notify the lighting listeners too (after _reset_stale).
        self._update_state()

    def _sync_mode(self) -> None:
        """Show the controls of the current mode (Upscale or Restore Photos)."""
        restoring = self.restoring
        settings = self.app.settings
        _colorize, upscale = rs.preset_flags(settings.restore_preset)
        self.restore_panel.set_visible(restoring)
        self.scale_box.set_visible(not restoring or upscale)
        if restoring:
            self.empty_title.set_label("Bring old photographs back")
            self.empty_description.set_label(
                "Drop faded, scratched or damaged photos — black and white or colour."
            )
            self.privacy_label.set_label(PRIVACY_TEXT)
        else:
            self.empty_title.set_label("Drop photos to begin")
            self.empty_description.set_label(
                "Upscale with AI, then give them light and a camera look. "
                "PNG, JPEG, WebP, TIFF or BMP — single images or whole folders."
            )
            self.privacy_label.set_label("🔒 Your images stay on your computer.")
        self._update_bw_banner()

    def _on_mode_toggled(self, button: Gtk.ToggleButton, mode: str) -> None:
        if self._syncing or not button.get_active() or self.app.settings.mode == mode:
            return
        self.app.settings.mode = mode
        self.app.settings_changed()

    def show_restoration_options(self) -> None:
        RestorationDialog(self.app).present(self)

    def _expected_restoration(self, row: QueueRow) -> str | None:
        """The restoration tag a current result for ``row`` would have (None: upscaling)."""
        options = self.app.settings.processing_options()
        restoration = options.restoration
        if restoration is None:
            return None
        colorize = row.monochrome and restoration.colorize and restoration.colorize_strength > 0
        return restoration_tag(options, colorize)

    def _reset_stale(self) -> None:
        """Results made with other lighting, camera look or restoration settings
        (or in the other mode) are out of date: queue them again.

        Their files stay on disk (the settings are part of the file name), so
        switching back just finds and skips them.
        """
        if self.running:
            return
        tag = self.app.settings.lighting().tag()
        look = self.app.settings.camera_look_settings().tag()
        stale = [
            r
            for r in self.rows
            if r.item.status in (ItemStatus.DONE, ItemStatus.SKIPPED)
            and r.item.result is not None
            and (
                r.item.result.lighting != tag
                or r.item.result.look != look
                or r.item.result.restoration != self._expected_restoration(r)
            )
        ]
        for row in stale:
            row.item.reset()
            row.refresh()
        if stale:
            self._update_state()

    def _on_look_changed(self) -> None:
        self._update_tab_values()
        self._refresh_stage()

    def _on_scale_selected(self, index: int) -> None:
        if self.restoring:
            self.app.settings.restore_scale = SCALES[index]
        else:
            self.app.settings.scale = SCALES[index]
        self.app.settings_changed()

    def _on_format_selected(self, index: int) -> None:
        self.app.settings.output_format = FORMATS[index][0]
        self.app.settings_changed()

    def _on_model_changed(self, *_args: object) -> None:
        if self._syncing:
            return
        families = self._families()
        if self.model_dd.get_selected() < len(families):
            self.app.settings.model = families[self.model_dd.get_selected()][0]
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

    def _show_device(self, device: dm.DeviceInfo) -> None:
        label = device.label()
        self.device_label.set_label(label)
        self.device_btn.set_tooltip_text(f"Processing device: {label}\nChange it in Preferences")

    def _on_devices(self, _report: dm.DeviceReport) -> None:
        if self.running and self._upscaler is not None:
            device = self._upscaler.device
        else:
            device = self.app.selected_device()
        self._show_device(device)

    # --- the stage ---------------------------------------------------------
    def _on_row_selected(self, _listbox: Gtk.ListBox, row: Gtk.ListBoxRow | None) -> None:
        if row is None:
            return  # keep showing the last photo until another is chosen
        assert isinstance(row, QueueRow)
        self._stage_row = row
        self._refresh_stage()

    def _select_some_row(self) -> None:
        """Keep a photo on the stage: the first one ready, when none is."""
        if self._stage_row in self.rows:
            return
        self._stage_row = None
        row = next((r for r in self.rows if r.ready), None)
        if row is not None:
            self.listbox.select_row(row)
        else:
            self._refresh_stage()

    def _stage_target(self, row: QueueRow) -> tuple[Path, bool]:
        """The file the stage shows for ``row`` and whether it is the result."""
        result = row.item.result
        if (
            result is not None
            and row.item.status in (ItemStatus.DONE, ItemStatus.SKIPPED)
            and result.output.exists()
        ):
            return result.output, True
        return row.item.path, False

    def _refresh_stage(self) -> None:
        """Show the selected photo: its result once processed, otherwise the
        original with the lighting and camera look previewed live."""
        row = self._stage_row
        if row is None or not row.ready:
            self._stage_source = None
            self._stage_base = self._stage_base_texture = self._stage_look_texture = None
            self._stage_generation += 1
            self.stage_picture.set_paintable(None)
            self.stage_spinner.set_spinning(row is not None and row.load_error is None)
            for widget in (self.stage_badge, self.stage_info, self.stage_tools):
                widget.set_visible(False)
            return
        target, is_output = self._stage_target(row)
        self._update_stage_overlays()
        if target != self._stage_source:
            self._stage_source = target
            self._stage_base = self._stage_base_texture = self._stage_look_texture = None
            self._stage_shown_look = None
            self._stage_generation += 1
            generation = self._stage_generation
            self.stage_picture.set_paintable(None)
            self.stage_spinner.set_spinning(True)

            def done(result: tuple) -> None:
                if generation != self._stage_generation:
                    return
                image, _size = result
                self.stage_spinner.set_spinning(False)
                self._stage_base = image
                self._stage_is_output = is_output
                self._stage_base_texture = texture_from_pil(image)
                self._stage_look_texture = None
                self._stage_shown_look = None
                self._render_stage_look()
                self._update_stage_overlays()

            def failed(exc: BaseException) -> None:
                if generation == self._stage_generation:
                    self.stage_spinner.set_spinning(False)
                    self.stage_badge.set_label("Preview Unavailable")
                    self.stage_badge.remove_css_class("finished")
                    log.warning("Stage preview failed: %s", exc)

            run_in_thread(
                load_preview, target, STAGE_SIZE, on_done=done, on_error=failed, name="stage"
            )
            return
        self._render_stage_look()

    def _wanted_stage_look(self) -> tuple | None:
        """(lighting, look, output size) to preview on the stage; None: nothing."""
        row = self._stage_row
        if self._stage_is_output or row is None or row.info is None:
            return None
        settings = self.app.settings
        adjustments = settings.lighting().adjustments()
        recipe = settings.camera_look_settings().recipe()
        if adjustments.is_neutral and recipe.is_neutral:
            return None
        scale = settings.processing_options().output_scale
        return adjustments, recipe, (row.info.width * scale, row.info.height * scale)

    def _render_stage_look(self) -> None:
        base = self._stage_base
        if base is None:
            return
        wanted = self._wanted_stage_look()
        if wanted is None or wanted == self._stage_shown_look:
            if wanted is None:
                self._stage_look_texture = None
                self._stage_shown_look = None
            self._show_stage_texture()
            return
        if self._stage_busy:
            # Picked up when the running render finishes; meanwhile a newly
            # loaded photo shows without the look rather than not at all.
            self._show_stage_texture()
            return
        self._stage_busy = True
        adjustments, recipe, output = wanted

        def done(img: Image.Image) -> None:
            self._stage_busy = False
            if base is self._stage_base:
                self._stage_look_texture = texture_from_pil(img)
                self._stage_shown_look = wanted
            self._render_stage_look()  # the settings or photo may have moved on

        def failed(exc: BaseException) -> None:
            self._stage_busy = False
            log.warning("Stage look preview failed: %s", exc)
            if base is self._stage_base and self._wanted_stage_look() == wanted:
                # Nothing valid to show for these settings: the plain photo.
                self._stage_look_texture = None
                self._stage_shown_look = None
                self._show_stage_texture()
            else:
                self._render_stage_look()  # moved on meanwhile: render that

        run_in_thread(
            lambda: cl.apply_to_pil(base, recipe, adjustments, quick=True, output=output),
            on_done=done,
            on_error=failed,
            name="stage-look",
        )

    def _show_stage_texture(self) -> None:
        looked = self._stage_look_texture is not None
        original = looked and self.original_btn.get_active()
        texture = self._stage_look_texture if looked and not original else self._stage_base_texture
        self.stage_picture.set_paintable(texture)
        self._update_stage_overlays()

    def _update_stage_overlays(self) -> None:
        row = self._stage_row
        if row is None or not row.ready:
            return
        _target, is_output = self._stage_target(row)
        looked = self._stage_look_texture is not None and not is_output
        if is_output and row.item.result is not None:
            badge = "Restored" if row.item.result.restored else "Upscaled"
        elif looked and self.original_btn.get_active():
            badge = "Original"
        elif looked:
            badge = "Live Preview"
        else:
            badge = "Original"
        self.stage_badge.set_label(badge)
        if is_output:
            self.stage_badge.add_css_class("finished")
        else:
            self.stage_badge.remove_css_class("finished")
        self.stage_info.set_label(f"{row.item.path.name}   {row.details.get_label()}")
        self.stage_info.set_tooltip_text(str(row.item.path))
        self.original_btn.set_visible(looked)
        self.stage_folder_btn.set_visible(is_output)
        self.stage_compare_btn.set_tooltip_text(
            "Compare Before / After" if is_output else "Compare and Zoom"
        )
        for widget in (self.stage_badge, self.stage_info, self.stage_tools):
            widget.set_visible(True)

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
        if not self.ready:
            self._pending_paths.extend(paths)  # opened with files: added once shown
            return
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
        row.set_scale(
            self.app.settings.processing_options().output_scale, show_monochrome=self.restoring
        )
        self.rows.append(row)
        self.listbox.append(row)

        def ready(info: object, thumb: object, mono: bool, r: QueueRow = row) -> None:
            r.set_info(info, thumb, mono)
            if r is self._stage_row:
                self._refresh_stage()
            self._select_some_row()
            self._update_look_sample()
            self._update_bw_banner()
            self._update_state()

        def failed(err: UpscalerError, r: QueueRow = row) -> None:
            r.set_load_error(err)
            if r is self._stage_row:
                self._refresh_stage()  # stops the spinner
            self._update_state()

        self.thumbnails.request(item.path, ready, failed)

    def remove_row(self, row: QueueRow) -> None:
        if self._remove(row):
            self._after_removal()

    def _remove(self, row: QueueRow) -> bool:
        if row.item.status in (ItemStatus.PROCESSING, ItemStatus.QUEUED):
            return False
        self.rows.remove(row)
        self.listbox.remove(row)
        return True

    def _after_removal(self) -> None:
        self._select_some_row()
        self._update_look_sample()
        self._update_bw_banner()
        self._update_state()

    def _update_look_sample(self) -> None:
        """The camera look cards show the first photo in the queue (uncropped,
        decoded at card resolution in the background)."""
        path = next((r.item.path for r in self.rows if r.ready), None)
        if path == self._look_sample_path:
            return
        self._look_sample_path = path
        if path is None:
            self.camera_look.set_sample(None)
            return

        def done(result: tuple, wanted: Path = path) -> None:
            if wanted == self._look_sample_path:
                self.camera_look.set_sample(result[0])

        run_in_thread(
            load_preview,
            path,
            LOOK_SAMPLE_SIZE,
            on_done=done,
            on_error=lambda e: log.warning("Look sample failed: %s", e),
            name="look-sample",
        )

    def clear_finished(self) -> None:
        removed = [
            row
            for row in list(self.rows)
            if row.item.status in (ItemStatus.DONE, ItemStatus.SKIPPED) and self._remove(row)
        ]
        if removed:
            self._after_removal()  # once: no stage decodes for rows on their way out

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

    def preview_restorer(self) -> tuple:
        """(restorer, restoration settings, options) for a preview, models checked.

        Shares the batch's engine, so models loaded for a preview are reused.
        """
        from pixelift.core.restoration.pipeline import Restorer

        options = self.app.settings.processing_options()
        restoration = options.restoration or self.app.settings.restoration()
        options.validate()
        restorer = Restorer(self._get_upscaler())  # restore() reports missing models
        if restoration.ai_scale():
            self.app.model_manager.resolve(options.model, restoration.ai_scale())
        return restorer, restoration, options

    def _on_device_fallback(self, device: dm.DeviceInfo, reason: str) -> None:
        self._show_device(device)
        self.toasts.add_toast(Adw.Toast(title=f"{reason} — continuing on the CPU", timeout=6))
        self._upscaler_key = None  # rebuild with the preferred device next time

    def start(
        self, rows: list[QueueRow] | None = None, without: frozenset[str] = frozenset()
    ) -> None:
        """Process ``rows`` (default: the whole queue).

        ``without``: model ids to do without for this batch only (their stages
        are skipped; the saved settings stay as they are).
        """
        if self.running:
            return
        if self.app.device_report is None:
            self.toasts.add_toast(Adw.Toast(title="Still detecting the processing device…"))
            self.app.when_devices_ready(lambda _r: self.start(rows, without))
            return
        settings = self.app.settings
        options = settings.processing_options()
        if options.restoration is not None and without:
            options.restoration = _without_models(options.restoration, without)
        candidates = rows if rows is not None else self.rows
        todo = [
            r
            for r in candidates
            if r.ready and r.item.status not in (ItemStatus.DONE, ItemStatus.SKIPPED)
        ]
        try:
            options.validate()
            restoration = options.restoration
            if restoration is None:
                self.app.model_manager.resolve(options.model, options.scale)
            elif restoration.ai_scale():
                self.app.model_manager.resolve(options.model, restoration.ai_scale())
        except ModelNotInstalledError as err:
            self._offer_model_download(err)
            return
        except (ValueError, UpscalerError) as err:
            show_error(self, err if isinstance(err, UpscalerError) else UpscalerError(str(err)))
            return
        if options.restoration is not None and not self._restoration_models_ready(
            options.restoration, any(r.monochrome for r in todo), rows, without
        ):
            return

        if not todo:
            verb = "restore" if options.restoration is not None else "upscale"
            self.toasts.add_toast(
                Adw.Toast(
                    title=f"Nothing to {verb} — add some images first"
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
        self._show_device(upscaler.device)
        self._inhibit_cookie = self.app.inhibit(
            self,
            Gtk.ApplicationInhibitFlags.SUSPEND,
            "Restoring photos" if options.restoration is not None else "Upscaling images",
        )
        self.processor.start([r.item for r in todo], others=[r.item for r in self.rows])
        self._update_state()

    def _restoration_models_ready(
        self,
        restoration: rs.RestorationSettings,
        any_mono: bool,
        rows: list[QueueRow] | None,
        without: frozenset[str],
    ) -> bool:
        """Ask before downloading missing restoration models (never silently)."""
        from pixelift.core.restoration.pipeline import required_models

        manager = self.app.model_manager
        missing = [
            manager.spec(model_id)
            for model_id in required_models(restoration, monochrome=any_mono)
            if not manager.is_installed(model_id)
        ]
        if not missing:
            return True
        total = sum(s.size_bytes for s in missing)
        lines = "\n".join(
            f"• {s.name} {s.version} — {s.size_mb:.0f} MB, {s.license}" for s in missing
        )
        dialog = Adw.AlertDialog(
            heading="AI models needed",
            body=f"This restoration uses AI models that are not installed yet:\n\n{lines}\n\n"
            f"Download them now ({total / 1e6:.0f} MB in total, once)? They come from the "
            "projects' official releases and are verified. Your photos are still processed "
            "only on this computer.",
        )
        dialog.add_response("cancel", "Cancel")
        dialog.add_response("without", "Continue Without")
        dialog.add_response("download", "Download")
        dialog.set_response_appearance("download", Adw.ResponseAppearance.SUGGESTED)
        dialog.set_default_response("download")

        def on_response(_d: Adw.AlertDialog, response: str) -> None:
            if response == "download":
                for spec in missing:
                    self.app.downloads.start(spec.id)
                self.app.show_preferences(page="models")
                self.toasts.add_toast(
                    Adw.Toast(title="Downloading AI models — start again when they are ready")
                )
            elif response == "without":
                self.start(rows, without | {s.id for s in missing})

        dialog.connect("response", on_response)
        dialog.present(self)
        return False

    # --- black and white photos -----------------------------------------
    def _update_bw_banner(self) -> None:
        count = sum(
            1
            for r in self.rows
            if r.ready
            and r.monochrome
            and r.item.status not in (ItemStatus.DONE, ItemStatus.SKIPPED)
        )
        if self.restoring and count and not self._bw_answered:
            self.bw_banner.show_for(count)
        else:
            self.bw_banner.set_reveal_child(False)

    def _on_bw_choice(self, colorize: bool) -> None:
        """Colorize only with consent: the user picked on the banner."""
        settings = self.app.settings
        self._bw_answered = True
        _current, upscale = rs.preset_flags(settings.restore_preset)
        if colorize:
            settings.restore_preset = rs.PRESET_FULL if upscale else rs.PRESET_COLORIZE
        else:
            settings.restore_preset = rs.PRESET_UPSCALE if upscale else rs.PRESET_RESTORE
        self.app.settings_changed()
        self._update_bw_banner()

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
            if row is self._stage_row and ev.kind is not EventKind.ITEM_PROGRESS:
                self._refresh_stage()  # shows the result as soon as it is saved
        self.overall_bar.set_fraction(ev.fraction)
        self.overall_label.set_label(f"{ev.completed} of {ev.total} · {ev.fraction * 100:.0f}%")
        if ev.kind in (EventKind.ITEM_STARTED, EventKind.ITEM_PROGRESS) and ev.item:
            paused = " (paused)" if self.processor and self.processor.paused else ""
            stage = f" — {ev.item.stage}" if ev.item.stage and ev.item.stage != "Starting" else ""
            self.current_label.set_label(f"{ev.item.path.name}{stage}{paused}")
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
        restored = self.processor is not None and self.processor.options.restoration is not None
        parts = [f"{summary.done} {'restored' if restored else 'upscaled'}"]
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
            notification = Gio.Notification.new(
                "Restoration finished" if restored else "Upscaling finished"
            )
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
        self.queue_subtitle.set_label(
            f"{count} image{'s' if count != 1 else ''}" + (f" · {done} done" if done else "")
        )
        self.sidebar_btn.set_visible(count > 0)
        self.start_btn.set_visible(not running)
        self.start_btn.set_sensitive(pending > 0)
        verb = "Restore" if self.restoring else "Upscale"
        if pending > 1:
            self.start_btn.set_label(f"{verb} {pending} Photos")
        elif pending == 1 and count > 1:
            self.start_btn.set_label(f"{verb} 1 Photo")
        else:
            self.start_btn.set_label(verb)
        self.pause_btn.set_visible(running)
        self.pause_btn.set_label("Resume" if self.processor and self.processor.paused else "Pause")
        self.cancel_btn.set_visible(running)
        self.cancel_btn.set_sensitive(running and not self.processor.control.cancelled)
        self.retry_btn.set_visible(not running and failed > 0)
        self.clear_btn.set_sensitive(count > 0 and not running)
        self.panel_stack.set_sensitive(not running)
        self.mode_box.set_sensitive(not running)
        from pixelift.ui.preview import PreviewWindow

        for window in self.app.get_windows():
            if isinstance(window, PreviewWindow):
                window.set_lighting_editable(not running)
                if running:
                    window.cancel_restoration()  # the batch needs the models and device
        self.progress_revealer.set_reveal_child(self.processor is not None)

    def _on_close_request(self, _window: Gtk.Window) -> bool:
        if not self.running:
            self.thumbnails.shutdown()
            return False
        dialog = Adw.AlertDialog(
            heading="Stop restoring?" if self.restoring else "Stop upscaling?",
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


def _without_models(
    restoration: rs.RestorationSettings, ids: frozenset[str]
) -> rs.RestorationSettings:
    """``restoration`` with the stages that need the models ``ids`` turned off."""
    import dataclasses

    from pixelift.models.restoration import COLORIZE_MODELS, FACE_MODELS

    if ids & set(FACE_MODELS) and restoration.stages().face != rs.FACE_OFF:
        custom = dataclasses.replace(restoration.stages(), face=rs.FACE_OFF)
        restoration = dataclasses.replace(restoration, level=rs.CUSTOM, custom=custom)
    if ids & set(COLORIZE_MODELS):
        restoration = dataclasses.replace(restoration, colorize=False)
    return restoration
