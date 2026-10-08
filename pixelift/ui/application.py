"""GTK application object: global state, actions, theme and startup flow."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")

from gi.repository import Adw, Gdk, Gio, GLib, Gtk  # noqa: E402

from pixelift import APP_ID, APP_NAME, __version__  # noqa: E402
from pixelift.core import device_manager as dm  # noqa: E402
from pixelift.core.model_manager import ModelManager  # noqa: E402
from pixelift.storage import paths  # noqa: E402
from pixelift.storage.settings import Settings, load_settings, save_settings  # noqa: E402
from pixelift.ui.async_utils import run_in_thread  # noqa: E402
from pixelift.ui.downloads import DownloadTracker  # noqa: E402

log = logging.getLogger(__name__)

_PKG_DIR = Path(__file__).resolve().parent.parent


class UpscalerApplication(Adw.Application):
    def __init__(self) -> None:
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.HANDLES_OPEN)
        GLib.set_application_name(APP_NAME)
        self.settings: Settings = load_settings()
        self.model_manager = ModelManager()
        self.downloads = DownloadTracker(self.model_manager)
        self.device_report: dm.DeviceReport | None = None
        self._device_waiters: list[Callable[[dm.DeviceReport], None]] = []
        self._settings_listeners: list[Callable[[], None]] = []
        self._lighting_listeners: list[Callable[[], None]] = []

    # --- lifecycle ---------------------------------------------------------
    def do_startup(self) -> None:
        Adw.Application.do_startup(self)
        display = Gdk.Display.get_default()
        if display is not None:
            Gtk.IconTheme.get_for_display(display).add_search_path(str(_PKG_DIR / "data" / "icons"))
            css = Gtk.CssProvider()
            css.load_from_path(str(_PKG_DIR / "ui" / "style.css"))
            Gtk.StyleContext.add_provider_for_display(
                display, css, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )
        Gtk.Window.set_default_icon_name(APP_ID)
        self.apply_theme()
        self._add_actions()
        # Importing PyTorch takes a second or two: never on the UI thread.
        run_in_thread(
            dm.detect_devices,
            on_done=self._on_devices,
            on_error=self._on_devices_failed,
            name="device-detect",
        )

    def do_activate(self) -> None:
        from pixelift.ui.main_window import MainWindow

        window = self.props.active_window
        if window is None:
            window = MainWindow(self)
        window.present()
        if not self.settings.first_run_complete:
            from pixelift.ui.first_run import FirstRunDialog

            FirstRunDialog(self).present(window)

    def do_open(self, files: list[Gio.File], _n_files: int, _hint: str) -> None:
        self.activate()
        window = self.props.active_window
        paths_ = [Path(f.get_path()) for f in files if f.get_path()]
        if window is not None and paths_:
            window.add_paths(paths_)

    # --- devices -----------------------------------------------------------
    def _on_devices(self, report: dm.DeviceReport) -> None:
        self.device_report = report
        waiters, self._device_waiters = self._device_waiters, []
        for waiter in waiters:
            waiter(report)

    def _on_devices_failed(self, exc: BaseException) -> None:
        log.error("Device detection failed: %s", exc)
        self._on_devices(dm.DeviceReport([dm.CPU_DEVICE], ["Device detection failed; using CPU."]))

    def when_devices_ready(self, callback: Callable[[dm.DeviceReport], None]) -> None:
        if self.device_report is not None:
            callback(self.device_report)
        else:
            self._device_waiters.append(callback)

    def selected_device(self) -> dm.DeviceInfo:
        """The device to use; CPU until background detection has finished."""
        report = self.device_report or dm.DeviceReport([dm.CPU_DEVICE])
        return dm.resolve_device(
            self.settings.device, report, gpu_enabled=self.settings.gpu_enabled
        )

    # --- settings ----------------------------------------------------------
    def on_settings_changed(self, listener: Callable[[], None]) -> None:
        self._settings_listeners.append(listener)

    def off_settings_changed(self, listener: Callable[[], None]) -> None:
        if listener in self._settings_listeners:
            self._settings_listeners.remove(listener)

    def settings_changed(self) -> None:
        log.debug("Settings changed: %s", self.settings)
        self.settings.normalise()
        save_settings(self.settings)
        self.apply_theme()
        for listener in list(self._settings_listeners) + list(self._lighting_listeners):
            listener()

    def on_lighting_changed(self, listener: Callable[[], None]) -> None:
        """Called when the lighting values change (also on any settings change)."""
        self._lighting_listeners.append(listener)

    def off_lighting_changed(self, listener: Callable[[], None]) -> None:
        if listener in self._lighting_listeners:
            self._lighting_listeners.remove(listener)

    def lighting_changed(self) -> None:
        """Notify lighting listeners only, without saving (cheap enough for slider drags)."""
        self.settings.normalise()
        for listener in list(self._lighting_listeners):
            listener()

    def save_settings(self) -> None:
        save_settings(self.settings)

    def apply_theme(self) -> None:
        scheme = {
            "light": Adw.ColorScheme.FORCE_LIGHT,
            "dark": Adw.ColorScheme.FORCE_DARK,
        }.get(self.settings.theme, Adw.ColorScheme.DEFAULT)
        Adw.StyleManager.get_default().set_color_scheme(scheme)

    # --- actions -----------------------------------------------------------
    def _add_actions(self) -> None:
        def add(name: str, callback: Callable[[], None], accels: list[str] | None = None) -> None:
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", lambda *_: callback())
            self.add_action(action)
            if accels:
                self.set_accels_for_action(f"app.{name}", accels)

        add("preferences", self.show_preferences, ["<Control>comma"])
        add("models", lambda: self.show_preferences(page="models"))
        add("about", self.show_about)
        add("open-log", self.open_log_folder)
        add("first-run", self.show_first_run)
        add("quit", self.quit_app, ["<Control>q"])

    def show_preferences(self, page: str | None = None) -> None:
        from pixelift.ui.settings import PreferencesDialog

        dialog = PreferencesDialog(self)
        if page:
            dialog.show_page(page)
        dialog.present(self.props.active_window)

    def show_first_run(self) -> None:
        from pixelift.ui.first_run import FirstRunDialog

        FirstRunDialog(self).present(self.props.active_window)

    def show_about(self) -> None:
        about = Adw.AboutDialog(
            application_name=APP_NAME,
            application_icon=APP_ID,
            version=__version__,
            developer_name="Pixelift contributors",
            license_type=Gtk.License.MIT_X11,
            comments="AI image upscaling and photo restoration that run entirely on your "
            "computer.\n"
            "Your images stay on your computer.",
            website="https://github.com/xinntao/Real-ESRGAN",
        )
        about.add_legal_section(
            "Real-ESRGAN",
            "Copyright © 2021 Xintao Wang",
            Gtk.License.BSD_3,
            None,
        )
        about.add_legal_section(
            "GFPGAN",
            "Copyright © 2021 THL A29 Limited, a Tencent company",
            Gtk.License.APACHE_2_0,
            None,
        )
        about.add_legal_section(
            "facexlib (RetinaFace)", "Copyright © 2020 Xintao Wang", Gtk.License.MIT_X11, None
        )
        about.add_legal_section(
            "DeOldify", "Copyright © 2018 Jason Antic", Gtk.License.MIT_X11, None
        )
        about.add_credit_section(
            "AI models",
            [
                "Real-ESRGAN by Xintao Wang et al.",
                "GFPGAN by Xintao Wang et al. (Tencent ARC)",
                "RetinaFace by Jiankang Deng et al.",
                "DeOldify by Jason Antic",
            ],
        )
        about.present(self.props.active_window)

    def open_log_folder(self) -> None:
        folder = paths.state_dir()
        folder.mkdir(parents=True, exist_ok=True)
        Gtk.FileLauncher.new(Gio.File.new_for_path(str(folder))).launch(
            self.props.active_window, None, None
        )

    def quit_app(self) -> None:
        window = self.props.active_window
        if window is not None:
            window.close()
        else:
            self.quit()


def run(files: list[str]) -> int:
    GLib.set_prgname(APP_ID)  # matches the .desktop file (X11 WM_CLASS)
    app = UpscalerApplication()
    return app.run([sys.argv[0], *files])
