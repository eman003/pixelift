"""First-run assistant: device check, model download offer and a test upscale."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from gi.repository import Adw, Gtk

from pixelift import APP_ID, APP_NAME
from pixelift.core import device_manager as dm
from pixelift.core.errors import UpscalerError, friendly_error
from pixelift.core.image_processor import test_pattern
from pixelift.core.upscaler import TorchUpscaler
from pixelift.models import get_spec
from pixelift.models.realesrgan import RECOMMENDED_CPU_MODEL, RECOMMENDED_MODEL
from pixelift.ui.async_utils import run_in_thread
from pixelift.ui.widgets.model_row import ModelRow

if TYPE_CHECKING:
    from pixelift.ui.application import UpscalerApplication


def _page(title: str, description: str, icon: str | None = None) -> Adw.StatusPage:
    page = Adw.StatusPage(title=title, description=description, vexpand=True)
    if icon:
        page.set_icon_name(icon)
    page.add_css_class("compact")
    return page


class FirstRunDialog(Adw.Dialog):
    def __init__(self, app: UpscalerApplication) -> None:
        super().__init__(title=f"Welcome to {APP_NAME}", content_width=560, content_height=620)
        self.app = app
        self._models_filled = False
        self._spec_name = ""
        self.connect("closed", lambda _d: self._mark_complete())

        toolbar = Adw.ToolbarView()
        toolbar.add_top_bar(Adw.HeaderBar(show_title=False))
        self.stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.SLIDE_LEFT)
        toolbar.set_content(self.stack)
        self.set_child(toolbar)

        self._build_welcome()
        self._build_models()
        self._build_test()
        app.when_devices_ready(self._on_devices)

    def _footer(self, *buttons: Gtk.Button) -> Gtk.Box:
        box = Gtk.Box(spacing=12, halign=Gtk.Align.CENTER, margin_bottom=24, margin_top=12)
        for button in buttons:
            button.add_css_class("pill")
            box.append(button)
        return box

    # --- step 1: welcome + hardware -----------------------------------------
    def _build_welcome(self) -> None:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        page = _page(
            f"Welcome to {APP_NAME}",
            "Make images larger and sharper with AI.\n"
            "Everything runs on this computer — no account, no uploads, no tracking.",
            APP_ID,
        )
        inner = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        group = Adw.PreferencesGroup()
        self.device_row = Adw.ActionRow(title="Processing device", subtitle="Detecting…")
        self.device_row.add_prefix(Gtk.Image(icon_name="computer-symbolic"))
        self.device_spinner = Gtk.Spinner(spinning=True)
        self.device_row.add_suffix(self.device_spinner)
        group.add(self.device_row)
        privacy = Adw.ActionRow(
            title="Your images stay on your computer.", subtitle="AI processing happens locally."
        )
        privacy.add_prefix(Gtk.Image(icon_name="security-high-symbolic"))
        group.add(privacy)
        inner.append(group)
        self.hints = Gtk.Label(wrap=True, xalign=0, visible=False)
        self.hints.add_css_class("dim-label")
        self.hints.add_css_class("caption")
        inner.append(self.hints)
        page.set_child(Adw.Clamp(maximum_size=440, child=inner))
        box.append(page)
        next_btn = Gtk.Button(label="Continue")
        next_btn.add_css_class("suggested-action")
        next_btn.connect("clicked", lambda _b: self._show_models())
        box.append(self._footer(next_btn))
        self.stack.add_named(box, "welcome")

    def _on_devices(self, report: dm.DeviceReport) -> None:
        device = self.app.selected_device()
        self.device_spinner.set_visible(False)
        self.device_row.set_subtitle(device.label())
        if report.hints:
            self.hints.set_label("\n\n".join(report.hints))
            self.hints.set_visible(True)

    # --- step 2: models -----------------------------------------------------
    def _build_models(self) -> None:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        page = _page(
            "Download an AI model",
            "Upscaling needs an AI model file. Models are downloaded once from the official "
            "Real-ESRGAN project (BSD-3-Clause license) and verified. After that, no internet "
            "connection is needed.",
            "folder-download-symbolic",
        )
        self.models_group = Adw.PreferencesGroup()
        page.set_child(Adw.Clamp(maximum_size=440, child=self.models_group))
        box.append(page)
        skip = Gtk.Button(label="Skip for Now")
        skip.connect("clicked", lambda _b: self.close())
        self.models_next = Gtk.Button(label="Continue")
        self.models_next.add_css_class("suggested-action")
        self.models_next.connect("clicked", lambda _b: self._show_test())
        box.append(self._footer(skip, self.models_next))
        self.stack.add_named(box, "models")
        self.app.downloads.subscribe(lambda *_: self._update_models_next())

    def _show_models(self) -> None:
        device = self.app.selected_device()
        recommended = [RECOMMENDED_MODEL]
        if not device.is_gpu:
            # On CPU the compact model is ~10× faster; offer it first.
            recommended = [RECOMMENDED_CPU_MODEL, RECOMMENDED_MODEL]
        if not self._models_filled:
            for model_id in recommended:
                row = ModelRow(get_spec(model_id), self.app.downloads, allow_remove=False)
                if model_id == recommended[0]:
                    row.set_title(f"{row.get_title()} — recommended")
                self.models_group.add(row)
            self._models_filled = True
        if not device.is_gpu:
            self.models_group.set_description(
                "No supported GPU was found, so images will be processed on the CPU. "
                "The fast model is recommended; the full Real-ESRGAN model gives slightly "
                "better detail but is much slower on a CPU."
            )
        self._update_models_next()
        self.stack.set_visible_child_name("models")

    def _update_models_next(self) -> None:
        installed = bool(self.app.model_manager.installed())
        self.models_next.set_sensitive(installed)
        self.models_next.set_tooltip_text(None if installed else "Download a model first")

    # --- step 3: test upscale ----------------------------------------------
    def _build_test(self) -> None:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.test_page = _page("Testing…", "Running a small test upscale on this computer.")
        self.test_spinner = Gtk.Spinner(
            spinning=True, width_request=48, height_request=48, halign=Gtk.Align.CENTER
        )
        self.test_page.set_child(self.test_spinner)
        box.append(self.test_page)
        self.done_btn = Gtk.Button(label=f"Start Using {APP_NAME}", sensitive=False)
        self.done_btn.add_css_class("suggested-action")
        self.done_btn.connect("clicked", lambda _b: self.close())
        box.append(self._footer(self.done_btn))
        self.stack.add_named(box, "test")

    def _show_test(self) -> None:
        self.stack.set_visible_child_name("test")
        manager = self.app.model_manager
        device = self.app.selected_device()
        installed = manager.installed()
        settings = self.app.settings
        try:
            spec = manager.resolve(settings.model, settings.scale)
        except UpscalerError:
            spec = installed[0]
            # Make the default model one that is actually installed.
            from pixelift.models import all_families

            for family in all_families():
                if spec.id in family.variants.values():
                    settings.model = family.id
                    self.app.settings_changed()
                    break

        def work() -> tuple[float, tuple[int, int], str]:
            upscaler = TorchUpscaler(manager, device)
            image = test_pattern(48)
            started = time.monotonic()
            out = upscaler.upscale(image, 4, spec.id)
            elapsed = time.monotonic() - started
            label = upscaler.device_label
            upscaler.release()
            return elapsed, (out.shape[1], out.shape[0]), label

        run_in_thread(
            work, on_done=self._on_test_done, on_error=self._on_test_failed, name="self-test"
        )
        self._spec_name = spec.name

    def _on_test_done(self, result: tuple[float, tuple[int, int], str]) -> None:
        elapsed, (w, h), label = result
        self.test_spinner.set_visible(False)
        self.test_page.set_icon_name("emblem-ok-symbolic")
        self.test_page.set_title("Ready!")
        self.test_page.set_description(
            f"A 48×48 test image was upscaled to {w}×{h} with {self._spec_name} "
            f"in {elapsed:.1f} s on {label}.\n\nDrop images into the window to get started."
        )
        self.done_btn.set_sensitive(True)
        self._mark_complete()

    def _on_test_failed(self, exc: BaseException) -> None:
        error = friendly_error(exc)
        self.test_spinner.set_visible(False)
        self.test_page.set_icon_name("dialog-warning-symbolic")
        self.test_page.set_title("Test failed")
        self.test_page.set_description(
            error.reason + "\n\nYou can change the processing device in Preferences and try again."
        )
        self.done_btn.set_label("Close")
        self.done_btn.set_sensitive(True)

    def _mark_complete(self) -> None:
        if not self.app.settings.first_run_complete:
            self.app.settings.first_run_complete = True
            self.app.settings_changed()
