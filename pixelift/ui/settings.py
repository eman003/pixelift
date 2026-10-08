"""Preferences dialog: processing, output, performance and model management."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from gi.repository import Adw, Gio, GLib, Gtk

from pixelift.core import device_manager as dm
from pixelift.models import KIND_LABELS, UPSCALE, all_families
from pixelift.ui.widgets.model_row import ModelRow
from pixelift.utils import system
from pixelift.utils.image_utils import render_filename

if TYPE_CHECKING:
    from pixelift.ui.application import UpscalerApplication

TILE_LABELS = (("Automatic", 0), ("256 px", 256), ("512 px", 512), ("1024 px", 1024))
FORMAT_LABELS = (("PNG (lossless)", "png"), ("JPEG", "jpeg"), ("WebP", "webp"))
EXISTING_LABELS = (
    ("Skip the image", "skip"),
    ("Overwrite", "overwrite"),
    ("Add a number", "rename"),
)
THEME_LABELS = (("Follow system", "system"), ("Light", "light"), ("Dark", "dark"))
JOB_LABELS = (("Automatic", 0), ("1", 1), ("2", 2), ("3", 3), ("4", 4))


def _combo(title: str, labels: list[str], subtitle: str = "") -> Adw.ComboRow:
    row = Adw.ComboRow(title=title, model=Gtk.StringList.new(labels))
    if subtitle:
        row.set_subtitle(subtitle)
    return row


class PreferencesDialog(Adw.PreferencesDialog):
    def __init__(self, app: UpscalerApplication) -> None:
        super().__init__(title="Preferences", search_enabled=False)
        self.app = app
        self.settings = app.settings
        self._pages: dict[str, Adw.PreferencesPage] = {}
        self._loading = True
        self._build_processing()
        self._build_output()
        self._build_performance()
        self._build_models()
        self._loading = False

    def show_page(self, name: str) -> None:
        if name in self._pages:
            self.set_visible_page(self._pages[name])

    def _changed(self) -> None:
        if not self._loading:
            self.app.settings_changed()

    def _page(self, key: str, title: str, icon: str) -> Adw.PreferencesPage:
        page = Adw.PreferencesPage(title=title, icon_name=icon)
        self.add(page)
        self._pages[key] = page
        return page

    # --- processing --------------------------------------------------------
    def _build_processing(self) -> None:
        page = self._page("processing", "Processing", "applications-science-symbolic")
        group = Adw.PreferencesGroup(title="AI Model")
        page.add(group)

        families = all_families()
        model_row = _combo("Default model", [f.name for f in families])
        ids = [f.id for f in families]
        model_row.set_selected(ids.index(self.settings.model) if self.settings.model in ids else 0)
        model_row.connect(
            "notify::selected", lambda r, _p: self._set("model", ids[r.get_selected()])
        )
        group.add(model_row)

        scale_row = _combo("Default scale", ["2×", "4×"])
        scale_row.set_selected(0 if self.settings.scale == 2 else 1)
        scale_row.connect(
            "notify::selected", lambda r, _p: self._set("scale", (2, 4)[r.get_selected()])
        )
        group.add(scale_row)

        self.device_group = Adw.PreferencesGroup(title="Device")
        page.add(self.device_group)
        self.device_row = _combo("Processing device", ["Detecting…"])
        self.device_row.set_sensitive(False)
        self.device_group.add(self.device_row)
        self.device_ids: list[str] = []

        tile_row = _combo(
            "Tile size",
            [label for label, _ in TILE_LABELS],
            "Large images are processed in tiles. Smaller tiles use less memory; "
            "automatic mode adapts to free memory and shrinks on errors.",
        )
        values = [v for _, v in TILE_LABELS]
        tile_row.set_selected(
            values.index(self.settings.tile_size) if self.settings.tile_size in values else 0
        )
        tile_row.connect(
            "notify::selected", lambda r, _p: self._set("tile_size", values[r.get_selected()])
        )
        self.device_group.add(tile_row)

        self.memory_row = Adw.SpinRow.new_with_range(0, 65536, 256)
        self.memory_row.set_title("GPU memory limit (MB)")
        self.memory_row.set_subtitle("0 = no limit (NVIDIA/AMD GPUs only)")
        self.memory_row.set_value(self.settings.gpu_memory_limit_mb)
        self.memory_row.connect(
            "notify::value", lambda r, _p: self._set("gpu_memory_limit_mb", int(r.get_value()))
        )
        self.device_group.add(self.memory_row)
        self.app.when_devices_ready(self._fill_devices)

    def _fill_devices(self, report: dm.DeviceReport) -> None:
        labels = [f"Automatic ({report.best.label()})"]
        self.device_ids = ["auto"]
        for dev in report.devices:
            labels.append(dev.label())
            self.device_ids.append(dev.id)
        self._loading = True
        self.device_row.set_model(Gtk.StringList.new(labels))
        current = self.settings.device if self.settings.device in self.device_ids else "auto"
        self.device_row.set_selected(self.device_ids.index(current))
        self._loading = False
        self.device_row.set_sensitive(True)
        self.device_row.connect(
            "notify::selected", lambda r, _p: self._set("device", self.device_ids[r.get_selected()])
        )
        self.memory_row.set_sensitive(any(d.kind == "cuda" for d in report.devices))
        if report.hints:
            self.device_group.set_description("\n".join(report.hints))

    # --- output ------------------------------------------------------------
    def _build_output(self) -> None:
        page = self._page("output", "Output", "document-save-symbolic")
        group = Adw.PreferencesGroup(title="Files")
        page.add(group)

        self.folder_row = Adw.ActionRow(title="Output folder")
        choose = Gtk.Button(
            icon_name="folder-open-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Choose folder"
        )
        choose.add_css_class("flat")
        choose.connect("clicked", lambda _b: self._choose_folder())
        self.folder_reset = Gtk.Button(
            icon_name="edit-undo-symbolic",
            valign=Gtk.Align.CENTER,
            tooltip_text="Save next to the originals",
        )
        self.folder_reset.add_css_class("flat")
        self.folder_reset.connect("clicked", lambda _b: self._set_folder(""))
        self.folder_row.add_suffix(self.folder_reset)
        self.folder_row.add_suffix(choose)
        self.folder_row.set_activatable_widget(choose)
        self._update_folder_row()
        group.add(self.folder_row)

        self.template_row = Adw.EntryRow(
            title="Filename template (upscaling)", text=self.settings.filename_template
        )
        self.template_row.set_tooltip_text(
            "Restored photos are named <name>_restored[_colorized][_2x].<ext>"
        )
        self.template_row.connect("changed", self._on_template_changed)
        group.add(self.template_row)
        self.template_hint = Adw.ActionRow(title="", subtitle_selectable=False)
        self.template_hint.add_css_class("property")
        group.add(self.template_hint)
        self._update_template_hint()

        existing = _combo("If the output file exists", [label for label, _ in EXISTING_LABELS])
        evalues = [v for _, v in EXISTING_LABELS]
        existing.set_selected(evalues.index(self.settings.existing))
        existing.connect(
            "notify::selected", lambda r, _p: self._set("existing", evalues[r.get_selected()])
        )
        group.add(existing)

        fmt_group = Adw.PreferencesGroup(title="Format")
        page.add(fmt_group)
        fmt = _combo("Default format", [label for label, _ in FORMAT_LABELS])
        fvalues = [v for _, v in FORMAT_LABELS]
        fmt.set_selected(fvalues.index(self.settings.output_format))
        fmt.connect(
            "notify::selected", lambda r, _p: self._set("output_format", fvalues[r.get_selected()])
        )
        fmt_group.add(fmt)
        quality = Adw.SpinRow.new_with_range(1, 100, 1)
        quality.set_title("JPEG / WebP quality")
        quality.set_value(self.settings.quality)
        quality.connect("notify::value", lambda r, _p: self._set("quality", int(r.get_value())))
        fmt_group.add(quality)
        meta = Adw.SwitchRow(
            title="Keep metadata",
            subtitle="Copy EXIF data and colour profiles to the output",
            active=self.settings.preserve_metadata,
        )
        meta.connect("notify::active", lambda r, _p: self._set("preserve_metadata", r.get_active()))
        fmt_group.add(meta)

    def _update_folder_row(self) -> None:
        folder = self.settings.output_dir
        self.folder_row.set_subtitle(
            folder or "Next to each original, in an “upscaled” (or “restored”) folder"
        )
        self.folder_reset.set_visible(bool(folder))

    def _choose_folder(self) -> None:
        dialog = Gtk.FileDialog(title="Choose Output Folder", modal=True)

        def done(dlg: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
            try:
                folder = dlg.select_folder_finish(result)
            except GLib.Error:
                return
            if folder and folder.get_path():
                self._set_folder(folder.get_path())

        dialog.select_folder(self.get_root(), None, done)

    def _set_folder(self, folder: str) -> None:
        self._set("output_dir", folder)
        self._update_folder_row()

    def _on_template_changed(self, row: Adw.EntryRow) -> None:
        text = row.get_text()
        try:
            render_filename(text, Path("photo.jpg"), 4, "realesrgan", 7680, 4320, ".png")
        except (ValueError, IndexError, KeyError):
            row.add_css_class("error")
            self.template_hint.set_title(
                "Unknown field — use {name} {scale} {model} {width} {height} {ext} {lighting} "
                "{look}"
            )
            return
        row.remove_css_class("error")
        self._set("filename_template", text)
        self._update_template_hint()

    def _update_template_hint(self) -> None:
        ext = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}[self.settings.output_format]
        example = render_filename(
            self.settings.filename_template,
            Path("photo.jpg"),
            self.settings.scale,
            self.settings.model,
            7680,
            4320,
            ext,
            self.settings.lighting().tag(),
            self.settings.camera_look_settings().tag(),
        )
        self.template_hint.set_title(f"Example: photo.jpg → {example}")

    # --- performance -------------------------------------------------------
    def _build_performance(self) -> None:
        page = self._page("performance", "Performance", "speedometer-symbolic")
        group = Adw.PreferencesGroup(title="Performance")
        page.add(group)
        gpu = Adw.SwitchRow(
            title="Enable GPU acceleration",
            subtitle="Turn off to always use the CPU",
            active=self.settings.gpu_enabled,
        )
        gpu.connect("notify::active", lambda r, _p: self._set("gpu_enabled", r.get_active()))
        group.add(gpu)

        jobs = _combo(
            "Concurrent jobs",
            [label for label, _ in JOB_LABELS],
            "Automatic runs one image at a time per device, which is fastest and "
            "uses the least memory.",
        )
        jvalues = [v for _, v in JOB_LABELS]
        jobs.set_selected(
            jvalues.index(self.settings.concurrent_jobs)
            if self.settings.concurrent_jobs in jvalues
            else 0
        )
        jobs.connect(
            "notify::selected",
            lambda r, _p: self._set("concurrent_jobs", jvalues[r.get_selected()]),
        )
        group.add(jobs)

        threads = Adw.SpinRow.new_with_range(0, system.cpu_count(), 1)
        threads.set_title("CPU threads")
        threads.set_subtitle(f"0 = use all {system.cpu_count()} cores")
        threads.set_value(self.settings.cpu_threads)
        threads.connect("notify::value", lambda r, _p: self._set("cpu_threads", int(r.get_value())))
        group.add(threads)

        look = Adw.PreferencesGroup(title="Appearance")
        page.add(look)
        theme = _combo("Style", [label for label, _ in THEME_LABELS])
        tvalues = [v for _, v in THEME_LABELS]
        theme.set_selected(tvalues.index(self.settings.theme))
        theme.connect(
            "notify::selected", lambda r, _p: self._set("theme", tvalues[r.get_selected()])
        )
        look.add(theme)

    # --- models ------------------------------------------------------------
    def _build_models(self) -> None:
        page = self._page("models", "Models", "folder-download-symbolic")
        manager = self.app.model_manager
        descriptions = {
            "Upscaling": "Models are downloaded once from the projects' official releases and "
            "verified with SHA-256 checksums. Images are always processed locally — your "
            "images stay on your computer.",
            "Face Restoration": "Used by Restore Photos to restore faces: GFPGAN restores, "
            "RetinaFace finds the faces. Both are needed.",
            "Photo Colorization": "Used by Restore Photos to colorize black-and-white "
            "photographs, only when you ask for it.",
        }
        specs = manager.specs()
        for label in dict.fromkeys(KIND_LABELS.values()):
            group_specs = [
                s for s in specs if KIND_LABELS.get(s.kind, KIND_LABELS[UPSCALE]) == label
            ]
            if not group_specs:
                continue
            group = Adw.PreferencesGroup(
                title=f"{label} Models", description=descriptions.get(label, "")
            )
            page.add(group)
            for spec in group_specs:
                group.add(ModelRow(spec, self.app.downloads))

        folder_group = Adw.PreferencesGroup()
        page.add(folder_group)
        folder = Adw.ActionRow(title="Models folder", subtitle=str(manager.models_dir))
        open_btn = Gtk.Button(
            icon_name="folder-open-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Open folder"
        )
        open_btn.add_css_class("flat")

        def open_folder(_b: Gtk.Button) -> None:
            manager.models_dir.mkdir(parents=True, exist_ok=True)
            Gtk.FileLauncher.new(Gio.File.new_for_path(str(manager.models_dir))).launch(
                self.get_root(), None, None
            )

        open_btn.connect("clicked", open_folder)
        folder.add_suffix(open_btn)
        folder.set_activatable_widget(open_btn)
        folder_group.add(folder)

    # --- helpers -----------------------------------------------------------
    def _set(self, key: str, value: object) -> None:
        if self._loading:
            return
        setattr(self.settings, key, value)
        self._changed()
        if key in ("output_format", "scale", "model") and hasattr(self, "template_hint"):
            self._update_template_hint()
