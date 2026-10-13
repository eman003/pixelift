"""HDR output controls for the Export panel: SDR or HDR (PQ) files, bound to the settings.

The common choice is one click (Range: SDR / HDR); the HDR format and peak
luminance sit next to it, and the technical options (colour space, SDR white
level, highlight expansion) are in the Advanced popover. A status line says
what happens to the selected photo's dynamic range (``conversion_status``) and
warns when HDR information is discarded or an encoder is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gi.repository import GLib, Gtk

from pixelift.core.hdr_output import codecs
from pixelift.core.hdr_output import pipeline as hp
from pixelift.ui.async_utils import run_in_thread

if TYPE_CHECKING:
    from pixelift.ui.application import UpscalerApplication

FORMAT_IDS = list(hp.FORMATS)
PRIMARY_IDS = list(hp.OUTPUT_PRIMARIES)
TRANSFER_IDS = list(hp.TRANSFERS)
PRIMARIES_TIP = (
    "Rec. 2020 is the HDR standard; Display P3 suits Apple and many laptop displays "
    "(PQ only: HLG is always Rec. 2020, Ultra HDR JPEG has an sRGB base)."
)
SAVE_DELAY_MS = 300  # spin buttons: save once the value settles


class HdrOutputControls(Gtk.Box):
    def __init__(self, app: UpscalerApplication, label_width: int = 6) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.app = app
        self._syncing = False
        self._source_hdr: bool | None = None  # the selected photo (None: no photo)
        self._gain_map = False  # the selected photo is a gain-map JPEG
        # Whether ffmpeg can write AVIF: probed off the main thread (it runs
        # ``ffmpeg -encoders``); None until known.
        self._avif: bool | None = None
        self._save_source = 0
        self._shut = False

        row = Gtk.Box(spacing=12)
        title = Gtk.Label(label="Range", xalign=0, width_chars=label_width)
        title.add_css_class("dim-label")
        row.append(title)
        self.range_buttons: dict[str, Gtk.ToggleButton] = {}
        group: Gtk.ToggleButton | None = None
        segment = Gtk.Box(css_classes=["linked", "segmented"], halign=Gtk.Align.START)
        for mode, label, tip in (
            ("sdr", "SDR", "Standard dynamic range files (PNG, JPEG or WebP)"),
            ("hdr", "HDR", "HDR files: PQ or HLG (16-bit PNG, 10-bit AVIF) or Ultra HDR JPEG"),
        ):
            button = Gtk.ToggleButton(label=label, group=group, tooltip_text=tip)
            group = group or button
            button.connect("toggled", self._on_range_toggled, mode)
            segment.append(button)
            self.range_buttons[mode] = button
        row.append(segment)

        self.hdr_box = Gtk.Box(spacing=12)
        self.format_dd = Gtk.DropDown.new_from_strings([self._format_label(f) for f in FORMAT_IDS])
        self.format_model = self.format_dd.get_model()
        self.format_dd.set_tooltip_text(
            "16-bit PNG: lossless, keeps transparency.\n"
            "10-bit AVIF: much smaller, keeps transparency; needs ffmpeg with libaom-av1.\n"
            "Ultra HDR JPEG: an ordinary sRGB JPEG every viewer shows, plus a gain map "
            "that HDR-capable viewers use; no transparency."
        )
        self.format_dd.update_property([Gtk.AccessibleProperty.LABEL], ["HDR file format"])
        self.format_dd.connect("notify::selected", self._on_format_changed)
        self.hdr_box.append(self.format_dd)

        self.transfer_dd = Gtk.DropDown.new_from_strings(list(hp.TRANSFERS.values()))
        self.transfer_dd.set_tooltip_text(
            "PQ (SMPTE ST 2084): absolute luminance, the HDR10 / photo standard.\n"
            "HLG (BT.2100): relative, broadcast style; always Rec. 2020 with a "
            "1000 cd/m² reference display."
        )
        self.transfer_dd.update_property([Gtk.AccessibleProperty.LABEL], ["Transfer function"])
        self.transfer_dd.connect("notify::selected", self._on_transfer_changed)
        self.hdr_box.append(self.transfer_dd)

        peak_box = Gtk.Box(spacing=6)
        self.peak = Gtk.SpinButton.new_with_range(*hp.PEAK_RANGE, 100)
        self.peak.set_tooltip_text(
            "Peak luminance: the brightest value written. Brighter HDR highlights "
            "are rolled off (ITU-R BT.2390). 1000 cd/m² suits most HDR displays."
        )
        self.peak.update_property([Gtk.AccessibleProperty.LABEL], ["Peak luminance in nits"])
        self.peak.connect("value-changed", self._on_peak_changed)
        peak_box.append(Gtk.Label(label="Peak", css_classes=["dim-label"]))
        peak_box.append(self.peak)
        peak_box.append(Gtk.Label(label="cd/m²", css_classes=["dim-label", "caption"]))
        self.hdr_box.append(peak_box)

        self.advanced_btn = Gtk.MenuButton(
            icon_name="emblem-system-symbolic",
            tooltip_text="Advanced HDR Output Settings",
            popover=self._build_advanced_popover(),
        )
        self.hdr_box.append(self.advanced_btn)
        row.append(self.hdr_box)
        self.append(row)

        self.status = Gtk.Label(xalign=0, wrap=True)
        self.status.add_css_class("caption")
        self.append(self.status)

        app.on_settings_changed(self.sync)
        self.sync()
        run_in_thread(codecs.avif_available, on_done=self._on_avif_probed, name="avif-probe")

    def shutdown(self) -> None:
        """Stop following the settings (the window is going away or rebuilt)."""
        self._shut = True
        self._flush_pending()
        self.app.off_settings_changed(self.sync)

    def _format_label(self, fmt: str) -> str:
        label = hp.FORMATS[fmt][0]
        if fmt == "avif" and self._avif is False:
            label += " (unavailable)"
        return label

    def _on_avif_probed(self, available: bool) -> None:
        if self._shut:
            return
        self._avif = available
        index = FORMAT_IDS.index("avif")
        self._syncing = True  # replacing the selected item may move the selection
        try:
            self.format_model.splice(index, 1, [self._format_label("avif")])
            self.format_dd.set_selected(FORMAT_IDS.index(self.app.settings.hdr_output_format))
        finally:
            self._syncing = False
        self._update_status()

    def _build_advanced_popover(self) -> Gtk.Popover:
        grid = Gtk.Grid(
            row_spacing=8,
            column_spacing=12,
            margin_top=8,
            margin_bottom=8,
            margin_start=8,
            margin_end=8,
        )
        self.primaries_dd = Gtk.DropDown.new_from_strings(list(hp.OUTPUT_PRIMARIES.values()))
        self.primaries_dd.update_property([Gtk.AccessibleProperty.LABEL], ["HDR colour space"])
        self.primaries_dd.connect("notify::selected", self._on_primaries_changed)
        self.white = Gtk.SpinButton.new_with_range(*hp.SDR_WHITE_RANGE, 1)
        self.white.update_property([Gtk.AccessibleProperty.LABEL], ["SDR white in nits"])
        self.white.connect("value-changed", self._on_white_changed)
        self.expand = Gtk.Switch(halign=Gtk.Align.START, valign=Gtk.Align.CENTER)
        self.expand.update_property([Gtk.AccessibleProperty.LABEL], ["Expand highlights"])
        self.expand.connect("notify::active", self._on_expand_changed)
        rows = (
            ("Colour space", self.primaries_dd, PRIMARIES_TIP),
            ("SDR white (cd/m²)", self.white, "Where SDR white (and 1.0 in EXR / .hdr files) "
             "is placed. 203 cd/m² is the ITU-R BT.2408 reference."),
            ("Expand highlights", self.expand, "SDR sources only: stretch the brightest tones "
             "up to 4× SDR white. A creative choice — it cannot recover clipped detail."),
        )  # fmt: skip
        for i, (label, widget, tip) in enumerate(rows):
            title = Gtk.Label(label=label, xalign=0, tooltip_text=tip)
            widget.set_tooltip_text(tip)
            grid.attach(title, 0, i, 1, 1)
            grid.attach(widget, 1, i, 1, 1)
        note = Gtk.Label(
            label="HDR files are checked after saving (and with ffprobe when installed). "
            "The stage shows an SDR rendition: this display path cannot show HDR.",
            wrap=True,
            max_width_chars=40,
            xalign=0,
        )
        note.add_css_class("caption")
        note.add_css_class("dim-label")
        grid.attach(note, 0, len(rows), 2, 1)
        return Gtk.Popover(child=grid)

    # --- settings <-> controls -------------------------------------------------
    def sync(self) -> None:
        s = self.app.settings
        self._syncing = True
        try:
            self.range_buttons[s.output_range].set_active(True)
            self.transfer_dd.set_selected(TRANSFER_IDS.index(s.hdr_transfer))
            self.format_dd.set_selected(FORMAT_IDS.index(s.hdr_output_format))
            self.peak.set_value(s.hdr_peak_nits)
            self.primaries_dd.set_selected(PRIMARY_IDS.index(s.hdr_output_primaries))
            self.white.set_value(s.hdr_sdr_white)
            self.expand.set_active(s.hdr_expand_highlights)
        finally:
            self._syncing = False
        restoring = s.mode == "restore"
        self.set_sensitive(not restoring)
        self.hdr_box.set_visible(s.output_range == "hdr" and not restoring)
        out = s.hdr_output()
        self.transfer_dd.set_visible(not out.gain_map)
        self.peak.set_sensitive(not out.hlg)
        self.peak.set_tooltip_text(
            "HLG uses its 1000 cd/m² reference display"
            if out.hlg
            else "Peak luminance: the brightest value written. Brighter HDR highlights "
            "are rolled off (ITU-R BT.2390). 1000 cd/m² suits most HDR displays."
        )
        fixed = out.hlg or out.gain_map
        self.primaries_dd.set_sensitive(not fixed)
        if fixed:
            self.primaries_dd.set_tooltip_text(
                "Ultra HDR JPEG uses an sRGB base" if out.gain_map else "HLG uses Rec. 2020"
            )
        else:
            self.primaries_dd.set_tooltip_text(PRIMARIES_TIP)
        self._update_status()

    def set_source(self, source_hdr: bool | None, gain_map: bool = False) -> None:
        """Tell the status line whether the selected photo is HDR (None: no photo)
        and whether it is a gain-map JPEG."""
        if (source_hdr, gain_map) != (self._source_hdr, self._gain_map):
            self._source_hdr, self._gain_map = source_hdr, gain_map
            self._update_status()

    def _update_status(self) -> None:
        s = self.app.settings
        out = s.hdr_output()
        restoring = s.mode == "restore"
        warn = False
        if restoring:
            text = "Photo restoration writes SDR files."
        else:
            parts = []
            if self._source_hdr is not None:
                parts.append(hp.conversion_status(self._source_hdr, out, self._gain_map))
                if self._source_hdr and not out.enabled:
                    parts.append(
                        "its gain map is not used" if self._gain_map
                        else "the HDR range will be discarded"
                    )  # fmt: skip
                    warn = True
            if out.enabled:
                parts.append(out.describe())
                if out.format == "avif" and self._avif is False:
                    parts.append("AVIF needs ffmpeg with libaom-av1 — choose 16-bit PNG")
                    warn = True
                if self._source_hdr is False:
                    parts.append("SDR conversion: no lost detail is recovered")
            text = " · ".join(parts)
        self.status.set_label(text)
        self.status.set_visible(bool(text))
        if warn:
            self.status.add_css_class("warning")
            self.status.remove_css_class("dim-label")
        else:
            self.status.remove_css_class("warning")
            self.status.add_css_class("dim-label")

    def _changed(self, key: str, value: object, debounce: bool = False) -> None:
        """Store a value; spin buttons (``debounce``) save once they settle,
        not on every step of a held arrow."""
        if self._syncing or getattr(self.app.settings, key) == value:
            return
        setattr(self.app.settings, key, value)
        self._cancel_pending()
        if not debounce:
            self.app.settings_changed()
            return
        self._update_status()

        def save() -> bool:
            self._save_source = 0
            self.app.settings_changed()
            return GLib.SOURCE_REMOVE

        self._save_source = GLib.timeout_add(SAVE_DELAY_MS, save)

    def _cancel_pending(self) -> bool:
        pending = bool(self._save_source)
        if pending:
            GLib.source_remove(self._save_source)
        self._save_source = 0
        return pending

    def _flush_pending(self) -> None:
        if self._cancel_pending():
            self.app.settings_changed()

    def _on_range_toggled(self, button: Gtk.ToggleButton, mode: str) -> None:
        if button.get_active():
            self._changed("output_range", mode)

    def _on_transfer_changed(self, *_args: object) -> None:
        index = self.transfer_dd.get_selected()
        if index < len(TRANSFER_IDS):
            self._changed("hdr_transfer", TRANSFER_IDS[index])

    def _on_format_changed(self, *_args: object) -> None:
        index = self.format_dd.get_selected()
        if index < len(FORMAT_IDS):
            self._changed("hdr_output_format", FORMAT_IDS[index])

    def _on_peak_changed(self, *_args: object) -> None:
        self._changed("hdr_peak_nits", int(self.peak.get_value()), debounce=True)

    def _on_primaries_changed(self, *_args: object) -> None:
        index = self.primaries_dd.get_selected()
        if index < len(PRIMARY_IDS):
            self._changed("hdr_output_primaries", PRIMARY_IDS[index])

    def _on_white_changed(self, *_args: object) -> None:
        self._changed("hdr_sdr_white", int(self.white.get_value()), debounce=True)

    def _on_expand_changed(self, *_args: object) -> None:
        self._changed("hdr_expand_highlights", self.expand.get_active())
