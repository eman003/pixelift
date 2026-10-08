"""Friendly error dialogs (never stack traces)."""

from __future__ import annotations

from gi.repository import Adw, Gtk

from pixelift.core.errors import UpscalerError


def show_error(parent: Gtk.Widget, error: UpscalerError) -> None:
    body = error.reason
    if error.suggestions:
        body += "\n\nTry:\n" + "\n".join(f"• {tip}" for tip in error.suggestions)
    dialog = Adw.AlertDialog(heading=error.title, body=body)
    dialog.add_response("log", "Open Log Folder")
    dialog.add_response("close", "Close")
    dialog.set_default_response("close")
    dialog.set_close_response("close")

    def on_response(_d: Adw.AlertDialog, response: str) -> None:
        if response == "log":
            app = parent.get_root().get_application() if parent.get_root() else None
            if app is not None:
                app.activate_action("open-log", None)

    dialog.connect("response", on_response)
    dialog.present(parent)
