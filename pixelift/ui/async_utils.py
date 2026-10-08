"""Run blocking work off the GTK main thread and deliver results back to it."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from gi.repository import GLib

log = logging.getLogger(__name__)


def idle(func: Callable[..., Any], *args: Any) -> None:
    """Call ``func(*args)`` once on the main loop (safe from any thread)."""

    def _once() -> bool:
        try:
            func(*args)
        except Exception:
            log.exception("Main-loop callback failed")
        return GLib.SOURCE_REMOVE

    GLib.idle_add(_once)


def run_in_thread(
    func: Callable[..., Any],
    *args: Any,
    on_done: Callable[[Any], None] | None = None,
    on_error: Callable[[BaseException], None] | None = None,
    name: str = "worker",
) -> threading.Thread:
    def worker() -> None:
        try:
            result = func(*args)
        except BaseException as exc:
            log.debug("Background task %s failed", name, exc_info=True)
            if on_error:
                idle(on_error, exc)
            return
        if on_done:
            idle(on_done, result)

    thread = threading.Thread(target=worker, name=name, daemon=True)
    thread.start()
    return thread
