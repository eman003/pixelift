"""Cooperative pause / cancel control shared between a job and its owner."""

from __future__ import annotations

import threading

from pixelift.core.errors import CancelledError


class JobControl:
    """Checked by long-running work between steps (e.g. between tiles)."""

    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._running = threading.Event()
        self._running.set()

    def cancel(self) -> None:
        self._cancelled.set()
        self._running.set()  # wake paused workers so they can exit

    def pause(self) -> None:
        if not self._cancelled.is_set():
            self._running.clear()

    def resume(self) -> None:
        self._running.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def paused(self) -> bool:
        return not self._running.is_set()

    def check(self) -> None:
        """Block while paused; raise CancelledError if cancelled."""
        self._running.wait()
        if self._cancelled.is_set():
            raise CancelledError()
