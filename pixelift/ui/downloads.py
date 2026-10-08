"""App-wide model download tracking, so progress survives closing dialogs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pixelift.core.control import JobControl
from pixelift.core.errors import CancelledError, UpscalerError, friendly_error
from pixelift.core.model_manager import ModelManager
from pixelift.ui.async_utils import idle, run_in_thread


@dataclass
class DownloadState:
    model_id: str
    done: int = 0
    total: int = 0
    active: bool = True
    error: UpscalerError | None = None
    control: JobControl | None = None

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0


Listener = Callable[[str, DownloadState | None], None]


class DownloadTracker:
    def __init__(self, manager: ModelManager) -> None:
        self.manager = manager
        self.states: dict[str, DownloadState] = {}
        self._listeners: list[Listener] = []

    def subscribe(self, listener: Listener) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    def _notify(self, model_id: str) -> None:
        state = self.states.get(model_id)
        for listener in list(self._listeners):
            listener(model_id, state)

    def is_active(self, model_id: str) -> bool:
        state = self.states.get(model_id)
        return bool(state and state.active)

    def start(self, model_id: str) -> None:
        if self.is_active(model_id):
            return
        state = DownloadState(model_id, control=JobControl())
        self.states[model_id] = state

        def progress(done: int, total: int) -> None:
            state.done, state.total = done, total
            idle(self._notify, model_id)

        def finished(_path: object) -> None:
            self.states.pop(model_id, None)
            self._notify(model_id)

        def failed(exc: BaseException) -> None:
            state.active = False
            if isinstance(exc, CancelledError):
                self.states.pop(model_id, None)
            else:
                state.error = friendly_error(exc)
            self._notify(model_id)

        run_in_thread(
            self.manager.download,
            model_id,
            progress,
            state.control,
            on_done=finished,
            on_error=failed,
            name=f"download-{model_id}",
        )
        self._notify(model_id)

    def cancel(self, model_id: str) -> None:
        state = self.states.get(model_id)
        if state and state.control:
            state.control.cancel()

    def remove(self, model_id: str) -> None:
        self.manager.remove(model_id)
        self.states.pop(model_id, None)
        self._notify(model_id)
