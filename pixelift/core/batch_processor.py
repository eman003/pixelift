"""Background batch processing with a bounded worker pool.

The processor is GUI-agnostic: it reports through a listener callable that is
invoked on worker threads. The GTK layer forwards events to the main loop with
``GLib.idle_add``; the CLI prints them directly.
"""

from __future__ import annotations

import enum
import itertools
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from pixelift.core.control import JobControl
from pixelift.core.errors import CancelledError, UpscalerError, friendly_error
from pixelift.core.image_processor import (
    ProcessingOptions,
    ProcessResult,
    output_path_for,
    process_image,
)
from pixelift.core.upscaler import Upscaler
from pixelift.utils.image_utils import probe_image

log = logging.getLogger(__name__)


class ItemStatus(enum.Enum):
    PENDING = "Ready"
    QUEUED = "Queued"
    PROCESSING = "Processing"
    DONE = "Done"
    SKIPPED = "Skipped"
    FAILED = "Failed"
    CANCELLED = "Cancelled"

    @property
    def finished(self) -> bool:
        return self in (
            ItemStatus.DONE,
            ItemStatus.SKIPPED,
            ItemStatus.FAILED,
            ItemStatus.CANCELLED,
        )


_ids = itertools.count(1)


@dataclass(eq=False)
class QueueItem:
    path: Path
    id: int = field(default_factory=lambda: next(_ids))
    status: ItemStatus = ItemStatus.PENDING
    progress: float = 0.0
    stage: str = ""
    result: ProcessResult | None = None
    error: UpscalerError | None = None

    def reset(self) -> None:
        self.status = ItemStatus.PENDING
        self.progress = 0.0
        self.stage = ""
        self.result = None
        self.error = None


class EventKind(enum.Enum):
    BATCH_STARTED = "batch_started"
    ITEM_STARTED = "item_started"
    ITEM_PROGRESS = "item_progress"
    ITEM_FINISHED = "item_finished"
    PAUSED = "paused"
    RESUMED = "resumed"
    DEVICE_CHANGED = "device_changed"
    BATCH_FINISHED = "batch_finished"


@dataclass
class BatchEvent:
    kind: EventKind
    item: QueueItem | None = None
    completed: int = 0
    total: int = 0
    fraction: float = 0.0  # overall progress 0..1
    message: str = ""


@dataclass
class BatchSummary:
    total: int
    done: int
    skipped: int
    failed: int
    cancelled: int
    seconds: float


Listener = Callable[[BatchEvent], None]


class BatchProcessor:
    """Runs ``process_image`` over a list of items with ``workers`` threads."""

    def __init__(
        self,
        upscaler: Upscaler,
        options: ProcessingOptions,
        listener: Listener | None = None,
        workers: int = 1,
        progress_interval: float = 0.1,
    ) -> None:
        self.upscaler = upscaler
        self.options = options
        self.listener = listener or (lambda _e: None)
        self.workers = max(1, workers)
        self.control = JobControl()
        self.progress_interval = progress_interval
        self._items: list[QueueItem] = []
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._finished = threading.Event()
        self._started_at = 0.0
        self._remaining = 0
        self._claims: dict[Path, Path] = {}  # output -> source, guarded by _lock
        self._planned = False  # guarded by _lock
        self.summary: BatchSummary | None = None

    # --- control -----------------------------------------------------------
    @property
    def running(self) -> bool:
        return bool(self._threads) and not self._finished.is_set()

    @property
    def paused(self) -> bool:
        return self.control.paused

    def pause(self) -> None:
        if self.running and not self.control.paused:
            self.control.pause()
            self._emit(EventKind.PAUSED)

    def resume(self) -> None:
        if self.control.paused:
            self.control.resume()
            self._emit(EventKind.RESUMED)

    def cancel(self) -> None:
        self.control.cancel()

    def wait(self, timeout: float | None = None) -> bool:
        return self._finished.wait(timeout)

    # --- running -----------------------------------------------------------
    def start(self, items: list[QueueItem], others: Iterable[QueueItem] = ()) -> None:
        """Process ``items`` in background threads (returns immediately).

        Items that are already DONE or SKIPPED are left untouched. Their outputs,
        and those of ``others`` (finished items outside this batch), are never
        reused for a different source file.
        """
        if self.running:
            raise RuntimeError("batch already running")
        self._items = [i for i in items if i.status not in (ItemStatus.DONE, ItemStatus.SKIPPED)]
        self._claims = {
            _key(i.result.output): _key(i.path)
            for i in itertools.chain(items, others)
            if i.status in (ItemStatus.DONE, ItemStatus.SKIPPED) and i.result is not None
        }
        self._planned = False
        self._finished.clear()
        self._started_at = time.monotonic()
        work: queue.SimpleQueue[QueueItem] = queue.SimpleQueue()
        for item in self._items:
            item.reset()
            item.status = ItemStatus.QUEUED
            work.put(item)
        self._remaining = self.workers
        self._emit(EventKind.BATCH_STARTED)
        self._threads = [
            threading.Thread(target=self._worker, args=(work,), name=f"upscale-{n}", daemon=True)
            for n in range(self.workers)
        ]
        for thread in self._threads:
            thread.start()

    def run(self, items: list[QueueItem]) -> BatchSummary:
        """Blocking variant used by the CLI and tests."""
        self.start(items)
        self.wait()
        assert self.summary is not None
        return self.summary

    def _worker(self, work: queue.SimpleQueue[QueueItem]) -> None:
        try:
            self._plan_outputs()
            while True:
                try:
                    item = work.get_nowait()
                except queue.Empty:
                    break
                if self.control.cancelled:
                    item.status = ItemStatus.CANCELLED
                    self._emit(EventKind.ITEM_FINISHED, item)
                    continue
                self._process(item)
        finally:
            with self._lock:
                self._remaining -= 1
                last = self._remaining == 0
            if last:
                self._finish()

    def _process(self, item: QueueItem) -> None:
        item.status = ItemStatus.PROCESSING
        item.stage = "Starting"
        self._emit(EventKind.ITEM_STARTED, item)
        last_emit = 0.0

        def on_progress(fraction: float, stage: str) -> None:
            nonlocal last_emit
            item.progress = fraction
            item.stage = stage
            now = time.monotonic()
            if now - last_emit >= self.progress_interval or fraction >= 1.0:
                last_emit = now
                self._emit(EventKind.ITEM_PROGRESS, item)

        try:
            # Wait here (not mid-file) if paused before the job begins.
            self.control.check()
            result = process_image(
                item.path,
                self.options,
                self.upscaler,
                on_progress,
                self.control,
                claim=lambda output: self._claim(output, item.path),
            )
            item.result = result
            item.status = ItemStatus.SKIPPED if result.skipped else ItemStatus.DONE
            item.progress = 1.0
            item.stage = result.note
        except CancelledError:
            item.status = ItemStatus.CANCELLED
            item.stage = ""
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
            error = friendly_error(exc)
            log.warning("Failed %s: %s", item.path, error.reason)
            item.error = error
            item.status = ItemStatus.FAILED
            item.stage = error.reason
        self._emit(EventKind.ITEM_FINISHED, item)

    def _plan_outputs(self) -> None:
        """Claim each item's preferred output name in queue order.

        Done once, before any job starts, so when two sources map to the same
        name the earlier one always gets it, however the workers interleave.
        Runs on a worker thread because probing touches every file.
        """
        with self._lock:
            if self._planned:
                return
            self._planned = True
            for item in self._items:
                if self.control.cancelled:
                    return
                try:
                    info = probe_image(item.path)
                    output = output_path_for(item.path, info.width, info.height, self.options)
                except Exception:  # noqa: BLE001 - the job reports it properly
                    continue
                self._claims.setdefault(_key(output), _key(item.path))

    def _claim(self, output: Path, source: Path) -> bool:
        output, source = _key(output), _key(source)
        with self._lock:
            return self._claims.setdefault(output, source) == source

    def _finish(self) -> None:
        items = self._items
        for item in items:  # never leave items stuck in "Queued"
            if item.status is ItemStatus.QUEUED:
                item.status = ItemStatus.CANCELLED
        counts = {s: sum(1 for i in items if i.status is s) for s in ItemStatus}
        self.summary = BatchSummary(
            total=len(items),
            done=counts[ItemStatus.DONE],
            skipped=counts[ItemStatus.SKIPPED],
            failed=counts[ItemStatus.FAILED],
            cancelled=counts[ItemStatus.CANCELLED],
            seconds=time.monotonic() - self._started_at,
        )
        try:
            self.upscaler.release_memory()
        except Exception:
            log.debug("release_memory failed", exc_info=True)
        self._finished.set()
        self._emit(EventKind.BATCH_FINISHED)

    # --- events ------------------------------------------------------------
    def overall(self) -> tuple[int, int, float]:
        items = self._items
        total = len(items)
        completed = sum(1 for i in items if i.status.finished)
        partial = sum(i.progress for i in items if i.status is ItemStatus.PROCESSING)
        fraction = (completed + partial) / total if total else 1.0
        return completed, total, fraction

    def notify_device_change(self, message: str) -> None:
        self._emit(EventKind.DEVICE_CHANGED, message=message)

    def _emit(self, kind: EventKind, item: QueueItem | None = None, message: str = "") -> None:
        completed, total, fraction = self.overall()
        try:
            self.listener(BatchEvent(kind, item, completed, total, fraction, message))
        except Exception:
            log.exception("Batch listener failed")


def _key(path: Path) -> Path:
    return Path(path).resolve()
