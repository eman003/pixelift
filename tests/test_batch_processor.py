from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from pixelift.core.batch_processor import (
    BatchProcessor,
    EventKind,
    ItemStatus,
    QueueItem,
)
from pixelift.core.image_processor import ProcessingOptions
from pixelift.core.upscaler import Upscaler


class FakeUpscaler(Upscaler):
    """Nearest-neighbour upscaler with an optional per-tile delay."""

    def __init__(self, delay: float = 0.0, tiles: int = 4) -> None:
        self.delay = delay
        self.tiles = tiles
        self.calls = 0
        self.released = 0

    def upscale(self, image, scale, model, *, progress=None, control=None):
        self.calls += 1
        for i in range(self.tiles):
            if control:
                control.check()
            time.sleep(self.delay)
            if progress:
                progress(i + 1, self.tiles)
        return np.repeat(np.repeat(image, scale, 0), scale, 1)

    def release_memory(self) -> None:
        self.released += 1


def make_items(image_factory, n: int) -> list[QueueItem]:
    return [QueueItem(image_factory(f"img{i}.png")) for i in range(n)]


def test_batch_processes_all(image_factory):
    items = make_items(image_factory, 3)
    events = []
    upscaler = FakeUpscaler()
    proc = BatchProcessor(upscaler, ProcessingOptions(scale=2), events.append)
    summary = proc.run(items)
    assert summary.done == 3 and summary.failed == 0
    assert all(i.status is ItemStatus.DONE for i in items)
    assert all(i.result.output_size == (80, 60) for i in items)
    kinds = [e.kind for e in events]
    assert kinds[0] is EventKind.BATCH_STARTED and kinds[-1] is EventKind.BATCH_FINISHED
    assert kinds.count(EventKind.ITEM_FINISHED) == 3
    assert events[-1].fraction == 1.0 and events[-1].completed == 3
    assert upscaler.released == 1


def test_failed_image_does_not_stop_batch(image_factory, tmp_path):
    items = make_items(image_factory, 2)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"nope")
    items.insert(1, QueueItem(bad))
    summary = BatchProcessor(FakeUpscaler(), ProcessingOptions(scale=2)).run(items)
    assert (summary.done, summary.failed) == (2, 1)
    assert items[1].status is ItemStatus.FAILED
    assert "bad.png" in items[1].error.reason


def test_skip_completed_items_and_existing_outputs(image_factory):
    items = make_items(image_factory, 2)
    up = FakeUpscaler()
    BatchProcessor(up, ProcessingOptions(scale=2)).run(items)
    assert up.calls == 2
    # Completed items are not re-run in the same queue.
    BatchProcessor(up, ProcessingOptions(scale=2)).run(items)
    assert up.calls == 2
    # New queue items whose outputs exist are skipped (default policy).
    fresh = [QueueItem(i.path) for i in items]
    summary = BatchProcessor(up, ProcessingOptions(scale=2)).run(fresh)
    assert summary.skipped == 2 and up.calls == 2


def test_cancel(image_factory):
    items = make_items(image_factory, 5)
    proc = BatchProcessor(FakeUpscaler(delay=0.05), ProcessingOptions(scale=2))
    proc.start(items)
    time.sleep(0.15)
    proc.cancel()
    assert proc.wait(10)
    statuses = [i.status for i in items]
    assert ItemStatus.CANCELLED in statuses
    assert ItemStatus.QUEUED not in statuses and ItemStatus.PROCESSING not in statuses
    assert proc.summary.cancelled >= 1
    # Nothing half-written for the cancelled item
    for item in items:
        if item.status is ItemStatus.CANCELLED:
            assert item.result is None


def test_pause_and_resume(image_factory):
    items = make_items(image_factory, 2)
    events = []
    proc = BatchProcessor(FakeUpscaler(delay=0.03), ProcessingOptions(scale=2), events.append)
    proc.start(items)
    time.sleep(0.05)
    proc.pause()
    assert proc.paused
    time.sleep(0.1)  # the tile in flight finishes; pause applies at tile boundaries
    snapshot = [(i.status, i.progress) for i in items]
    time.sleep(0.3)
    assert [(i.status, i.progress) for i in items] == snapshot  # no progress while paused
    proc.resume()
    assert proc.wait(10)
    assert all(i.status is ItemStatus.DONE for i in items)
    kinds = {e.kind for e in events}
    assert {EventKind.PAUSED, EventKind.RESUMED} <= kinds


def test_retry_failed(image_factory, tmp_path):
    path = tmp_path / "later.png"
    path.write_bytes(b"not yet an image")
    item = QueueItem(path)
    up = FakeUpscaler()
    BatchProcessor(up, ProcessingOptions(scale=2)).run([item])
    assert item.status is ItemStatus.FAILED
    image_factory("later.png").replace(path)  # fix the file, then retry
    BatchProcessor(up, ProcessingOptions(scale=2)).run([item])
    assert item.status is ItemStatus.DONE and item.error is None


def test_bounded_concurrency(image_factory):
    active = 0
    peak = 0
    lock = threading.Lock()

    class Counting(FakeUpscaler):
        def upscale(self, image, scale, model, **kw):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.05)
                return super().upscale(image, scale, model, **kw)
            finally:
                with lock:
                    active -= 1

    items = make_items(image_factory, 6)
    BatchProcessor(Counting(), ProcessingOptions(scale=2), workers=2).run(items)
    assert peak == 2


def test_listener_errors_do_not_break_batch(image_factory):
    def broken(_event):
        raise RuntimeError("listener bug")

    summary = BatchProcessor(FakeUpscaler(), ProcessingOptions(scale=2), broken).run(
        make_items(image_factory, 1)
    )
    assert summary.done == 1


def test_cannot_start_twice(image_factory):
    proc = BatchProcessor(FakeUpscaler(delay=0.05), ProcessingOptions(scale=2))
    proc.start(make_items(image_factory, 2))
    with pytest.raises(RuntimeError):
        proc.start([])
    proc.cancel()
    proc.wait(10)


@pytest.mark.parametrize("workers", [1, 3])
def test_same_stem_sources_get_distinct_outputs(image_factory, workers):
    items = [QueueItem(image_factory("photo.jpg")), QueueItem(image_factory("photo.png"))]
    proc = BatchProcessor(FakeUpscaler(), ProcessingOptions(scale=2), workers=workers)
    summary = proc.run(items)
    assert summary.done == 2 and summary.skipped == 0
    # The earlier item in the queue always wins the plain name.
    assert [i.result.output.name for i in items] == ["photo_2x.png", "photo_2x (2).png"]

    # A rerun skips both, each against its own output.
    again = [QueueItem(i.path) for i in items]
    summary = BatchProcessor(FakeUpscaler(), ProcessingOptions(scale=2), workers=workers).run(again)
    assert summary.skipped == 2
    assert [i.result.output for i in again] == [i.result.output for i in items]


def test_output_names_follow_queue_order_not_job_timing(image_factory, monkeypatch):
    """photo.png's job reaches its output first, but photo.jpg is first in the queue."""
    from pixelift.utils import image_utils as iu

    probe = iu.probe_image

    def slow_jpg_probe(path):
        if path.suffix == ".jpg":
            time.sleep(0.2)
        return probe(path)

    monkeypatch.setattr(iu, "probe_image", slow_jpg_probe)  # the job's probe only
    items = [QueueItem(image_factory("photo.jpg")), QueueItem(image_factory("photo.png"))]
    BatchProcessor(FakeUpscaler(), ProcessingOptions(scale=2), workers=2).run(items)
    assert [i.result.output.name for i in items] == ["photo_2x.png", "photo_2x (2).png"]


def test_finished_items_outside_batch_keep_their_output(image_factory):
    first = QueueItem(image_factory("photo.jpg"))
    BatchProcessor(FakeUpscaler(), ProcessingOptions(scale=2)).run([first])
    second = QueueItem(image_factory("photo.png"))
    proc = BatchProcessor(FakeUpscaler(), ProcessingOptions(scale=2))
    proc.start([second], others=[first, second])
    proc.wait()
    assert second.status is ItemStatus.DONE
    assert second.result.output.name == "photo_2x (2).png"
