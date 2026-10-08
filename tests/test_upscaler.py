from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from pixelift.core import device_manager as dm
from pixelift.core.control import JobControl
from pixelift.core.errors import CancelledError, ModelNotInstalledError
from pixelift.core.tiling import auto_tile_size, paste_tile, plan_tiles, tile_starts
from pixelift.core.upscaler import TorchUpscaler


def rgb(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


# --- dimensions -------------------------------------------------------------------
@pytest.mark.parametrize(
    ("model", "scale"),
    [("test-x4", 4), ("test-x2", 2), ("test-x4", 2), ("test-family", 2), ("test-family", 4)],
)
def test_output_dimensions(upscaler, model, scale):
    out = upscaler.upscale(rgb(23, 37), scale, model)
    assert out.shape == (23 * scale, 37 * scale, 3)
    assert out.dtype == np.uint8


def test_family_picks_native_variant(manager):
    assert manager.resolve("test-family", 2).id == "test-x2"
    assert manager.resolve("test-family", 4).id == "test-x4"


def test_odd_sizes_with_pixel_unshuffle_model(upscaler):
    # x2 RRDBNet needs even dimensions internally; odd inputs must be padded.
    out = upscaler.upscale(rgb(1, 1), 2, "test-x2")
    assert out.shape == (2, 2, 3)
    out = upscaler.upscale(rgb(65, 33), 2, "test-x2")
    assert out.shape == (130, 66, 3)


def test_rejects_bad_input(upscaler):
    with pytest.raises(ValueError):
        upscaler.upscale(np.zeros((4, 4), np.uint8), 4, "test-x4")
    with pytest.raises(ValueError):
        upscaler.upscale(np.zeros((4, 4, 3), np.float32), 4, "test-x4")


def test_model_loaded_once_and_reused(upscaler, monkeypatch):
    calls = []
    original = torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: calls.append(1) or original(*a, **k))
    upscaler.upscale(rgb(8, 8), 4, "test-x4")
    upscaler.upscale(rgb(8, 8, 1), 4, "test-x4")
    assert len(calls) == 1
    upscaler.release()
    upscaler.upscale(rgb(8, 8), 4, "test-x4")
    assert len(calls) == 2


def test_missing_model(manager):
    manager.remove("test-x4")
    up = TorchUpscaler(manager)
    with pytest.raises(ModelNotInstalledError):
        up.upscale(rgb(8, 8), 4, "test-x4")


def test_damaged_model_file(manager, upscaler):
    manager.path_for("test-x4").write_bytes(b"garbage")
    with pytest.raises(ModelNotInstalledError, match="damaged"):
        upscaler.upscale(rgb(8, 8), 4, "test-x4")


# --- tiling -------------------------------------------------------------------------
def test_tile_starts_cover_and_overlap():
    for length in (1, 63, 64, 65, 500, 1000, 1023):
        for tile in (64, 128, 512):
            starts = tile_starts(length, tile, 16)
            assert starts[0] == 0
            assert min(starts[-1] + tile, length) == length
            for a, b in itertools.pairwise(starts):
                assert b - a <= tile - min(16, tile // 2)  # overlap respected


def test_plan_tiles_records_overlaps():
    tiles = plan_tiles(300, 300, 128, 16)
    assert tiles[0].overlap_top == 0 and tiles[0].overlap_left == 0
    assert all(t.overlap_left >= 16 for t in tiles if t.x0 > 0)
    assert all(t.overlap_top >= 16 for t in tiles if t.y0 > 0)


def test_paste_tile_feathers_seam():
    out = np.zeros((10, 20, 3), np.uint8)
    paste_tile(out, np.full((10, 12, 3), 0, np.uint8), 0, 0, 0, 0)
    paste_tile(out, np.full((10, 12, 3), 200, np.uint8), 0, 8, 0, 4)
    row = out[5, :, 0]
    assert row[7] == 0 and row[12] == 200
    assert 0 < row[8] < row[9] < row[10] < row[11] < 200  # smooth ramp


class NearestX4(nn.Module):
    """Deterministic stand-in network so tiled output must equal untiled output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=4, mode="nearest")


def _nearest_upscaler(manager, tile: int) -> TorchUpscaler:
    up = TorchUpscaler(manager, tile_size=tile)
    up._nets["test-x4"] = NearestX4()
    return up


def test_tiled_equals_untiled(manager):
    image = rgb(150, 210)
    whole = _nearest_upscaler(manager, 1024).upscale(image, 4, "test-x4")
    tiled = _nearest_upscaler(manager, 64).upscale(image, 4, "test-x4")
    assert np.array_equal(whole, np.repeat(np.repeat(image, 4, 0), 4, 1))
    assert np.abs(whole.astype(int) - tiled.astype(int)).max() <= 1


def test_tiled_real_network_is_seamless(upscaler):
    image = rgb(96, 96, seed=3)
    image = (image // 64 * 64).astype(np.uint8)  # blocky = smooth regions
    upscaler.tile_size = 1024
    whole = upscaler.upscale(image, 4, "test-x4").astype(np.float32)
    upscaler.tile_size = 64
    tiled = upscaler.upscale(image, 4, "test-x4").astype(np.float32)
    assert np.abs(whole - tiled).mean() < 2.0


def test_auto_tile_size():
    assert auto_tile_size(4000, 4000, 16_000, 32 * 1024**3) == 1024
    assert auto_tile_size(4000, 4000, 16_000, 1 * 1024**3) == 256
    assert auto_tile_size(4000, 4000, 16_000, 1) == 64
    assert auto_tile_size(4000, 4000, 16_000, 32 * 1024**3, cap=512) == 512
    assert auto_tile_size(100, 100, 16_000, 200_000_000) == 1024  # whole image fits


def test_progress_and_cancel(manager):
    up = _nearest_upscaler(manager, 64)
    seen = []
    up.upscale(rgb(130, 130), 4, "test-x4", progress=lambda d, t: seen.append((d, t)))
    assert seen[0][0] == 0 and seen[-1][0] == seen[-1][1] > 1

    control = JobControl()

    def cancel_after_first(done: int, _total: int) -> None:
        if done == 1:
            control.cancel()

    with pytest.raises(CancelledError):
        up.upscale(rgb(130, 130), 4, "test-x4", progress=cancel_after_first, control=control)


# --- memory & device recovery -------------------------------------------------------
class OOMAbove(nn.Module):
    """Raises an allocator OOM for inputs larger than ``limit`` pixels per side."""

    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit
        self.sizes: list[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.sizes.append(x.shape[-1])
        if max(x.shape[-2:]) > self.limit:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return F.interpolate(x, scale_factor=4, mode="nearest")


def test_recovers_from_oom_by_shrinking_tiles(manager):
    up = TorchUpscaler(manager, tile_size=512)
    net = OOMAbove(limit=128)
    up._nets["test-x4"] = net
    out = up.upscale(rgb(300, 300), 4, "test-x4")
    assert out.shape == (1200, 1200, 3)
    assert max(net.sizes) == 300 and net.sizes[-1] <= 128


def test_cpu_oom_at_min_tile_is_friendly(manager):
    from pixelift.core.errors import OutOfMemoryError

    up = TorchUpscaler(manager, tile_size=128)
    up._nets["test-x4"] = OOMAbove(limit=0)
    with pytest.raises(OutOfMemoryError):
        up.upscale(rgb(100, 100), 4, "test-x4")


def test_gpu_failure_falls_back_to_cpu(manager):
    """A GPU device that cannot run (no CUDA here, or a broken driver) falls back."""
    if torch.cuda.is_available():
        pytest.skip("needs a machine where cuda:0 is unusable")
    fake_gpu = dm.DeviceInfo("cuda:0", "Fake GPU", "CUDA", 4 * 1024**3)
    events = []
    up = TorchUpscaler(
        manager,
        fake_gpu,
        tile_size=64,
        on_device_change=lambda dev, why: events.append((dev.id, why)),
    )
    out = up.upscale(rgb(20, 20), 4, "test-x4")
    assert out.shape == (80, 80, 3)
    assert up.device.id == "cpu"
    assert events and events[0][0] == "cpu"


def test_no_fallback_when_disabled(manager):
    from pixelift.core.errors import DeviceError

    if torch.cuda.is_available():
        pytest.skip("needs a machine where cuda:0 is unusable")
    fake_gpu = dm.DeviceInfo("cuda:0", "Fake GPU", "CUDA", 4 * 1024**3)
    up = TorchUpscaler(manager, fake_gpu, tile_size=64, cpu_fallback=False)
    with pytest.raises(DeviceError):
        up.upscale(rgb(20, 20), 4, "test-x4")


@pytest.mark.gpu
def test_gpu_upscale(manager):
    report = dm.detect_devices()
    if not report.best.is_gpu:
        pytest.skip("no GPU available")
    up = TorchUpscaler(manager, report.best)
    assert up.upscale(rgb(64, 64), 4, "test-x4").shape == (256, 256, 3)
    assert up.device.is_gpu
