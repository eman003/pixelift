"""AI upscaling engine.

``Upscaler`` is the abstraction the rest of the app depends on; it knows
nothing about GTK, files or settings. ``TorchUpscaler`` implements it with
PyTorch, tiled inference, OOM recovery and automatic CPU fallback.
"""

from __future__ import annotations

import gc
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from pixelift.core import device_manager as dm
from pixelift.core.control import JobControl
from pixelift.core.errors import (
    CancelledError,
    DeviceError,
    ImageTooLargeError,
    ModelNotInstalledError,
    OutOfMemoryError,
    UpscalerError,
    gpu_oom_error,
    is_oom,
)
from pixelift.core.model_manager import ModelManager
from pixelift.core.tiling import (
    MIN_TILE,
    auto_tile_size,
    paste_tile,
    plan_tiles,
    smaller_tile,
)
from pixelift.models.base import ModelSpec, extract_state_dict

if TYPE_CHECKING:
    from torch import nn

log = logging.getLogger(__name__)

TileProgress = Callable[[int, int], None]  # (tiles_done, tiles_total)
T = TypeVar("T")

CPU_TILE_CAP = 512
DEFAULT_OVERLAP = 32

# CUDA devices whose per-process memory fraction we lowered. The setting is
# process-wide, so it must be undone explicitly when the limit is removed.
_limited_cuda_devices: set[str] = set()


def _uses_half(device: dm.DeviceInfo) -> bool:
    return device.kind in ("cuda", "xpu")


class _DeviceChangedError(Exception):
    """The upscaler moved to another device before an attempt could start."""


class Upscaler(ABC):
    """Upscale an RGB ``uint8`` array of shape (H, W, 3)."""

    @abstractmethod
    def upscale(
        self,
        image: np.ndarray,
        scale: int,
        model: str,
        *,
        progress: TileProgress | None = None,
        control: JobControl | None = None,
    ) -> np.ndarray: ...

    def release(self) -> None:  # noqa: B027 - optional hook
        """Free loaded models and cached memory."""

    def release_memory(self) -> None:  # noqa: B027 - optional hook
        """Free cached working memory between batches; keep models loaded."""

    @property
    def device_label(self) -> str:
        return "CPU"


class TorchUpscaler(Upscaler):
    def __init__(
        self,
        model_manager: ModelManager,
        device: dm.DeviceInfo = dm.CPU_DEVICE,
        tile_size: int = 0,
        memory_limit_mb: int = 0,
        cpu_threads: int = 0,
        cpu_fallback: bool = True,
        on_device_change: Callable[[dm.DeviceInfo, str], None] | None = None,
    ) -> None:
        import torch

        self.models = model_manager
        self.device = device
        self.tile_size = tile_size  # 0 = automatic
        self.memory_limit_mb = memory_limit_mb
        self.cpu_fallback = cpu_fallback
        self.on_device_change = on_device_change
        self._nets: dict[str, nn.Module] = {}
        self._lock = threading.RLock()
        if cpu_threads > 0:
            torch.set_num_threads(cpu_threads)
        self._apply_memory_limit()

    # --- properties --------------------------------------------------------
    @property
    def device_label(self) -> str:
        return self.device.label()

    @property
    def half(self) -> bool:
        return _uses_half(self.device)

    def _apply_memory_limit(self) -> None:
        if self.device.kind != "cuda":
            return
        import torch

        device_id = self.device.id
        if self.memory_limit_mb <= 0 or not self.device.total_memory:
            if device_id in _limited_cuda_devices:
                torch.cuda.set_per_process_memory_fraction(1.0, torch.device(device_id))
                _limited_cuda_devices.discard(device_id)
                log.info("Removed the memory limit on %s", device_id)
            return
        fraction = min(1.0, self.memory_limit_mb * 1024**2 / self.device.total_memory)
        torch.cuda.set_per_process_memory_fraction(fraction, torch.device(device_id))
        _limited_cuda_devices.add(device_id)
        log.info("Limiting %s to %.0f%% of its memory", device_id, fraction * 100)

    # --- model loading -----------------------------------------------------
    def load(self, spec: ModelSpec) -> nn.Module:
        """Load ``spec`` onto the device once; later calls reuse it.

        Cached networks always live on ``self.device``: ``_switch_to_cpu``
        clears the cache under the same lock when it changes the device.
        """
        import torch

        with self._lock:
            net = self._nets.get(spec.id)
            if net is not None:
                return net
            path = self.models.path_for(spec)
            if not path.is_file():
                raise ModelNotInstalledError(f"{spec.name} is not installed.")
            log.info("Loading model %s on %s", spec.id, self.device.id)
            try:
                with torch.serialization.safe_globals(list(spec.safe_globals)):
                    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                state = extract_state_dict(checkpoint, spec.state_key)
                if spec.convert is not None:
                    state = spec.convert(state)
                net = spec.build()
                net.load_state_dict(state, strict=True)
                del state
            except Exception as exc:
                log.exception("Failed to load %s", path)
                raise ModelNotInstalledError(
                    f"The file for {spec.name} is damaged or incompatible.",
                    ["Remove the model in Settings → Models and download it again"],
                    title="Unable to load AI model.",
                ) from exc
            del checkpoint
            net.eval().requires_grad_(False)
            net = net.to(dm.to_torch(self.device))
            if self.half:
                net = net.half()
            self._nets[spec.id] = net
            return net

    def release(self) -> None:
        with self._lock:
            self._nets.clear()
        gc.collect()
        dm.empty_cache(self.device)

    def release_memory(self) -> None:
        gc.collect()
        dm.empty_cache(self.device)

    def _switch_to_cpu(self, failed: dm.DeviceInfo, reason: str) -> None:
        with self._lock:
            if self.device != failed:
                return  # another job already switched away from this device
            log.warning("Falling back to CPU: %s", reason)
            self._nets.clear()
            dm.empty_cache(self.device)
            self.device = dm.CPU_DEVICE
        if self.on_device_change:
            self.on_device_change(self.device, reason)

    # --- inference ---------------------------------------------------------
    def run_model(
        self,
        spec: ModelSpec,
        fn: Callable[[nn.Module, dm.DeviceInfo], T],
        control: JobControl | None = None,
        release: bool = True,
    ) -> T:
        """Run ``fn(net, device)`` with ``spec`` loaded, for non-tiled networks.

        Shares the upscaling engine's model cache and its error handling: on a
        GPU out-of-memory or driver error the work is retried on the CPU (if
        CPU fallback is enabled), and every model then moves to the CPU too.
        ``release=False`` keeps the GPU memory cache for a run of calls (the
        caller then calls :meth:`release_memory` once at the end).
        """
        while True:
            if control is not None:
                control.check()
            with self._lock:
                device = self.device
                net = self.load(spec)
            try:
                return fn(net, device)
            except Exception as exc:
                if not device.is_gpu:
                    if is_oom(exc):
                        raise OutOfMemoryError(
                            f"The computer ran out of memory while running {spec.name}.",
                            ["Close other applications", "Use a smaller image"],
                        ) from exc
                    raise
                if isinstance(exc, (UpscalerError, CancelledError)):
                    raise
                oom = is_oom(exc)
                if not oom:
                    log.exception("GPU inference failed (%s)", spec.id)
                if not self.cpu_fallback:
                    if oom:
                        raise gpu_oom_error() from exc
                    raise DeviceError(
                        f"The GPU reported an error: {str(exc).splitlines()[0][:200]}",
                        ["Switch the processing device to CPU in Settings"],
                    ) from exc
                gc.collect()
                dm.empty_cache(device)
                self._switch_to_cpu(device, "GPU ran out of memory" if oom else "GPU error")
            finally:
                if release and device.is_gpu:
                    dm.empty_cache(device)

    def choose_tile_size(
        self, spec: ModelSpec, height: int, width: int, device: dm.DeviceInfo
    ) -> int:
        if self.tile_size > 0:
            return self.tile_size
        bytes_per_px = spec.memory_per_pixel * (0.5 if _uses_half(device) else 1.0)
        if device.is_gpu:
            budget = int(dm.free_memory(device) * 0.6)
            cap = 1024
        else:
            budget = int(min(dm.free_memory(device) * 0.3, 3 * 1024**3))
            cap = CPU_TILE_CAP
        return auto_tile_size(height, width, bytes_per_px, budget, cap)

    def upscale(
        self,
        image: np.ndarray,
        scale: int,
        model: str,
        *,
        progress: TileProgress | None = None,
        control: JobControl | None = None,
    ) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("expected an RGB uint8 array of shape (H, W, 3)")
        if scale < 1:
            raise ValueError("scale must be >= 1")
        spec = self.models.resolve(model, scale)
        height, width = image.shape[:2]
        try:
            out = np.empty((height * scale, width * scale, 3), dtype=np.uint8)
        except (MemoryError, ValueError) as exc:
            raise ImageTooLargeError(
                "There is not enough memory to hold the upscaled image.",
                ["Use 2× instead of 4×", "Close other applications"],
            ) from exc

        tile = 0
        tile_device: dm.DeviceInfo | None = None
        while True:
            # Other jobs share this upscaler and may switch it to the CPU at any
            # moment, so each attempt uses one consistent device + network pair,
            # with a tile sized for that device.
            with self._lock:
                device = self.device
            if device != tile_device:
                tile = self.choose_tile_size(spec, height, width, device)
                tile_device = device
            try:
                self._run_tiled(spec, device, image, out, scale, tile, progress, control)
                return out
            except _DeviceChangedError:
                continue
            except Exception as exc:
                if not device.is_gpu and not is_oom(exc):
                    raise
                if is_oom(exc):
                    gc.collect()
                    dm.empty_cache(device)
                    if tile > MIN_TILE:
                        new_tile = smaller_tile(tile)
                        log.warning("Out of memory at tile %d; retrying with %d", tile, new_tile)
                        tile = new_tile
                        continue
                    if not device.is_gpu:
                        raise OutOfMemoryError(
                            "The computer ran out of memory even with the smallest tile size.",
                            ["Close other applications", "Use a smaller image"],
                        ) from exc
                    if not self.cpu_fallback:
                        raise gpu_oom_error() from exc
                    self._switch_to_cpu(device, "GPU ran out of memory")
                elif not isinstance(exc, (UpscalerError, CancelledError)):
                    # Driver / kernel / runtime failures on the GPU (not OOM).
                    log.exception("GPU inference failed")
                    if not self.cpu_fallback:
                        raise DeviceError(
                            f"The GPU reported an error: {str(exc).splitlines()[0][:200]}",
                            ["Switch the processing device to CPU in Settings"],
                        ) from exc
                    self._switch_to_cpu(device, "GPU error")
                else:
                    raise
            finally:
                if device.is_gpu:
                    dm.empty_cache(device)

    def _run_tiled(
        self,
        spec: ModelSpec,
        device: dm.DeviceInfo,
        image: np.ndarray,
        out: np.ndarray,
        scale: int,
        tile: int,
        progress: TileProgress | None,
        control: JobControl | None,
    ) -> None:
        with self._lock:
            if self.device != device:
                # Switched to the CPU since this attempt began: let upscale() retry.
                raise _DeviceChangedError
            net = self.load(spec)
        height, width = image.shape[:2]
        overlap = 0 if tile >= max(height, width) else min(DEFAULT_OVERLAP, tile // 4)
        tiles = plan_tiles(height, width, tile, overlap)
        log.debug("Upscaling %dx%d with %d tile(s) of %d px", width, height, len(tiles), tile)
        if progress:
            progress(0, len(tiles))
        for done, t in enumerate(tiles, start=1):
            if control is not None:
                control.check()
            patch = self._infer(net, device, spec, image[t.y0 : t.y1, t.x0 : t.x1], scale)
            paste_tile(
                out,
                patch,
                t.y0 * scale,
                t.x0 * scale,
                t.overlap_top * scale,
                t.overlap_left * scale,
            )
            del patch
            if progress:
                progress(done, len(tiles))

    def _infer(
        self, net: nn.Module, device: dm.DeviceInfo, spec: ModelSpec, patch: np.ndarray, scale: int
    ) -> np.ndarray:
        import torch
        import torch.nn.functional as F  # noqa: N812

        th, tw = patch.shape[:2]
        native = spec.native_scale
        dtype = torch.float16 if _uses_half(device) else torch.float32
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(patch)).to(dm.to_torch(device))
            x = x.permute(2, 0, 1).unsqueeze(0).to(dtype).div_(255.0)
            pad_h = -th % spec.size_multiple
            pad_w = -tw % spec.size_multiple
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
            y = net(x)[:, :, : th * native, : tw * native]
            del x
            if native != scale:
                y = F.interpolate(
                    y.float(), size=(th * scale, tw * scale), mode="bicubic", antialias=True
                )
            y = y.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
            result = y[0].permute(1, 2, 0).contiguous().cpu().numpy()
            del y
        return result
