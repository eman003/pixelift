"""Processing-device detection and selection (CUDA, ROCm, Intel XPU, CPU).

PyTorch is imported lazily so the GUI can start before it finishes loading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pixelift.utils import system

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    id: str  # torch device string: "cpu", "cuda:0", "xpu:0"
    name: str
    backend: str  # "CPU", "CUDA", "ROCm", "Intel XPU"
    total_memory: int = 0  # bytes; 0 = unknown / system RAM

    @property
    def is_gpu(self) -> bool:
        return self.id != "cpu"

    @property
    def kind(self) -> str:
        return self.id.split(":", 1)[0]

    def label(self) -> str:
        if not self.is_gpu:
            return f"CPU — {self.name}" if self.name != "CPU" else "CPU"
        return f"{self.name} — {self.backend}"


@dataclass
class DeviceReport:
    devices: list[DeviceInfo]
    # Human-readable notes, e.g. "NVIDIA GPU found but CUDA is not available".
    hints: list[str] = field(default_factory=list)
    torch_version: str = ""

    @property
    def best(self) -> DeviceInfo:
        return self.devices[0]

    def find(self, device_id: str) -> DeviceInfo | None:
        for dev in self.devices:
            if dev.id == device_id or dev.kind == device_id:
                return dev
        return None


CPU_DEVICE = DeviceInfo("cpu", system.cpu_name(), "CPU", system.total_ram_bytes())


def detect_devices() -> DeviceReport:
    """List usable devices, best first. CPU is always present and last."""
    devices: list[DeviceInfo] = []
    hints: list[str] = []
    try:
        import torch
    except ImportError:
        return DeviceReport(
            [CPU_DEVICE], ["PyTorch is not installed — AI upscaling is unavailable."], ""
        )

    try:
        if torch.cuda.is_available():
            backend = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append(DeviceInfo(f"cuda:{i}", props.name, backend, props.total_memory))
    except Exception:
        log.exception("CUDA/ROCm detection failed")

    try:
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            for i in range(xpu.device_count()):
                props = xpu.get_device_properties(i)
                total = getattr(props, "total_memory", 0)
                devices.append(DeviceInfo(f"xpu:{i}", props.name, "Intel XPU", total))
    except Exception:
        log.exception("Intel XPU detection failed")

    devices.sort(key=lambda d: d.total_memory, reverse=True)
    devices.append(CPU_DEVICE)

    present = system.pci_gpu_vendors()
    backends = {d.backend for d in devices}
    if "NVIDIA" in present and "CUDA" not in backends:
        hints.append(
            "An NVIDIA GPU was found but CUDA is not available. Install the NVIDIA driver "
            "and a CUDA build of PyTorch to enable GPU acceleration."
        )
    if "AMD" in present and "ROCm" not in backends:
        hints.append(
            "An AMD GPU was found. GPU acceleration requires a ROCm build of PyTorch "
            "and a supported Radeon GPU."
        )
    if "Intel" in present and "Intel XPU" not in backends:
        hints.append(
            "An Intel GPU was found. Acceleration requires an XPU build of PyTorch and "
            "a supported Intel Arc / Core Ultra GPU."
        )
    log.info("Devices: %s", ", ".join(d.label() for d in devices))
    return DeviceReport(devices, hints, torch.__version__)


def resolve_device(
    preference: str, report: DeviceReport | None = None, gpu_enabled: bool = True
) -> DeviceInfo:
    """Turn a user preference ("auto", "cpu", "cuda", "cuda:1", "xpu") into a device.

    Falls back to the CPU (with a log warning) if the requested device is missing.
    """
    report = report or detect_devices()
    if not gpu_enabled or preference == "cpu":
        return CPU_DEVICE
    if preference in ("", "auto"):
        return report.best
    found = report.find(preference)
    if found is None:
        log.warning("Requested device %r not available; using CPU", preference)
        return CPU_DEVICE
    return found


def free_memory(device: DeviceInfo) -> int:
    """Currently free memory on the device in bytes (best effort)."""
    import torch

    try:
        if device.kind == "cuda":
            free, _total = torch.cuda.mem_get_info(torch.device(device.id))
            return int(free)
        if device.kind == "xpu" and hasattr(torch.xpu, "mem_get_info"):
            free, _total = torch.xpu.mem_get_info(torch.device(device.id))
            return int(free)
    except Exception:
        log.debug("Could not query free memory for %s", device.id, exc_info=True)
    if device.is_gpu:
        return int(device.total_memory * 0.7)
    return system.available_ram_bytes()


def empty_cache(device: DeviceInfo) -> None:
    try:
        import torch

        if device.kind == "cuda":
            torch.cuda.empty_cache()
        elif device.kind == "xpu":
            torch.xpu.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def to_torch(device: DeviceInfo) -> torch.device:
    import torch

    return torch.device(device.id)


def recommended_concurrency(device: DeviceInfo) -> int:
    """Concurrent images to process.

    One job per device: PyTorch already parallelises a single job across all CPU
    cores / GPU units, and extra jobs only multiply memory use.
    """
    return 1
