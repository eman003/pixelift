from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from pixelift.core import device_manager as dm
from pixelift.utils import system


def test_cpu_always_available():
    report = dm.detect_devices()
    assert report.devices[-1].id == "cpu"
    assert report.torch_version


def test_resolve_device_preferences():
    gpu = dm.DeviceInfo("cuda:0", "RTX 3060", "CUDA", 12 * 1024**3)
    report = dm.DeviceReport([gpu, dm.CPU_DEVICE])
    assert dm.resolve_device("auto", report) is gpu
    assert dm.resolve_device("cuda", report) is gpu
    assert dm.resolve_device("cpu", report).id == "cpu"
    assert dm.resolve_device("auto", report, gpu_enabled=False).id == "cpu"
    assert dm.resolve_device("xpu", report).id == "cpu"  # missing -> CPU fallback
    assert gpu.label() == "RTX 3060 — CUDA"


def test_cuda_detection_mocked(monkeypatch):
    props = SimpleNamespace(name="NVIDIA GeForce RTX 3060", total_memory=12 * 1024**3)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    report = dm.detect_devices()
    assert report.best.id == "cuda:0"
    assert report.best.label() == "NVIDIA GeForce RTX 3060 — CUDA"
    monkeypatch.setattr(torch.version, "hip", "6.0", raising=False)
    assert dm.detect_devices().best.backend == "ROCm"


def test_broken_driver_does_not_crash(monkeypatch):
    def boom() -> bool:
        raise RuntimeError("driver exploded")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    assert dm.detect_devices().best.id == "cpu"


def test_pci_vendor_hints(tmp_path):
    for name, cls, vendor in (
        ("0000:01:00.0", "0x030200", "0x10de"),
        ("0000:00:02.0", "0x030000", "0x8086"),
        ("0000:00:1f.0", "0x060100", "0x8086"),
    ):
        dev = tmp_path / name
        dev.mkdir()
        (dev / "class").write_text(cls)
        (dev / "vendor").write_text(vendor)
    assert system.pci_gpu_vendors(tmp_path) == ["NVIDIA", "Intel"] or set(
        system.pci_gpu_vendors(tmp_path)
    ) == {"NVIDIA", "Intel"}


def test_system_helpers():
    assert system.available_ram_bytes() > 0
    assert system.cpu_count() >= 1
    assert system.cpu_name()


@pytest.mark.gpu
def test_real_gpu_detected():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA/ROCm device")
    assert dm.detect_devices().best.is_gpu
