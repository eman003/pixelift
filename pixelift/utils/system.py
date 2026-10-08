"""Small helpers for querying the host system without extra dependencies."""

from __future__ import annotations

import os
from pathlib import Path


def available_ram_bytes() -> int:
    """MemAvailable from /proc/meminfo (falls back to free pages)."""
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return 4 * 1024**3


def total_ram_bytes() -> int:
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return 8 * 1024**3


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "CPU"


def cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


_GPU_VENDORS = {"0x10de": "NVIDIA", "0x1002": "AMD", "0x8086": "Intel"}


def pci_gpu_vendors(sysfs: Path = Path("/sys/bus/pci/devices")) -> list[str]:
    """Vendors of display controllers (PCI class 0x03xxxx) present in the machine."""
    vendors: list[str] = []
    try:
        devices = list(sysfs.iterdir())
    except OSError:
        return vendors
    for dev in devices:
        try:
            if not (dev / "class").read_text().strip().startswith("0x03"):
                continue
            vendor = _GPU_VENDORS.get((dev / "vendor").read_text().strip())
        except OSError:
            continue
        if vendor and vendor not in vendors:
            vendors.append(vendor)
    return vendors
