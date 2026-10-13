"""On-disk cache of upscaled images, so HDR changes do not rerun the AI model.

HDR Enhancement runs after the upscaler, so the upscaled image (lighting and
camera-look grade included) does not depend on the HDR settings. When HDR
Enhancement or HDR output is on, that image is kept here as a raw ``.npy``
array (``uint16`` for the HDR proxy of HDR sources); processing the photo
again with other HDR settings memory-maps it instead of upscaling again. The
memory map also means the HDR stage reads the large image straight from the
page cache without a second full-size copy in memory.

Entries are keyed by everything the upscaled pixels depend on (the source
file's identity, model, scale, device, lighting and look grade). The cache is
bounded (``MAX_BYTES``, least recently used first out) and best-effort: any
failure just means upscaling again. Originals are never touched. Preferences
show its size and can clear it.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from pixelift.storage import paths
from pixelift.utils.image_utils import disk_free

log = logging.getLogger(__name__)

MAX_BYTES = 4 * 1024**3  # the whole cache
_SPARE_DISK = 1024**3  # leave at least this much disk free
_TMP_PREFIX = ".tmp-"
_STALE_TMP_SECONDS = 6 * 3600  # far longer than writing any image takes
_lock = threading.Lock()


def cache_dir() -> Path:
    return paths.cache_dir() / "upscaled"


def source_identity(source: Path) -> str | None:
    """The source file's identity (path, size, mtime, inode), or None when unreadable.

    Taken *before* the file is decoded: a key built after a long upscale would
    otherwise name a file edited meanwhile and file the old pixels under it.
    """
    try:
        stat = source.stat()
        resolved = source.resolve()
    except OSError:
        return None
    return "|".join(str(p) for p in (resolved, stat.st_size, stat.st_mtime_ns, stat.st_ino))


def cache_key(source_id: str | None, *parts: object) -> str | None:
    """Identifies an upscaled image: ``source_identity()`` plus ``parts``.

    None when the source could not be read (nothing is cached then).
    """
    if source_id is None:
        return None
    text = "|".join(str(p) for p in (source_id, *parts))
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def load(
    key: str, shape: tuple[int, int, int], dtype: type[np.generic] = np.uint8
) -> np.ndarray | None:
    """The cached image memory-mapped read-only, or None (missing, damaged or
    of another size)."""
    path = cache_dir() / f"{key}.npy"
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        return None
    if array.shape != shape or array.dtype != dtype:
        log.info("Discarding cached upscale %s: unexpected shape", path.name)
        with contextlib.suppress(OSError):
            path.unlink()
        return None
    with contextlib.suppress(OSError):
        os.utime(path)  # recently used
    return array


def store(key: str, rgb: np.ndarray) -> bool:
    """Keep ``rgb`` (best effort); returns whether it was stored."""
    size = rgb.nbytes + 128
    if size > MAX_BYTES:
        return False
    folder = cache_dir()
    try:
        with _lock:  # only the bookkeeping: jobs write their files in parallel
            folder.mkdir(parents=True, exist_ok=True)
            os.chmod(folder, 0o700)  # the user's photos: private
            _remove_stale_temporaries(folder)
            _evict(folder, MAX_BYTES - size)
            if disk_free(folder) < size + _SPARE_DISK:
                log.info("Not caching the upscaled image: low disk space")
                return False
            fd, tmp = tempfile.mkstemp(prefix=_TMP_PREFIX, suffix=".npy", dir=folder)
        try:
            with os.fdopen(fd, "wb") as fh:
                np.save(fh, np.ascontiguousarray(rgb), allow_pickle=False)
            os.replace(tmp, folder / f"{key}.npy")
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
    except OSError as exc:
        log.info("Could not cache the upscaled image: %s", exc)
        return False
    return True


def clear() -> None:
    """Remove every cached upscale (and leftovers of interrupted writes)."""
    folder = cache_dir()
    with _lock:
        for path in [*_entries(folder), *_temporaries(folder)]:
            with contextlib.suppress(OSError):
                path.unlink()


def total_bytes() -> int:
    """Disk space the cache uses, unfinished writes included."""
    folder = cache_dir()
    total = 0
    for path in [*_entries(folder), *_temporaries(folder)]:
        with contextlib.suppress(OSError):
            total += path.stat().st_size
    return total


def _entries(folder: Path) -> list[Path]:
    try:
        return [p for p in folder.iterdir() if p.suffix == ".npy" and not p.name.startswith(".")]
    except OSError:
        return []


def _temporaries(folder: Path) -> list[Path]:
    try:
        return [p for p in folder.iterdir() if p.name.startswith(_TMP_PREFIX)]
    except OSError:
        return []


def _remove_stale_temporaries(folder: Path) -> None:
    """Remove files left by writes that never finished (the app was killed).

    Only old ones: a younger file may be another job's write in progress.
    """
    cutoff = time.time() - _STALE_TMP_SECONDS
    for path in _temporaries(folder):
        with contextlib.suppress(OSError):
            if path.stat().st_mtime < cutoff:
                path.unlink()


def _evict(folder: Path, budget: int) -> None:
    """Remove the least recently used entries until the rest fit ``budget``."""
    entries = []
    for path in _entries(folder):
        with contextlib.suppress(OSError):
            stat = path.stat()
            entries.append((stat.st_mtime, stat.st_size, path))
    entries.sort()
    total = sum(size for _, size, _ in entries)
    for _, size, path in entries:
        if total <= budget:
            break
        with contextlib.suppress(OSError):
            path.unlink()
            total -= size
