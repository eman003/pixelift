from __future__ import annotations

import dataclasses
import types
from pathlib import Path

import pytest

from pixelift.core.control import JobControl
from pixelift.core.errors import (
    CancelledError,
    ModelDownloadError,
    ModelNotInstalledError,
)
from pixelift.core.model_manager import ModelManager, sha256_file
from pixelift.models import all_families, all_specs, candidate_specs, get_spec
from pixelift.storage import paths


def test_ensure_verified_hashes_each_file_version_once(manager, tiny_specs, monkeypatch):
    import pixelift.core.model_manager as mm

    calls = []
    real = mm.sha256_file
    monkeypatch.setattr(mm, "sha256_file", lambda p: calls.append(p) or real(p))
    spec = tiny_specs["x4"]
    path = manager.ensure_verified(spec)
    assert manager.ensure_verified(spec) == path
    assert len(calls) == 1
    path.write_bytes(b"tampered")
    with pytest.raises(ModelNotInstalledError, match="not the official release"):
        manager.ensure_verified(spec)
    path.unlink()
    with pytest.raises(ModelNotInstalledError, match="not installed"):
        manager.ensure_verified(spec)


def test_builtin_catalogue_is_consistent():
    specs = {s.id: s for s in all_specs()}
    for spec_id in ("realesrgan-x4plus", "realesrgan-x2plus", "realesr-general-x4v3"):
        spec = specs[spec_id]
        assert len(spec.sha256) == 64
        assert spec.url.startswith("https://github.com/xinntao/Real-ESRGAN/")
        assert spec.license == "BSD-3-Clause"
    for family in all_families():
        for scale, spec_id in family.variants.items():
            assert specs[spec_id].native_scale == scale


@pytest.mark.parametrize(
    "spec_id",
    [
        "realesrgan-x4plus",
        "realesrgan-x2plus",
        "realesr-general-x4v3",
        "realesrgan-x4plus-anime",
        "realesr-animevideov3",
    ],
)
def test_builtin_architectures_build(spec_id):
    net = get_spec(spec_id).build()
    assert sum(p.numel() for p in net.parameters()) > 10_000


def test_candidate_order_prefers_native_then_larger():
    assert [s.id for s in candidate_specs("realesrgan", 2)] == [
        "realesrgan-x2plus",
        "realesrgan-x4plus",
    ]
    assert [s.id for s in candidate_specs("realesrgan", 4)] == [
        "realesrgan-x4plus",
        "realesrgan-x2plus",
    ]


def test_models_dir_is_app_specific(monkeypatch, tmp_path):
    monkeypatch.delenv("PIXELIFT_MODELS_DIR")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert paths.models_dir() == tmp_path / "pixelift" / "models"


def test_statuses_and_installed(manager, tiny_specs):
    statuses = {s.spec.id: s.installed for s in manager.statuses()}
    assert statuses == {"test-x4": True, "test-x2": True}
    manager.remove("test-x2")
    assert not manager.is_installed("test-x2")
    assert [s.id for s in manager.installed()] == ["test-x4"]


def test_resolve_falls_back_and_explains(manager):
    manager.remove("test-x2")
    assert manager.resolve("test-family", 2).id == "test-x4"
    manager.remove("test-x4")
    with pytest.raises(ModelNotInstalledError, match="not installed"):
        manager.resolve("test-family", 2)
    with pytest.raises(ModelNotInstalledError):
        manager.resolve("no-such-model", 4)


def test_download_verifies_checksum(tmp_path, tiny_specs):
    spec = tiny_specs["x4"]
    target_dir = tmp_path / "dl"
    mgr = ModelManager(target_dir, [spec])
    progress = []
    path = mgr.download(spec, lambda d, t: progress.append((d, t)))
    assert path.is_file() and mgr.verify(spec)
    assert sha256_file(path) == spec.sha256
    assert progress[-1][0] == spec.size_bytes


def test_download_rejects_bad_checksum(tmp_path, tiny_specs):
    bad = dataclasses.replace(tiny_specs["x4"], sha256="0" * 64)
    mgr = ModelManager(tmp_path / "dl", [bad])
    with pytest.raises(ModelDownloadError, match="checksum"):
        mgr.download(bad)
    assert not mgr.is_installed(bad)
    assert not list((tmp_path / "dl").iterdir())  # temp file cleaned up


def test_download_network_error(tmp_path, tiny_specs):
    missing = dataclasses.replace(tiny_specs["x4"], url=(tmp_path / "nope.pth").as_uri())
    mgr = ModelManager(tmp_path / "dl", [missing])
    with pytest.raises(ModelDownloadError):
        mgr.download(missing)


def test_download_cancel(tmp_path, tiny_specs):
    control = JobControl()
    control.cancel()
    mgr = ModelManager(tmp_path / "dl", [tiny_specs["x4"]])
    with pytest.raises(CancelledError):
        mgr.download(tiny_specs["x4"], control=control)
    assert not mgr.is_installed(tiny_specs["x4"])


def test_install_manual_file(tmp_path, tiny_specs, models_dir):
    spec = tiny_specs["x4"]
    mgr = ModelManager(tmp_path / "manual", [spec])
    mgr.install_file(spec, models_dir / spec.filename)
    assert mgr.verify(spec)
    other = tmp_path / "other.pth"
    other.write_bytes(b"wrong")
    with pytest.raises(ModelDownloadError):
        mgr.install_file(spec, other)


def test_download_dropped_connection_is_a_download_error(tmp_path, tiny_specs, monkeypatch):
    import http.client
    import urllib.request

    class Dropping:
        headers = types.MappingProxyType({})

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self, _n):
            raise http.client.IncompleteRead(b"partial", 100)

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: Dropping())
    mgr = ModelManager(tmp_path / "dl", [tiny_specs["x4"]])
    with pytest.raises(ModelDownloadError, match="Could not download"):
        mgr.download(tiny_specs["x4"])
    assert not list((tmp_path / "dl").iterdir())


def test_install_manual_file_is_atomic(tmp_path, tiny_specs, models_dir, monkeypatch):
    import shutil

    spec = tiny_specs["x4"]
    mgr = ModelManager(tmp_path / "manual", [spec])

    def interrupted(_src, dst):
        Path(dst).write_bytes(b"trunc")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(shutil, "copyfile", interrupted)
    with pytest.raises(ModelDownloadError, match="disk space"):
        mgr.install_file(spec, models_dir / spec.filename)
    assert not mgr.is_installed(spec)
    assert not list((tmp_path / "manual").iterdir())


def test_stale_partial_downloads_are_removed(tmp_path, tiny_specs):
    import os
    import time

    folder = tmp_path / "dl"
    folder.mkdir()
    old, fresh = folder / ".download-old", folder / ".download-fresh"
    old.write_bytes(b"x")
    fresh.write_bytes(b"x")
    long_ago = time.time() - 24 * 3600
    os.utime(old, (long_ago, long_ago))
    ModelManager(folder, [tiny_specs["x4"]])
    assert not old.exists() and fresh.exists()  # a running download may own it
