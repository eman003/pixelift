"""Shared fixtures. Tests never touch the user's real config, logs or models."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pixelift.core.model_manager import ModelManager
from pixelift.models.base import ModelFamily, ModelSpec, register, register_family
from pixelift.storage import paths

# Resolved before the env is isolated, for the optional real-model tests.
REAL_MODELS_DIR = paths.models_dir()


@pytest.fixture(autouse=True)
def _isolate_xdg(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    base = tmp_path_factory.mktemp("xdg")
    for var in ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME"):
        monkeypatch.setenv(var, str(base / var.lower()))
    monkeypatch.setenv("PIXELIFT_MODELS_DIR", str(base / "models"))


def _tiny_compact():
    from pixelift.models.archs import SRVGGNetCompact

    torch.manual_seed(0)
    return SRVGGNetCompact(num_feat=8, num_conv=1, upscale=4)


def _tiny_rrdb_x2():
    from pixelift.models.archs import RRDBNet

    torch.manual_seed(0)
    return RRDBNet(scale=2, num_feat=8, num_block=1, num_grow_ch=4)


def _make_spec(model_id: str, build, native: int, multiple: int = 1) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        name=f"Test {model_id}",
        description="tiny random network for tests",
        native_scale=native,
        filename=f"{model_id}.pth",
        url="",
        sha256="",
        size_bytes=0,
        license="test",
        license_url="",
        build=build,
        memory_per_pixel=1000,
        size_multiple=multiple,
        state_key="params",
    )


def _write_weights(spec: ModelSpec, folder: Path) -> ModelSpec:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / spec.filename
    torch.save({"params": spec.build().state_dict()}, path)
    data = path.read_bytes()
    final = ModelSpec(
        **{
            **spec.__dict__,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "url": path.as_uri(),
        }
    )
    return final


@pytest.fixture
def models_dir() -> Path:
    return paths.models_dir()  # the isolated per-test PIXELIFT_MODELS_DIR


@pytest.fixture
def tiny_specs(models_dir: Path) -> dict[str, ModelSpec]:
    x4 = _write_weights(_make_spec("test-x4", _tiny_compact, 4), models_dir)
    x2 = _write_weights(_make_spec("test-x2", _tiny_rrdb_x2, 2, multiple=2), models_dir)
    register(x4)
    register(x2)
    register_family(ModelFamily("test-family", "Test family", "tests", {4: x4.id, 2: x2.id}))
    return {"x4": x4, "x2": x2}


@pytest.fixture
def manager(models_dir: Path, tiny_specs: dict[str, ModelSpec]) -> ModelManager:
    return ModelManager(models_dir, list(tiny_specs.values()))


@pytest.fixture
def upscaler(manager: ModelManager):
    from pixelift.core.upscaler import TorchUpscaler

    return TorchUpscaler(manager, cpu_threads=2)


def make_image(path: Path, size=(40, 30), mode="RGB", **save_kwargs) -> Path:
    w, h = size
    y, x = np.mgrid[0:h, 0:w]
    rgb = np.dstack(
        [(x * 255 // max(w - 1, 1)), (y * 255 // max(h - 1, 1)), ((x + y) % 32) * 8]
    ).astype(np.uint8)
    img = Image.fromarray(rgb, "RGB")
    if mode == "RGBA":
        alpha = Image.fromarray(((x * 255) // max(w - 1, 1)).astype(np.uint8), "L")
        img.putalpha(alpha)
    elif mode != "RGB":
        img = img.convert(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, **save_kwargs)
    return path


@pytest.fixture
def image_factory(tmp_path: Path):
    def factory(name: str = "photo.png", **kwargs) -> Path:
        return make_image(tmp_path / "in" / name, **kwargs)

    return factory
