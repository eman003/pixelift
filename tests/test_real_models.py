"""Integration tests with the real Real-ESRGAN weights.

Skipped unless the models are installed (run `pixelift --download-model
recommended --download-model realesr-general-x4v3` first).
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import REAL_MODELS_DIR, make_image
from PIL import Image

from pixelift.core.image_processor import ProcessingOptions, process_image
from pixelift.core.model_manager import ModelManager
from pixelift.core.upscaler import TorchUpscaler

pytestmark = pytest.mark.models


def _manager_with(model_id: str) -> ModelManager:
    manager = ModelManager(REAL_MODELS_DIR)
    if not manager.is_installed(model_id):
        pytest.skip(f"{model_id} not installed in {REAL_MODELS_DIR}")
    return manager


@pytest.mark.parametrize(
    ("model_id", "scale"),
    [("realesr-general-x4v3", 4), ("realesr-general-x4v3", 2), ("realesrgan-x4plus", 4)],
)
def test_real_model_end_to_end(tmp_path, model_id, scale):
    manager = _manager_with(model_id)
    src = make_image(tmp_path / "photo.jpg", size=(48, 32), quality=90)
    upscaler = TorchUpscaler(manager)
    result = process_image(src, ProcessingOptions(scale=scale, model=model_id), upscaler)
    with Image.open(result.output) as out:
        assert out.size == (48 * scale, 32 * scale)
        arr = np.asarray(out, dtype=np.float32)
    # Sanity: the result resembles the source (not noise / not blank).
    ref = np.asarray(Image.open(src).resize(out.size, Image.Resampling.BICUBIC), np.float32)
    assert np.abs(arr - ref).mean() < 25
    assert arr.std() > 10


def test_real_model_checksum(tmp_path):
    manager = _manager_with("realesr-general-x4v3")
    assert manager.verify("realesr-general-x4v3")
