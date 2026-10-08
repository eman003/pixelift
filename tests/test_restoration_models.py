"""Photo restoration with the real AI models (GFPGAN, RetinaFace, DeOldify).

Skipped unless the models are installed, e.g.:

    pixelift --download-model gfpgan-v1.4 --download-model retinaface-resnet50 \\
             --download-model deoldify-artistic

The test photographs are public domain (see tests/data/README.md).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from conftest import REAL_MODELS_DIR
from PIL import Image

from pixelift.core.image_processor import ProcessingOptions, process_image
from pixelift.core.model_manager import ModelManager
from pixelift.core.restoration import faces
from pixelift.core.restoration import filters as fl
from pixelift.core.restoration import settings as rs
from pixelift.core.restoration.pipeline import Restorer
from pixelift.core.upscaler import TorchUpscaler

pytestmark = pytest.mark.models

DATA = Path(__file__).parent / "data"
FACE_ONLY = rs.RestorationSettings(
    level=rs.CUSTOM, custom=rs.Stages(face=rs.FACE_NATURAL), modern=rs.MODERN_OFF
)


def _restorer(*model_ids: str) -> Restorer:
    manager = ModelManager(REAL_MODELS_DIR)
    missing = [m for m in model_ids if not manager.is_installed(m)]
    if missing:
        pytest.skip(f"not installed in {REAL_MODELS_DIR}: {', '.join(missing)}")
    return Restorer(TorchUpscaler(manager))


def _load(name: str) -> np.ndarray:
    return np.asarray(Image.open(DATA / name).convert("RGB"))


def _face_region(rgb: np.ndarray, face: faces.Face) -> np.ndarray:
    x0, y0, x1, y1 = (int(v) for v in face.box)
    return rgb[max(0, y0) : y1, max(0, x0) : x1, 0].astype(np.float32)


def _structure_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of blurred brightness: same face shape and lighting?"""
    import torch

    def blur(x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(x)[None, None]
        return fl.gaussian_blur(t, 3.0)[0, 0].numpy().ravel()

    return float(np.corrcoef(blur(a), blur(b))[0, 1])


def test_checksums_and_strict_loading():
    restorer = _restorer("gfpgan-v1.4", "retinaface-resnet50", "deoldify-artistic")
    engine = restorer.upscaler
    for model_id in ("gfpgan-v1.4", "retinaface-resnet50", "deoldify-artistic"):
        assert engine.models.verify(model_id)
        assert engine.load(engine.models.spec(model_id)) is not None  # strict=True inside


def test_detects_portrait_and_group_faces():
    restorer = _restorer("retinaface-resnet50")
    engine = restorer.upscaler
    spec = engine.models.spec("retinaface-resnet50")
    portrait = engine.run_model(
        spec, lambda n, d: faces.detect_faces(n, d, _load("portrait_bw.jpg"))
    )
    assert len(portrait) == 1 and portrait[0].score > 0.97
    left_eye, right_eye = portrait[0].landmarks[0], portrait[0].landmarks[1]
    assert left_eye[0] < right_eye[0] and abs(left_eye[1] - right_eye[1]) < 15
    group = engine.run_model(spec, lambda n, d: faces.detect_faces(n, d, _load("group_bw.jpg")))
    assert len(group) >= 15  # the 1927 Solvay conference: 29 people


def test_face_restoration_preserves_identity():
    restorer = _restorer("gfpgan-v1.4", "retinaface-resnet50")
    src = _load("portrait_bw.jpg")
    natural = restorer.restore(src, FACE_ONLY, model="realesrgan")
    strong_settings = dataclasses.replace(
        FACE_ONLY, custom=rs.Stages(face=rs.FACE_STRONG), fidelity=100
    )
    strong = restorer.restore(src, strong_settings, model="realesrgan")
    assert natural.faces.restored == 1 and natural.monochrome
    engine = restorer.upscaler
    face = engine.run_model(
        engine.models.spec("retinaface-resnet50"), lambda n, d: faces.detect_faces(n, d, src)
    )[0]
    original = _face_region(src, face)
    for result in (natural, strong):
        assert _structure_corr(original, _face_region(result.rgb, face)) > 0.9
        # Overall skin tone / brightness kept.
        assert abs(_face_region(result.rgb, face).mean() - original.mean()) < 8
    change = lambda r: np.abs(_face_region(r.rgb, face) - original).mean()  # noqa: E731
    assert 0 < change(natural) < change(strong)
    # Outside the face nothing changes.
    assert np.array_equal(natural.rgb[:20, :20], src[:20, :20])


def test_group_photo_restores_many_faces():
    restorer = _restorer("gfpgan-v1.4", "retinaface-resnet50")
    result = restorer.restore(_load("group_bw.jpg"), FACE_ONLY, model="realesrgan")
    assert result.faces.restored >= 15
    # Small faces are restored only lightly (identity protection).
    assert result.faces.protected >= 1


def test_colorization_and_strength():
    restorer = _restorer("deoldify-artistic")
    src = _load("portrait_bw.jpg")
    settings = rs.RestorationSettings(
        level=rs.CUSTOM, custom=rs.Stages(), modern=rs.MODERN_OFF, colorize=True
    )
    colored = restorer.restore(src, settings, model="realesrgan")
    assert colored.colorized and not colored.monochrome
    rgb = colored.rgb.astype(np.float32)
    chroma = np.abs(rgb[..., 0] - rgb[..., 2]).mean()
    assert 2 < chroma < 40  # visible but natural, not garish
    # Preserve original tones: the brightness is the original's.
    luma = rgb @ np.array(fl.LUMA, dtype=np.float32)
    assert np.abs(luma - src[..., 0]).mean() < 2.5
    grey = restorer.restore(
        src, dataclasses.replace(settings, colorize_strength=0), model="realesrgan"
    )
    assert not grey.colorized and np.array_equal(grey.rgb, src)
    off = restorer.restore(src, dataclasses.replace(settings, colorize=False), model="realesrgan")
    assert not off.colorized and off.monochrome


def test_full_restoration_with_upscaling(tmp_path):
    _restorer("gfpgan-v1.4", "retinaface-resnet50", "deoldify-artistic", "realesr-general-x4v3")
    manager = ModelManager(REAL_MODELS_DIR)
    src = tmp_path / "grandma_1962.jpg"
    Image.open(DATA / "portrait_bw.jpg").save(src, quality=92)
    restoration = dataclasses.replace(rs.RestorationSettings(), colorize=True, scale=2)
    options = ProcessingOptions(
        model="realesr-general-x4v3", restoration=restoration, output_format="jpeg"
    )
    result = process_image(src, options, TorchUpscaler(manager))
    assert result.output.name == "grandma_1962_restored_colorized_2x.jpg"
    assert result.output_size == (628, 800) and result.colorized and result.faces == 1


def test_colorized_portrait_with_camera_look():
    """Real DeOldify colorization, then a portrait look: skin stays natural."""
    from pixelift.core import camera_looks as cl
    from pixelift.models.restoration import DEOLDIFY_ARTISTIC

    restorer = _restorer(DEOLDIFY_ARTISTIC.id)
    settings = rs.RestorationSettings(
        level=rs.CUSTOM, custom=rs.Stages(fading=30), colorize=True, modern=rs.MODERN_OFF
    )
    src = _load("portrait_bw.jpg")
    plain = restorer.restore(src, settings, model="realesrgan")
    look = cl.CameraLookSettings("canon-portrait", 100).recipe()
    graded = restorer.restore(src, settings, model="realesrgan", look=look)
    assert plain.colorized and graded.colorized and graded.rgb.shape == src.shape
    expected = cl.finish_look(cl.apply_look(plain.rgb, look), look)
    assert np.array_equal(graded.rgb, expected)  # the look follows colorization
    h, w = src.shape[:2]
    face = (slice(h // 4, h // 2), slice(w * 2 // 5, w * 3 // 5))
    before = plain.rgb[face].astype(float).mean((0, 1))
    after = graded.rgb[face].astype(float).mean((0, 1))
    # Skin protection: the face keeps the colorizer's skin tone (red dominant).
    assert after[0] > after[1] and after[0] > after[2]
    assert np.abs(after - before).max() < 8
