"""The restoration pipeline: one engine shared by every image in a batch.

Stage order (chosen so each step works on the cleanest possible input):

1. Black-and-white photos become neutral grey (removes toning, yellowed paper
   and chroma noise in one go).
2. Dust, then scratches — before denoising, which would smear specks into blobs.
3. Noise reduction — before the tonal stretch, which would amplify noise.
4. Faded-image recovery and colour-cast correction.
5. Colorization (only with consent, only black-and-white photos) — after
   cleanup, so the network sees a clean, well-exposed photo.
6. Manual colour correction, Modern Finish and the lighting profile — at the
   input resolution (cheap), so the AI upscaler builds on the final tones,
   as in normal upscaling.
7. AI upscaling (or AI detail reconstruction without upscaling) with
   Real-ESRGAN, blended with a conventional resize according to fidelity.
8. Face restoration at the output resolution: GFPGAN's 512 px faces keep
   their detail instead of being shrunk and re-upscaled.
9. Sharpening at the output resolution.

Models are loaded once by the shared :class:`TorchUpscaler` and reused for
every image; nothing ever leaves the computer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from pixelift.core.control import JobControl
from pixelift.core.errors import ModelNotInstalledError, UpscalerError
from pixelift.core.lighting import Adjustments
from pixelift.core.restoration import cleanup, colorize, faces, tones
from pixelift.core.restoration import filters as fl
from pixelift.core.restoration.analysis import MonoInfo, detect_monochrome, estimate_noise
from pixelift.core.restoration.settings import FACE_OFF, MODERN_OFF, RestorationSettings
from pixelift.core.upscaler import TorchUpscaler, Upscaler
from pixelift.models.base import ModelSpec
from pixelift.models.restoration import (
    COLORIZE_MODELS,
    DEOLDIFY_ARTISTIC,
    FACE_MODELS,
    GFPGAN_V14,
    RETINAFACE,
)

log = logging.getLogger(__name__)

Progress = Callable[[float, str], None]


@dataclass
class RestoreResult:
    rgb: np.ndarray
    monochrome: bool  # the result is neutral black-and-white (save it as greyscale)
    colorized: bool
    faces: faces.FaceReport = field(default_factory=faces.FaceReport)


def required_models(settings: RestorationSettings, monochrome: bool | None = None) -> list[str]:
    """Model ids the settings need (``monochrome=None``: colorization may be needed)."""
    needed: list[str] = []
    if settings.stages().face != FACE_OFF:
        needed += FACE_MODELS
    if settings.colorize and settings.colorize_strength > 0 and monochrome is not False:
        needed += COLORIZE_MODELS
    return needed


def upscale_weight(fidelity: int) -> float:
    """Share of the AI upscaler's output vs. a plain resize (rest)."""
    return 0.5 + 0.5 * min(100, max(0, fidelity)) / 100


class Restorer:
    """Restores RGB images. Shares models and device handling with ``upscaler``."""

    def __init__(self, upscaler: Upscaler) -> None:
        self.upscaler = upscaler

    # --- models ----------------------------------------------------------------
    def _engine(self) -> TorchUpscaler:
        if not isinstance(self.upscaler, TorchUpscaler):
            raise UpscalerError("AI restoration needs the PyTorch engine.")
        return self.upscaler

    def _spec(self, spec: ModelSpec) -> ModelSpec:
        engine = self._engine()
        if not engine.models.is_installed(spec):
            raise ModelNotInstalledError(
                f"{spec.name} ({spec.size_mb:.0f} MB) is needed for this restoration but is not "
                "installed.",
                [
                    "Open Settings → Models and download it",
                    f"Or run: pixelift --download-model {spec.id}",
                    "Or turn the stage off in the restoration options",
                ],
            )
        return spec

    def check_models(self, settings: RestorationSettings, monochrome: bool | None = None) -> None:
        engine = self._engine()
        for model_id in required_models(settings, monochrome):
            self._spec(engine.models.spec(model_id))

    # --- pipeline ----------------------------------------------------------------
    def restore(
        self,
        rgb: np.ndarray,
        settings: RestorationSettings,
        *,
        model: str,
        lighting: Adjustments | None = None,
        mono: MonoInfo | None = None,
        source_grayscale: bool = False,
        progress: Progress | None = None,
        control: JobControl | None = None,
    ) -> RestoreResult:
        """Restore ``rgb`` (H, W, 3 uint8). The input array is not modified.

        ``model`` is the upscaling model family used for upscaling and AI
        detail reconstruction; ``lighting`` the lighting profile's adjustments.
        """
        settings.validate()
        report = progress or (lambda _f, _s: None)
        lighting = lighting or Adjustments()
        stages = settings.stages()

        def check() -> None:
            if control is not None:
                control.check()

        if mono is None:
            mono = MonoInfo(True, False, 0.0) if source_grayscale else detect_monochrome(rgb)
        monochrome = mono.monochrome
        colorize_now = monochrome and settings.colorize and settings.colorize_strength > 0
        self.check_models(settings, monochrome)

        plan = _Plan(settings, stages, colorize_now, lighting, monochrome)
        image = rgb
        # Neutral grey from here on? (Then it is saved as a greyscale image.)
        gray = monochrome and _is_neutral(rgb)
        if monochrome and not gray and not settings.is_identity():
            report(plan.at("prepare"), "Converting to black and white")
            image = fl.to_gray(image)
            gray = True
        check()

        noise = estimate_noise(image) if (stages.dust or stages.scratches or stages.noise) else 0.0
        if stages.dust:
            report(plan.at("dust"), "Removing dust")
            image = cleanup.remove_dust(image, stages.dust, noise, control)
        if stages.scratches:
            report(plan.at("scratches"), "Reducing scratches")
            image = cleanup.reduce_scratches(image, stages.scratches, noise, control)
        if stages.noise:
            report(plan.at("noise"), "Reducing noise")
            image = cleanup.denoise(image, stages.noise, noise, monochrome, control)
        check()
        if stages.fading or stages.auto_color:
            report(plan.at("tones"), "Restoring faded tones" if monochrome else "Restoring colours")
            image = tones.restore_tones(image, stages.fading, stages.auto_color, monochrome)
        check()

        if colorize_now:
            report(plan.at("colorize"), "Colorizing")
            spec = self._spec(DEOLDIFY_ARTISTIC)
            gray_rgb = fl.to_gray(image)
            predicted = self._engine().run_model(
                spec, lambda net, dev: colorize.predict_color(net, dev, gray_rgb[..., 0]), control
            )
            image = colorize.apply_color(
                gray_rgb,
                predicted,
                settings.colorize_strength,
                settings.colorize_vivid,
                settings.preserve_tones,
            )
            del gray_rgb, predicted
        check()

        finish = (
            tones.MODERN_FINISHES.get(settings.modern) if settings.modern != MODERN_OFF else None
        )
        if not settings.color.is_neutral or finish or not lighting.is_neutral:
            report(plan.at("look"), "Adjusting colour and lighting")
            image = tones.adjust(image, settings.color)
            if finish is not None:
                image = tones.adjust(image, finish.adjustments)
                image = cleanup.clarity(image, finish.clarity, control)
            image = tones.adjust(image, lighting)
        check()

        if settings.ai_scale():
            image = self._upscale(image, settings, model, plan, report, control)
        check()

        face_report = faces.FaceReport()
        if stages.face != FACE_OFF:
            image, face_report = self._restore_faces(
                image, settings, stages.face, plan, report, control
            )
            if gray and not colorize_now and face_report.restored:
                image = fl.to_gray(image)  # GFPGAN may tint faces slightly
        check()

        if stages.sharpness:
            report(plan.at("sharpen"), "Enhancing detail")
            out_noise = estimate_noise(image)
            image = cleanup.sharpen(image, stages.sharpness, out_noise, control)

        if image is rgb:
            image = rgb.copy()
        still_gray = gray and not colorize_now and not lighting.changes_colour
        still_gray = still_gray and settings.color.temperature == 0 and settings.color.tint == 0
        return RestoreResult(image, still_gray, colorize_now, face_report)

    def _upscale(
        self,
        image: np.ndarray,
        settings: RestorationSettings,
        model: str,
        plan: _Plan,
        report: Progress,
        control: JobControl | None,
    ) -> np.ndarray:
        """AI upscaling (scale > 1) or detail reconstruction (2× then back down)."""
        height, width = image.shape[:2]
        scale = settings.ai_scale()
        start, span = plan.at("upscale"), plan.span("upscale")
        label = "Upscaling" if settings.scale > 1 else "Reconstructing detail"

        def on_tiles(done: int, total: int) -> None:
            report(start + span * done / max(total, 1), f"{label} (tile {done}/{total})")

        report(start, label)
        ai = self.upscaler.upscale(image, scale, model, progress=on_tiles, control=control)
        if settings.scale == 1:
            ai = fl.resize(ai, (width, height))
            base = image
        else:
            base = fl.resize(image, (ai.shape[1], ai.shape[0]))
        weight = upscale_weight(settings.fidelity)
        if weight >= 1:
            return ai
        return _blend(base, ai, weight)

    def _restore_faces(
        self,
        image: np.ndarray,
        settings: RestorationSettings,
        mode: str,
        plan: _Plan,
        report: Progress,
        control: JobControl | None,
    ) -> tuple[np.ndarray, faces.FaceReport]:
        engine = self._engine()
        detector = self._spec(RETINAFACE)
        restorer = self._spec(GFPGAN_V14)
        start, span = plan.at("faces"), plan.span("faces")
        report(start, "Finding faces")
        try:
            return self._restore_found_faces(
                image, settings, mode, detector, restorer, start, span, report, control
            )
        finally:
            engine.release_memory()  # once per image, not after every face

    def _restore_found_faces(
        self,
        image: np.ndarray,
        settings: RestorationSettings,
        mode: str,
        detector: ModelSpec,
        restorer: ModelSpec,
        start: float,
        span: float,
        report: Progress,
        control: JobControl | None,
    ) -> tuple[np.ndarray, faces.FaceReport]:
        engine = self._engine()
        found = engine.run_model(
            detector, lambda net, dev: faces.detect_faces(net, dev, image), control, release=False
        )
        result = faces.FaceReport(found=len(found))
        if not found:
            return image, result
        out = image.copy()  # never write into an earlier stage's (or the caller's) array
        mask = faces.ellipse_mask()
        base = faces.base_weight(mode, settings.fidelity)
        scale_hint = settings.scale if settings.scale > 1 else 1
        for index, face in enumerate(found, start=1):
            if control is not None:
                control.check()
            report(
                start + span * (index - 1) / len(found), f"Restoring faces ({index}/{len(found)})"
            )
            image_to_face = faces.similarity_transform(face.landmarks, faces.TEMPLATE)
            crop = faces.crop_face(out, image_to_face)
            restored = engine.run_model(
                restorer,
                lambda net, dev, c=crop: faces.run_gfpgan(net, dev, c),
                control,
                release=False,
            )
            original_t = _face_tensor(crop)
            restored_t = _face_tensor(restored)
            guard = faces.identity_factor(original_t, restored_t, mask)
            need = faces.detail_factor(original_t, restored_t, mask)
            weight = (
                base
                * guard
                * need
                * faces.size_factor(face.eye_distance / scale_hint)
                * faces.score_factor(face.score)
            )
            if guard * faces.size_factor(face.eye_distance / scale_hint) < 0.75:
                result.protected += 1  # identity protection held this face back
            if weight <= 0.02:
                continue
            change = faces.harmonise(original_t, restored_t, mode) - original_t
            faces.paste_face(out, change[0].permute(1, 2, 0).numpy(), image_to_face, mask * weight)
            result.restored += 1
            log.debug(
                "Face %d: score %.3f, eyes %.0f px, identity %.2f, need %.2f, weight %.2f",
                index, face.score, face.eye_distance, guard, need, weight,
            )  # fmt: skip
        return out, result


def _face_tensor(face: np.ndarray):  # noqa: ANN202
    import torch

    return (
        torch.from_numpy(np.ascontiguousarray(face, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0)
    )


def _is_neutral(rgb: np.ndarray) -> bool:
    sample = fl.proxy(rgb, 256).astype(np.int16)
    return bool(
        np.abs(sample[..., 0] - sample[..., 1]).max() <= 1
        and np.abs(sample[..., 1] - sample[..., 2]).max() <= 1
    )


def _blend(base: np.ndarray, ai: np.ndarray, weight: float) -> np.ndarray:
    """base + weight × (ai − base), in bands of rows (in place on ``ai``)."""
    rows = max(1, (1 << 20) // max(1, ai.shape[1]))
    for y in range(0, ai.shape[0], rows):
        a = ai[y : y + rows].astype(np.float32)
        b = base[y : y + rows].astype(np.float32)
        a -= b
        a *= weight
        a += b + 0.5
        np.clip(a, 0, 255, out=a)
        ai[y : y + rows] = a.astype(np.uint8)
    return ai


class _Plan:
    """Progress fractions for the stages that will actually run."""

    WEIGHTS: ClassVar[dict[str, float]] = {
        "prepare": 0.01,
        "dust": 0.06,
        "scratches": 0.08,
        "noise": 0.04,
        "tones": 0.02,
        "colorize": 0.06,
        "look": 0.03,
        "upscale": 0.5,
        "faces": 0.2,
        "sharpen": 0.03,
    }

    def __init__(
        self,
        settings: RestorationSettings,
        stages: object,
        colorize_now: bool,
        lighting: Adjustments,
        monochrome: bool,
    ) -> None:
        s = stages
        active = {
            "prepare": monochrome,
            "dust": bool(s.dust),  # type: ignore[attr-defined]
            "scratches": bool(s.scratches),  # type: ignore[attr-defined]
            "noise": bool(s.noise),  # type: ignore[attr-defined]
            "tones": bool(s.fading or s.auto_color),  # type: ignore[attr-defined]
            "colorize": colorize_now,
            "look": True,
            "upscale": settings.scale > 1 or bool(s.detail),  # type: ignore[attr-defined]
            "faces": s.face != FACE_OFF,  # type: ignore[attr-defined]
            "sharpen": bool(s.sharpness),  # type: ignore[attr-defined]
        }
        total = sum(w for k, w in self.WEIGHTS.items() if active[k]) or 1.0
        self._start: dict[str, float] = {}
        self._span: dict[str, float] = {}
        pos = 0.0
        for key, weight in self.WEIGHTS.items():
            share = weight / total if active[key] else 0.0
            self._start[key] = pos
            self._span[key] = share
            pos += share

    def at(self, stage: str) -> float:
        return self._start[stage]

    def span(self, stage: str) -> float:
        return self._span[stage]
