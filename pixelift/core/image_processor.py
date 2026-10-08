"""Single-image pipeline: load -> lighting -> upscale -> restore alpha/mode -> save.

Shared by the GUI and the CLI.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from pixelift.core.control import JobControl
from pixelift.core.errors import ImageTooLargeError, OutputError, UpscalerError
from pixelift.core.lighting import LightingSettings, apply_lighting
from pixelift.core.upscaler import Upscaler
from pixelift.utils import image_utils as iu
from pixelift.utils.metadata import save_kwargs
from pixelift.utils.system import available_ram_bytes

log = logging.getLogger(__name__)

_UMASK = os.umask(0o022)
os.umask(_UMASK)

ExistingPolicy = Literal["skip", "overwrite", "rename"]
Progress = Callable[[float, str], None]  # (fraction 0..1, stage text)
# Reserve an output path for this source; False if another source in the same
# batch already writes there (e.g. photo.jpg and photo.png -> photo_4x.png).
OutputClaim = Callable[[Path], bool]


@dataclass
class ProcessingOptions:
    scale: int = 4
    model: str = "realesrgan"
    output_format: str = "png"  # png | jpeg | webp
    quality: int = 92
    output_dir: Path | None = None  # None -> <source dir>/upscaled
    filename_template: str = iu.DEFAULT_TEMPLATE
    existing: ExistingPolicy = "skip"
    preserve_metadata: bool = True
    jpeg_background: tuple[int, int, int] = (255, 255, 255)
    lighting: LightingSettings = field(default_factory=LightingSettings)

    def validate(self) -> None:
        if self.scale not in (2, 3, 4):
            raise ValueError("scale must be 2, 3 or 4")
        if self.output_format not in iu.OUTPUT_FORMATS:
            raise ValueError(f"unsupported output format: {self.output_format}")
        if not 1 <= self.quality <= 100:
            raise ValueError("quality must be between 1 and 100")


@dataclass
class ProcessResult:
    source: Path
    output: Path
    input_size: tuple[int, int]
    output_size: tuple[int, int]
    seconds: float = 0.0
    skipped: bool = False
    note: str = ""
    lighting: str = ""  # LightingSettings.tag() the output was made with


def output_path_for(source: Path, width: int, height: int, options: ProcessingOptions) -> Path:
    _, ext = iu.OUTPUT_FORMATS[options.output_format]
    folder = options.output_dir or iu.default_output_dir(source)
    name = iu.render_filename(
        options.filename_template,
        source,
        options.scale,
        options.model,
        width * options.scale,
        height * options.scale,
        ext,
        options.lighting.tag(),
    )
    return folder / name


def reserve_output(output: Path, existing: ExistingPolicy, claim: OutputClaim | None) -> Path:
    """``output``, or ``output (2)``… when another source already claimed it."""
    for candidate in iu.numbered_paths(output):
        if existing == "rename" and candidate.exists():
            continue
        if claim is None or claim(candidate):
            return candidate
    raise OutputError(
        f"Could not find a free file name for {output.name}.",
        ["Choose a different output folder in Settings"],
    )


def check_feasible(width: int, height: int, options: ProcessingOptions, output: Path) -> None:
    """Fail early (before minutes of work) on impossible jobs."""
    out_w, out_h = width * options.scale, height * options.scale
    if options.output_format == "webp" and max(out_w, out_h) > iu.WEBP_MAX_DIMENSION:
        raise ImageTooLargeError(
            f"The result ({out_w}×{out_h}) exceeds WebP's maximum of "
            f"{iu.WEBP_MAX_DIMENSION} pixels per side.",
            ["Choose PNG output", "Use 2× instead of 4×"],
        )
    if options.output_format == "jpeg" and max(out_w, out_h) > iu.JPEG_MAX_DIMENSION:
        raise ImageTooLargeError(
            f"The result ({out_w}×{out_h}) exceeds JPEG's maximum size.", ["Choose PNG output"]
        )
    # Output buffer + one copy while encoding, plus the decoded input.
    needed = out_w * out_h * 4 * 2 + width * height * 4
    available = available_ram_bytes()
    if needed > available:
        raise ImageTooLargeError(
            f"Upscaling to {out_w}×{out_h} needs about {iu.human_size(needed)} of memory, "
            f"but only {iu.human_size(available)} is available.",
            ["Use 2× instead of 4×", "Close other applications", "Split the image into parts"],
        )
    # Rough PNG size estimate: raw / 2.
    if iu.disk_free(output.parent) < out_w * out_h * 2:
        raise OutputError(
            f"Not enough free disk space in {output.parent}.",
            ["Free up disk space", "Choose another output folder"],
        )


def _save_atomic(img: Image.Image, target: Path, fmt: str, kwargs: dict[str, object]) -> None:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputError(
            f"Cannot create the output folder {target.parent}: {exc.strerror}",
            ["Choose a different output folder in Settings"],
        ) from exc
    fd, tmp_name = tempfile.mkstemp(prefix=".upscaling-", suffix=target.suffix, dir=target.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        img.save(tmp, fmt, **kwargs)
        os.chmod(tmp, 0o666 & ~_UMASK)  # mkstemp creates 0600; honour the umask
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def process_image(
    source: Path,
    options: ProcessingOptions,
    upscaler: Upscaler,
    progress: Progress | None = None,
    control: JobControl | None = None,
    claim: OutputClaim | None = None,
) -> ProcessResult:
    """Upscale one file and write the result. Raises UpscalerError subclasses."""
    started = time.monotonic()
    source = Path(source)
    options.validate()
    report = progress or (lambda _f, _s: None)

    info = iu.probe_image(source)
    output = reserve_output(
        output_path_for(source, info.width, info.height, options), options.existing, claim
    )
    if options.existing == "skip" and output.exists():
        log.info("Skipping %s: %s exists", source, output)
        out_size = (info.width * options.scale, info.height * options.scale)
        return ProcessResult(
            source,
            output,
            (info.width, info.height),
            out_size,
            skipped=True,
            note="Output already exists",
            lighting=options.lighting.tag(),
        )
    check_feasible(info.width, info.height, options, output)

    report(0.0, "Loading")
    loaded = iu.load_image(source)
    if control:
        control.check()

    # Lighting goes before the AI model: it is cheaper at the input resolution,
    # and the model then reconstructs detail from the corrected tones (which is
    # also what the preview shows). In place, so no extra full-size copy.
    lighting = options.lighting.adjustments()
    if not lighting.is_neutral:
        report(0.01, "Adjusting lighting")
        apply_lighting(loaded.rgb, lighting, out=loaded.rgb)
        if control:
            control.check()

    def on_tiles(done: int, total: int) -> None:
        report(0.02 + 0.9 * done / max(total, 1), f"Upscaling (tile {done}/{total})")

    report(0.02, "Upscaling")
    rgb = upscaler.upscale(
        loaded.rgb, options.scale, options.model, progress=on_tiles, control=control
    )
    out_h, out_w = rgb.shape[:2]
    in_size = (loaded.width, loaded.height)
    del loaded.rgb  # free the decoded input before encoding the output

    report(0.93, "Saving")
    img = Image.fromarray(rgb, "RGB")
    del rgb
    note = ""
    if loaded.alpha is not None:
        alpha = Image.fromarray(loaded.alpha, "L").resize((out_w, out_h), Image.Resampling.LANCZOS)
        if options.output_format == "jpeg":
            background = Image.new("RGB", img.size, options.jpeg_background)
            background.paste(img, mask=alpha)
            img = background
            note = "Transparency flattened (JPEG has no alpha channel)"
        else:
            img.putalpha(alpha)
    elif loaded.grayscale and not lighting.changes_colour:
        # A warmed, cooled or tinted grey source keeps that colour (as previewed).
        img = img.convert("L")

    fmt, _ = iu.OUTPUT_FORMATS[options.output_format]
    meta = loaded.metadata if options.preserve_metadata else None
    kwargs = save_kwargs(fmt, meta, options.quality, options.scale)
    if control:
        control.check()
    try:
        _save_atomic(img, output, fmt, kwargs)
    except UpscalerError:
        raise
    except OSError as exc:
        log.exception("Saving %s failed", output)
        raise OutputError(
            f"Could not write {output.name}: {exc.strerror or exc}",
            ["Check the output folder is writable", "Make sure the disk is not full"],
        ) from exc
    finally:
        img.close()

    elapsed = time.monotonic() - started
    log.info("Upscaled %s -> %s (%dx%d) in %.1fs", source, output, out_w, out_h, elapsed)
    report(1.0, "Done")
    return ProcessResult(
        source,
        output,
        in_size,
        (out_w, out_h),
        elapsed,
        note=note,
        lighting=options.lighting.tag(),
    )


def test_pattern(size: int = 48) -> np.ndarray:
    """Small synthetic RGB image used for the first-run self test."""
    y, x = np.mgrid[0:size, 0:size]
    r = (x * 255 // max(size - 1, 1)).astype(np.uint8)
    g = (y * 255 // max(size - 1, 1)).astype(np.uint8)
    b = (((x // 6 + y // 6) % 2) * 255).astype(np.uint8)
    return np.dstack([r, g, b])
