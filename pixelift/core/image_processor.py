"""Single-image pipeline: load -> lighting -> camera look -> upscale -> look
finish (sharpening, grain) -> restore alpha/mode -> save.

With ``ProcessingOptions.restoration`` set, the photo-restoration pipeline
(which applies the lighting and camera look and upscales itself) replaces the
middle steps;
loading, metadata, output naming and saving are shared.

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
from typing import TYPE_CHECKING, Literal

import numpy as np
from PIL import Image

from pixelift.core.camera_looks import CameraLookSettings, apply_look, finish_look
from pixelift.core.control import JobControl
from pixelift.core.errors import ImageTooLargeError, OutputError, UpscalerError
from pixelift.core.lighting import LightingSettings, apply_lighting
from pixelift.core.restoration.settings import RestorationSettings
from pixelift.core.upscaler import Upscaler
from pixelift.utils import image_utils as iu

if TYPE_CHECKING:
    from pixelift.core.restoration.pipeline import Restorer
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
    camera_look: CameraLookSettings = field(default_factory=CameraLookSettings)
    # Photo restoration instead of plain upscaling (None = upscaling).
    restoration: RestorationSettings | None = None

    @property
    def output_scale(self) -> int:
        """How much larger the output is than the input."""
        return self.restoration.scale if self.restoration is not None else self.scale

    def validate(self) -> None:
        if self.restoration is not None:
            self.restoration.validate()
        elif self.scale not in (2, 3, 4):
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
    look: str = ""  # CameraLookSettings.tag() the output was made with
    restoration: str | None = None  # restoration_tag() the output was made with; None: upscaled
    colorized: bool = False
    faces: int = 0  # faces restored

    @property
    def restored(self) -> bool:
        return self.restoration is not None


def will_colorize(source: Path, options: ProcessingOptions) -> bool:
    """Whether restoring ``source`` with ``options`` colorizes it.

    Only black-and-white photos are ever colorized, and only when asked to.
    """
    restoration = options.restoration
    if restoration is None or not restoration.colorize or restoration.colorize_strength <= 0:
        return False
    from pixelift.core.restoration.analysis import detect_monochrome_file

    try:
        return detect_monochrome_file(source).monochrome
    except (OSError, UpscalerError):
        return False


def restoration_tag(options: ProcessingOptions, colorized: bool) -> str:
    """Identifies what a restored output was made with (for staleness checks)."""
    assert options.restoration is not None
    return f"{options.restoration.tag(colorized)}|{int(colorized)}|{options.restoration.scale}"


def restoration_template(restoration: RestorationSettings, colorized: bool) -> str:
    """``{name}_restored[-tag][_colorized][_{scale}x]`` (lighting/look tags appended later)."""
    tag = restoration.tag(colorized)
    template = "{name}_restored" + (f"-{tag}" if tag else "")
    if colorized:
        template += "_colorized"
    if restoration.scale > 1:
        template += "_{scale}x"
    return template


def output_path_for(
    source: Path,
    width: int,
    height: int,
    options: ProcessingOptions,
    colorized: bool | None = None,
) -> Path:
    _, ext = iu.OUTPUT_FORMATS[options.output_format]
    restoration = options.restoration
    folder = options.output_dir or iu.default_output_dir(source, restored=restoration is not None)
    scale = options.output_scale
    if restoration is not None:
        if colorized is None:
            colorized = will_colorize(source, options)
        template = restoration_template(restoration, colorized)
    else:
        template = options.filename_template
    name = iu.render_filename(
        template,
        source,
        scale,
        options.model,
        width * scale,
        height * scale,
        ext,
        options.lighting.tag(),
        options.camera_look.tag(),
    )
    return folder / name


def reserve_output(
    output: Path, existing: ExistingPolicy, claim: OutputClaim | None, source: Path | None = None
) -> Path:
    """``output``, or ``output (2)``… when another source already claimed it.

    Never the ``source`` file itself: originals are never overwritten, whatever
    the output folder and file-name template.
    """
    source_key = _same_file_key(source) if source is not None else None
    for candidate in iu.numbered_paths(output):
        if source_key is not None and _same_file_key(candidate) == source_key:
            continue
        if existing == "rename" and candidate.exists():
            continue
        if claim is None or claim(candidate):
            return candidate
    raise OutputError(
        f"Could not find a free file name for {output.name}.",
        ["Choose a different output folder in Settings"],
    )


def _same_file_key(path: Path) -> tuple[int, int] | Path:
    try:
        stat = path.stat()
        return (stat.st_dev, stat.st_ino)
    except OSError:
        return path.resolve()


def check_feasible(width: int, height: int, options: ProcessingOptions, output: Path) -> None:
    """Fail early (before minutes of work) on impossible jobs."""
    out_w, out_h = width * options.output_scale, height * options.output_scale
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
    if options.restoration is not None:
        # Each stage keeps its input and output (3 bytes/pixel each), and
        # upscaling blends the AI result with a resized copy.
        needed += width * height * 3 * 4 + out_w * out_h * 3 * 2
        ai_scale = options.restoration.ai_scale()
        if ai_scale > options.output_scale:
            # Detail reconstruction: the AI result at 2× before it is resized back.
            needed += width * height * ai_scale * ai_scale * 3
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
    restorer: Restorer | None = None,
) -> ProcessResult:
    """Upscale (or restore) one file and write the result. Raises UpscalerError subclasses.

    ``restorer`` is the batch's shared restoration engine (created on demand).
    The source file is only ever read.
    """
    started = time.monotonic()
    source = Path(source)
    options.validate()
    report = progress or (lambda _f, _s: None)
    scale = options.output_scale
    restoring = options.restoration is not None
    colorize = will_colorize(source, options) if restoring else False
    tag = restoration_tag(options, colorize) if restoring else None

    info = iu.probe_image(source)
    output = reserve_output(
        output_path_for(source, info.width, info.height, options, colorize),
        options.existing,
        claim,
        source,
    )
    if options.existing == "skip" and output.exists():
        log.info("Skipping %s: %s exists", source, output)
        out_size = (info.width * scale, info.height * scale)
        return ProcessResult(
            source,
            output,
            (info.width, info.height),
            out_size,
            skipped=True,
            note="Output already exists",
            lighting=options.lighting.tag(),
            look=options.camera_look.tag(),
            restoration=tag,
            colorized=colorize,
        )
    check_feasible(info.width, info.height, options, output)

    report(0.0, "Loading")
    loaded = iu.load_image(source)
    if control:
        control.check()

    lighting = options.lighting.adjustments()
    look = options.camera_look.recipe()
    note = ""
    faces = 0
    if restoring:
        rgb, keep_gray, faces, note = _restore(
            source, loaded, options, upscaler, restorer, colorize, report, control
        )
    else:
        # Lighting goes before the AI model: it is cheaper at the input
        # resolution, and the model then reconstructs detail from the corrected
        # tones (which is also what the preview shows). In place, so no extra
        # full-size copy.
        if not lighting.is_neutral:
            report(0.01, "Adjusting lighting")
            apply_lighting(loaded.rgb, lighting, out=loaded.rgb)
            if control:
                control.check()
        # The camera look follows the lighting, for the same reasons; only its
        # sharpening and grain wait for the upscaled image (the model would
        # smooth grain away and exaggerate sharpening).
        if not look.grade_neutral:
            report(0.015, "Applying camera look")
            apply_look(loaded.rgb, look, out=loaded.rgb)
            if control:
                control.check()

        def on_tiles(done: int, total: int) -> None:
            report(0.02 + 0.9 * done / max(total, 1), f"Upscaling (tile {done}/{total})")

        report(0.02, "Upscaling")
        rgb = upscaler.upscale(
            loaded.rgb, options.scale, options.model, progress=on_tiles, control=control
        )
        if not look.finish_neutral:
            report(0.92, "Finishing camera look")
            rgb = finish_look(rgb, look, out=rgb if rgb.flags.writeable else None)
        # A warmed, cooled or tinted grey source keeps that colour (as previewed).
        keep_gray = loaded.grayscale and not lighting.changes_colour and not look.changes_colour
    out_h, out_w = rgb.shape[:2]
    in_size = (loaded.width, loaded.height)
    del loaded.rgb  # free the decoded input before encoding the output

    report(0.93, "Saving")
    img = Image.fromarray(rgb, "RGB")
    del rgb
    if loaded.alpha is not None:
        alpha = Image.fromarray(loaded.alpha, "L").resize((out_w, out_h), Image.Resampling.LANCZOS)
        if options.output_format == "jpeg":
            background = Image.new("RGB", img.size, options.jpeg_background)
            background.paste(img, mask=alpha)
            img = background
            note = _join(note, "Transparency flattened (JPEG has no alpha channel)")
        else:
            img.putalpha(alpha)
    elif keep_gray:
        img = img.convert("L")

    fmt, _ = iu.OUTPUT_FORMATS[options.output_format]
    meta = loaded.metadata if options.preserve_metadata else None
    kwargs = save_kwargs(fmt, meta, options.quality, scale)
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
    verb = "Restored" if restoring else "Upscaled"
    log.info("%s %s -> %s (%dx%d) in %.1fs", verb, source, output, out_w, out_h, elapsed)
    report(1.0, "Done")
    return ProcessResult(
        source,
        output,
        in_size,
        (out_w, out_h),
        elapsed,
        note=note,
        lighting=options.lighting.tag(),
        look=options.camera_look.tag(),
        restoration=tag,
        colorized=colorize,
        faces=faces,
    )


def _restore(
    source: Path,
    loaded: iu.LoadedImage,
    options: ProcessingOptions,
    upscaler: Upscaler,
    restorer: Restorer | None,
    colorize: bool,
    report: Progress,
    control: JobControl | None,
) -> tuple[np.ndarray, bool, int, str]:
    """Run the restoration pipeline: (rgb, save as grey, faces restored, note)."""
    from pixelift.core.restoration.analysis import MonoInfo, detect_monochrome_file
    from pixelift.core.restoration.pipeline import Restorer

    assert options.restoration is not None
    restorer = restorer or Restorer(upscaler)
    # The same (cached) detection that named the output file.
    mono = MonoInfo(True, False, 0.0) if loaded.grayscale else detect_monochrome_file(source)
    settings = options.restoration
    if mono.monochrome != colorize and settings.colorize and settings.colorize_strength > 0:
        # Only possible if the file changed since it was named; follow the name.
        mono = MonoInfo(colorize, mono.toned, mono.colorfulness)
    result = restorer.restore(
        loaded.rgb,
        settings,
        model=options.model,
        lighting=options.lighting.adjustments(),
        look=options.camera_look.recipe(),
        mono=mono,
        progress=lambda f, stage: report(0.01 + 0.91 * f, stage),
        control=control,
    )
    notes = []
    if result.colorized:
        notes.append("Colorized")
    elif result.monochrome:
        notes.append("Black & white")
    if result.faces.restored:
        count = result.faces.restored
        notes.append(f"{count} face{'s' if count != 1 else ''} restored")
    return result.rgb, result.monochrome, result.faces.restored, " · ".join(notes)


def _join(first: str, second: str) -> str:
    return f"{first} · {second}" if first else second


def test_pattern(size: int = 48) -> np.ndarray:
    """Small synthetic RGB image used for the first-run self test."""
    y, x = np.mgrid[0:size, 0:size]
    r = (x * 255 // max(size - 1, 1)).astype(np.uint8)
    g = (y * 255 // max(size - 1, 1)).astype(np.uint8)
    b = (((x // 6 + y // 6) % 2) * 255).astype(np.uint8)
    return np.dstack([r, g, b])
