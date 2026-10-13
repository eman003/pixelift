"""Single-image pipeline: load -> lighting -> camera look -> upscale -> HDR
enhancement -> look finish (sharpening, grain) -> restore alpha/mode -> save.

With HDR output on (``ProcessingOptions.hdr_output``), the SDR result is
converted to HDR before saving, or, for HDR sources, the HDR pixels go
through the AI model as an invertible 16-bit proxy instead (see
``core/hdr_output``); the file is written as PQ, HLG or a gain-map JPEG
and verified.

With ``ProcessingOptions.restoration`` set, the photo-restoration pipeline
(which applies the lighting and camera look and upscales itself) replaces the
middle steps;
loading, metadata, output naming and saving are shared.

Shared by the GUI and the CLI.
"""

from __future__ import annotations

import dataclasses
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

from pixelift.core import upscale_cache
from pixelift.core.camera_looks import CameraLookSettings, LookRecipe, apply_look, finish_look
from pixelift.core.control import JobControl
from pixelift.core.errors import ImageTooLargeError, OutputError, UpscalerError
from pixelift.core.hdr import HdrSettings, analyse, apply_hdr
from pixelift.core.hdr_output import codecs as hdr_codecs
from pixelift.core.hdr_output import gainmap
from pixelift.core.hdr_output.colour import ProxyCurve
from pixelift.core.hdr_output.pipeline import (
    HdrOutputSettings,
    LinearImage,
    ProxyReconstructed,
    SdrConverted,
    make_proxy,
    sdr_colour_space,
    write_hdr,
)
from pixelift.core.lighting import Adjustments, LightingSettings, apply_lighting
from pixelift.core.restoration.settings import RestorationSettings
from pixelift.core.upscaler import Upscaler
from pixelift.utils import image_utils as iu

if TYPE_CHECKING:
    from pixelift.core.restoration.pipeline import Restorer
from pixelift.utils.metadata import ImageMetadata, save_kwargs, without_location
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
    strip_location: bool = False  # leave GPS coordinates out of copied EXIF
    jpeg_background: tuple[int, int, int] = (255, 255, 255)
    lighting: LightingSettings = field(default_factory=LightingSettings)
    camera_look: CameraLookSettings = field(default_factory=CameraLookSettings)
    # HDR enhancement after upscaling: Upscale mode only, so read it through
    # effective_hdr() (neutral when restoring).
    hdr: HdrSettings = field(default_factory=HdrSettings)
    # SDR or HDR (PQ) output files; Upscale mode only (effective_hdr_output()).
    hdr_output: HdrOutputSettings = field(default_factory=HdrOutputSettings)
    # Keep upscales made with HDR on for later HDR changes (off for one-shot
    # runs such as the CLI or batches larger than the cache).
    cache_upscale: bool = True
    # Photo restoration instead of plain upscaling (None = upscaling).
    restoration: RestorationSettings | None = None

    def effective_hdr(self) -> HdrSettings:
        """The HDR settings that apply: none when restoring."""
        return HdrSettings() if self.restoration is not None else self.hdr

    def effective_hdr_output(self) -> HdrOutputSettings:
        """The output dynamic range that applies: SDR when restoring."""
        if self.restoration is not None:
            return HdrOutputSettings()
        return self.hdr_output

    @property
    def output_scale(self) -> int:
        """How much larger the output is than the input."""
        return self.restoration.scale if self.restoration is not None else self.scale

    def validate(self) -> None:
        if self.restoration is not None:
            self.restoration.validate()
        else:
            if self.scale not in (2, 3, 4):
                raise ValueError("scale must be 2, 3 or 4")
            iu.check_template(self.filename_template)
        if self.output_format not in iu.OUTPUT_FORMATS:
            raise ValueError(f"unsupported output format: {self.output_format}")
        if not 1 <= self.quality <= 100:
            raise ValueError("quality must be between 1 and 100")
        self.hdr_output.validate()


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
    hdr: str = ""  # hdr_tag() the output was made with
    restoration: str | None = None  # restoration_tag() the output was made with; None: upscaled
    colorized: bool = False
    faces: int = 0  # faces restored
    output_settings: str = ""  # output_tag() the output was made with

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


def output_tag(options: ProcessingOptions) -> str:
    """Identifies the model, scale and file settings an output was made with
    (for staleness checks; lighting, look and restoration have their own tags)."""
    return "|".join(
        str(part)
        for part in (
            options.model,
            options.output_scale,
            options.output_format,
            options.quality,
            options.output_dir or "",
            options.filename_template,
            options.preserve_metadata,
            options.strip_location,
            options.effective_hdr_output().format if options.effective_hdr_output().enabled else "",
        )
    )


def hdr_tag(options: ProcessingOptions, kept_hdr: bool = False) -> str:
    """The HDR tag of an output made with ``options``: HDR Enhancement's and
    HDR output's (``pq1000``…), "" when neither changes anything.

    ``kept_hdr``: the source is HDR and written as HDR, so HDR Enhancement
    (an SDR tool) does not apply to it.
    """
    enhancement = "" if kept_hdr else options.effective_hdr().tag()
    parts = (enhancement, options.effective_hdr_output().tag())
    return "_".join(part for part in parts if part)


def keeps_hdr(source: Path, options: ProcessingOptions) -> bool:
    """Whether ``source`` is an HDR file that ``options`` write as HDR from its
    own HDR data (lighting, camera look and HDR Enhancement are then skipped)."""
    if not options.effective_hdr_output().enabled:
        return False
    found = iu.hdr_source(source, gain_maps=True)
    if found is not None and found.encoding == "gainmap":
        return gainmap.usable(source)  # an unusable one is read as an SDR photo
    return found is not None


def result_tags(options: ProcessingOptions, kept_hdr: bool) -> tuple[str, str, str]:
    """(lighting, look, HDR) tags of an output: what it was really made with."""
    if kept_hdr:
        return "", "", hdr_tag(options, kept_hdr=True)
    return options.lighting.tag(), options.camera_look.tag(), hdr_tag(options)


def output_extension(options: ProcessingOptions) -> str:
    hdr_output = options.effective_hdr_output()
    if hdr_output.enabled:
        return hdr_output.extension
    return iu.OUTPUT_FORMATS[options.output_format][1]


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
    ext = output_extension(options)
    restoration = options.restoration
    folder = options.output_dir or iu.default_output_dir(source, restored=restoration is not None)
    scale = options.output_scale
    if restoration is not None:
        if colorized is None:
            colorized = will_colorize(source, options)
        template = restoration_template(restoration, colorized)
    else:
        template = options.filename_template
    lighting, look, hdr = result_tags(options, keeps_hdr(source, options))
    name = iu.render_filename(
        template, source, scale, options.model, width * scale, height * scale, ext,
        lighting, look, hdr,
    )  # fmt: skip
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
    hdr_output = options.effective_hdr_output()
    if hdr_output.enabled:
        _check_hdr_feasible(out_w, out_h, hdr_output, output)
    elif options.output_format == "webp" and max(out_w, out_h) > iu.WEBP_MAX_DIMENSION:
        raise ImageTooLargeError(
            f"The result ({out_w}×{out_h}) exceeds WebP's maximum of "
            f"{iu.WEBP_MAX_DIMENSION} pixels per side.",
            ["Choose PNG output", "Use 2× instead of 4×"],
        )
    if (
        not hdr_output.enabled
        and options.output_format == "jpeg"
        and max(out_w, out_h) > iu.JPEG_MAX_DIMENSION
    ):
        raise ImageTooLargeError(
            f"The result ({out_w}×{out_h}) exceeds JPEG's maximum size.", ["Choose PNG output"]
        )
    # Output buffer + one copy while encoding, plus the decoded input.
    needed = out_w * out_h * 4 * 2 + width * height * 4
    if hdr_output.enabled:
        # The 16-bit proxy (HDR sources) or the AVIF planes, plus HDR decoding.
        needed += out_w * out_h * 6 + width * height * 40
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
    # Rough, generous file size estimates: PNG about raw / 2; JPEG and WebP
    # are lossy and far smaller.
    bytes_per_pixel = 2.0 if options.output_format == "png" else 0.5
    if hdr_output.enabled:
        bytes_per_pixel = 4.0 if hdr_output.format == "png" else 1.0
    if iu.disk_free(output.parent) < out_w * out_h * bytes_per_pixel:
        raise OutputError(
            f"Not enough free disk space in {output.parent}.",
            ["Free up disk space", "Choose another output folder"],
        )


def _check_hdr_feasible(
    out_w: int, out_h: int, hdr_output: HdrOutputSettings, output: Path
) -> None:
    if hdr_output.format == "avif" and not hdr_codecs.avif_available():
        raise OutputError(
            "HDR AVIF export needs ffmpeg with the libaom-av1 encoder, which was not found.",
            ["Choose 16-bit PNG as the HDR format", "Install ffmpeg (sudo apt install ffmpeg)"],
        )
    if hdr_output.gain_map and max(out_w, out_h) > iu.JPEG_MAX_DIMENSION:
        raise ImageTooLargeError(
            f"The result ({out_w}×{out_h}) exceeds JPEG's maximum size.",
            ["Choose 16-bit PNG as the HDR format", "Use 2× instead of 4×"],
        )
    if hdr_output.format == "avif" and max(out_w, out_h) > AVIF_MAX_DIMENSION:
        raise ImageTooLargeError(
            f"The result ({out_w}×{out_h}) exceeds AVIF's maximum of "
            f"{AVIF_MAX_DIMENSION} pixels per side.",
            ["Choose 16-bit PNG as the HDR format", "Use 2× instead of 4×"],
        )


AVIF_MAX_DIMENSION = 65536  # AV1 frame size limit (one image, no grid)


def _save_hdr(
    linear: LinearImage,
    target: Path,
    options: ProcessingOptions,
    alpha: np.ndarray | None,
    meta: ImageMetadata | None,
    control: JobControl | None,
) -> str:
    """Write ``linear`` as an HDR file (temporary name, verified, then renamed)."""
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
    dpi = meta.scaled_dpi(options.output_scale) if meta is not None else None
    try:
        result = write_hdr(
            tmp,
            linear,
            options.effective_hdr_output(),
            alpha=alpha,
            exif=meta.scaled_exif(options.output_scale) if meta is not None else None,
            dpi=dpi,
            quality=options.quality,
            check=control.check if control else None,
        )
        os.chmod(tmp, 0o666 & ~_UMASK)
        os.replace(tmp, target)
    except hdr_codecs.HdrCodecError as exc:
        log.error("HDR export of %s failed: %s", target, exc)
        raise OutputError(str(exc), exc.suggestions) from exc
    except OSError as exc:
        log.exception("Saving %s failed", target)
        raise OutputError(
            f"Could not write {target.name}: {exc.strerror or exc}",
            ["Check the output folder is writable", "Make sure the disk is not full"],
        ) from exc
    finally:
        tmp.unlink(missing_ok=True)
    return result.summary()


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
    tags = result_tags(options, keeps_hdr(source, options))
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
            lighting=tags[0],
            look=tags[1],
            hdr=tags[2],
            restoration=tag,
            colorized=colorize,
            output_settings=output_tag(options),
        )
    check_feasible(info.width, info.height, options, output)

    report(0.0, "Loading")
    hdr_output = options.effective_hdr_output()
    source_id = upscale_cache.source_identity(source)  # before reading: see source_identity
    loaded = iu.load_image(
        source, scene_white=hdr_output.sdr_white, gain_maps=hdr_output.enabled and not restoring
    )
    if control:
        control.check()

    lighting = options.lighting.adjustments()
    look = options.camera_look.recipe()
    note = ""
    faces = 0
    note = loaded.note
    linear: LinearImage | None = None  # the HDR result, when writing HDR
    hdr_alpha: np.ndarray | None = None  # float alpha of an HDR source
    if loaded.hdr is not None and not (hdr_output.enabled and not restoring):
        note = _join(note, "HDR source tone-mapped to SDR (its HDR range is not kept)")
        loaded.hdr = None
    if restoring:
        rgb, keep_gray, faces, restore_note = _restore(
            source, loaded, options, upscaler, restorer, colorize, report, control
        )
        note = _join(note, restore_note) if restore_note else note
    elif loaded.hdr is not None:
        linear, hdr_alpha, hdr_note = _upscale_hdr_source(
            source, source_id, loaded, options, upscaler, report, control
        )
        note = _join(note, hdr_note) if hdr_note else note
        rgb = np.empty((0, 0, 3), dtype=np.uint8)
        keep_gray = False
    else:
        hdr = options.effective_hdr().recipe()
        # The upscaled image is cached while HDR is on: other HDR settings (or
        # switching HDR off again) then reuse it instead of running the AI
        # model again. Only upscalers that can identify their output exactly
        # (weights, tile size, device) take part.
        grade = dataclasses.replace(look, sharpen=0.0, grain=0.0)
        size = (loaded.height, loaded.width)
        identity = upscaler.cache_identity(options.model, options.scale, size)
        key = None
        if identity is not None:
            key = upscale_cache.cache_key(source_id, options.scale, lighting, grade, *identity)
        shape = (loaded.height * options.scale, loaded.width * options.scale, 3)
        rgb = upscale_cache.load(key, shape) if key else None
        if rgb is not None:
            log.info("Reusing the cached upscale of %s", source)
            report(0.02, "Using the cached upscale")
        else:
            later = not hdr.is_neutral or hdr_output.enabled  # work after the model
            caching = key is not None and options.cache_upscale and later
            rgb = _upscale(loaded, options, upscaler, lighting, look, report, control, later)
            # Identified again: the upscaler may have fallen back to the CPU,
            # or used smaller tiles after running out of memory.
            identity = upscaler.cache_identity(options.model, options.scale, size)
            if caching and identity is not None:
                report(0.85, "Caching the upscaled image")
                key = upscale_cache.cache_key(source_id, options.scale, lighting, grade, *identity)
                if key:
                    upscale_cache.store(key, rgb)
        if control:
            control.check()
        # HDR after the AI model: the model would otherwise treat boosted
        # local contrast and compression artefacts as detail, and the cached
        # upscale stays valid while the HDR settings change.
        if not hdr.is_neutral:
            report(0.86, "Enhancing dynamic range")
            out = rgb if rgb.flags.writeable else None  # a cached upscale is read-only
            rgb = apply_hdr(rgb, hdr, out=out, plan=analyse(rgb), check=control and control.check)
        if not look.finish_neutral:
            report(0.92, "Finishing camera look")
            rgb = finish_look(rgb, look, out=rgb if rgb.flags.writeable else None)
        # A warmed, cooled or tinted grey source keeps that colour (as previewed).
        keep_gray = loaded.grayscale and not lighting.changes_colour and not look.changes_colour
    in_size = (loaded.width, loaded.height)
    if hdr_output.enabled and not restoring:
        if linear is None:
            rgb, space, space_note = sdr_colour_space(rgb, loaded.metadata.icc_profile)
            linear = SdrConverted(rgb, space, hdr_output)
            note = _join(note, "Converted from SDR to HDR (no detail is recovered)")
            if space_note:
                note = _join(note, space_note)
        out_w, out_h = linear.width, linear.height
        del loaded.rgb
        meta = loaded.metadata if options.preserve_metadata else None
        if meta is not None and options.strip_location:
            meta = without_location(meta)
        report(0.93, "Encoding HDR")
        alpha = hdr_alpha if hdr_alpha is not None else loaded.alpha
        summary = _save_hdr(linear, output, options, alpha, meta, control)
        note = _join(note, summary)
        del linear, rgb
        elapsed = time.monotonic() - started
        log.info("Upscaled %s -> %s (%dx%d, HDR) in %.1fs", source, output, out_w, out_h, elapsed)
        report(1.0, "Done")
        return ProcessResult(
            source,
            output,
            in_size,
            (out_w, out_h),
            elapsed,
            note=note,
            lighting=tags[0],
            look=tags[1],
            hdr=tags[2],
            restoration=tag,
            colorized=colorize,
            faces=faces,
            output_settings=output_tag(options),
        )
    out_h, out_w = rgb.shape[:2]
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
    if meta is not None and options.strip_location:
        meta = without_location(meta)
    kwargs = save_kwargs(fmt, meta, options.quality, scale, img.mode)
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
        lighting=tags[0],
        look=tags[1],
        hdr=tags[2],
        restoration=tag,
        colorized=colorize,
        faces=faces,
        output_settings=output_tag(options),
    )


def _upscale(
    loaded: iu.LoadedImage,
    options: ProcessingOptions,
    upscaler: Upscaler,
    lighting: Adjustments,
    look: LookRecipe,
    report: Progress,
    control: JobControl | None,
    hdr_active: bool,
) -> np.ndarray:
    """Lighting, camera-look grade and the AI model (``hdr_active``: leave
    room on the progress bar for caching and HDR)."""
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
    share = 0.83 if hdr_active else 0.9

    def on_tiles(done: int, total: int) -> None:
        report(0.02 + share * done / max(total, 1), f"Upscaling (tile {done}/{total})")

    report(0.02, "Upscaling")
    return upscaler.upscale(
        loaded.rgb, options.scale, options.model, progress=on_tiles, control=control
    )


def _upscale_hdr_source(
    source: Path,
    source_id: str | None,
    loaded: iu.LoadedImage,
    options: ProcessingOptions,
    upscaler: Upscaler,
    report: Progress,
    control: JobControl | None,
) -> tuple[LinearImage, np.ndarray | None, str]:
    """AI-upscale an HDR source through the invertible proxy: (HDR result, alpha, note).

    The model receives ``ProxyCurve.forward`` of the HDR pixels as 16-bit
    sRGB-encoded values (everything up to ~100 cd/m² unchanged, brighter
    tones compressed logarithmically) and its output is mapped back with the
    exact inverse. Lighting, camera looks and HDR Enhancement are SDR tools
    and are not applied. The proxy upscale is cached like an SDR upscale, so
    changing the peak, colour space or format does not run the model again.
    """
    hdr = loaded.hdr
    assert hdr is not None
    notes = []
    sdr_tools = (
        not options.lighting.adjustments().is_neutral
        or not options.camera_look.recipe().grade_neutral
        or not options.camera_look.recipe().finish_neutral
        or not options.effective_hdr().recipe().is_neutral
    )
    if sdr_tools:
        notes.append("Lighting, camera look and HDR Enhancement skipped (SDR-only tools)")
    curve = ProxyCurve(hdr.peak)
    dtype = np.uint16 if upscaler.supports_uint16 else np.uint8
    scene = options.effective_hdr_output().sdr_white if hdr.source.encoding == "linear" else 0
    parts = ("hdr-proxy", round(curve.peak, 6), scene, np.dtype(dtype).name)
    size = hdr.linear.shape[:2]
    wide = dtype == np.uint16
    identity = upscaler.cache_identity(options.model, options.scale, size, wide)
    key = upscale_cache.cache_key(source_id, options.scale, *parts, *identity) if identity else None
    shape = (hdr.linear.shape[0] * options.scale, hdr.linear.shape[1] * options.scale, 3)
    alpha = hdr.alpha
    up = upscale_cache.load(key, shape, dtype) if key else None
    if up is not None:
        log.info("Reusing the cached HDR proxy upscale of %s", source)
        report(0.02, "Using the cached upscale")
    else:
        report(0.01, "Preparing HDR for the AI model")
        proxy = make_proxy(hdr.linear, curve)
        if dtype == np.uint8:
            proxy = (proxy // 257).astype(np.uint8)
        loaded.hdr = None  # the proxy holds everything the model needs
        del hdr
        if control:
            control.check()

        def on_tiles(done: int, total: int) -> None:
            report(0.02 + 0.83 * done / max(total, 1), f"Upscaling (tile {done}/{total})")

        report(0.02, "Upscaling")
        up = upscaler.upscale(
            proxy, options.scale, options.model, progress=on_tiles, control=control
        )
        del proxy
        identity = upscaler.cache_identity(options.model, options.scale, size, wide)
        if options.cache_upscale and identity is not None:
            report(0.85, "Caching the upscaled image")
            key = upscale_cache.cache_key(source_id, options.scale, *parts, *identity)
            if key:
                upscale_cache.store(key, up)
    if dtype == np.uint8:
        up = up.astype(np.uint16) * 257
    return ProxyReconstructed(up, curve), alpha, " · ".join(notes)


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
