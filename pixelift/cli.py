"""Command-line interface. Uses the same engine as the GUI."""

from __future__ import annotations

import argparse
import shutil
import struct
import sys
from pathlib import Path

from pixelift import APP_NAME, __version__
from pixelift.core import camera_looks as cl
from pixelift.core import device_manager as dm
from pixelift.core import hdr
from pixelift.core.batch_processor import BatchEvent, BatchProcessor, EventKind, QueueItem
from pixelift.core.control import JobControl
from pixelift.core.errors import CancelledError, UpscalerError, friendly_error
from pixelift.core.hdr_output import codecs as hdr_codecs
from pixelift.core.hdr_output import pipeline as hdr_output
from pixelift.core.lighting import all_profiles, get_profile
from pixelift.core.model_manager import ModelManager
from pixelift.core.restoration import settings as rs
from pixelift.core.tiling import MIN_TILE
from pixelift.core.upscaler import TorchUpscaler
from pixelift.models import KIND_LABELS, all_families
from pixelift.storage.settings import Settings, load_settings
from pixelift.utils.image_utils import OUTPUT_FORMATS, collect_images, human_size
from pixelift.utils.logging import setup_logging


def _percent(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid percentage: {text!r}") from None
    if not 0 <= value <= 100:
        raise argparse.ArgumentTypeError(f"must be between 0 and 100, not {value}")
    return value


def _non_negative(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid number: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, not {value}")
    return value


def _tile_size(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid tile size: {text!r}") from None
    if value != 0 and value < MIN_TILE:
        # One model call per few pixels would run for hours.
        raise argparse.ArgumentTypeError(
            f"must be 0 (automatic) or at least {MIN_TILE}, not {value}"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        prog="pixelift",
        description=f"{APP_NAME} — upscale images locally with AI (Real-ESRGAN). "
        "Run without arguments to open the desktop app. Your images stay on your computer.",
    )
    parser.add_argument("inputs", nargs="*", type=Path, help="image files or folders")
    restore = parser.add_argument_group(
        "photo restoration",
        "Restore old photographs instead of only upscaling them. Everything runs "
        "locally; originals are never modified (results go to <input dir>/restored).",
    )
    restore.add_argument(
        "-r",
        "--restore",
        action="store_true",
        help="restore photos (dust, scratches, colour, faces)",
    )
    restore.add_argument(
        "--restore-level",
        choices=rs.LEVELS[:-1],
        default=None,
        help="light, standard (default) or heavy; 'custom' uses the values set in the app",
    )
    restore.add_argument(
        "--colorize",
        action="store_true",
        help="colorize black-and-white photos (colour photos are never colorized)",
    )
    restore.add_argument(
        "--colorize-strength", type=_percent, metavar="PERCENT", help="0 = grey … 100 = full colour"
    )
    restore.add_argument(
        "--upscale", action="store_true", help="also upscale restored photos by --scale"
    )
    restore.add_argument(
        "--face", choices=rs.FACE_MODES, help="face restoration (default: natural)"
    )
    restore.add_argument(
        "--fidelity",
        type=_percent,
        metavar="PERCENT",
        help="0 keeps the original pixels, 100 trusts the AI fully (default: 50)",
    )
    restore.add_argument("--modern", choices=rs.MODERN_MODES, help="Modern Finish look")
    parser.add_argument("-s", "--scale", type=int, choices=(2, 4), default=settings.scale)
    parser.add_argument(
        "-m",
        "--model",
        default=settings.model,
        help="model family or model id (see --list-models); default: %(default)s",
    )
    parser.add_argument(
        "-d",
        "--device",
        default=settings.device,
        help="auto, cpu, cuda, cuda:N, xpu (default: %(default)s)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="output directory (default: <input dir>/upscaled)"
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=[*OUTPUT_FORMATS, "jpg"],
        help=f"SDR output format (default: {settings.output_format}); implies --dynamic-range sdr",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=settings.quality,
        help="JPEG/WebP quality 1-100 (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--tile-size",
        type=_tile_size,
        default=settings.tile_size,
        help="tile size in pixels, 0 = automatic (default: %(default)s)",
    )
    parser.add_argument(
        "--template",
        default=settings.filename_template,
        help="output filename template, fields: {name} {scale} {model} "
        "{width} {height} {ext} {lighting} {look} {hdr} (default: %(default)s)",
    )
    parser.add_argument(
        "-l",
        "--lighting",
        choices=[p.id for p in all_profiles()],
        default=settings.lighting_profile,
        help="lighting profile applied before upscaling; 'custom' uses the values "
        "set in the app (default: %(default)s)",
    )
    parser.add_argument(
        "--lighting-intensity",
        type=_percent,
        metavar="PERCENT",
        help="lighting profile strength 0-100; not used with 'custom' "
        f"(default: {settings.lighting_intensity})",
    )
    parser.add_argument(
        "--look",
        choices=[look.id for look in cl.all_looks()]
        + [cl.USER_PREFIX + name for name in settings.camera_look_saved],
        default=settings.camera_look,
        metavar="LOOK",
        help="camera-inspired look applied after the lighting (see --list-looks); "
        "'custom' uses the values set in the app, 'user:NAME' a look saved there "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--look-intensity",
        type=_percent,
        metavar="PERCENT",
        help="camera look strength 0-100; not used with 'custom' "
        f"(default: {settings.camera_look_intensity})",
    )
    parser.add_argument(
        "--grain",
        choices=cl.GRAIN_CHOICES,
        help="film grain: 'auto' uses the look's own grain "
        f"(default: {settings.camera_look_grain})",
    )
    parser.add_argument(
        "--hdr",
        choices=["off", *(p.id for p in hdr.all_profiles())],
        default=settings.hdr_profile if settings.hdr_enabled else "off",
        help="HDR-like enhancement after upscaling (shadow recovery, highlight "
        "compression, local contrast) of the standard-range picture; it does not "
        "make an HDR file (see --dynamic-range). "
        "The profile selected in the app keeps its Advanced settings; others are "
        "used as designed (default: %(default)s)",
    )
    parser.add_argument(
        "--hdr-intensity",
        type=_percent,
        metavar="PERCENT",
        help="HDR strength 0-100 (default: the profile's own; for the profile "
        f"selected in the app, the app's {settings.hdr_intensity})",
    )
    hdr_group = parser.add_argument_group(
        "HDR output",
        "Write true HDR files (PQ or HLG as 16-bit PNG with cICP or 10-bit AVIF, or "
        "an Ultra HDR gain-map JPEG) "
        "instead of SDR ones. SDR sources are converted (no detail is recovered); "
        "HDR sources (PQ/HLG PNG or AVIF, OpenEXR, Radiance .hdr) keep their range.",
    )
    hdr_group.add_argument(
        "--dynamic-range",
        choices=["sdr", "hdr"],
        help="output dynamic range (default: "
        f"{'hdr' if settings.output_range == 'hdr' else 'sdr'}, "
        "or sdr with --format)",
    )
    hdr_group.add_argument(
        "--hdr-format",
        choices=list(hdr_output.FORMATS),
        default=settings.hdr_output_format,
        help="HDR file format: png = 16-bit PNG (lossless, alpha), avif = 10-bit "
        "AVIF (needs ffmpeg with libaom-av1), jpeg = Ultra HDR gain-map JPEG "
        "(an sRGB JPEG every viewer shows, plus a gain map) (default: %(default)s)",
    )
    hdr_group.add_argument(
        "--hdr-transfer",
        choices=list(hdr_output.TRANSFERS),
        default=settings.hdr_transfer,
        help="HDR transfer function for png/avif: pq = SMPTE ST 2084, hlg = BT.2100 "
        "HLG (Rec. 2020, 1000 cd/m² reference) (default: %(default)s)",
    )
    hdr_group.add_argument(
        "--hdr-colour-space",
        "--hdr-color-space",
        dest="hdr_colour_space",
        choices=list(hdr_output.OUTPUT_PRIMARIES),
        default=settings.hdr_output_primaries,
        help="HDR primaries (default: %(default)s)",
    )
    hdr_group.add_argument(
        "--hdr-peak",
        type=int,
        metavar="NITS",
        default=settings.hdr_peak_nits,
        help=f"brightest luminance written, {hdr_output.PEAK_RANGE[0]}-"
        f"{hdr_output.PEAK_RANGE[1]} cd/m² (default: %(default)s)",
    )
    hdr_group.add_argument(
        "--sdr-white",
        type=int,
        metavar="NITS",
        default=settings.hdr_sdr_white,
        help="luminance of SDR white (and of 1.0 in EXR/.hdr files), "
        f"{hdr_output.SDR_WHITE_RANGE[0]}-{hdr_output.SDR_WHITE_RANGE[1]} cd/m² "
        "(default: %(default)s, ITU-R BT.2408)",
    )
    hdr_group.add_argument(
        "--expand-highlights",
        action=argparse.BooleanOptionalAction,
        default=settings.hdr_expand_highlights,
        help="SDR sources: stretch the brightest tones above SDR white "
        "(a creative choice, not a recovery) (default: %(default)s)",
    )
    hdr_group.add_argument(
        "--inspect-hdr",
        metavar="FILE",
        type=Path,
        help="print the colour description of an HDR PNG or AVIF and exit",
    )
    parser.add_argument(
        "--list-looks",
        action="store_true",
        help="list the camera looks (and your saved looks) and exit",
    )
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--overwrite", action="store_const", const="overwrite", dest="existing")
    existing.add_argument(
        "--rename",
        action="store_const",
        const="rename",
        dest="existing",
        help="add a number if the output exists",
    )
    parser.set_defaults(existing="skip")
    parser.add_argument("--no-metadata", action="store_true", help="do not copy EXIF/ICC metadata")
    parser.add_argument(
        "--strip-location",
        action=argparse.BooleanOptionalAction,
        default=settings.strip_location,
        help="leave GPS location out of the copied EXIF",
    )
    parser.add_argument(
        "--no-recursive", action="store_true", help="do not descend into subfolders"
    )
    parser.add_argument(
        "--threads",
        type=_non_negative,
        default=settings.cpu_threads,
        help="CPU threads (0 = all)",
    )
    parser.add_argument("--list-models", action="store_true", help="show models and exit")
    parser.add_argument("--list-devices", action="store_true", help="show devices and exit")
    parser.add_argument(
        "--download-model",
        metavar="ID",
        action="append",
        help="download a model (repeatable; 'recommended' for the default)",
    )
    parser.add_argument("--remove-model", metavar="ID", action="append")
    parser.add_argument(
        "--install-model-file",
        nargs=2,
        metavar=("ID", "FILE"),
        help="install a manually downloaded weights file (checksum verified)",
    )
    parser.add_argument("--gui", action="store_true", help="open the desktop app")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


class _Printer:
    """Progress output: a live bar on terminals, plain lines otherwise."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.tty = sys.stderr.isatty()
        self.width = min(40, max(10, shutil.get_terminal_size().columns - 50))

    def __call__(self, ev: BatchEvent) -> None:
        if ev.kind is EventKind.ITEM_PROGRESS and self.tty and ev.item:
            filled = int(ev.fraction * self.width)
            bar = "█" * filled + "░" * (self.width - filled)
            sys.stderr.write(
                f"\r{bar} {ev.fraction * 100:3.0f}%  {ev.completed}/{ev.total}  "
                f"{ev.item.path.name[:30]}  {ev.item.stage[:24]:<24}"
            )
            sys.stderr.flush()
        elif ev.kind is EventKind.ITEM_FINISHED and ev.item:
            item = ev.item
            if self.tty:
                sys.stderr.write("\r\033[K")
            if item.result and not item.result.skipped:
                w, h = item.result.output_size
                print(f"✓ {item.path} → {item.result.output} ({w}×{h}, {item.result.seconds:.1f}s)")
            elif item.result:
                print(f"– {item.path}: skipped ({item.result.note})")
            elif item.error:
                print(f"✗ {item.path}: {item.error.reason}", file=sys.stderr)
                for tip in item.error.suggestions:
                    print(f"    • {tip}", file=sys.stderr)
            else:
                print(f"– {item.path}: {item.status.value}")
        elif ev.kind is EventKind.DEVICE_CHANGED:
            print(f"! {ev.message}", file=sys.stderr)


def _list_models(manager: ModelManager) -> None:
    print("Model families (use with --model):")
    for fam in all_families():
        variants = ", ".join(f"{s}×" for s in sorted(fam.variants))
        print(f"  {fam.id:<24} {fam.name} [{variants}]")
    statuses = manager.statuses()
    for label in dict.fromkeys(KIND_LABELS.values()):  # unique, in order
        group = [st for st in statuses if KIND_LABELS.get(st.spec.kind) == label]
        if not group:
            continue
        print(f"\n{label} models:")
        for status in group:
            spec = status.spec
            mark = "✓" if status.installed else "○"
            state = "installed" if status.installed else "not installed"
            print(
                f"  {mark} {spec.id:<26} {spec.name:<34} {spec.version:<16} "
                f"{spec.size_mb:6.1f} MB  {state}  [{spec.license}]"
            )
    print(f"\nModels directory: {manager.models_dir}")


def _restoration_from_args(args: argparse.Namespace, settings: object) -> rs.RestorationSettings:
    """Restoration settings: the app's saved values, overridden by the options."""
    import dataclasses

    base = settings.restoration()  # type: ignore[attr-defined]
    level = args.restore_level or base.level
    custom = base.custom
    if args.face is not None:
        stages = custom if level == rs.CUSTOM else rs.LEVEL_STAGES[level]
        if stages.face != args.face:
            custom = dataclasses.replace(stages, face=args.face)
            level = rs.CUSTOM
    return dataclasses.replace(
        base,
        level=level,
        custom=custom,
        colorize=args.colorize,
        colorize_strength=(
            args.colorize_strength if args.colorize_strength is not None else base.colorize_strength
        ),
        fidelity=args.fidelity if args.fidelity is not None else base.fidelity,
        modern=args.modern or base.modern,
        scale=args.scale if args.upscale else 1,
    )


def _check_restoration_models(manager: ModelManager, options: object, images: list[Path]) -> None:
    from pixelift.core.errors import ModelNotInstalledError
    from pixelift.core.image_processor import will_colorize
    from pixelift.core.restoration.pipeline import required_models

    restoration = options.restoration  # type: ignore[attr-defined]
    ai_scale = restoration.ai_scale()
    if ai_scale:  # upscaling or AI detail reconstruction
        manager.resolve(options.model, ai_scale)  # type: ignore[attr-defined]
    # Colorization is only needed when some input is black and white.
    any_mono = restoration.colorize and any(will_colorize(p, options) for p in images)
    missing = [
        manager.spec(model_id)
        for model_id in required_models(restoration, monochrome=any_mono)
        if not manager.is_installed(model_id)
    ]
    if missing:
        names = ", ".join(f"{s.name} ({s.size_mb:.0f} MB)" for s in missing)
        downloads = " ".join(f"--download-model {s.id}" for s in missing)
        raise ModelNotInstalledError(
            f"This restoration needs AI models that are not installed: {names}.",
            [f"Run: pixelift {downloads}", "Or use --face off (and no --colorize)"],
        )


def _download(manager: ModelManager, ids: list[str]) -> int:
    from pixelift.models.realesrgan import RECOMMENDED_MODEL

    for model_id in ids:
        model_id = RECOMMENDED_MODEL if model_id == "recommended" else model_id
        try:
            spec = manager.spec(model_id)
        except KeyError:
            print(f"Unknown model: {model_id}", file=sys.stderr)
            return 2
        if manager.is_installed(spec):
            print(f"{spec.name} is already installed.")
            continue
        print(f"Downloading {spec.name} ({human_size(spec.size_bytes)}, {spec.license}) …")

        def progress(done: int, total: int) -> None:
            if sys.stderr.isatty():
                sys.stderr.write(f"\r  {done * 100 // max(total, 1):3d}%  {human_size(done)}")
                sys.stderr.flush()

        try:
            path = manager.download(spec, progress, JobControl())
        except UpscalerError as err:
            print(f"\n{err.user_message()}", file=sys.stderr)
            return 1
        print(f"\r  verified and installed: {path}")
    return 0


def _list_looks(settings: Settings) -> None:
    print("Camera looks (camera-inspired, not official manufacturer presets):")
    for look in cl.all_looks():
        label = f" [{look.category_label}]" if look.category_label else ""
        print(f"  {look.id:<26} {look.name}{label} — {look.description}")
    if settings.camera_look_saved:
        print("Your saved looks:")
        for name in sorted(settings.camera_look_saved):
            print(f"  {cl.USER_PREFIX + name:<26} {name}")


def run_cli(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.lighting == "custom" and args.lighting_intensity is not None:
        parser.error("--lighting-intensity does not apply to --lighting custom")
    if args.look == cl.CUSTOM and args.look_intensity is not None:
        parser.error("--look-intensity does not apply to --look custom")
    if args.hdr == "off" and args.hdr_intensity is not None:
        parser.error("--hdr-intensity needs an --hdr profile")
    if args.inspect_hdr and args.inputs:
        parser.error("--inspect-hdr takes one FILE and no other inputs")
    if args.format is not None and args.dynamic_range == "hdr":
        parser.error("--format is for SDR output; choose the HDR file type with --hdr-format")
    restore_only = (
        "restore_level",
        "colorize",
        "colorize_strength",
        "upscale",
        "face",
        "fidelity",
        "modern",
    )
    if not args.restore and any(getattr(args, name) not in (None, False) for name in restore_only):
        parser.error("the photo restoration options need --restore")
    log_path = setup_logging(args.verbose, console=args.verbose)
    manager = ModelManager()

    if args.list_devices:
        report = dm.detect_devices()
        print(f"PyTorch {report.torch_version or 'not installed'}")
        for dev in report.devices:
            mem = f"  {human_size(dev.total_memory)}" if dev.total_memory else ""
            print(f"  {dev.id:<8} {dev.label()}{mem}")
        for hint in report.hints:
            print(f"  note: {hint}")
        return 0
    if args.list_looks:
        _list_looks(load_settings())
        return 0
    if args.inspect_hdr:
        return _inspect_hdr(args.inspect_hdr)
    if args.list_models:
        _list_models(manager)
        return 0
    if args.install_model_file:
        model_id, file = args.install_model_file
        try:
            print(f"Installed {manager.install_file(model_id, Path(file))}")
        except (UpscalerError, KeyError, OSError) as exc:
            print(friendly_error(exc).user_message(), file=sys.stderr)
            return 1
        return 0
    if args.remove_model:
        for model_id in args.remove_model:
            try:
                manager.remove(model_id)
            except KeyError:
                print(f"Unknown model: {model_id} (see --list-models)", file=sys.stderr)
                return 2
            print(f"Removed {model_id}")
        return 0
    if args.download_model:
        return _download(manager, args.download_model)
    if not args.inputs:
        parser.print_usage(sys.stderr)
        return 2

    images = collect_images(args.inputs, recursive=not args.no_recursive)
    if args.output is not None:
        # An -o folder inside a scanned folder: never upscale earlier results again.
        out_dir = args.output.resolve()
        images = [p for p in images if not p.resolve().is_relative_to(out_dir)]
    missing = [p for p in args.inputs if not p.exists()]
    for path in missing:
        print(f"✗ {path}: file not found", file=sys.stderr)
    found = {p.resolve() for p in images}
    unsupported = [p for p in args.inputs if p.is_file() and p.resolve() not in found]
    for path in unsupported:
        print(f"✗ {path}: not a supported image format", file=sys.stderr)
    skipped_inputs = len(missing) + len(unsupported)  # makes the exit status 1
    if not images:
        print(
            "No supported images found (PNG, JPEG, WebP, AVIF, TIFF, BMP, EXR, HDR).",
            file=sys.stderr,
        )
        return 1

    settings = load_settings()
    options = settings.processing_options()
    options.scale = args.scale
    options.model = args.model
    # An explicit --format asks for that SDR file, whatever range the app last used.
    if args.dynamic_range is None:
        hdr_default = settings.output_range == "hdr" and args.format is None
        args.dynamic_range = "hdr" if hdr_default else "sdr"
    output_format = args.format or settings.output_format
    options.output_format = "jpeg" if output_format == "jpg" else output_format
    options.quality = args.quality
    options.output_dir = args.output
    options.filename_template = args.template
    options.existing = args.existing
    options.preserve_metadata = not args.no_metadata
    options.strip_location = args.strip_location
    settings.lighting_profile = args.lighting
    if args.lighting_intensity is not None:
        settings.lighting_intensity = args.lighting_intensity
    options.lighting = settings.normalise().lighting()
    settings.camera_look = args.look
    if args.look_intensity is not None:
        settings.camera_look_intensity = args.look_intensity
    if args.grain is not None:
        settings.camera_look_grain = args.grain
    options.camera_look = settings.normalise().camera_look_settings()
    app_profile = settings.hdr_profile if settings.hdr_enabled else "off"
    if args.hdr != app_profile:
        # Another profile than the app's: as designed, at its own intensity,
        # without the app's Advanced fine-tuning of its profile.
        settings.hdr_enabled = args.hdr != "off"
        if settings.hdr_enabled:
            settings.hdr_profile = args.hdr
            settings.hdr_intensity = hdr.get_profile(args.hdr).default_intensity
            for name in hdr.ADVANCED_NAMES:
                setattr(settings, f"hdr_{name}", 0)
    if args.hdr_intensity is not None:
        settings.hdr_intensity = args.hdr_intensity
    options.hdr = settings.normalise().hdr()
    options.hdr_output = hdr_output.HdrOutputSettings(
        mode=args.dynamic_range,
        transfer=args.hdr_transfer,
        format=args.hdr_format,
        primaries=args.hdr_colour_space,
        peak_nits=args.hdr_peak,
        sdr_white=args.sdr_white,
        expand_highlights=args.expand_highlights,
    )
    # One-shot runs: nothing would reuse a cached upscale (one written by the
    # app is still used).
    options.cache_upscale = False
    # Only --restore restores: the app's last mode (Restore Photos) must not
    # turn a plain upscale on the command line into a restoration.
    options.restoration = _restoration_from_args(args, settings) if args.restore else None
    try:
        options.validate()
        if options.restoration is None:
            manager.resolve(options.model, options.scale)
        else:
            _check_restoration_models(manager, options, images)
    except ValueError as exc:
        print(f"Invalid option: {exc}", file=sys.stderr)
        return 2
    except UpscalerError as err:
        print(err.user_message(), file=sys.stderr)
        return 1

    device = dm.resolve_device(args.device, gpu_enabled=settings.gpu_enabled)
    print(f"Processing device: {device.label()}")
    lighting = ""
    if options.lighting.active:
        name = get_profile(options.lighting.profile).name
        lighting = f" · lighting {name}"
        if options.lighting.profile != "custom":
            lighting += f" {options.lighting.intensity}%"
    if options.camera_look.active:
        look = options.camera_look
        lighting += f" · look {look.name()}"
        if look.look != cl.CUSTOM:
            lighting += f" {look.intensity}%"
    hdr_settings = options.effective_hdr()
    if hdr_settings.active:
        name = hdr.get_profile(hdr_settings.profile).name
        lighting += f" · {name} {hdr_settings.intensity}%"
    output_kind = options.output_format.upper()
    if options.effective_hdr_output().enabled:
        output_kind = options.effective_hdr_output().describe()
    elif options.hdr_output.enabled:
        print("note: HDR output does not apply to photo restoration; writing SDR files")
    if options.restoration is not None:
        restoration = options.restoration
        parts = [rs.LEVEL_LABELS[restoration.level] + " restoration"]
        parts.append(f"faces {restoration.stages().face}")
        if restoration.colorize:
            parts.append("colorize B&W")
        if restoration.scale > 1:
            parts.append(f"{restoration.scale}× upscale")
        summary_text = " · ".join(parts)
        print(f"{len(images)} image(s) · {summary_text} · {output_kind}{lighting}")
    else:
        print(
            f"{len(images)} image(s) · {options.scale}× · model {options.model} · "
            f"{output_kind}{lighting}"
        )
    printer = _Printer(len(images))
    holder: dict[str, BatchProcessor] = {}
    upscaler = TorchUpscaler(
        manager,
        device,
        tile_size=args.tile_size,
        memory_limit_mb=settings.gpu_memory_limit_mb,
        cpu_threads=args.threads,
        on_device_change=lambda dev, why: holder["p"].notify_device_change(
            f"{why}; continuing on {dev.label()}"
        ),
    )
    processor = BatchProcessor(
        upscaler, options, printer, workers=dm.recommended_concurrency(device)
    )
    holder["p"] = processor
    items = [QueueItem(p) for p in images]
    try:
        summary = processor.run(items)
    except KeyboardInterrupt:
        processor.cancel()
        processor.wait()
        print("\nCancelled.", file=sys.stderr)
        return 130
    except CancelledError:
        return 130
    verb = "restored" if options.restoration is not None else "upscaled"
    print(
        f"\nFinished in {summary.seconds:.1f}s: {summary.done} {verb}, "
        f"{summary.skipped} skipped, {summary.failed} failed."
    )
    if summary.failed and log_path:
        print(f"Details were written to {log_path}", file=sys.stderr)
    if skipped_inputs:
        print(f"{skipped_inputs} input(s) could not be used (see above).", file=sys.stderr)
    return 0 if summary.failed == 0 and not skipped_inputs else 1


def _inspect_hdr(path: Path) -> int:
    """Print what an HDR file says about itself (``--inspect-hdr``).

    Exit status 0 for an HDR file (PQ, HLG or gain map), 3 otherwise.
    """
    from pixelift.core.hdr_output import gainmap

    try:
        with open(path, "rb") as fh:
            head = fh.read(12)
        if head.startswith(hdr_codecs.PNG_SIGNATURE):
            report = hdr_codecs.inspect_png(path)
        elif head[4:8] == b"ftyp":
            report = hdr_codecs.inspect_avif(path)
        elif head.startswith(b"\xff\xd8"):
            if not gainmap.is_gain_map_jpeg(path):
                print(f"{path}: an SDR JPEG (no gain map)")
                return 3
            report = gainmap.inspect(path)
        else:
            print(f"{path}: not a PNG, AVIF or JPEG file", file=sys.stderr)
            return 1
    except (OSError, hdr_codecs.HdrCodecError) as exc:
        print(f"{path}: {exc}", file=sys.stderr)
        return 1
    except (struct.error, IndexError, ValueError):
        print(f"{path}: the file is damaged (its HDR headers could not be read)", file=sys.stderr)
        return 1
    print(report.describe())
    if report.headroom_stops is not None:
        print(f"  Gain map: HDRCapacityMax {report.headroom_stops:.3f} stops (MPF + hdrgm XMP)")
        print("  HDR (gain map)")
        return 0
    print(
        f"  H.273 code points: primaries {report.primaries}, transfer {report.transfer}, "
        f"matrix {report.matrix}, full range {report.full_range}"
    )
    if report.max_cll is not None:
        print(f"  MaxCLL {report.max_cll:g} cd/m², MaxFALL {report.max_fall:g} cd/m²")
    if report.has_alpha or report.channels in (2, 4):
        print("  Alpha: yes")
    print(f"  ICC profile: {'yes' if report.has_icc else 'no'}")
    hdr = report.is_pq or report.is_hlg
    print(f"  {'HDR (' + ('PQ' if report.is_pq else 'HLG') + ')' if hdr else 'not an HDR image'}")
    return 0 if hdr else 3
