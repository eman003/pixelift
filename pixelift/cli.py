"""Command-line interface. Uses the same engine as the GUI."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from pixelift import APP_NAME, __version__
from pixelift.core import device_manager as dm
from pixelift.core.batch_processor import BatchEvent, BatchProcessor, EventKind, QueueItem
from pixelift.core.control import JobControl
from pixelift.core.errors import CancelledError, UpscalerError, friendly_error
from pixelift.core.lighting import all_profiles, get_profile
from pixelift.core.model_manager import ModelManager
from pixelift.core.restoration import settings as rs
from pixelift.core.upscaler import TorchUpscaler
from pixelift.models import KIND_LABELS, all_families
from pixelift.storage.settings import load_settings
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
        "-f", "--format", choices=[*OUTPUT_FORMATS, "jpg"], default=settings.output_format
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
        type=int,
        default=settings.tile_size,
        help="tile size in pixels, 0 = automatic (default: %(default)s)",
    )
    parser.add_argument(
        "--template",
        default=settings.filename_template,
        help="output filename template, fields: {name} {scale} {model} "
        "{width} {height} {ext} {lighting} (default: %(default)s)",
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
        "--no-recursive", action="store_true", help="do not descend into subfolders"
    )
    parser.add_argument(
        "--threads", type=int, default=settings.cpu_threads, help="CPU threads (0 = all)"
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


def run_cli(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.lighting == "custom" and args.lighting_intensity is not None:
        parser.error("--lighting-intensity does not apply to --lighting custom")
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
            manager.remove(model_id)
            print(f"Removed {model_id}")
        return 0
    if args.download_model:
        return _download(manager, args.download_model)
    if not args.inputs:
        parser.print_usage(sys.stderr)
        return 2

    images = collect_images(args.inputs, recursive=not args.no_recursive)
    missing = [p for p in args.inputs if not p.exists()]
    for path in missing:
        print(f"✗ {path}: file not found", file=sys.stderr)
    if not images:
        print("No supported images found (PNG, JPEG, WebP, TIFF, BMP).", file=sys.stderr)
        return 1

    settings = load_settings()
    options = settings.processing_options()
    options.scale = args.scale
    options.model = args.model
    options.output_format = "jpeg" if args.format == "jpg" else args.format
    options.quality = args.quality
    options.output_dir = args.output
    options.filename_template = args.template
    options.existing = args.existing
    options.preserve_metadata = not args.no_metadata
    settings.lighting_profile = args.lighting
    if args.lighting_intensity is not None:
        settings.lighting_intensity = args.lighting_intensity
    options.lighting = settings.normalise().lighting()
    if args.restore:
        options.restoration = _restoration_from_args(args, settings)
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
    if options.restoration is not None:
        restoration = options.restoration
        parts = [rs.LEVEL_LABELS[restoration.level] + " restoration"]
        parts.append(f"faces {restoration.stages().face}")
        if restoration.colorize:
            parts.append("colorize B&W")
        if restoration.scale > 1:
            parts.append(f"{restoration.scale}× upscale")
        summary_text = " · ".join(parts)
        print(
            f"{len(images)} image(s) · {summary_text} · {options.output_format.upper()}{lighting}"
        )
    else:
        print(
            f"{len(images)} image(s) · {options.scale}× · model {options.model} · "
            f"{options.output_format.upper()}{lighting}"
        )
    printer = _Printer(len(images))
    holder: dict[str, BatchProcessor] = {}
    upscaler = TorchUpscaler(
        manager,
        device,
        tile_size=max(0, args.tile_size),
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
    return 0 if summary.failed == 0 else 1
