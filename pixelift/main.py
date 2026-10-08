"""Entry point: GUI when started without arguments (or with --gui), CLI otherwise."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or "--gui" in argv:
        files = [a for a in argv if a != "--gui" and not a.startswith("-")]
        return gui_main(files)
    from pixelift.cli import run_cli

    return run_cli(argv)


def gui_main(files: list[str] | None = None) -> int:
    from pixelift.utils.logging import setup_logging

    setup_logging()
    try:
        from pixelift.ui.application import run
    except (ImportError, ValueError) as exc:
        print(
            "The desktop interface needs GTK 4 and libadwaita (>= 1.5) with PyGObject.\n"
            "On Ubuntu install: sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 1
    return run(files or [])


if __name__ == "__main__":
    raise SystemExit(main())
