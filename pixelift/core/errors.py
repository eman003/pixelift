"""User-facing error types.

Every error shown to a user is an :class:`UpscalerError` with a short title,
a plain-language reason and a list of suggestions. Raw exceptions are logged,
never displayed.
"""

from __future__ import annotations

import errno
import logging

log = logging.getLogger(__name__)


class UpscalerError(Exception):
    title = "Unable to process image."

    def __init__(
        self,
        reason: str,
        suggestions: list[str] | tuple[str, ...] = (),
        title: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.suggestions = list(suggestions)
        if title:
            self.title = title

    def user_message(self) -> str:
        text = f"{self.title}\n\nReason:\n{self.reason}"
        if self.suggestions:
            text += "\n\nTry:\n" + "\n".join(f"• {s}" for s in self.suggestions)
        return text


class InvalidImageError(UpscalerError):
    title = "Unable to open image."


class ModelNotInstalledError(UpscalerError):
    title = "AI model not installed."


class ModelDownloadError(UpscalerError):
    title = "Model download failed."


class OutOfMemoryError(UpscalerError):
    title = "Not enough memory."


class ImageTooLargeError(UpscalerError):
    title = "Image too large."


class OutputError(UpscalerError):
    title = "Unable to save image."


class DeviceError(UpscalerError):
    title = "Processing device unavailable."


class CancelledError(Exception):
    """Raised inside a job when the user cancels it. Not an error for the user."""


def is_oom(exc: BaseException) -> bool:
    """True for CUDA/ROCm/XPU/CPU allocator out-of-memory errors."""
    if isinstance(exc, MemoryError):
        return True
    name = type(exc).__name__
    if name == "OutOfMemoryError":
        return True
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and (
        "out of memory" in text or "can't allocate memory" in text or "not enough memory" in text
    )


def gpu_oom_error() -> OutOfMemoryError:
    return OutOfMemoryError(
        "The image requires more GPU memory than available.",
        ["Enable automatic tiling", "Reduce tile size", "Switch to CPU"],
    )


def friendly_error(exc: BaseException) -> UpscalerError:
    """Convert any exception into a user-facing error (and log the details)."""
    if isinstance(exc, UpscalerError):
        return exc
    log.error("Unexpected error", exc_info=exc)
    if is_oom(exc):
        return OutOfMemoryError(
            "The computer ran out of memory while processing this image.",
            ["Reduce tile size in Settings", "Close other applications", "Use 2× instead of 4×"],
        )
    if isinstance(exc, PermissionError):
        return OutputError(
            f"Permission denied: {getattr(exc, 'filename', '') or exc}",
            ["Choose a different output folder"],
        )
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return OutputError(
            "The disk is full.", ["Free up disk space", "Choose another output folder"]
        )
    if isinstance(exc, OSError):
        return UpscalerError(
            exc.strerror or str(exc), ["Check that the file still exists and is readable"]
        )
    return UpscalerError(
        "An unexpected internal error occurred. Details were written to the log file.",
        ["Try again", "Switch the processing device to CPU in Settings"],
    )
