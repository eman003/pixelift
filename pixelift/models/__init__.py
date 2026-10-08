"""AI model definitions. Importing this package registers all built-in models."""

from pixelift.models import realesrgan, restoration  # noqa: F401  (registers models)
from pixelift.models.base import (
    COLORIZE,
    FACE_DETECT,
    FACE_RESTORE,
    KIND_LABELS,
    UPSCALE,
    ModelFamily,
    ModelSpec,
    all_families,
    all_specs,
    candidate_specs,
    get_family,
    get_spec,
    register,
    register_family,
    specs_of_kind,
)

__all__ = [
    "COLORIZE",
    "FACE_DETECT",
    "FACE_RESTORE",
    "KIND_LABELS",
    "UPSCALE",
    "ModelFamily",
    "ModelSpec",
    "all_families",
    "all_specs",
    "candidate_specs",
    "get_family",
    "get_spec",
    "register",
    "register_family",
    "specs_of_kind",
]
