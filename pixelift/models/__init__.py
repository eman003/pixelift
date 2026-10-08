"""AI model definitions. Importing this package registers all built-in models."""

from pixelift.models import realesrgan  # noqa: F401  (registers models)
from pixelift.models.base import (
    ModelFamily,
    ModelSpec,
    all_families,
    all_specs,
    candidate_specs,
    get_family,
    get_spec,
    register,
    register_family,
)

__all__ = [
    "ModelFamily",
    "ModelSpec",
    "all_families",
    "all_specs",
    "candidate_specs",
    "get_family",
    "get_spec",
    "register",
    "register_family",
]
