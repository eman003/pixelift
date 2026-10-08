"""Model descriptions and the model registry.

A :class:`ModelSpec` describes one downloadable weights file and how to build
the network for it. A :class:`ModelFamily` groups specs by purpose (e.g.
"Real-ESRGAN photo" has a native 2× and a native 4× variant), so the UI can
offer "model + scale" without knowing about individual files.

To add a new model (Real-CUGAN, SwinIR, …): implement its network, create
``ModelSpec`` objects whose ``build`` returns an ``nn.Module`` taking a
``[N, 3, H, W]`` float tensor in ``[0, 1]`` and returning the upscaled tensor,
and call :func:`register` from a module imported in ``models/__init__``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    description: str
    native_scale: int
    filename: str
    url: str
    sha256: str
    size_bytes: int
    license: str
    license_url: str
    build: Callable[[], nn.Module]
    # Rough activation memory per *input* pixel in fp32, used to pick tile sizes.
    memory_per_pixel: int = 16_000
    # Pixel dimensions of tiles must be multiples of this (pixel-unshuffle models).
    size_multiple: int = 1
    # Key inside the checkpoint dict holding the weights (None = try common keys).
    state_key: str | None = None
    tags: tuple[str, ...] = ()

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1_000_000


@dataclass(frozen=True)
class ModelFamily:
    id: str
    name: str
    description: str
    # native scale -> ModelSpec id, preferred order of fallback is by closeness.
    variants: dict[int, str] = field(default_factory=dict)


_SPECS: dict[str, ModelSpec] = {}
_FAMILIES: dict[str, ModelFamily] = {}


def register(spec: ModelSpec) -> ModelSpec:
    _SPECS[spec.id] = spec
    return spec


def register_family(family: ModelFamily) -> ModelFamily:
    _FAMILIES[family.id] = family
    return family


def all_specs() -> list[ModelSpec]:
    return list(_SPECS.values())


def all_families() -> list[ModelFamily]:
    return list(_FAMILIES.values())


def get_spec(model_id: str) -> ModelSpec:
    try:
        return _SPECS[model_id]
    except KeyError:
        raise KeyError(f"Unknown model: {model_id}") from None


def get_family(family_id: str) -> ModelFamily:
    try:
        return _FAMILIES[family_id]
    except KeyError:
        raise KeyError(f"Unknown model: {family_id}") from None


def candidate_specs(model: str, scale: int) -> list[ModelSpec]:
    """Specs able to serve ``model`` at ``scale``, best first.

    ``model`` may be a family id or a concrete spec id. Within a family the
    variant with the requested native scale is preferred; otherwise a larger
    native scale (downsampled afterwards) beats a smaller one.
    """
    if model in _SPECS:
        return [_SPECS[model]]
    family = get_family(model)
    order = sorted(
        family.variants.items(),
        key=lambda kv: (kv[0] != scale, kv[0] < scale, abs(kv[0] - scale)),
    )
    return [_SPECS[spec_id] for _, spec_id in order]


def extract_state_dict(checkpoint: Any, key: str | None) -> dict[str, Any]:
    if key and isinstance(checkpoint, dict) and key in checkpoint:
        return checkpoint[key]
    if isinstance(checkpoint, dict):
        for candidate in ("params_ema", "params", "state_dict", "model"):
            if candidate in checkpoint and isinstance(checkpoint[candidate], dict):
                return checkpoint[candidate]
    return checkpoint
