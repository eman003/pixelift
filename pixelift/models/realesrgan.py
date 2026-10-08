"""Real-ESRGAN model catalogue.

Weights are published by the Real-ESRGAN project under the BSD-3-Clause
license: https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE
They are *not* bundled with the application; the model manager downloads
them on request and verifies their SHA-256 checksums.
"""

from __future__ import annotations

from pixelift.models.base import ModelFamily, ModelSpec, register, register_family

_RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"
_LICENSE = "BSD-3-Clause"
_LICENSE_URL = "https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE"


def _rrdb(scale: int, num_block: int = 23):  # noqa: ANN202
    def build():  # noqa: ANN202
        from pixelift.models.archs import RRDBNet

        return RRDBNet(scale=scale, num_block=num_block)

    return build


def _compact(num_conv: int):  # noqa: ANN202
    def build():  # noqa: ANN202
        from pixelift.models.archs import SRVGGNetCompact

        return SRVGGNetCompact(num_conv=num_conv, upscale=4)

    return build


X4PLUS = register(
    ModelSpec(
        id="realesrgan-x4plus",
        name="Real-ESRGAN x4",
        description="Best quality for photos and general images (native 4×).",
        native_scale=4,
        filename="RealESRGAN_x4plus.pth",
        url=f"{_RELEASES}/v0.1.0/RealESRGAN_x4plus.pth",
        sha256="4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
        size_bytes=67_040_989,
        license=_LICENSE,
        license_url=_LICENSE_URL,
        build=_rrdb(4),
        memory_per_pixel=16_000,
        state_key="params_ema",
        tags=("recommended", "photo"),
    )
)

X2PLUS = register(
    ModelSpec(
        id="realesrgan-x2plus",
        name="Real-ESRGAN x2",
        description="Photos and general images, native 2× (faster than 4×).",
        native_scale=2,
        filename="RealESRGAN_x2plus.pth",
        url=f"{_RELEASES}/v0.2.1/RealESRGAN_x2plus.pth",
        sha256="49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
        size_bytes=67_061_725,
        license=_LICENSE,
        license_url=_LICENSE_URL,
        build=_rrdb(2),
        memory_per_pixel=6_000,
        size_multiple=2,
        state_key="params_ema",
        tags=("photo",),
    )
)

GENERAL_V3 = register(
    ModelSpec(
        id="realesr-general-x4v3",
        name="Real-ESRGAN General v3 (fast)",
        description="Small, fast model — a good choice for CPU-only computers.",
        native_scale=4,
        filename="realesr-general-x4v3.pth",
        url=f"{_RELEASES}/v0.2.5.0/realesr-general-x4v3.pth",
        sha256="8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
        size_bytes=4_885_111,
        license=_LICENSE,
        license_url=_LICENSE_URL,
        build=_compact(32),
        memory_per_pixel=3_000,
        state_key="params",
        tags=("fast", "photo"),
    )
)

ANIME_6B = register(
    ModelSpec(
        id="realesrgan-x4plus-anime",
        name="Real-ESRGAN Anime",
        description="Optimised for anime and illustrations (native 4×).",
        native_scale=4,
        filename="RealESRGAN_x4plus_anime_6B.pth",
        url=f"{_RELEASES}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        sha256="f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
        size_bytes=17_938_799,
        license=_LICENSE,
        license_url=_LICENSE_URL,
        build=_rrdb(4, num_block=6),
        memory_per_pixel=16_000,
        state_key="params_ema",
        tags=("anime",),
    )
)

ANIME_VIDEO_V3 = register(
    ModelSpec(
        id="realesr-animevideov3",
        name="Real-ESRGAN Anime Video v3 (fast)",
        description="Tiny, very fast model for anime-style content.",
        native_scale=4,
        filename="realesr-animevideov3.pth",
        url=f"{_RELEASES}/v0.2.5.0/realesr-animevideov3.pth",
        sha256="b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d",
        size_bytes=2_504_012,
        license=_LICENSE,
        license_url=_LICENSE_URL,
        build=_compact(16),
        memory_per_pixel=2_500,
        state_key="params",
        tags=("fast", "anime"),
    )
)

register_family(
    ModelFamily(
        id="realesrgan",
        name="Real-ESRGAN",
        description="High quality photo upscaling",
        variants={4: X4PLUS.id, 2: X2PLUS.id},
    )
)
register_family(
    ModelFamily(
        id="realesrgan-general",
        name="Real-ESRGAN General (fast)",
        description="Fast general-purpose upscaling",
        variants={4: GENERAL_V3.id},
    )
)
register_family(
    ModelFamily(
        id="realesrgan-anime",
        name="Real-ESRGAN Anime",
        description="Anime and illustrations",
        variants={4: ANIME_6B.id},
    )
)
register_family(
    ModelFamily(
        id="realesrgan-anime-fast",
        name="Real-ESRGAN Anime (fast)",
        description="Fast anime upscaling",
        variants={4: ANIME_VIDEO_V3.id},
    )
)

RECOMMENDED_MODEL = X4PLUS.id
RECOMMENDED_CPU_MODEL = GENERAL_V3.id
