"""Photo-restoration model catalogue: face restoration, face detection, colorization.

Like the Real-ESRGAN weights these are *not* bundled: the model manager
downloads them on request from the projects' official release locations and
verifies their SHA-256 checksums. All processing happens locally.

- GFPGAN v1.4 — Apache-2.0 — https://github.com/TencentARC/GFPGAN
- RetinaFace (facexlib) — MIT — https://github.com/xinntao/facexlib
- DeOldify Artistic — MIT — https://github.com/jantic/DeOldify
"""

from __future__ import annotations

from pixelift.models.base import COLORIZE, FACE_DETECT, FACE_RESTORE, ModelSpec, register


def _gfpgan():  # noqa: ANN202
    from pixelift.models.face_archs import GFPGANv1Clean

    return GFPGANv1Clean()


def _retinaface():  # noqa: ANN202
    from pixelift.models.face_archs import RetinaFace

    return RetinaFace()


def _deoldify():  # noqa: ANN202
    from pixelift.models.deoldify_arch import DeOldifyDeep

    return DeOldifyDeep(nf_factor=1.5)


def _strip_module(state):  # noqa: ANN001, ANN202
    from pixelift.models.face_archs import strip_module_prefix

    return strip_module_prefix(state)


def _fold_norms(state):  # noqa: ANN001, ANN202
    from pixelift.models.deoldify_arch import convert_state_dict

    return convert_state_dict(state)


GFPGAN_V14 = register(
    ModelSpec(
        id="gfpgan-v1.4",
        name="GFPGAN",
        description="Restores blurry and damaged faces; identity safeguards keep the "
        "person recognisable.",
        native_scale=1,
        filename="GFPGANv1.4.pth",
        url="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        sha256="e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad",
        size_bytes=348_632_874,
        license="Apache-2.0",
        license_url="https://github.com/TencentARC/GFPGAN/blob/master/LICENSE",
        build=_gfpgan,
        memory_per_pixel=0,
        state_key="params_ema",
        tags=("restoration",),
        kind=FACE_RESTORE,
        version="v1.4",
    )
)

RETINAFACE = register(
    ModelSpec(
        id="retinaface-resnet50",
        name="RetinaFace face detector",
        description="Finds faces (also small ones in group photos) for face restoration.",
        native_scale=1,
        filename="detection_Resnet50_Final.pth",
        url="https://github.com/xinntao/facexlib/releases/download/v0.1.0/"
        "detection_Resnet50_Final.pth",
        sha256="6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d",
        size_bytes=109_497_761,
        license="MIT",
        license_url="https://github.com/xinntao/facexlib/blob/master/LICENSE",
        build=_retinaface,
        memory_per_pixel=0,
        tags=("restoration",),
        kind=FACE_DETECT,
        version="facexlib v0.1.0",
        convert=_strip_module,
    )
)

DEOLDIFY_ARTISTIC = register(
    ModelSpec(
        id="deoldify-artistic",
        name="DeOldify",
        description="Colorizes black-and-white photographs (Artistic model). Pixelift "
        "tones its colours down for a natural, historical look.",
        native_scale=1,
        filename="ColorizeArtistic_gen.pth",
        url="https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth",
        sha256="3f750246fa220529323b85a8905f9b49c0e5d427099185334d048fb5b5e22477",
        size_bytes=255_144_681,
        license="MIT",
        license_url="https://github.com/jantic/DeOldify/blob/master/LICENSE",
        build=_deoldify,
        memory_per_pixel=0,
        tags=("restoration",),
        kind=COLORIZE,
        version="Artistic (2019)",
        convert=_fold_norms,
        safe_globals=(slice,),
    )
)

# What each optional restoration stage needs installed.
FACE_MODELS = (GFPGAN_V14.id, RETINAFACE.id)
COLORIZE_MODELS = (DEOLDIFY_ARTISTIC.id,)
