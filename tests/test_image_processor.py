from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixelift.core.errors import ImageTooLargeError, InvalidImageError, OutputError
from pixelift.core.image_processor import (
    ProcessingOptions,
    check_feasible,
    output_path_for,
    process_image,
    reserve_output,
)
from pixelift.utils import image_utils as iu

ORIENTATION = 0x0112


# --- loading ------------------------------------------------------------------
def test_load_rgb(image_factory):
    loaded = iu.load_image(image_factory("a.png", size=(40, 30)))
    assert loaded.rgb.shape == (30, 40, 3)
    assert loaded.rgb.dtype == np.uint8
    assert loaded.alpha is None
    assert loaded.rgb.flags.writeable


@pytest.mark.parametrize("ext", ["png", "jpg", "webp", "tiff", "bmp"])
def test_supported_formats_load(image_factory, ext):
    path = image_factory(f"a.{ext}")
    assert iu.is_supported(path)
    assert iu.load_image(path).rgb.shape == (30, 40, 3)
    info = iu.probe_image(path)
    assert (info.width, info.height) == (40, 30)
    assert info.file_size == path.stat().st_size


def test_invalid_image_raises_friendly_error(tmp_path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"definitely not a png")
    with pytest.raises(InvalidImageError) as err:
        iu.load_image(bad)
    assert "broken.png" in err.value.reason
    with pytest.raises(InvalidImageError):
        iu.probe_image(bad)


def test_truncated_image(tmp_path, image_factory):
    path = image_factory("t.jpg", size=(200, 200))
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 3])
    with pytest.raises(InvalidImageError):
        iu.load_image(path)


def test_missing_file(tmp_path):
    with pytest.raises(InvalidImageError):
        iu.load_image(tmp_path / "nope.png")


def test_collect_images_filters_and_skips_output(tmp_path, image_factory):
    image_factory("a.png")
    image_factory("sub/b.jpg")
    image_factory("upscaled/a_4x.png")
    (tmp_path / "in" / "notes.txt").write_text("x")
    found = iu.collect_images([tmp_path / "in"])
    assert sorted(p.name for p in found) == ["a.png", "b.jpg"]
    assert [p.name for p in iu.collect_images([tmp_path / "in"], recursive=False)] == ["a.png"]


# --- EXIF orientation -----------------------------------------------------------
def _rotated_jpeg(path: Path) -> Path:
    img = Image.new("RGB", (40, 20), "red")
    img.paste((0, 0, 255), (0, 0, 10, 20))  # blue strip on the left
    exif = Image.Exif()
    exif[ORIENTATION] = 6  # rotate 90° CW for display
    exif[0x010F] = "TestCam"  # Make
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, exif=exif.tobytes(), quality=95)
    return path


def test_exif_orientation_applied_on_load(tmp_path):
    path = _rotated_jpeg(tmp_path / "rot.jpg")
    info = iu.probe_image(path)
    assert (info.width, info.height) == (20, 40)
    loaded = iu.load_image(path)
    assert loaded.rgb.shape == (40, 20, 3)
    # After a 90° CW rotation the left blue strip ends up at the top.
    assert loaded.rgb[2, 10, 2] > 200 and loaded.rgb[-2, 10, 0] > 200


def test_output_is_not_rotated_twice(tmp_path, upscaler, tiny_specs):
    path = _rotated_jpeg(tmp_path / "rot.jpg")
    options = ProcessingOptions(scale=2, model="test-x2", output_dir=tmp_path / "out")
    result = process_image(path, options, upscaler)
    with Image.open(result.output) as out:
        assert out.size == (40, 80)
        exif = out.getexif()
        assert exif.get(ORIENTATION, 1) == 1  # absent or 1 both mean "normal"
        assert exif.get(0x010F) == "TestCam"  # other metadata preserved


# --- alpha / modes ----------------------------------------------------------------
def test_alpha_preserved_png(image_factory, upscaler, tiny_specs, tmp_path):
    path = image_factory("alpha.png", mode="RGBA")
    result = process_image(path, ProcessingOptions(scale=4, model="test-x4"), upscaler)
    with Image.open(result.output) as out:
        assert out.mode == "RGBA"
        assert out.size == (160, 120)
        alpha = np.asarray(out.getchannel("A"))
        assert alpha[:, :8].mean() < 40 and alpha[:, -8:].mean() > 215


def test_alpha_flattened_for_jpeg(image_factory, upscaler, tiny_specs):
    path = image_factory("alpha.png", mode="RGBA")
    options = ProcessingOptions(scale=4, model="test-x4", output_format="jpeg")
    result = process_image(path, options, upscaler)
    assert "flattened" in result.note
    with Image.open(result.output) as out:
        assert out.mode == "RGB" and out.format == "JPEG"


def test_opaque_alpha_is_dropped(image_factory):
    path = image_factory("opaque.png")
    Image.open(path).convert("RGBA").save(path)
    assert iu.load_image(path).alpha is None


def test_grayscale_stays_grayscale(image_factory, upscaler, tiny_specs):
    path = image_factory("gray.png", mode="L")
    result = process_image(path, ProcessingOptions(scale=4, model="test-x4"), upscaler)
    with Image.open(result.output) as out:
        assert out.mode == "L"


def test_palette_and_16bit(tmp_path, image_factory):
    pal = image_factory("pal.png", mode="P")
    assert iu.load_image(pal).rgb.shape == (30, 40, 3)
    arr = (np.arange(40 * 30, dtype=np.uint16).reshape(30, 40) * 50).astype(np.uint16)
    path = tmp_path / "deep.png"
    Image.fromarray(arr, "I;16").save(path)
    loaded = iu.load_image(path)
    assert loaded.grayscale
    # scaled (>> 8), not clipped to 255
    assert loaded.rgb[..., 0].max() == arr.max() >> 8


# --- output formats & metadata --------------------------------------------------
@pytest.mark.parametrize(
    ("fmt", "pil_format", "ext"),
    [("png", "PNG", ".png"), ("jpeg", "JPEG", ".jpg"), ("webp", "WEBP", ".webp")],
)
def test_output_format_conversion(image_factory, upscaler, tiny_specs, fmt, pil_format, ext):
    path = image_factory("photo.jpg")
    options = ProcessingOptions(scale=4, model="test-x4", output_format=fmt, quality=80)
    result = process_image(path, options, upscaler)
    assert result.output.suffix == ext
    assert result.output.parent == path.parent / "upscaled"
    with Image.open(result.output) as out:
        assert out.format == pil_format
        assert out.size == (160, 120)


def test_icc_profile_and_dpi_preserved(image_factory, upscaler, tiny_specs):
    from PIL import ImageCms

    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    path = image_factory("icc.jpg", icc_profile=icc, dpi=(72, 72))
    result = process_image(path, ProcessingOptions(scale=4, model="test-x4"), upscaler)
    with Image.open(result.output) as out:
        assert out.info.get("icc_profile") == icc
        assert round(out.info["dpi"][0]) == 288


def test_metadata_can_be_disabled(image_factory, upscaler, tiny_specs):
    from PIL import ImageCms

    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    path = image_factory("icc.jpg", icc_profile=icc)
    options = ProcessingOptions(scale=4, model="test-x4", preserve_metadata=False)
    with Image.open(process_image(path, options, upscaler).output) as out:
        assert "icc_profile" not in out.info


def test_output_file_permissions_follow_umask(image_factory, upscaler, tiny_specs):
    path = image_factory("perm.png")
    result = process_image(path, ProcessingOptions(scale=4, model="test-x4"), upscaler)
    umask = os.umask(0)
    os.umask(umask)
    assert result.output.stat().st_mode & 0o777 == 0o666 & ~umask
    assert not list(result.output.parent.glob(".upscaling-*"))  # no temp files left


# --- naming & existing files ------------------------------------------------------
def test_filename_template():
    src = Path("/photos/photo.jpg")
    assert iu.render_filename("{name}_{scale}x", src, 4, "m", 1, 1, ".png") == "photo_4x.png"
    assert (
        iu.render_filename(
            "{name}-{width}x{height}-{model}", src, 2, "realesrgan", 800, 600, ".webp"
        )
        == "photo-800x600-realesrgan.webp"
    )
    assert iu.render_filename("", src, 4, "m", 1, 1, ".png") == "photo_4x.png"
    assert iu.render_filename("a/{name}", src, 4, "m", 1, 1, ".png") == "a_photo.png"
    with pytest.raises(ValueError, match="Unknown field"):
        iu.render_filename("{nme}", src, 4, "m", 1, 1, ".png")
    # A lighting tag is appended unless the template places it.
    assert iu.render_filename("", src, 4, "m", 1, 1, ".png", "vivid") == "photo_4x_vivid.png"
    assert iu.render_filename("{lighting}-{name}", src, 4, "m", 1, 1, ".png", "vivid") == (
        "vivid-photo.png"
    )


def test_output_path_default_and_custom(tmp_path):
    src = tmp_path / "photo.jpg"
    opts = ProcessingOptions(scale=4)
    assert output_path_for(src, 10, 10, opts) == tmp_path / "upscaled" / "photo_4x.png"
    opts = ProcessingOptions(scale=2, output_dir=tmp_path / "x", output_format="jpeg")
    assert output_path_for(src, 10, 10, opts) == tmp_path / "x" / "photo_2x.jpg"


def test_existing_policies(image_factory, upscaler, tiny_specs):
    path = image_factory("p.png")
    first = process_image(path, ProcessingOptions(scale=4, model="test-x4"), upscaler)
    skipped = process_image(path, ProcessingOptions(scale=4, model="test-x4"), upscaler)
    assert skipped.skipped and skipped.output == first.output
    renamed = process_image(
        path, ProcessingOptions(scale=4, model="test-x4", existing="rename"), upscaler
    )
    assert renamed.output.name == "p_4x (2).png"
    over = process_image(
        path, ProcessingOptions(scale=4, model="test-x4", existing="overwrite"), upscaler
    )
    assert over.output == first.output and not over.skipped


def test_reserve_output(tmp_path):
    target = tmp_path / "a.png"
    assert reserve_output(target, "rename", None) == target
    target.write_bytes(b"x")
    assert reserve_output(target, "rename", None).name == "a (2).png"
    assert reserve_output(target, "skip", None) == target
    taken = {target}
    assert reserve_output(target, "skip", lambda p: p not in taken).name == "a (2).png"
    with pytest.raises(OutputError):
        reserve_output(target, "skip", lambda _p: False)


def test_webp_dimension_limit_checked_early(tmp_path):
    opts = ProcessingOptions(scale=4, output_format="webp")
    with pytest.raises(ImageTooLargeError, match="WebP"):
        check_feasible(5000, 1000, opts, tmp_path / "x.webp")


def test_memory_check_rejects_absurd_sizes(tmp_path):
    with pytest.raises(ImageTooLargeError, match="memory"):
        check_feasible(400_000, 400_000, ProcessingOptions(scale=4), tmp_path / "x.png")


def test_options_validation():
    with pytest.raises(ValueError):
        ProcessingOptions(scale=5).validate()
    with pytest.raises(ValueError):
        ProcessingOptions(quality=0).validate()
    with pytest.raises(ValueError):
        ProcessingOptions(output_format="gif").validate()
