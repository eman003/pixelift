from __future__ import annotations

import json

import pytest
from PIL import Image

from pixelift import cli
from pixelift.storage.settings import Settings, load_settings, save_settings, settings_path


def test_settings_roundtrip(tmp_path):
    path = tmp_path / "s.json"
    s = Settings(scale=2, output_format="webp", quality=70, theme="dark")
    save_settings(s, path)
    loaded = load_settings(path)
    assert loaded == s


def test_settings_invalid_values_are_normalised(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(
        json.dumps(
            {
                "scale": 7,
                "quality": 500,
                "output_format": "gif",
                "theme": 3,
                "unknown": True,
                "tile_size": 333,
            }
        )
    )
    s = load_settings(path)
    assert (s.scale, s.quality, s.output_format, s.theme, s.tile_size) == (
        4,
        100,
        "png",
        "system",
        0,
    )


def test_corrupt_settings_file(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("{not json")
    assert load_settings(path) == Settings()


def test_settings_default_location():
    assert settings_path().name == "settings.json"
    assert "pixelift" in str(settings_path())


def test_settings_to_processing_options(tmp_path):
    opts = Settings(output_dir=str(tmp_path), scale=2).processing_options()
    assert opts.output_dir == tmp_path and opts.scale == 2


def test_cli_upscale_single_file(image_factory, tiny_specs, tmp_path, capsys):
    src = image_factory("photo.jpg")
    out_dir = tmp_path / "out"
    code = cli.run_cli(
        [
            str(src),
            "--scale",
            "4",
            "--model",
            "test-x4",
            "--output",
            str(out_dir),
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    with Image.open(out_dir / "photo_4x.png") as img:
        assert img.size == (160, 120)
    assert "1 upscaled" in capsys.readouterr().out


def test_cli_batch_directory_and_options(image_factory, tiny_specs, tmp_path):
    image_factory("a.png")
    image_factory("b.png")
    code = cli.run_cli(
        [
            str(tmp_path / "in"),
            "-s",
            "2",
            "-m",
            "test-family",
            "-f",
            "jpg",
            "-q",
            "80",
            "--template",
            "{name}-big",
            "--tile-size",
            "64",
        ]
    )
    assert code == 0
    outs = sorted((tmp_path / "in" / "upscaled").iterdir())
    assert [p.name for p in outs] == ["a-big.jpg", "b-big.jpg"]


def test_cli_reports_failures(tmp_path, tiny_specs, capsys):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"x")
    assert cli.run_cli([str(bad), "-m", "test-x4"]) == 1
    assert "not a supported image" in capsys.readouterr().err


def test_cli_missing_model_message(image_factory, tiny_specs, capsys):
    src = image_factory("p.png")
    assert cli.run_cli([str(src), "--model", "realesrgan-x4plus"]) == 1
    err = capsys.readouterr().err
    assert "not installed" in err and "--download-model" in err


def test_cli_no_images(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert cli.run_cli([str(tmp_path / "empty")]) == 1


def test_cli_list_models_and_devices(capsys):
    assert cli.run_cli(["--list-models"]) == 0
    out = capsys.readouterr().out
    assert "realesrgan-x4plus" in out and "BSD-3-Clause" in out
    assert cli.run_cli(["--list-devices"]) == 0
    assert "cpu" in capsys.readouterr().out


def test_main_dispatch_uses_cli(monkeypatch):
    from pixelift import main as main_mod

    called = {}
    monkeypatch.setattr(cli, "run_cli", lambda argv: called.setdefault("argv", argv) and 0)
    main_mod.main(["x.png", "--scale", "2"])
    assert called["argv"] == ["x.png", "--scale", "2"]


def test_cli_upscales_even_when_the_app_was_left_in_restore_mode(
    image_factory, tiny_specs, tmp_path, capsys
):
    save_settings(Settings(mode="restore"))
    src = image_factory("photo.jpg")
    args = [str(src), "-s", "4", "-m", "test-x4", "-o", str(tmp_path / "out"), "--device", "cpu"]
    assert cli.run_cli(args) == 0
    with Image.open(tmp_path / "out" / "photo_4x.png") as img:
        assert img.size == (160, 120)
    assert "1 upscaled" in capsys.readouterr().out


@pytest.mark.parametrize("content", ["[]", "null", "1", '"text"'])
def test_settings_file_that_is_not_an_object(tmp_path, content):
    path = tmp_path / "settings.json"
    path.write_text(content)
    assert load_settings(path) == Settings()


def test_settings_bool_and_int_are_not_mixed_up(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"quality": true, "preserve_metadata": 0}')
    s = load_settings(path)
    assert s.quality == Settings().quality
    assert s.preserve_metadata is Settings().preserve_metadata


def test_invalid_filename_template_is_reset(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"filename_template": "{name"}')
    assert load_settings(path).filename_template == Settings().filename_template


def test_cli_rejects_invalid_template(image_factory, tiny_specs, capsys):
    src = image_factory()
    assert cli.run_cli([str(src), "-m", "test-x4", "--template", "{name"]) == 2
    assert "template" in capsys.readouterr().err


def test_cli_remove_unknown_model(capsys):
    assert cli.run_cli(["--remove-model", "no-such-model"]) == 2
    assert "Unknown model" in capsys.readouterr().err


def test_restoration_model_is_not_an_upscaler(tmp_path):
    from pixelift.core.errors import ModelNotInstalledError
    from pixelift.core.model_manager import ModelManager
    from pixelift.models import UPSCALE, all_specs

    other = next(s for s in all_specs() if s.kind != UPSCALE)
    with pytest.raises(ModelNotInstalledError, match="not a known model"):
        ModelManager(tmp_path).resolve(other.id, 4)


def test_gui_entry_point_opens_files(monkeypatch):
    import sys

    from pixelift import main
    from pixelift.ui import application

    opened = []
    monkeypatch.setattr(application, "run", lambda files: opened.append(files) or 0)
    monkeypatch.setattr(sys, "argv", ["pixelift-gui", "a.jpg", "--verbose", "b.png"])
    assert main.gui_main() == 0
    assert opened == [["a.jpg", "b.png"]]


def test_cli_strip_location(tmp_path, tiny_specs):
    src = tmp_path / "gps.jpg"
    exif = Image.Exif()
    exif[0x010F] = "Camera Maker"
    exif.get_ifd(0x8825)[1] = "N"
    Image.new("RGB", (8, 8)).save(src, exif=exif.tobytes())
    out_dir = tmp_path / "out"
    args = [str(src), "--model", "test-x4", "--output", str(out_dir), "--device", "cpu"]
    assert cli.run_cli([*args, "--strip-location"]) == 0
    with Image.open(out_dir / "gps_4x.png") as img:
        assert img.getexif()[0x010F] == "Camera Maker"
        assert not img.getexif().get_ifd(0x8825)
