from __future__ import annotations

import json

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
