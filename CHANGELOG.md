# Changelog

All notable changes to Pixelift are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [1.3.0] - 2026-10-08

### Added

- **Camera Looks**: 30 camera- and film-inspired renderings — Sony-, Canon-,
  Nikon-, Fujifilm-, Leica- and Hasselblad-inspired picture styles, Kodak-,
  Portra- and Ektar-inspired film, Classic/Modern Film, Cinematic and Black &
  White — clearly labelled as inspired, not official manufacturer presets.
  - Each look combines white balance, tone curve, highlight roll-off, matte
    fade, vibrance, eight-band hue/saturation/luminance, split toning,
    black-and-white channel mixing, micro-contrast, sharpening and optional
    film grain, with skin-tone protection.
  - Intensity 0–100 % (default 50 %), Grain Look Default/Off/Low/Medium/High,
    a Custom look (exposure, contrast, highlights, shadows, temperature, tint,
    saturation, vibrance, eight colour bands, sharpness) that can be saved as
    your own looks, and favorites.
  - Gallery of look cards previewed on your own photo, grouped by category;
    live before/after in the preview window, combined with the lighting.
  - Applied after the lighting and (in Restore Photos mode) after
    colorization, before upscaling; sharpening and grain go on the final
    image. Part of the output name (`{look}`), so differently graded results
    never overwrite each other. `--look`, `--look-intensity`, `--grain` and
    `--list-looks` on the command line.

### Fixed

- The command line restored photos instead of upscaling them when the app had
  last been left in Restore Photos mode; only `--restore` restores now.

## [1.2.0] - 2026-10-08

### Added

- **AI Photo Restoration** ("Restore Photos" mode in the app, `--restore` on the
  command line) for old scanned photographs, entirely on the computer:
  - Scan cleanup with conventional image processing: dust and speck removal,
    scratch reduction, noise reduction (gentle on grain, strong on colour
    noise), faded-image recovery, automatic correction of yellowing and colour
    casts, noise-aware sharpening.
  - Face restoration with GFPGAN v1.4 (Off / Natural / Strong) with identity
    protection: only the facial area is replaced, the original's shape, skin
    tone and lighting are kept, and tiny, uncertain or badly damaged faces are
    restored only lightly. *Restoration fidelity* (Original ↔ AI enhanced).
  - Black-and-white detection (also sepia-toned and yellowed prints), with a
    "Restore in B&W / Restore & Colorize" choice — never colorized without
    consent; colour photos are never colorized.
  - Colorization with DeOldify (Artistic) toned for natural, historical
    colour: style, strength (0 % = grey) and *Preserve original tones*.
  - Levels Light / Standard / Heavy / Custom, presets Restore / Restore +
    Colorize / Restore + Upscale / Full Restoration, manual colour correction
    (using the lighting engine), and Modern Finish (Natural, Clean, Vivid,
    Professional).
  - Restoration + Real-ESRGAN upscaling, or AI detail reconstruction without
    upscaling (Heavy).
  - "Preview Restoration" in the compare window; restored results compare as
    *Original Scan* / *Restored*.
  - Output in a `restored` folder: `photo_restored.jpg`,
    `photo_restored_colorized.jpg`, `photo_restored_4x.jpg`.
- Model manager: GFPGAN (Apache-2.0), RetinaFace (MIT) and DeOldify (MIT)
  models, grouped by purpose and shown with version, size and license; asked
  before downloading.
- CLI: `--restore`, `--restore-level`, `--colorize`, `--colorize-strength`,
  `--upscale`, `--face`, `--fidelity`, `--modern`; `--list-models` groups models.

### Changed

- Originals are never overwritten, whatever the output folder and file-name
  template (previously possible with `{name}` + the original's folder + Overwrite).
- Restoration results in `restored` folders are skipped when adding folders, like
  `upscaled` folders (other images in a folder of that name are still added).

## [1.1.0] - 2026-10-08

### Added

- **Lighting profiles**, applied before upscaling: Natural Daylight, Bright &
  Clean, Golden Hour, Studio, Cinematic, Low Light Recovery, Cool Daylight,
  Vivid, High Contrast, and Custom (exposure, brightness, contrast, highlights,
  shadows, temperature, tint, saturation). Profiles have an intensity of 0–100 %.
- Lighting controls in the main window and in the preview window, with a live
  "Lighting" side in the before/after view for images not yet upscaled.
- CLI options `-l` / `--lighting` and `--lighting-intensity`.
- `{lighting}` filename template field.

### Changed

- When lighting is active, its name is part of the output file name (e.g.
  `photo_4x_golden-hour.png`), so results made with different lighting never
  overwrite or get mistaken for each other. Output names with *Original*
  lighting are unchanged.
- Changing the lighting puts finished images back in the queue.
- Grayscale images that a profile warms, cools or tints are saved in colour,
  as the preview shows.
- Settings are saved once a lighting slider drag settles instead of on every
  step, and dragging no longer refreshes the whole queue.

### Fixed

- Images in one batch that map to the same output name (e.g. `photo.jpg` and
  `photo.png`) no longer overwrite each other or swap names between runs; the
  earlier image in the queue always gets the plain name.
- A job no longer fails, or runs on a mix of devices, when another job
  switches the shared upscaler from the GPU to the CPU. Tiles are sized for the
  device each attempt actually runs on.
- Removing the GPU memory limit in Preferences now lifts it again.
- `--lighting-intensity` outside 0–100, or combined with `--lighting custom`,
  is rejected instead of silently changed or ignored.
- The lighting preview recovers after a failed render.
- The preview window's lighting controls are locked while a batch runs.
- Opening the preview of a large image no longer briefly freezes the window.

## [1.0.0] - 2026-10-08

### Added

- First release: local AI image upscaler for Ubuntu (GTK 4 + libadwaita) with
  Real-ESRGAN models, CUDA / ROCm / XPU / CPU support, batch queue, before/after
  preview, model manager and a command-line interface.

[1.3.0]: https://github.com/eman003/pixelift/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/eman003/pixelift/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/eman003/pixelift/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/eman003/pixelift/releases/tag/v1.0.0
