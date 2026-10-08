# Changelog

All notable changes to Pixelift are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

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

[1.1.0]: https://github.com/eman003/pixelift/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/eman003/pixelift/releases/tag/v1.0.0
