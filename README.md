# Pixelift

**Make images larger and sharper with AI — entirely on your own computer.**

Pixelift is a native Ubuntu desktop app (GTK 4 + libadwaita) and command-line
tool that upscales images 2× or 4× with [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
super-resolution.

> 🔒 **Your images stay on your computer.** Processing is 100 % local. There
> is no account, no telemetry and no analytics. The internet is only used —
> when you ask — to download AI model files.

## Features

- Drag-and-drop or file picker; single files, multiple files or whole folders
- PNG, JPEG, WebP, TIFF and BMP input; PNG, JPEG or WebP output
- 2× and 4× upscaling, several Real-ESRGAN models (photo, fast, anime)
- NVIDIA (CUDA), AMD (ROCm) and Intel (XPU) GPU acceleration, with a reliable
  CPU fallback — GPUs are never required
- Tiled processing for very large images, automatic tile sizing and
  automatic recovery from out-of-memory errors
- Batch queue with progress, pause / resume / cancel, retry failed, skip completed
- Before/after comparison: slider, toggle, zoom, pan, fit, 100 %
- Keeps EXIF orientation (no accidental rotation), EXIF metadata, ICC colour
  profiles, DPI, transparency and grayscale
- Model manager with SHA-256-verified downloads
- Light, dark and system theme; responsive layout
- A CLI that uses exactly the same engine as the app

## Requirements

| | |
|---|---|
| OS | Ubuntu 24.04 LTS or newer (or any distro with Python 3.12+, GTK ≥ 4.12, libadwaita ≥ 1.5) |
| RAM | 8 GB recommended; very large 4× results need more (Pixelift checks before starting) |
| Disk | ~1 GB for the app with PyTorch (CPU), plus 5–70 MB per model |
| GPU | Optional — see [GPU acceleration](#gpu-acceleration) |

## Installation

### Option A — `.deb` package (Ubuntu 24.04)

```bash
sudo apt install ./pixelift_1.0.0_amd64.deb
```

Pixelift then appears in the application launcher, and the `pixelift`
command is available in the terminal. The package bundles the CPU build of
PyTorch; GPU support is an optional extra step (below).

### Option B — AppImage

```bash
chmod +x Pixelift-1.0.0-x86_64.AppImage
./Pixelift-1.0.0-x86_64.AppImage            # desktop app
./Pixelift-1.0.0-x86_64.AppImage photo.jpg  # CLI
```

The AppImage bundles the app, PyTorch, numpy and Pillow, and uses the system's
Python 3.12 and GTK 4 / libadwaita (present on Ubuntu 24.04 desktops; otherwise
`sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1`). Running AppImages
needs FUSE 2 (`sudo apt install libfuse2t64`), or run it with
`--appimage-extract-and-run`.

### Option C — from source (development)

```bash
sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1 python3-venv
git clone https://github.com/eman003/pixelift.git
cd pixelift

# --system-site-packages makes the distro's PyGObject/GTK visible in the venv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# CPU-only PyTorch (small). For NVIDIA use .../whl/cu126, AMD .../whl/rocm7.2,
# Intel .../whl/xpu — see https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

pixelift                               # start the desktop app
./scripts/install_desktop_entry.sh     # optional: add it to the app launcher
```

> If `python3 -m venv` says *ensurepip is not available*, install
> `python3-venv` (or `python3.12-venv`), or create the venv with
> `--without-pip` and bootstrap pip with `get-pip.py`.

## AI models

Models are **not** bundled. On first launch Pixelift detects your hardware,
explains that a model is needed, offers to download the recommended one
(nothing is downloaded without your consent), and runs a small test upscale.
You can manage models later in **Preferences → Models**.

| Model id | Name | Native scale | Size | Best for |
|---|---|---|---|---|
| `realesrgan-x4plus` | Real-ESRGAN x4 (recommended) | 4× | 67 MB | Photos, general images |
| `realesrgan-x2plus` | Real-ESRGAN x2 | 2× | 67 MB | Photos at 2× |
| `realesr-general-x4v3` | Real-ESRGAN General v3 (fast) | 4× | 4.9 MB | CPU-only computers |
| `realesrgan-x4plus-anime` | Real-ESRGAN Anime | 4× | 18 MB | Anime, illustrations |
| `realesr-animevideov3` | Real-ESRGAN Anime Video v3 (fast) | 4× | 2.5 MB | Anime, fast |

In the app you pick a *model family* (e.g. "Real-ESRGAN") and a scale; Pixelift
uses the matching native-scale weights if installed, otherwise a larger-scale
model followed by high-quality downsampling.

**Licenses.** All model weights are published by the Real-ESRGAN project under
the [BSD 3-Clause license](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE)
and are downloaded from its official GitHub releases. Every file is verified
against a SHA-256 checksum recorded in `pixelift/models/realesrgan.py`.

**Command line / offline installation:**

```bash
pixelift --list-models
pixelift --download-model recommended --download-model realesr-general-x4v3
python scripts/download_models.py --all

# Air-gapped machine: download the .pth file elsewhere, then
pixelift --install-model-file realesrgan-x4plus ~/Downloads/RealESRGAN_x4plus.pth
```

Models are stored in `~/.local/share/pixelift/models/` (override with
`PIXELIFT_MODELS_DIR`).

## Usage

### Desktop app

1. Drop images or folders onto the window (or click **Select Images**, Ctrl+O).
2. Choose **Scale**, **Model**, **Format** and **Output** folder in the bottom bar.
3. Click **Start Upscaling**. Use **Pause**, **Resume**, **Cancel** and
   **Retry Failed** as needed. Images whose result already exists are skipped.
4. Click an image (or its compare button) to open the before/after view:
   drag the divider, scroll to zoom, drag to pan, **Space** toggles
   before/after, **1** = 100 %, **0** = fit, **+/−** zoom.

Results go to `<original folder>/upscaled/` by default, named with the template
`{name}_{scale}x` (e.g. `photo.jpg → photo_4x.png`). Template fields: `{name}`,
`{scale}`, `{model}`, `{width}`, `{height}`, `{ext}`.

### Command line

```bash
pixelift input.jpg --scale 4

pixelift ./photos \
    --scale 4 \
    --model realesrgan \
    --output ./upscaled
```

| Option | Meaning |
|---|---|
| `--scale 2\|4` | Upscale factor |
| `--model MODEL` | Family (`realesrgan`, `realesrgan-general`, `realesrgan-anime`, `realesrgan-anime-fast`) or a model id |
| `--device auto\|cpu\|cuda\|cuda:N\|xpu` | Processing device |
| `--output DIRECTORY` | Output folder (default `<input dir>/upscaled`) |
| `--format png\|jpg\|webp` | Output format |
| `--quality 1-100` | JPEG/WebP quality |
| `--tile-size SIZE` | Tile size in px (`0` = automatic) |
| `--template TEMPLATE` | Output filename template |
| `--overwrite` / `--rename` | What to do if the output exists (default: skip) |
| `--no-metadata` | Don't copy EXIF/ICC |
| `--threads N` | CPU threads |
| `--list-devices`, `--list-models` | Show hardware / models |

`image-upscaler` is installed as an alias of `pixelift`. Running `pixelift`
with no arguments (or with `--gui [files…]`) opens the desktop app.

## GPU acceleration

Pixelift picks the best device automatically and shows it in the title bar
(e.g. *Processing device: NVIDIA GeForce RTX 3060 — CUDA*). You can override it
in **Preferences → Processing** or with `--device`. If the GPU fails or runs out
of memory, Pixelift shrinks the tile size and, as a last resort, continues on
the CPU.

| Hardware | Requirement |
|---|---|
| NVIDIA | Proprietary driver (`sudo ubuntu-drivers install`) + CUDA build of PyTorch |
| AMD | ROCm-supported Radeon GPU + ROCm build of PyTorch |
| Intel | Arc / Core Ultra GPU + XPU build of PyTorch |
| CPU | Always works (the compact *General v3* model is ~10× faster on CPU) |

The `.deb`/AppImage ship the CPU build of PyTorch to keep downloads small. To
enable a GPU for the packaged app (installs per-user, nothing system-wide):

```bash
/usr/lib/pixelift/install-gpu-runtime.sh cu126     # NVIDIA (or cu130)
/usr/lib/pixelift/install-gpu-runtime.sh rocm7.2   # AMD
/usr/lib/pixelift/install-gpu-runtime.sh xpu       # Intel
/usr/lib/pixelift/install-gpu-runtime.sh remove    # back to CPU build
pixelift --list-devices
```

For a source install, simply install the matching PyTorch build in the venv.
`pixelift --list-devices` also explains when a GPU is present but unusable
(e.g. "An NVIDIA GPU was found but CUDA is not available").

**Memory settings.** *Tile size* (Automatic / 256 / 512 / 1024) trades speed for
memory; *GPU memory limit* caps PyTorch's share of an NVIDIA/AMD GPU.

## Troubleshooting

| Problem | Fix |
|---|---|
| "AI model not installed" | Preferences → Models → Download, or `pixelift --download-model recommended` |
| "Not enough memory" / GPU out of memory | Lower *Tile size*, use 2×, switch to CPU, close other apps |
| "Image too large" for WebP | WebP is limited to 16383 px per side — choose PNG |
| GPU not used | `pixelift --list-devices`; install the driver and a GPU build of PyTorch |
| Very slow | Expected on CPU with Real-ESRGAN x4 (~10–30 s per megapixel); try *Real-ESRGAN General (fast)* |
| App doesn't start | Needs GTK 4 + libadwaita ≥ 1.5: `sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1` |
| Something else | Menu → **Open Log Folder**. Errors are logged in detail to `~/.local/state/pixelift/app.log` |

Settings are stored in `~/.config/pixelift/settings.json`.

## Development

```bash
source .venv/bin/activate
pytest                 # full suite; GPU tests auto-skip without a GPU
pytest -m "not models" # skip tests needing real downloaded weights
pytest -m gpu          # GPU tests only
ruff check . && ruff format --check .
```

Test markers: `gpu` (skipped unless CUDA/ROCm/XPU is usable), `models` (skipped
unless real weights are installed), `gui` (skipped without a display). The core
tests use tiny randomly-initialised networks, so no download or GPU is needed.

### Architecture

```text
pixelift/
├── main.py              entry point: GUI without args, CLI otherwise
├── cli.py               command-line interface
├── core/                ── no GTK imports anywhere in here ──
│   ├── upscaler.py      Upscaler ABC + TorchUpscaler (tiling, OOM recovery, CPU fallback)
│   ├── tiling.py        tile layout + seam feathering (pure numpy)
│   ├── image_processor.py  load → upscale → restore alpha/mode → save (one image)
│   ├── batch_processor.py  bounded worker pool, pause/resume/cancel, events
│   ├── device_manager.py   CUDA / ROCm / XPU / CPU detection
│   ├── model_manager.py    download, checksum, install, remove
│   ├── control.py       cooperative pause/cancel
│   └── errors.py        user-facing error types
├── models/              model catalogue + network architectures (add new models here)
├── storage/             XDG paths, JSON settings
├── utils/               image I/O, metadata, logging, system info
└── ui/                  GTK 4 / libadwaita front-end
```

The UI only talks to `core` through `Upscaler`, `BatchProcessor` (events are
marshalled to the main loop with `GLib.idle_add`) and `ModelManager`, so the
engine could be reused from another front-end (e.g. Rust) or replaced.

**Adding a model:** implement the network (or reuse `RRDBNet` /
`SRVGGNetCompact`), create `ModelSpec`s with URL, SHA-256 and license in a new
module under `pixelift/models/`, register a `ModelFamily`, and import the module
in `pixelift/models/__init__.py`.

### Packaging

```bash
packaging/build_deb.sh        # → build/pixelift_<version>_amd64.deb
packaging/build_appimage.sh   # → build/Pixelift-<version>-x86_64.AppImage
# Flatpak (local build, needs network during build):
flatpak-builder --user --install --force-clean --build-args=--share=network \
    build/flatpak packaging/flatpak/io.github.pixelift.Pixelift.yml
```

Both scripts bundle the CPU build of PyTorch for the build machine's Python
version (build on Ubuntu 24.04 for 24.04 targets).

### Known limitations

- 16-bit and floating-point images are processed at 8-bit precision.
- Very large outputs (≈ 100+ megapixels) need several GB of RAM for the final
  image buffer; Pixelift refuses jobs that would not fit instead of crashing.
- When zoomed into a huge result, the preview decodes the visible region from
  disk on demand, which can take a moment for very large PNGs.

## License

Pixelift is MIT-licensed (see `LICENSE`). The Real-ESRGAN network architectures
are re-implemented from BSD-3-Clause code © 2021 Xintao Wang; model weights are
downloaded separately under the same BSD-3-Clause license.
