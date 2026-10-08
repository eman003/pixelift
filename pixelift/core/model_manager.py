"""Download, verify, list and remove AI model weight files."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pixelift import __version__
from pixelift.core.control import JobControl
from pixelift.core.errors import ModelDownloadError, ModelNotInstalledError
from pixelift.models import ModelSpec, all_specs, candidate_specs, get_spec
from pixelift.storage import paths

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]  # (bytes_done, bytes_total)


@dataclass(frozen=True)
class ModelStatus:
    spec: ModelSpec
    installed: bool
    path: Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


class ModelManager:
    def __init__(
        self, models_dir: Path | None = None, specs: list[ModelSpec] | None = None
    ) -> None:
        self.models_dir = models_dir or paths.models_dir()
        self._specs = {s.id: s for s in (specs if specs is not None else all_specs())}

    # --- queries -----------------------------------------------------------
    def spec(self, model_id: str) -> ModelSpec:
        if model_id in self._specs:
            return self._specs[model_id]
        return get_spec(model_id)

    def specs(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def path_for(self, spec: ModelSpec | str) -> Path:
        spec = self.spec(spec) if isinstance(spec, str) else spec
        return self.models_dir / spec.filename

    def is_installed(self, spec: ModelSpec | str) -> bool:
        path = self.path_for(spec)
        return path.is_file() and path.stat().st_size > 0

    def statuses(self) -> list[ModelStatus]:
        return [ModelStatus(s, self.is_installed(s), self.path_for(s)) for s in self.specs()]

    def installed(self) -> list[ModelSpec]:
        return [s for s in self.specs() if self.is_installed(s)]

    def resolve(self, model: str, scale: int) -> ModelSpec:
        """Pick the installed weights that best serve ``model`` at ``scale``."""
        try:
            candidates = (
                [self._specs[model]] if model in self._specs else candidate_specs(model, scale)
            )
        except KeyError:
            raise ModelNotInstalledError(
                f"“{model}” is not a known model.",
                ["Run with --list-models to see available models"],
                title="Unknown model.",
            ) from None
        for spec in candidates:
            if self.is_installed(spec):
                return spec
        names = ", ".join(f"{s.name} ({s.size_mb:.0f} MB)" for s in candidates)
        raise ModelNotInstalledError(
            f"The model needed for this job is not installed: {names}.",
            [
                "Open Settings → Models and download it",
                "Or run: pixelift --download-model " + candidates[0].id,
            ],
        )

    # --- mutations ---------------------------------------------------------
    def verify(self, spec: ModelSpec | str) -> bool:
        spec = self.spec(spec) if isinstance(spec, str) else spec
        path = self.path_for(spec)
        return path.is_file() and sha256_file(path) == spec.sha256

    def remove(self, spec: ModelSpec | str) -> None:
        path = self.path_for(spec)
        path.unlink(missing_ok=True)
        log.info("Removed model %s", path)

    def install_file(self, spec: ModelSpec | str, source: Path) -> Path:
        """Install a manually downloaded weights file after verifying its checksum."""
        spec = self.spec(spec) if isinstance(spec, str) else spec
        digest = sha256_file(source)
        if digest != spec.sha256:
            raise ModelDownloadError(
                f"{source.name} does not match the expected checksum for {spec.name}.",
                ["Download the file again from the official release page"],
            )
        self.models_dir.mkdir(parents=True, exist_ok=True)
        target = self.path_for(spec)
        shutil.copyfile(source, target)
        return target

    def download(
        self,
        spec: ModelSpec | str,
        progress: ProgressCallback | None = None,
        control: JobControl | None = None,
        timeout: float = 30.0,
    ) -> Path:
        """Download weights to a temp file, verify SHA-256, then move into place."""
        spec = self.spec(spec) if isinstance(spec, str) else spec
        self.models_dir.mkdir(parents=True, exist_ok=True)
        target = self.path_for(spec)
        log.info("Downloading %s from %s", spec.id, spec.url)
        request = urllib.request.Request(
            spec.url, headers={"User-Agent": f"pixelift/{__version__}"}
        )
        fd, tmp_name = tempfile.mkstemp(prefix=".download-", dir=self.models_dir)
        tmp = Path(tmp_name)
        digest = hashlib.sha256()
        try:
            with (
                os.fdopen(fd, "wb") as out,
                urllib.request.urlopen(request, timeout=timeout) as response,
            ):
                total = int(response.headers.get("Content-Length") or spec.size_bytes)
                done = 0
                while block := response.read(1 << 16):
                    if control is not None:
                        control.check()
                    out.write(block)
                    digest.update(block)
                    done += len(block)
                    if progress:
                        progress(done, total)
            if digest.hexdigest() != spec.sha256:
                raise ModelDownloadError(
                    f"The downloaded file for {spec.name} failed checksum verification.",
                    ["Try downloading again", "Check your internet connection"],
                )
            os.replace(tmp, target)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            log.error("Download of %s failed: %s", spec.id, exc)
            raise ModelDownloadError(
                f"Could not download {spec.name}: {getattr(exc, 'reason', exc)}",
                ["Check your internet connection", "Try again later"],
            ) from exc
        finally:
            tmp.unlink(missing_ok=True)
        log.info("Installed model %s (%s)", spec.id, target)
        return target
