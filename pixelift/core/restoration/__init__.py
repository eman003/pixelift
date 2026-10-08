"""AI photo restoration: scan cleanup, colour recovery, faces, colorization.

``settings`` describes what to do and ``pipeline.Restorer`` does it. The
conventional stages (``cleanup``, ``tones``) need no AI models; faces and
colorization use the models from ``pixelift.models.restoration``.

Importing this package (and ``settings`` / ``analysis``) does not import
PyTorch, so the GUI can start quickly; ``pipeline`` and the stages do.
"""
