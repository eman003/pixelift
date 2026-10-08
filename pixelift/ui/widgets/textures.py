"""Convert Pillow images to GDK textures."""

from __future__ import annotations

from gi.repository import Gdk, GLib
from PIL import Image


def texture_from_pil(img: Image.Image) -> Gdk.Texture:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    width, height = img.size
    data = GLib.Bytes.new(img.tobytes())
    return Gdk.MemoryTexture.new(width, height, Gdk.MemoryFormat.R8G8B8A8, data, width * 4)
