"""An image held in memory, as the PNG bytes Visin stores.

A prediction overlay is usually an array, a tensor, a PIL image or a Matplotlib
figure, and saving it only to hand Visin a path is a step every script repeats.
Arrays and tensors are encoded here with ``zlib`` alone, so this needs neither
Pillow nor Matplotlib; those two are used only when the object is one of theirs.

A float array is read as 0 to 1, or as 0 to 255 when it holds a larger value, so
both ``image / 255`` and a raw ``uint8``-range float come out right. A
``(channels, height, width)`` array, which is what PyTorch holds, is turned to
``(height, width, channels)`` when its first axis is 1 to 4 and the other two are larger than 4.
"""

from __future__ import annotations

import importlib
import io
import struct
import zlib
from typing import Any

COLOR_TYPES = {1: 0, 2: 4, 3: 2, 4: 6}


def _numpy() -> Any:
    """NumPy, imported by name: its type stubs need a newer Python than mypy is held to."""
    return importlib.import_module("numpy")


def is_image(obj: Any) -> bool:
    """Whether ``obj`` is something ``encode_png`` takes: not a path."""
    return not isinstance(obj, (str, bytes)) and not hasattr(obj, "__fspath__")


def encode_png(image: Any) -> bytes:
    """``image`` as PNG bytes. Raises ``TypeError`` or ``ValueError`` for one it cannot read."""
    if hasattr(image, "savefig"):
        buffer = io.BytesIO()
        image.savefig(buffer, format="png")
        return buffer.getvalue()
    if hasattr(image, "save") and hasattr(image, "mode"):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    return _encode_array(_as_array(image))


def _as_array(image: Any) -> Any:
    np: Any = _numpy()

    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[0] in (1, 2, 3, 4) and min(array.shape[1:]) > 4:
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[-1] not in (2, 3, 4)):
        raise ValueError(f"cannot make an image of shape {tuple(array.shape)}: expected HxW, HxWx3 or HxWx4")
    if array.dtype == bool:
        return array.astype(np.uint8) * 255
    if array.dtype == np.uint8:
        return array
    if array.dtype.kind == "f":
        scale = 255.0 if float(np.nanmax(array, initial=0.0)) <= 1.0 else 1.0
        array = np.nan_to_num(array) * scale
    elif array.dtype.kind not in "iu":
        raise TypeError(f"cannot make an image of {array.dtype} values")
    return np.clip(array, 0, 255).round().astype(np.uint8)


def _chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def _encode_array(array: Any) -> bytes:
    np: Any = _numpy()

    height, width = array.shape[:2]
    channels = 1 if array.ndim == 2 else array.shape[2]
    rows = np.ascontiguousarray(array).reshape(height, width * channels)
    raw = np.hstack([np.zeros((height, 1), dtype=np.uint8), rows]).tobytes()
    header = struct.pack(">IIBBBBB", width, height, 8, COLOR_TYPES[channels], 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )
