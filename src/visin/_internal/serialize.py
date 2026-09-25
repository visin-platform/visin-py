"""Turning what a training loop has in hand into JSON Visin will accept.

A loop's metrics are rarely plain floats. They are NumPy scalars, zero-dimensional
tensors, the odd ``Decimal``; ``json.dumps`` refuses every one of them, and in
a background sender that refusal turns into a metric that silently never
arrives. Converting here, at the call, keeps the mistake visible and the
common cases working.

NaN and infinity have no JSON spelling. Sent anyway, the body is not valid JSON
and the server refuses the whole epoch, so they become ``null``, which Visin
already reads as "not measured" and charts as a gap.
"""

from __future__ import annotations

import dataclasses
import enum
import math
import uuid
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from pathlib import PurePath
from typing import Any, Callable


def to_jsonable(
    value: Any,
    nonfinite: list[str] | None = None,
    path: str = "",
    fallback: Callable[[Any], Any] | None = None,
) -> Any:
    """Return ``value`` as plain JSON types.

    ``nonfinite``, when given, collects the dotted path of every NaN or
    infinity that was replaced by ``None``, so the caller can say so.
    ``fallback`` converts what nothing else can; without it, that raises.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if nonfinite is not None:
            nonfinite.append(path or "<value>")
        return None
    if isinstance(value, Mapping):
        return {
            str(key): to_jsonable(item, nonfinite, f"{path}.{key}" if path else str(key), fallback)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            to_jsonable(item, nonfinite, f"{path}[{index}]", fallback) for index, item in enumerate(value)
        ]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return to_jsonable(float(value), nonfinite, path, fallback)
    if isinstance(value, (uuid.UUID, PurePath)):
        return str(value)
    if isinstance(value, enum.Enum):
        return to_jsonable(value.value, nonfinite, path, fallback)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(dataclasses.asdict(value), nonfinite, path, fallback)
    # NumPy scalars and arrays, and PyTorch/TensorFlow tensors, all offer
    # ``tolist``: a scalar comes back as a Python number, an array as nested
    # lists. Duck-typed so none of those libraries has to be installed.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return to_jsonable(tolist(), nonfinite, path, fallback)
    item = getattr(value, "item", None)
    if callable(item):
        return to_jsonable(item(), nonfinite, path, fallback)
    numpy = getattr(value, "numpy", None)  # an eager TensorFlow tensor
    if callable(numpy):
        return to_jsonable(numpy(), nonfinite, path, fallback)
    if fallback is not None:
        return fallback(value)
    raise TypeError(
        f"visin cannot send a {type(value).__name__}"
        f"{f' (at {path})' if path else ''}; convert it to a number, string, list or dict"
    )
