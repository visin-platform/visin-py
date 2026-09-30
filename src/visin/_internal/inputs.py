"""Turning what a caller passes into what Visin takes."""

from __future__ import annotations

import argparse
import dataclasses
import math
import operator
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def epoch_number(epoch: Any) -> int | float:
    """An epoch as the number it names, so ``3``, ``3.0`` and ``np.int64(3)``
    are one epoch with one UUID."""
    if isinstance(epoch, bool):
        raise TypeError("epoch must be a number")
    if isinstance(epoch, int):
        return epoch
    try:
        return operator.index(epoch)
    except TypeError:
        pass
    value = float(epoch)
    if not math.isfinite(value):
        raise ValueError(f"epoch must be finite, not {epoch!r}")
    return int(value) if value.is_integer() else value


def merge_results(
    results: Mapping[str, Any] | None,
    train: Mapping[str, Any] | None,
    val: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """``results`` with ``train`` and ``val`` folded in.

    ``results["train"]`` and ``results["val"]`` are where Visin's charts look
    for the training and the validation curve, so the two keywords are
    shorthand for them.
    """
    if results is not None and not isinstance(results, Mapping):
        raise TypeError(f"results must be a dict, not {type(results).__name__}")
    merged: dict[str, Any] = dict(results or {})
    for key, part in (("train", train), ("val", val)):
        if part is None:
            continue
        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(part)
        elif isinstance(existing, Mapping):
            merged[key] = {**existing, **part}
        else:
            raise ValueError(f"results[{key!r}] is not a dict, so {key}= cannot be added to it")
    if not merged:
        raise ValueError("log_epoch needs results, train= or val=")
    return merged


def as_mapping(config: Any) -> dict[str, Any]:
    """A training config as a dict, whatever shape it comes in.

    Takes a mapping, an ``argparse.Namespace``, a dataclass, a Hydra/OmegaConf
    config, or anything with ``model_dump`` (pydantic 2), ``to_dict`` or
    ``dict`` (pydantic 1).
    """
    if isinstance(config, Mapping):
        return dict(config)
    if isinstance(config, argparse.Namespace):
        return dict(vars(config))
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.asdict(config)
    if type(config).__module__.split(".")[0] == "omegaconf":
        from omegaconf import OmegaConf  # type: ignore[import-not-found]

        container = OmegaConf.to_container(config, resolve=True)
        if isinstance(container, Mapping):
            return dict(container)
    for method in ("model_dump", "to_dict", "dict"):
        convert = getattr(config, method, None)
        if callable(convert):
            value = convert()
            if isinstance(value, Mapping):
                return dict(value)
    raise TypeError(f"cannot read a config from a {type(config).__name__}; pass a dict")


def default_name() -> str:
    """The script's name and the time, for a run nobody named."""
    script = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] not in ("", "-c") else "run"
    return f"{script} {datetime.now():%Y-%m-%d %H:%M}"
