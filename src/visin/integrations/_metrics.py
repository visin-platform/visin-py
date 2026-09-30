"""Sorting a framework's flat metric names into Visin's ``train`` and ``val``.

Frameworks log ``val_loss``, ``val/iou``, ``train_acc_epoch``. Visin's charts
read ``train.loss`` and ``val.loss``, so the prefix becomes the group and the
rest becomes the metric.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_PREFIXES = (
    ("val", ("val_", "val/", "valid_", "valid/", "validation_", "validation/")),
    ("train", ("train_", "train/", "training_", "training/")),
    ("test", ("test_", "test/")),
)

# Keras derives a metric's name from its class, so MeanIoU logs as
# ``mean_io_u``. Visin's mean IoU chart reads ``mean_iou``.
_ALIASES = {"mean_io_u": "mean_iou", "io_u": "iou"}

_NOT_METRICS = frozenset({"lr", "learning_rate", "epoch", "step", "global_step"})


def split_metrics(metrics: Mapping[str, Any], default: str = "train") -> dict[str, dict[str, Any]]:
    """``{"val_loss": 1, "loss": 2}`` → ``{"val": {"loss": 1}, "train": {"loss": 2}}``.

    A name with no recognised prefix goes to ``default``. Lightning's
    ``_step`` copies of a metric are left out and its ``_epoch`` copies lose
    the suffix, since an epoch report wants the epoch value.
    """
    groups: dict[str, dict[str, Any]] = {}
    for raw, value in metrics.items():
        name = str(raw)
        if name in _NOT_METRICS or name.endswith("_step"):
            continue
        name = name.removesuffix("_epoch")
        group = default
        for candidate, prefixes in _PREFIXES:
            match = next((prefix for prefix in prefixes if name.startswith(prefix)), None)
            if match:
                group, name = candidate, name[len(match) :]
                break
        if not name:
            continue
        groups.setdefault(group, {})[_ALIASES.get(name, name)] = value
    return groups
