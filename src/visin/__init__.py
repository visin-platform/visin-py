"""Report training runs to a Visin instance, and read them back.

    import visin

    with visin.init("unet baseline", project="road-seg") as run:
        for epoch in range(1, epochs + 1):
            train_loss, val_loss, val_iou = train_one_epoch()
            run.log_epoch(epoch, train={"loss": train_loss}, val={"loss": val_loss, "mean_iou": val_iou})

``init`` reads ``VISIN_URL`` and ``VISIN_TOKEN``. With neither set it returns a
run that reports nothing, so the same script still runs on a laptop.
"""

from ._internal.config import Settings, read_settings
from ._internal.console import enable_console_logging
from ._version import __version__
from .api import Api, flatten
from .datasets import Datasets
from .errors import ApiError, ConfigurationError, TransportError, VisinError
from .offline import SyncResult, pending, sync
from .run import Run, epoch_uuid_for, init
from .system import system_info, system_metrics

__all__ = [
    "init",
    "Run",
    "Api",
    "Datasets",
    "sync",
    "pending",
    "SyncResult",
    "epoch_uuid_for",
    "flatten",
    "system_info",
    "system_metrics",
    "enable_console_logging",
    "Settings",
    "read_settings",
    "VisinError",
    "ApiError",
    "TransportError",
    "ConfigurationError",
    "__version__",
]
