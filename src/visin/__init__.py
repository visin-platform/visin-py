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
from .models import Benchmark, Dataset, Epoch, Project, TestResult, Training
from .offline import SyncResult, pending, sync
from .run import Run, epoch_uuid_for, init
from .system import system_info, system_metrics

__all__ = [
    "Api",
    "ApiError",
    "Benchmark",
    "ConfigurationError",
    "Dataset",
    "Datasets",
    "Epoch",
    "Project",
    "Run",
    "Settings",
    "SyncResult",
    "TestResult",
    "Training",
    "TransportError",
    "VisinError",
    "__version__",
    "enable_console_logging",
    "epoch_uuid_for",
    "flatten",
    "init",
    "pending",
    "read_settings",
    "sync",
    "system_info",
    "system_metrics",
]
