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
from .datasets import CachedDataset, Datasets, cached_datasets, remove_cached
from .errors import ApiError, ConfigurationError, SyncInProgressError, TransportError, VisinError
from .models import (
    Benchmark,
    Comparison,
    Configuration,
    Dataset,
    Epoch,
    Finding,
    MetricSummary,
    Project,
    Summary,
    TestResult,
    Training,
    Visualization,
)
from .offline import SyncResult, pending, sync
from .run import Run, epoch_uuid_for, get_run, init
from .shortcuts import (
    finish,
    log_benchmark,
    log_config,
    log_epoch,
    log_model,
    log_test_results,
    update,
    upload_visualization,
)
from .system import system_info, system_metrics

__all__ = [
    "Api",
    "ApiError",
    "Benchmark",
    "CachedDataset",
    "Comparison",
    "Configuration",
    "ConfigurationError",
    "Dataset",
    "Datasets",
    "Epoch",
    "Finding",
    "MetricSummary",
    "Project",
    "Run",
    "Settings",
    "Summary",
    "SyncInProgressError",
    "SyncResult",
    "TestResult",
    "Training",
    "TransportError",
    "VisinError",
    "Visualization",
    "__version__",
    "cached_datasets",
    "enable_console_logging",
    "epoch_uuid_for",
    "finish",
    "flatten",
    "get_run",
    "init",
    "log_benchmark",
    "log_config",
    "log_epoch",
    "log_model",
    "log_test_results",
    "pending",
    "read_settings",
    "remove_cached",
    "sync",
    "system_info",
    "system_metrics",
    "update",
    "upload_visualization",
]
