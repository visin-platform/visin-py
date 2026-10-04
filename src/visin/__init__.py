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
from .compare import Change, Diff, diff_evaluations
from .datasets import CachedDataset, Datasets, cached_datasets, remove_cached
from .errors import ApiError, ConfigurationError, SyncInProgressError, TransportError, VisinError
from .evaluation import evaluate, hub_checkpoint, local_checkpoint, promote, publish, sha256_of, withdraw
from .models import (
    Benchmark,
    Comparison,
    Configuration,
    Dataset,
    Epoch,
    Evaluation,
    Finding,
    Leaderboard,
    LeaderboardEntry,
    MetricSummary,
    Pagination,
    Project,
    Reason,
    Suite,
    Summary,
    TestResult,
    Training,
    Verdict,
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
from .suites import check_protocol, load_suite, manifest_digest, push_suite, read_split
from .system import system_info, system_metrics

__all__ = [
    "Api",
    "ApiError",
    "Benchmark",
    "CachedDataset",
    "Change",
    "Comparison",
    "Configuration",
    "ConfigurationError",
    "Dataset",
    "Datasets",
    "Diff",
    "Epoch",
    "Evaluation",
    "Finding",
    "Leaderboard",
    "LeaderboardEntry",
    "MetricSummary",
    "Pagination",
    "Project",
    "Reason",
    "Run",
    "Settings",
    "Suite",
    "Summary",
    "SyncInProgressError",
    "SyncResult",
    "TestResult",
    "Training",
    "TransportError",
    "Verdict",
    "VisinError",
    "Visualization",
    "__version__",
    "cached_datasets",
    "check_protocol",
    "diff_evaluations",
    "enable_console_logging",
    "epoch_uuid_for",
    "evaluate",
    "finish",
    "flatten",
    "get_run",
    "hub_checkpoint",
    "init",
    "load_suite",
    "local_checkpoint",
    "log_benchmark",
    "log_config",
    "log_epoch",
    "log_model",
    "log_test_results",
    "manifest_digest",
    "pending",
    "promote",
    "publish",
    "push_suite",
    "read_settings",
    "read_split",
    "remove_cached",
    "sha256_of",
    "sync",
    "system_info",
    "system_metrics",
    "update",
    "upload_visualization",
    "withdraw",
]
