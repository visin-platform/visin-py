"""Log through the current run, for code that cannot pass a ``Run`` around.

A helper deep in a training codebase can call ``visin.log_epoch(...)`` without
being handed the run: these functions use the run most recently made by
``visin.init`` or ``Run.attach`` (see :func:`visin.get_run`). With none, they
do nothing, like a disabled run. They take exactly what the ``Run`` methods
take; see those for the arguments.
"""

from __future__ import annotations

from typing import Any

from .run import get_run


def log_epoch(*args: Any, **kwargs: Any) -> str | None:
    """:meth:`Run.log_epoch` on the current run."""
    return get_run().log_epoch(*args, **kwargs)


def log_test_results(*args: Any, **kwargs: Any) -> str | None:
    """:meth:`Run.log_test_results` on the current run."""
    return get_run().log_test_results(*args, **kwargs)


def log_benchmark(*args: Any, **kwargs: Any) -> None:
    """:meth:`Run.log_benchmark` on the current run."""
    get_run().log_benchmark(*args, **kwargs)


def log_config(*args: Any, **kwargs: Any) -> None:
    """:meth:`Run.log_config` on the current run."""
    get_run().log_config(*args, **kwargs)


def log_model(*args: Any, **kwargs: Any) -> str | None:
    """:meth:`Run.log_model` on the current run."""
    return get_run().log_model(*args, **kwargs)


def upload_visualization(*args: Any, **kwargs: Any) -> None:
    """:meth:`Run.upload_visualization` on the current run."""
    get_run().upload_visualization(*args, **kwargs)


def update(*args: Any, **kwargs: Any) -> None:
    """:meth:`Run.update` on the current run."""
    get_run().update(*args, **kwargs)


def finish(*args: Any, **kwargs: Any) -> None:
    """:meth:`Run.finish` on the current run."""
    get_run().finish(*args, **kwargs)
