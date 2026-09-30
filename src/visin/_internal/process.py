"""What this process is: which rank it is, and how it ended."""

from __future__ import annotations

import os
import signal
import sys
import threading
from typing import Any


def rank() -> int:
    """This process's rank in a distributed job, 0 when it is not in one.

    Every rank of a data-parallel job runs the same script, so without this a
    four-GPU run would report each epoch four times.
    """
    for name in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK"):
        raw = (os.getenv(name) or "").strip()
        if raw.isdigit():
            return int(raw)
    return 0


class ProcessHooks:
    """Once per process: learn how it ended, so an unfinished run is marked right.

    Python's default answer to SIGTERM, which is what a scheduler sends before
    it kills a job, is to die on the spot: no ``finally``, no ``atexit``, so the
    epochs still queued were lost and the run stayed "running" forever. Turning
    it into ``SystemExit`` lets the ordinary shutdown path run. Only when nobody
    else has claimed the signal: Lightning, for one, installs its own handler to
    requeue on SLURM, and that must win.

    A platform without SIGTERM, or a thread that cannot install signal
    handlers, keeps Python's default.
    """

    installed = False
    crashed = False
    terminated = False

    @classmethod
    def install(cls) -> None:
        if cls.installed:
            return
        cls.installed = True
        previous = sys.excepthook

        def excepthook(exc_type: Any, exc: Any, tb: Any) -> None:
            cls.crashed = True
            previous(exc_type, exc, tb)

        sys.excepthook = excepthook
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
                signal.signal(signal.SIGTERM, cls._on_sigterm)
        except (AttributeError, OSError, ValueError):  # pragma: no cover
            pass

    @classmethod
    def _on_sigterm(cls, signum: int, _frame: Any) -> None:
        cls.terminated = True
        raise SystemExit(128 + signum)
