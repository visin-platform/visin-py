"""Showing visin's log lines on the console."""

from __future__ import annotations

import logging
import sys
from typing import TextIO

_MARK = "_visin_console"


def enable_console_logging(level: int | str = logging.INFO, stream: TextIO | None = None) -> None:
    """Print visin's log lines: the run it created or resumed, anything it could
    not send, and its summary at ``finish``.

    visin logs through :mod:`logging` under the ``visin`` logger and, as a
    library should, adds no handler of its own, so without this only warnings
    reach the console. Pass ``stream=sys.stdout`` to keep the lines in order
    with a script's own ``print`` output. Calling it again replaces the handler
    rather than printing every line twice.
    """
    logger = logging.getLogger("visin")
    for handler in list(logger.handlers):
        if getattr(handler, _MARK, False):
            logger.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    setattr(handler, _MARK, True)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
