"""A lock between processes, so two ``visin sync`` commands cannot send the same reports.

Benchmarks and configs carry no id of their own, so a report sent twice is
stored twice. Sync records each one as it is delivered, but that record is only
read at the start of a pass; two passes at once would both read it before
either wrote. An operating-system lock closes the gap, and is released when the
process ends, even if it is killed.

Windows has no ``fcntl``, and there the lock does nothing: sync stays safe to
repeat, just not safe to run twice at the same moment.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]


SYNC_LOCK = ".sync.lock"


@contextlib.contextmanager
def exclusive(path: Path) -> Iterator[bool]:
    """Hold an exclusive lock on ``path`` for the block. Yields ``False`` when another process has it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as handle:
        if fcntl is None:  # pragma: no cover - Windows
            yield True
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
