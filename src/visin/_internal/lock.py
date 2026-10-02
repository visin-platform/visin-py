"""A lock between processes, so two ``visin sync`` commands cannot send the same reports.

Benchmarks and configs carry no id of their own, so a report sent twice is
stored twice. Sync records each one as it is delivered, but that record is only
read at the start of a pass; two passes at once would both read it before
either wrote. An operating-system lock closes the gap, and is released when the
process ends, even if it is killed.

POSIX uses ``flock`` on the whole file; Windows locks its first byte with
``msvcrt.locking``. Both hold per open file, so a second ``open`` of the same
path is refused even inside one process.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import IO

SYNC_LOCK = ".sync.lock"


if sys.platform == "win32":  # pragma: no cover - exercised by the Windows CI job
    import msvcrt

    def _try_lock(handle: IO[bytes]) -> bool:
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(handle: IO[bytes]) -> None:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _try_lock(handle: IO[bytes]) -> bool:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _unlock(handle: IO[bytes]) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def exclusive(path: Path) -> Iterator[bool]:
    """Hold an exclusive lock on ``path`` for the block. Yields ``False`` when another process has it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as handle:
        if not _try_lock(handle):
            yield False
            return
        try:
            yield True
        finally:
            _unlock(handle)
