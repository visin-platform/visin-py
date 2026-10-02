"""Reports kept on disk until they can be sent.

A compute node often has no route to Visin at all, and a node that does can
lose it for an hour. Either way the run should train on and its reports should
arrive eventually, so they wait here, one JSON Lines file per run, and
``visin sync`` (or the run itself, once the server answers again) sends them.

Layout under the directory (``~/.visin`` unless ``VISIN_DIR`` says otherwise)::

    runs/<training_uuid>.jsonl            appended to while the run writes
    runs/<training_uuid>.<n>.sending      a batch a sync has claimed
    runs/<training_uuid>.<n>.sending.done line numbers of it already delivered
    runs/<training_uuid>.files/           copies of the files visualizations upload

A sync first renames the live file to a ``.sending`` batch. The rename is atomic,
so a run still appending simply starts a fresh ``.jsonl`` and nothing it writes
during the sync can be deleted with the batch. The ``.done`` file is what makes
an interrupted sync resume where it stopped rather than repeating a benchmark or
config, the two writes the server cannot recognise as repeats.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid as uuidlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import ApiError, ConfigurationError, VisinError
from .reports import KINDS, DeliveryContext, deliver, discard_staged
from .transport import HttpClient, worth_retrying_later

logger = logging.getLogger("visin")

RUNS = "runs"
# A run's UUID names its files, so it may not carry a path separator.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class Spool:
    """One run's reports on disk."""

    def __init__(self, directory: Path | str, training_uuid: str):
        if not training_uuid or not _SAFE_NAME.match(training_uuid):
            raise ConfigurationError(f"cannot keep reports for a run named {training_uuid!r} on disk")
        self.training_uuid = training_uuid
        self.root = Path(directory) / RUNS
        self.path = self.root / f"{training_uuid}.jsonl"
        self.files = self.root / f"{training_uuid}.files"
        self._lock = threading.Lock()

    def append(self, op: Mapping[str, Any]) -> None:
        line = (json.dumps(op, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with open(self.path, "ab+") as handle:
                # A process killed mid-write leaves a last line without its
                # newline. Appending straight after it would fuse this report
                # onto the torn one, and lose both.
                if handle.tell() > 0:
                    handle.seek(-1, os.SEEK_END)
                    if handle.read(1) != b"\n":
                        line = b"\n" + line
                handle.write(line)
                handle.flush()
                # A job killed right after this call should still find the line
                # on disk; one fsync per report is cheap at epoch frequency.
                os.fsync(handle.fileno())

    def stage(self, source: str | os.PathLike[str]) -> str:
        """Copy a file in, so the report no longer depends on the original.

        Scripts render each epoch's frames under the same file names, so by the
        time a queued upload runs, the original may already hold the next
        epoch's image. Returns the copy's name inside ``files``.
        """
        self.files.mkdir(parents=True, exist_ok=True)
        name = f"{uuidlib.uuid4().hex[:12]}-{os.path.basename(source)}"
        shutil.copyfile(source, self.files / name)
        return name

    def stage_bytes(self, data: bytes, filename: str) -> str:
        """Write ``data`` into ``files`` as a staged upload, like :meth:`stage`.

        For an image that exists only in memory. Returns the file's name inside ``files``.
        """
        self.files.mkdir(parents=True, exist_ok=True)
        name = f"{uuidlib.uuid4().hex[:12]}-{os.path.basename(filename)}"
        (self.files / name).write_bytes(data)
        return name

    def batches(self) -> list[Path]:
        """Claimed batches left by an earlier, interrupted sync, oldest first."""
        if not self.root.exists():
            return []
        prefix, suffix = f"{self.training_uuid}.", ".sending"
        # The stamp between the two must be all digits: run "a" must not take
        # the batches of a run named "a.b".
        found = [
            path
            for path in self.root.glob(f"{prefix}*{suffix}")
            if path.name[len(prefix) : -len(suffix)].isdigit()
        ]
        return sorted(found, key=lambda path: int(path.name[len(prefix) : -len(suffix)]))

    def claim(self) -> Path | None:
        """Turn the live file into a batch for sending. None when it is empty."""
        with self._lock:
            if not self.path.exists():
                return None
            batch = self.root / f"{self.training_uuid}.{time.time_ns()}.sending"
            try:
                os.replace(self.path, batch)
            except FileNotFoundError:  # another sync took it first
                return None
            except OSError as exc:  # Windows: the run has it open this instant
                logger.info("visin: %s is busy, leaving it for the next sync: %s", self.path.name, exc)
                return None
            return batch

    def pending(self) -> bool:
        return self.path.exists() or bool(self.batches())

    def count(self) -> int:
        """Reports not yet delivered, across the live file and any batches."""
        total = _count_lines(self.path)
        for batch in self.batches():
            total += _count_lines(batch) - len(_read_done(batch))
        return total

    def tidy(self) -> None:
        """Remove the files directory once nothing in it is still needed."""
        with contextlib.suppress(OSError):
            self.files.rmdir()


def _count_lines(path: Path) -> int:
    try:
        with open(path, encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except FileNotFoundError:
        return 0


def _done_path(batch: Path) -> Path:
    return batch.with_name(batch.name + ".done")


def _read_done(batch: Path) -> set[int]:
    try:
        text = _done_path(batch).read_text(encoding="utf-8")
    except FileNotFoundError:
        return set()
    return {int(line) for line in text.split() if line.isdigit()}


def _mark_done(batch: Path, index: int) -> None:
    with open(_done_path(batch), "a", encoding="utf-8") as handle:
        handle.write(f"{index}\n")


@dataclass
class SyncResult:
    """What one run's sync achieved."""

    training_uuid: str
    sent: int = 0
    rejected: list[str] = field(default_factory=list)
    remaining: int = 0
    kept_rejected: int = field(default=0, repr=False)

    @property
    def complete(self) -> bool:
        return self.remaining == 0 and not self.rejected


class _Unreachable(Exception):
    pass


def sync_spool(
    client: HttpClient,
    spool: Spool,
    *,
    drop_rejected: bool = False,
    context: DeliveryContext | None = None,
) -> SyncResult:
    """Send one run's pending reports, oldest first.

    Stops at the first failure that means the server is unreachable or
    overloaded, leaving the rest for next time. A report the server refuses
    outright is listed in ``rejected``; it stays on disk to be retried, unless
    ``drop_rejected`` is set, since a refusal leaves nothing stored and
    retrying it is harmless.
    """
    result = SyncResult(spool.training_uuid)
    context = context or DeliveryContext()
    context.files = spool.files
    batches = spool.batches()
    fresh = spool.claim()
    if fresh is not None:
        batches.append(fresh)
    try:
        for batch in batches:
            _sync_batch(client, batch, context, result, drop_rejected)
    except _Unreachable:
        pass
    # What is left on disk, less what the server refused: those are counted
    # as rejected, and "remaining" means waiting on the server.
    result.remaining = max(spool.count() - result.kept_rejected, 0)
    spool.tidy()
    return result


def _sync_batch(
    client: HttpClient,
    batch: Path,
    context: DeliveryContext,
    result: SyncResult,
    drop_rejected: bool,
) -> None:
    done = _read_done(batch)
    kept = False
    handled = 0
    while True:
        # Read again until nothing new appears: a writer that opened the live
        # file just before it was claimed finishes its line into this batch.
        lines = batch.read_text(encoding="utf-8").splitlines()
        if len(lines) == handled:
            break
        pending = [(index, lines[index]) for index in range(handled, len(lines))]
        handled = len(lines)
        kept = _send_lines(client, batch, pending, done, context, result, drop_rejected) or kept
    if kept:
        return
    batch.unlink()
    with contextlib.suppress(FileNotFoundError):
        _done_path(batch).unlink()


def _send_lines(
    client: HttpClient,
    batch: Path,
    pending: list[tuple[int, str]],
    done: set[int],
    context: DeliveryContext,
    result: SyncResult,
    drop_rejected: bool,
) -> bool:
    """Send one pass of a batch's lines. True if a refused report was kept."""
    kept = False
    for index, line in pending:
        if index in done or not line.strip():
            continue
        try:
            op = json.loads(line)
            if not isinstance(op, dict) or op.get("op") not in KINDS:
                raise ValueError(f"not a report visin knows: {line[:80]}")
        except ValueError as exc:
            # A torn last line from a process killed mid-write, or a file from
            # a newer visin. Neither will ever send, so it is not kept.
            result.rejected.append(f"line {index + 1} of {batch.name}: {exc}")
            _mark_done(batch, index)
            continue
        try:
            deliver(client, op, context)
        except VisinError as exc:
            if worth_retrying_later(exc):
                logger.info("visin: server unreachable, stopping sync: %s", exc)
                raise _Unreachable() from exc
            result.rejected.append(f"{op['op']}: {exc}")
            if drop_rejected:
                _mark_done(batch, index)
                discard_staged(op, context)
            else:
                kept = True
                result.kept_rejected += 1
            if op["op"] == "create_run" and isinstance(exc, ApiError):
                # Nothing after a refused run can land: every later report
                # names a run the server does not have.
                raise _Unreachable() from exc
            continue
        _mark_done(batch, index)
        discard_staged(op, context)
        result.sent += 1
    return kept


def pending_runs(directory: Path | str) -> list[str]:
    """Training UUIDs with reports waiting under ``directory``, oldest first."""
    root = Path(directory) / RUNS
    if not root.exists():
        return []
    seen: dict[str, float] = {}
    for path in root.iterdir():
        if path.name.endswith(".jsonl"):
            name = path.name[: -len(".jsonl")]
        elif path.name.endswith(".sending"):
            name = path.name.rsplit(".", 2)[0]
        else:
            continue
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue
        seen[name] = min(mtime, seen.get(name, mtime))
    return sorted(seen, key=lambda name: seen[name])
