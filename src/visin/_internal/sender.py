"""A background sender, so ``log_epoch`` costs the training loop nothing.

An epoch POST across a university network takes anywhere from 20ms to several
seconds, and it happens at exactly the moment the loop wants to start the next
epoch. Doing it inline makes reporting a tax on training time; worse, a
degraded Visin instance would slow the run down rather than merely losing
metrics.

So work is queued and a daemon thread drains it. The cost of that choice is
that failures surface late, which is why `flush` exists and why `Run.finish`
calls it.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from typing import Callable

logger = logging.getLogger("visin")


class _Barrier:
    """A marker the worker signals when it reaches this point in the queue."""

    __slots__ = ("event",)

    def __init__(self) -> None:
        self.event = threading.Event()


class Sender:
    """Serialises queued work onto one thread.

    One thread, not a pool: the writes are ordered (a run is created before its
    epochs, an epoch exists before its visualizations) and ordering is cheaper
    to guarantee than to recover from.
    """

    def __init__(self, max_queue: int = 10_000):
        self._queue: queue.Queue[Callable[[], None] | _Barrier | None] = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._drain, name="visin-sender", daemon=True)
        self._stopping = threading.Event()
        self._abandoned = threading.Event()
        self._started = False
        self._lock = threading.Lock()
        self.dropped = 0
        self.failed = 0
        # Work handed back by a stop that could not wait for it: never attempted.
        self.leftover: list[Callable[[], None]] = []

    def _ensure_started(self) -> None:
        with self._lock:
            if not self._started:
                self._thread.start()
                self._started = True

    def submit(self, work: Callable[[], None]) -> bool:
        """Queue one unit of work. Never blocks the caller.

        A full queue drops the item rather than blocking the training loop.
        Losing a metric is a bad outcome; stalling training because a metrics
        server is unreachable is a worse one, and the drop is counted so
        `finish` can report it. Returns whether the work was queued.
        """
        if self._stopping.is_set():
            return False
        self._ensure_started()
        try:
            self._queue.put_nowait(work)
        except queue.Full:
            self.dropped += 1
            logger.warning("visin: queue full, dropped a report (%d dropped so far)", self.dropped)
            return False
        return True

    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if isinstance(item, _Barrier):
                    item.event.set()
                    continue
                if self._abandoned.is_set():
                    self.leftover.append(item)
                    continue
                item()
            except Exception as exc:
                self.failed += 1
                logger.warning("visin: report failed: %s", exc)
            finally:
                self._queue.task_done()

    def flush(self, timeout: float = 30.0) -> bool:
        """Block until everything queued so far has been attempted.

        Implemented with a barrier rather than ``Queue.join`` because join has
        no timeout, and a hung flush at the end of a training run would turn a
        finished job into one the scheduler eventually kills.
        """
        if not self._started or not self._thread.is_alive():
            return True  # nothing queued will ever run, so there is nothing to wait for
        barrier = _Barrier()
        try:
            self._queue.put_nowait(barrier)
        except queue.Full:
            return False
        if not barrier.event.wait(timeout):
            logger.warning("visin: flush timed out after %.0fs; some reports may be unsent", timeout)
            return False
        return True

    def stop(self, timeout: float = 30.0) -> bool:
        """Flush, then retire the thread.

        When the flush times out, whatever is still queued is not attempted: it
        is moved to ``leftover``, so the caller can keep it for later rather
        than lose it with the process. The item in flight, if any, finishes on
        its own.
        """
        flushed = self.flush(timeout)
        self._stopping.set()
        if not self._started:
            return flushed
        if not flushed:
            self._abandoned.set()
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is not None and not isinstance(item, _Barrier):
                    self.leftover.append(item)
                self._queue.task_done()
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)
        self._thread.join(timeout=5.0 if flushed else 0.1)
        return flushed
