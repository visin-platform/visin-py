"""How one run's ops reach Visin, or wait on disk until they can.

An online run queues each op for a sender thread. When Visin stops answering,
the op is kept in the run's spool and the run is *spooling*: every later op is
kept too, so order is preserved, and once a minute the spool is sent again.
Registering the run and a catch-up both block the sender, so they use a short
retry budget: both fall back to the spool, so a short budget loses nothing,
where the client's full one held ``init`` for close to a minute against a
server that drops packets rather than refusing.

An offline run has no client: its ops go straight to the spool.

A catch-up takes the same lock as ``visin sync`` and waits for the next round
when it is held, since two senders of one spool could deliver a benchmark or
config twice.

A catch-up drops what Visin refuses rather than keeping it, because a run
cannot stop to ask; ``visin sync`` keeps refusals until told otherwise.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any

from ..errors import VisinError
from .lock import SYNC_LOCK, exclusive
from .reports import DeliveryContext, deliver, discard_staged
from .sender import Sender
from .spool import Spool, sync_spool
from .transport import HttpClient, worth_retrying_later

logger = logging.getLogger("visin")

CATCH_UP_INTERVAL = 60.0
CREATE_RETRIES = 1
CATCH_UP_RETRIES = 0


class Delivery:
    """The client, spool and sender of one run, and the state that joins them."""

    def __init__(self, client: HttpClient | None, spool: Spool | None, training_uuid: str | None):
        self.client = client
        self.spool = spool
        self.training_uuid = training_uuid
        self.context = DeliveryContext(files=spool.files if spool else None)
        self.sender = Sender() if client else None
        self.sent = 0
        self.spooling = False
        self._next_catch_up = 0.0
        self._warned_repeat = False
        self._delivered_epochs: set[str] = set()

    @property
    def mode(self) -> str:
        """``online``, ``offline`` or ``disabled``."""
        if self.client is not None:
            return "online"
        if self.spool is not None:
            return "offline"
        return "disabled"

    def register(self, op: dict[str, Any]) -> None:
        """Deliver ``op`` now, on the caller's thread, with the short budget.

        Raises what the server refused. A run that could not be reached goes to
        the spool instead, and the run starts spooling.
        """
        assert self.client is not None
        self.context.retries = CREATE_RETRIES
        try:
            deliver(self.client, op, self.context)
        finally:
            self.context.retries = None

    def keep(self, op: dict[str, Any]) -> None:
        """Write ``op`` to the spool."""
        assert self.spool is not None
        self.spool.append(op)

    def start_spooling(self) -> None:
        """Keep every op from now on, and try the server again after a pause."""
        self.spooling = True
        self._next_catch_up = time.monotonic() + CATCH_UP_INTERVAL

    def emit(self, op: dict[str, Any]) -> bool:
        """Queue ``op`` for the sender, or write it to the spool when offline.

        Returns whether it was taken: ``False`` when the sender has stopped or its
        queue is full, so the op was discarded. Raises ``OSError`` when an offline
        run cannot write to disk.
        """
        if self.client is not None:
            assert self.sender is not None
            # A partial, not a lambda: a stop that cannot wait hands queued work
            # back, and its op has to be readable to be kept on disk.
            if not self.sender.submit(functools.partial(self._send, op)):
                discard_staged(op, self.context)
                return False
        elif self.spool is not None:
            self.spool.append(op)
        else:
            return False
        return True

    def _send(self, op: dict[str, Any]) -> None:
        """Deliver one op. Runs on the sender thread."""
        assert self.client is not None
        if self.spooling and not self.catch_up():
            self.keep(op)
            return
        try:
            deliver(self.client, op, self.context)
        except VisinError as exc:
            if self.spool is not None and worth_retrying_later(exc):
                if not self.spooling:
                    logger.warning(
                        "visin: lost Visin (%s); keeping reports in %s until it answers",
                        exc,
                        self.spool.path,
                    )
                self.keep(op)
                self.start_spooling()
                return
            discard_staged(op, self.context)
            raise
        discard_staged(op, self.context)
        self.sent += 1
        if op.get("op") == "epoch":
            self._note_epoch(str(op["body"].get("epoch_uuid")), op["body"].get("epoch"))

    def _note_epoch(self, epoch_uuid: str, epoch: Any) -> None:
        """Say so, once, when an epoch was recorded before and the new copy dropped."""
        if epoch_uuid in self.context.repeated:
            self.context.repeated.discard(epoch_uuid)
            if epoch_uuid not in self._delivered_epochs and not self._warned_repeat:
                self._warned_repeat = True
                logger.warning(
                    "visin: epoch %s of run %s was already recorded, by an earlier attempt or an earlier "
                    "run of this script, and Visin keeps the first copy. A job that restarts its epoch "
                    "numbering should start a new run.",
                    epoch,
                    self.training_uuid,
                )
        self._delivered_epochs.add(epoch_uuid)

    def catch_up(self, force: bool = False) -> bool:
        """Send what was kept while Visin was away. True once nothing is left."""
        assert self.client is not None
        assert self.spool is not None
        if not force and time.monotonic() < self._next_catch_up:
            return False
        self.context.retries = CATCH_UP_RETRIES
        try:
            with exclusive(self.spool.root / SYNC_LOCK) as free:
                if not free:
                    self._next_catch_up = time.monotonic() + CATCH_UP_INTERVAL
                    return False
                result = sync_spool(self.client, self.spool, drop_rejected=True, context=self.context)
        finally:
            self.context.retries = None
        self.sent += result.sent
        for message in result.rejected:
            logger.warning("visin: Visin refused a kept report: %s", message)
        if result.remaining == 0:
            if self.spooling:
                logger.info("visin: Visin answers again; sent %d kept reports", result.sent)
            self.spooling = False
            return True
        self._next_catch_up = time.monotonic() + CATCH_UP_INTERVAL
        return False

    def flush(self, timeout: float) -> bool:
        """Wait until every op queued so far has been attempted."""
        return self.sender.flush(timeout) if self.sender else True

    def stop(self, timeout: float) -> bool:
        """Send what is left, keep on disk what could not be sent, and try the spool once more.

        Returns whether everything queued was attempted in time.
        """
        if self.sender is None:
            return True
        flushed = self.sender.stop(timeout)
        self._keep_leftovers()
        if flushed and self.spooling:
            try:
                self.catch_up(force=True)
            except Exception as exc:
                logger.warning("visin: could not send kept reports: %s", exc)
        return flushed

    def _keep_leftovers(self) -> None:
        assert self.sender is not None
        leftover = [
            item.args[0] for item in self.sender.leftover if isinstance(item, functools.partial) and item.args
        ]
        self.sender.leftover = []
        if not leftover:
            return
        if self.spool is None:
            self.sender.dropped += len(leftover)
            return
        for op in leftover:
            try:
                self.spool.append(op)
            except OSError as exc:
                self.sender.dropped += 1
                logger.warning("visin: could not keep a report on disk: %s", exc)

    def close(self) -> None:
        """Release the connection."""
        if self.client is not None:
            self.client.close()

    def summarise(self, status: str, flushed: bool) -> None:
        """Log how the run ended: all sent, or what is waiting, failed or dropped."""
        waiting = self.spool.count() if self.spool else 0
        if self.mode == "offline":
            logger.warning(
                "visin: run %s %s offline; %d reports kept in %s. Send them with `visin sync`.",
                self.training_uuid,
                status,
                waiting,
                self.spool.path if self.spool else "-",
            )
            return
        failed = self.sender.failed if self.sender else 0
        dropped = self.sender.dropped if self.sender else 0
        if waiting or failed or dropped or not flushed:
            logger.warning(
                "visin: run %s %s: %d reports sent, %d failed, %d dropped, %d waiting on disk%s",
                self.training_uuid,
                status,
                self.sent,
                failed,
                dropped,
                waiting,
                "; send them with `visin sync`" if waiting else "",
            )
        else:
            logger.info("visin: run %s %s, %d reports sent", self.training_uuid, status, self.sent)
