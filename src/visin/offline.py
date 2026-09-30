"""Runs that report without a server: offline mode, and sending later.

A run keeps its reports on disk when it is told to (``VISIN_MODE=offline``,
for a compute node with no route to Visin) or when it loses the server
mid-run. :func:`sync` sends them, and is what ``visin sync`` runs. The disk
format itself lives in ``visin._internal.spool``.
"""

from __future__ import annotations

from pathlib import Path

from ._internal.config import read_settings
from ._internal.spool import Spool, SyncResult, pending_runs, sync_spool
from ._internal.transport import HttpClient
from .errors import ConfigurationError

__all__ = ["SyncResult", "pending", "sync"]


def pending(directory: Path | str | None = None) -> dict[str, int]:
    """Runs with reports waiting on disk, and how many each, oldest run first."""
    settings = read_settings(directory=directory)
    return {uuid: Spool(settings.directory, uuid).count() for uuid in pending_runs(settings.directory)}


def sync(
    directory: Path | str | None = None,
    *,
    url: str | None = None,
    token: str | None = None,
    training_uuid: str | None = None,
    keep_rejected: bool = True,
) -> list[SyncResult]:
    """Send every run's reports kept on disk, or one run's with ``training_uuid``.

    What ``visin sync`` runs. Safe to repeat, and safe while an offline run is
    still writing: what it writes meanwhile waits for the next sync. Not safe to
    run twice at once over the same directory.
    """
    settings = read_settings(url=url, token=token, directory=directory)
    if not settings.configured:
        raise ConfigurationError("nowhere to send to: set VISIN_URL and VISIN_TOKEN, or pass url= and token=")
    assert settings.url
    client = HttpClient(settings.url, settings.token, verify=settings.verify_ssl)
    uuids = [training_uuid] if training_uuid else pending_runs(settings.directory)
    results = []
    try:
        for uuid in uuids:
            result = sync_spool(client, Spool(settings.directory, uuid), drop_rejected=not keep_rejected)
            results.append(result)
            if result.remaining and not result.rejected:
                break  # the server stopped answering; the rest would fail the same way
    finally:
        client.close()
    return results
