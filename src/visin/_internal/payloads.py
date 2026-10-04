"""The request bodies a run sends, built from what a caller passed.

Visin caps a run's name and description: the Zod schemas accept any string, but
the Mongoose model rejects an over-long one, so an untruncated name would fail
as a 400 halfway through a run rather than at the call that set it. A name
longer than the cap is often a config summary, the only description the run
has, so it is kept whole as the description when none was given.

A benchmark needs ``system_info`` to carry ``BENCHMARK_SYSTEM_FIELDS`` or the
API refuses it, so what a caller leaves out is filled in from this machine.

A benchmark that names an epoch by UUID does not also name its run: the server
finds the run through the epoch, and naming both can only disagree, which it
refuses as conflicting parents.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .. import system as _system
from .inputs import epoch_number, now
from .providers import HUB

logger = logging.getLogger("visin")

MAX_NOTES = 5000
MAX_NAME = 200
MAX_DESCRIPTION = 1000

BENCHMARK_SYSTEM_FIELDS = ("cpu_count", "cpu_count_logical", "memory_total_gb")


def tag_list(tags: Iterable[str] | str) -> list[str]:
    """``tags`` as a list; a lone string is one tag, not its characters."""
    return [tags] if isinstance(tags, str) else list(tags)


def capped_description(description: str) -> str:
    """``description`` cut to what Visin stores, saying so when it cuts."""
    if len(description) > MAX_DESCRIPTION:
        logger.warning("visin: description truncated to %d characters", MAX_DESCRIPTION)
    return description[:MAX_DESCRIPTION]


def run_payload(
    run_uuid: str,
    name: str,
    *,
    project: str | None = None,
    dataset: str | Mapping[str, Any] | None = None,
    model: str | None = None,
    config_id: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    description: str | None = None,
    tags: Iterable[str] | str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The body that registers a run. ``model`` travels under ``metadata``, and
    dataset strings fill ``datasetId`` and structured references fill ``dataset``."""
    combined: dict[str, Any] = dict(metadata or {})
    if model:
        combined["model"] = model
    if dataset:
        combined["dataset"] = dataset
    payload: dict[str, Any] = {
        "uuid": run_uuid,
        "name": name[:MAX_NAME],
        "status": "running",
        "startTime": now(),
    }
    if combined:
        payload["metadata"] = combined
    if dataset:
        if isinstance(dataset, Mapping):
            payload["dataset"] = dict(dataset)
        else:
            payload["datasetId"] = dataset
    if project:
        payload["projectId"] = project
    if config_id:
        payload["configId"] = config_id
    if provenance:
        payload["provenance"] = dict(provenance)
    if tags:
        payload["tags"] = tag_list(tags)
    if len(name) > MAX_NAME:
        logger.warning("visin: run name truncated to %d characters", MAX_NAME)
        description = description or name
    if description:
        payload["description"] = capped_description(description)
    return payload


def update_body(
    *,
    name: str | None = None,
    description: str | None = None,
    tags: Iterable[str] | str | None = None,
    metadata: Mapping[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """The body that changes a run: only what was given. A blank name is ignored; blank notes remove them."""
    body: dict[str, Any] = {}
    if name is not None and name.strip():
        body["name"] = name.strip()[:MAX_NAME]
    if description is not None:
        body["description"] = capped_description(description)
    if tags is not None:
        body["tags"] = tag_list(tags)
    if metadata is not None:
        body["metadata"] = dict(metadata)
    if notes is not None:
        if len(notes) > MAX_NOTES:
            logger.warning("visin: notes truncated to %d characters", MAX_NOTES)
        body["notes"] = notes[:MAX_NOTES]
    return body


def benchmark_payload(
    results: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    system_info: Mapping[str, Any] | None,
    *,
    training_uuid: str | None,
    epoch: int | float | None,
    epoch_uuid: str | None,
    name_epoch: Callable[[str, int | float], str],
    timestamp: Any,
) -> dict[str, Any]:
    """The body of a benchmark: one measurement or several, on this machine.

    ``name_epoch`` turns a run UUID and an epoch number into that epoch's UUID.
    """
    rows = [dict(results)] if isinstance(results, Mapping) else [dict(row) for row in results]
    info = dict(system_info or {})
    if any(field not in info for field in BENCHMARK_SYSTEM_FIELDS):
        info = {**_system.system_info(), **info}
    payload: dict[str, Any] = {"timestamp": timestamp or now(), "system_info": info, "results": rows}
    if training_uuid and not epoch_uuid:
        payload["training_uuid"] = training_uuid
    if epoch is not None:
        number = epoch_number(epoch)
        payload["epoch"] = number
        if epoch_uuid:
            payload["epoch_uuid"] = epoch_uuid
        elif training_uuid:
            payload["epoch_uuid"] = name_epoch(training_uuid, number)
    elif epoch_uuid:
        payload["epoch_uuid"] = epoch_uuid
    return payload


def model_payload(repo: str, revision: str, *, path: str | None, epoch: int | float | None) -> dict[str, Any]:
    """The body that links a Hub model to a run: a pointer pinned to one commit."""
    body: dict[str, Any] = {"provider": HUB, "kind": "model", "repo": repo, "revision": revision}
    if path:
        body["path"] = path
    if epoch is not None:
        body["epoch"] = int(epoch)
    return body
