"""What a run writes, as data, and how each kind of write is delivered.

Every write a run makes is first an *op*: a small JSON-able dict naming its
kind and carrying its request body. A live run delivers ops as it goes; an
offline run, or a live one that lost the server, writes them to disk and
``visin sync`` delivers them later through this same function. One code path
means an op replayed a week later lands exactly as it would have at the time.

Delivery is made safe to repeat wherever the API allows it. A run, an epoch and
a test result carry a caller-generated UUID, so a repeat is answered 409 and
counted as delivered. Configs, benchmarks and visualizations also carry client
identifiers, so a lost answer can be replayed safely.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..errors import ApiError, ConfigurationError
from .transport import HttpClient

logger = logging.getLogger("visin")

KINDS = frozenset(
    {"create_run", "epoch", "test_result", "benchmark", "config", "visualization", "update", "model"}
)


@dataclass
class DeliveryContext:
    """What delivering one run's ops needs to remember between ops."""

    files: Path | None = None
    training_ids: dict[str, str] = field(default_factory=dict)
    training_projects: dict[str, str] = field(default_factory=dict)
    # Attempts per request, when a caller cannot afford the client's full
    # budget: a run's blocking creation, or a catch-up that will be retried
    # in a minute anyway. None means the client's own.
    retries: int | None = None
    # Runs a create_run found already registered: resumed, not new.
    existing: set[str] = field(default_factory=set)
    # Epochs the server answered 409 for: recorded before this delivery.
    repeated: set[str] = field(default_factory=set)


def training_id(client: HttpClient, context: DeliveryContext, training_uuid: str) -> str:
    """The database id of a run, which ``PUT /trainings/:id`` needs."""
    known = context.training_ids.get(training_uuid)
    if known:
        return known
    found = client.request("GET", f"/trainings/uuid/{quote(training_uuid, safe='')}", retries=context.retries)
    ident = (found or {}).get("_id")
    if not ident:
        raise ApiError(f"no training with uuid {training_uuid}", status=404)
    if (found or {}).get("projectId"):
        context.training_projects[training_uuid] = str(found["projectId"])
    context.training_ids[training_uuid] = ident
    return str(ident)


def fetch_model_card(
    client: HttpClient, context: DeliveryContext, training_uuid: str, repo: str, epoch: int | None
) -> str | None:
    """The README Visin writes for a run's model, or ``None`` when it cannot be had.

    A card is a courtesy: an older server without the endpoint, a refusal or a
    dropped connection means the checkpoint is uploaded without one.
    """
    ident = training_id(client, context, training_uuid)
    params: dict[str, Any] = {"repo": repo}
    if epoch is not None:
        params["epoch"] = epoch
    card = client.request("GET", f"/trainings/{ident}/model-card", params=params, retries=context.retries)
    readme = (card or {}).get("readme")
    return readme if isinstance(readme, str) and readme else None


def _already_there(exc: ApiError) -> bool:
    return exc.status == 409


def deliver(client: HttpClient, op: dict[str, Any], context: DeliveryContext) -> Any:
    kind = op.get("op")
    body = op.get("body") or {}
    if kind == "create_run":
        return _create_run(client, body, context)
    if kind == "epoch":
        return _post_once(client, "/epochs/upload", body, context)
    if kind == "test_result":
        return _post_once(client, "/test-results/upload", body, context)
    if kind == "benchmark":
        return _post_once(client, "/benchmarks/upload", body, context)
    if kind == "config":
        return _config(client, op, body, context)
    if kind == "visualization":
        return _visualization(client, op, body, context)
    if kind == "model":
        ident = training_id(client, context, op["training_uuid"])
        return client.request(
            "POST", f"/trainings/{ident}/models", json=body, idempotent=True, retries=context.retries
        )
    if kind == "update":
        ident = training_id(client, context, op["training_uuid"])
        return client.request("PUT", f"/trainings/{ident}", json=body, retries=context.retries)
    raise ValueError(f"unknown report kind {kind!r}; was it written by a newer visin?")


def _post_once(client: HttpClient, path: str, body: dict[str, Any], context: DeliveryContext) -> Any:
    try:
        return client.request(
            "POST",
            path,
            json=body,
            idempotent=path != "/benchmarks/upload" or bool(body.get("benchmark_uuid")),
            retries=context.retries,
        )
    except ApiError as exc:
        # The server refuses a repeat of a UUID it already has. That is exactly
        # the outcome a retry should produce, so it counts as delivered.
        if _already_there(exc):
            logger.debug("visin: %s already recorded", path)
            if body.get("epoch_uuid") and path == "/epochs/upload":
                context.repeated.add(str(body["epoch_uuid"]))
            return None
        raise


def _create_run(client: HttpClient, body: dict[str, Any], context: DeliveryContext) -> Any:
    uuid = body["uuid"]
    try:
        training = client.request("POST", "/trainings", json=body, idempotent=True, retries=context.retries)
    except ApiError as exc:
        if not _already_there(exc):
            raise
        # Created by an earlier attempt, or by whatever launched this process.
        context.existing.add(uuid)
        training = client.request("GET", f"/trainings/uuid/{quote(uuid, safe='')}", retries=context.retries)
    ident = (training or {}).get("_id")
    if ident:
        context.training_ids[uuid] = str(ident)
    if (training or {}).get("projectId"):
        context.training_projects[uuid] = str(training["projectId"])
    return training


def _config(client: HttpClient, op: dict[str, Any], body: dict[str, Any], context: DeliveryContext) -> Any:
    training_uuid = op.get("training_uuid")
    # Use the run's actual project, including for resumed runs and offline replay.
    # A pipeline key may have put the run in a different project than requested.
    project = context.training_projects.get(training_uuid or "")
    if not project and training_uuid:
        training = (
            client.request("GET", f"/trainings/uuid/{quote(training_uuid, safe='')}", retries=context.retries)
            or {}
        )
        project = training.get("projectId")
        if project:
            context.training_projects[training_uuid] = str(project)
        if training.get("_id"):
            context.training_ids[training_uuid] = str(training["_id"])
    if project:
        body = {**body, "projectId": project}
    try:
        config = client.request(
            "POST",
            "/configs/upload",
            json=body,
            idempotent=bool(body.get("config_uuid")),
            retries=context.retries,
        )
    except ApiError as exc:
        if not _already_there(exc) or not body.get("config_uuid"):
            raise
        config = client.request(
            "GET", f"/configs/uuid/{quote(body['config_uuid'], safe='')}", retries=context.retries
        )
    config_id = (config or {}).get("_id")
    training_uuid = op.get("training_uuid")
    if config_id and training_uuid:
        # The config belongs to the project; linking it also shows it on this run.
        ident = training_id(client, context, training_uuid)
        client.request("PUT", f"/trainings/{ident}", json={"configId": config_id}, retries=context.retries)
    return config


def _visualization(
    client: HttpClient, op: dict[str, Any], body: dict[str, Any], context: DeliveryContext
) -> Any:
    path = staged_path(op, context)
    if not path.exists():
        raise ConfigurationError(f"the file for visualization {body.get('filename')!r} is gone: {path}")
    try:
        grant = client.request(
            "POST",
            "/visualizations/upload-url",
            json={
                **{key: body[key] for key in ("epoch_uuid", "filename", "type", "mimetype")},
                **(
                    {"visualization_uuid": body["visualization_uuid"]}
                    if body.get("visualization_uuid")
                    else {}
                ),
            },
            idempotent=bool(body.get("visualization_uuid")),
            retries=context.retries,
        )
    except ApiError as exc:
        if _already_there(exc) and body.get("visualization_uuid"):
            return None
        raise
    try:
        upload_url, visualization_uuid, file_id = (
            grant["uploadUrl"],
            grant["visualization_uuid"],
            grant["fileId"],
        )
    except (KeyError, TypeError) as exc:
        raise ApiError(f"upload-url answered without {exc}", status=None) from exc
    client.put_file(upload_url, str(path), body["mimetype"], retries=context.retries)
    record = {
        "epoch_uuid": body["epoch_uuid"],
        "visualization_uuid": visualization_uuid,
        "filename": body["filename"],
        "type": body["type"],
        "fileId": file_id,
        "mimetype": body["mimetype"],
        # The exact byte count: the server refuses a record whose size differs
        # from the file it received.
        "size": os.path.getsize(path),
    }
    if body.get("metadata") is not None:
        record["metadata"] = body["metadata"]
    return _post_once(client, "/visualizations", record, context)


def staged_path(op: dict[str, Any], context: DeliveryContext) -> Path:
    """Where the file an op uploads lives: its staged copy, or the original."""
    staged = op.get("staged")
    if staged and context.files is not None:
        return context.files / str(staged)
    return Path(str(op.get("path") or ""))


def discard_staged(op: dict[str, Any], context: DeliveryContext) -> None:
    """Remove an op's staged copy once nothing will need it again."""
    if op.get("op") != "visualization" or not op.get("staged") or context.files is None:
        return
    try:
        (context.files / op["staged"]).unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:  # pragma: no cover - a read-only or vanished directory
        logger.debug("visin: could not remove staged file: %s", exc)
