"""Regression tests for dataset credentials and repeated reports."""

import threading

import pytest
from fakes import ok, refused

from visin import Datasets, Run
from visin._internal.reports import DeliveryContext, deliver
from visin._internal.sender import Sender
from visin.errors import ApiError


def test_pipeline_read_retries_only_its_specific_refusal(client, session):
    session.route(
        "GET",
        "/datasets",
        refused(403, "This key is limited to one project and cannot reach this API."),
        ok([]),
    )
    datasets = Datasets(client=client)
    assert datasets.list() == []
    assert len(session.calls) == 2
    assert datasets._anonymous_reads
    session.route("GET", "/datasets", refused(403, "Private dataset"))
    with pytest.raises(ApiError):
        datasets.list()
    assert len(session.calls) == 3


def test_repeated_benchmark_counts_as_delivered(client, session):
    session.route("POST", "/benchmarks/upload", refused(409))
    assert deliver(client, {"op": "benchmark", "body": {"benchmark_uuid": "b"}}, DeliveryContext()) is None


def test_repeated_config_is_still_linked_to_its_run(client, session):
    session.route("POST", "/configs/upload", refused(409))
    session.route("GET", "/configs/uuid/c", ok({"_id": "config-id"}))
    context = DeliveryContext(training_ids={"run": "run-id"}, training_projects={"run": "project"})
    deliver(client, {"op": "config", "training_uuid": "run", "body": {"config_uuid": "c"}}, context)
    assert session.bodies("/trainings/run-id", method="PUT") == [{"configId": "config-id"}]


def test_repeated_visualization_skips_the_file_upload(client, session, uploads, tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(b"png")
    session.route("POST", "/visualizations/upload-url", refused(409))
    body = {
        "visualization_uuid": "v",
        "epoch_uuid": "e",
        "filename": "frame.png",
        "type": "prediction",
        "mimetype": "image/png",
    }
    assert (
        deliver(client, {"op": "visualization", "body": body, "path": str(path)}, DeliveryContext()) is None
    )
    assert not uploads.puts


def test_an_idle_sender_heartbeats_and_stops():
    called = threading.Event()
    sender = Sender()
    sender.start_heartbeat(called.set, interval=0.01)
    assert called.wait(1)
    assert sender.stop()
    assert not sender._thread.is_alive()


def test_anonymous_fallback_removes_only_the_request_authorization(client, session, monkeypatch):
    captured = []
    request = session.request

    def record(*args, **kwargs):
        captured.append(kwargs.get("headers") or {})
        return request(*args, **kwargs)

    monkeypatch.setattr(session, "request", record)
    session.route(
        "GET",
        "/datasets",
        refused(403, "This key is limited to one project and cannot reach this API."),
        ok([]),
    )
    Datasets(client=client).list()
    assert "Authorization" not in captured[0]
    assert captured[1]["Authorization"] is None
    assert session.headers["Authorization"] == "Bearer token"


def test_current_liveness_is_sent_without_becoming_a_report(client, session):
    from visin._internal.delivery import Delivery

    delivery = Delivery(client, None, "run")
    delivery.context.training_ids["run"] = "run-id"
    delivery.heartbeat()
    assert session.paths() == ["/trainings/run-id/heartbeat"]
    delivery.spooling = True
    delivery.heartbeat()
    assert len(session.calls) == 2
    assert Delivery(None, None, None).heartbeat() is None


def test_the_run_records_a_structured_dataset_and_its_downloaded_version(server, session):
    reference = {
        "source": "visin",
        "id": "0123456789abcdef01234567",
        "name": "ZOD",
        "revision": "archive-version",
    }
    session.route("POST", "/trainings", ok({"_id": "run-id", "uuid": "run"}, 201))
    run = Run.create("Dataset run", project="project", dataset=reference)
    body = session.bodies("/trainings")[0]
    assert body["dataset"] == reference
    assert "datasetId" not in body
    run.finish()


def test_an_idle_run_recovers_its_spool_and_heartbeats_after_an_outage(server, session):
    import requests

    session.route("POST", "/trainings", ok({"_id": "run-id", "uuid": "run"}, 201))
    run = Run.create("Recovery", project="project", training_uuid="run")
    session.route(
        "POST",
        "/epochs/upload",
        *[requests.ConnectionError("connection lost") for _ in range(server.retries + 1)],
    )
    run.log_epoch(1, {"loss": 1})
    assert run.flush()
    assert run._delivery.spooling
    run._delivery._next_catch_up = 0
    run._delivery.heartbeat()
    assert not run._delivery.spooling
    assert session.paths().count("/trainings/run-id/heartbeat") == 1
    assert run._delivery.spool.count() == 0
    run.finish()


def test_heartbeat_registers_an_unreachable_run_before_reporting_liveness(server, session):
    import time

    import requests

    session.route(
        "POST", "/trainings", requests.ConnectionError("offline"), requests.ConnectionError("offline")
    )
    run = Run.create("Initially offline", project="project", training_uuid="run")
    assert run._delivery.spooling
    run._delivery._next_catch_up = time.monotonic() + 60
    attempts = len(session.calls)
    run._delivery.heartbeat()
    assert len(session.calls) == attempts
    session.route("POST", "/trainings", ok({"_id": "run-id", "uuid": "run"}, 201))
    run._delivery._next_catch_up = 0
    run._delivery.heartbeat()
    assert not run._delivery.spooling
    assert session.paths()[-1] == "/trainings/run-id/heartbeat"
    assert run._delivery.spool.count() == 0
    run.finish()
