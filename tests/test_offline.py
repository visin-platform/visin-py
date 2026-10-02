"""Offline mode end to end: a run with no route to Visin, synced later."""

import json

import pytest
from fakes import ok, refused

import visin
from visin.errors import ConfigurationError


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIN_MODE", "offline")
    monkeypatch.setenv("VISIN_DIR", str(tmp_path))
    return tmp_path


def test_an_offline_run_writes_everything_to_disk(offline, session):
    frame = offline / "overlay.png"
    frame.write_bytes(b"png")
    with visin.init("compute node run", project="road-seg") as run:
        assert run.mode == "offline"
        run.log_epoch(1, train={"loss": 1.0})
        run.upload_visualization(1, frame, "overlay")
        run.log_test_results(1, {"overall": {"iou": 0.5}})
    lines = (offline / "runs" / f"{run.training_uuid}.jsonl").read_text().splitlines()
    kinds = [json.loads(line)["op"] for line in lines]
    assert kinds == ["create_run", "epoch", "visualization", "test_result", "update"]
    assert session.calls == []
    assert visin.pending(offline) == {run.training_uuid: 5}


def test_sync_sends_an_offline_run_in_order(offline, server, session, uploads, monkeypatch):
    with visin.init("compute node run") as run:
        run.log_epoch(1, train={"loss": 1.0})
        run.log_epoch(2, train={"loss": 0.5})
    session.route("GET", "/trainings/uuid/", ok({"_id": "t1"}))
    results = visin.sync()
    assert [r.training_uuid for r in results] == [run.training_uuid]
    assert results[0].sent == 4 and results[0].complete
    assert session.paths("POST") == ["/trainings", "/epochs/upload", "/epochs/upload"]
    assert session.bodies("/trainings/t1", "PUT")[0]["status"] == "completed"
    assert visin.pending(offline) == {}
    assert visin.sync() == []


def test_sync_stops_at_the_first_run_the_server_did_not_answer(offline, server, session):
    server.retries = 0
    first = visin.init("first")
    first.finish()
    second = visin.init("second")
    second.finish()
    session.default = refused(503, "down")
    results = visin.sync()
    assert len(results) == 1 and results[0].remaining == 2


def test_sync_one_run(offline, server, session):
    session.route("GET", "/trainings/uuid/", ok({"_id": "t1"}))
    wanted = visin.init("wanted")
    wanted.finish()
    other = visin.init("other")
    other.finish()
    results = visin.sync(training_uuid=wanted.training_uuid)
    assert [r.training_uuid for r in results] == [wanted.training_uuid]
    assert list(visin.pending(offline)) == [other.training_uuid]


def test_sync_needs_somewhere_to_send(offline):
    with pytest.raises(ConfigurationError, match=r"VISIN_URL.*vision-api\.visin\.eu"):
        visin.sync()


def test_attach_offline_records_no_creation(offline):
    run = visin.run.Run.attach("existing-run")
    run.log_epoch(1, {"loss": 1})
    run.finish()
    kinds = [
        json.loads(line)["op"] for line in (offline / "runs" / "existing-run.jsonl").read_text().splitlines()
    ]
    assert kinds == ["epoch", "update"]


def test_a_second_sync_over_the_same_directory_is_refused_while_one_runs(offline, server, session):
    from visin._internal.lock import exclusive

    session.route("GET", "/trainings/uuid/", ok({"_id": "t1"}))
    visin.init("waiting").finish()
    directory = visin.read_settings().directory
    with exclusive(directory / "runs" / ".sync.lock") as first:
        assert first
        with pytest.raises(visin.SyncInProgressError, match="already sending"):
            visin.sync()
    assert visin.sync()[0].sent == 2


def test_a_missing_file_while_listing_runs_is_not_an_error(tmp_path):
    from visin._internal.spool import pending_runs

    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "a.jsonl").write_text("{}\n")
    assert pending_runs(tmp_path) == ["a"]
