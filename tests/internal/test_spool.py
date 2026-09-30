import json

import pytest
from fakes import FakeResponse, ok, refused

from visin._internal.reports import DeliveryContext
from visin._internal.spool import Spool, pending_runs, sync_spool
from visin.errors import ConfigurationError, TransportError


def epoch(n, uuid="run-1"):
    return {
        "op": "epoch",
        "body": {"training_uuid": uuid, "epoch_uuid": f"e{n}", "epoch": n, "results": {"n": n}},
    }


def benchmark(n):
    return {"op": "benchmark", "body": {"n": n}}


@pytest.fixture
def spool(tmp_path):
    return Spool(tmp_path, "run-1")


def delivered(session, fragment):
    return [body.get("epoch", body) for body in session.bodies(fragment)]


# ---------------------------------------------------------------- the file


def test_reports_are_appended_one_json_line_each(spool):
    spool.append(epoch(1))
    spool.append(epoch(2))
    lines = spool.path.read_text().splitlines()
    assert [json.loads(line)["body"]["epoch"] for line in lines] == [1, 2]
    assert spool.count() == 2
    assert spool.pending()


def test_a_run_name_that_is_not_a_safe_file_name_is_refused(tmp_path):
    with pytest.raises(ConfigurationError):
        Spool(tmp_path, "../escape")


def test_stage_copies_the_file_so_the_original_may_change(spool, tmp_path):
    source = tmp_path / "frame.png"
    source.write_bytes(b"one")
    name = spool.stage(source)
    source.write_bytes(b"two")
    assert (spool.files / name).read_bytes() == b"one"
    assert name.endswith("-frame.png")


def test_pending_runs_lists_runs_oldest_first(tmp_path):
    Spool(tmp_path, "a").append(epoch(1, "a"))
    Spool(tmp_path, "b").append(epoch(1, "b"))
    assert pending_runs(tmp_path) == ["a", "b"]
    assert pending_runs(tmp_path / "nowhere") == []


# ---------------------------------------------------------------- syncing


def test_sync_sends_everything_in_order_and_cleans_up(client, session, spool):
    for n in range(3):
        spool.append(epoch(n))
    result = sync_spool(client, spool)
    assert result.sent == 3 and result.complete
    assert delivered(session, "/epochs/upload") == [0, 1, 2]
    assert not spool.pending()
    assert list(spool.root.iterdir()) == []


def test_sync_stops_when_the_server_goes_away_and_resumes_without_repeats(client, session, spool):
    client.retries = 0
    spool.append(benchmark(1))
    spool.append(benchmark(2))
    spool.append(benchmark(3))
    session.route("POST", "/benchmarks/upload", ok(), refused(503, "down"))
    first = sync_spool(client, spool)
    assert (first.sent, first.remaining) == (1, 2)
    second = sync_spool(client, spool)
    assert (second.sent, second.remaining) == (2, 0)
    # Benchmarks cannot be recognised as repeats, so each must be sent once.
    assert [body["n"] for body in session.bodies("/benchmarks/upload")] == [1, 2, 2, 3]
    assert len(session.bodies("/benchmarks/upload")) == 4  # one refused attempt, then 1, 2, 3 once each


def test_what_a_run_writes_during_a_sync_is_kept_for_the_next(client, session, spool):
    spool.append(epoch(1))
    batch = spool.claim()  # a sync has taken the file...
    spool.append(epoch(2))  # ...and the run carries on writing
    assert batch is not None and spool.path.exists()
    result = sync_spool(client, spool)
    assert result.sent == 2
    assert delivered(session, "/epochs/upload") == [1, 2]


def test_a_torn_last_line_is_dropped_and_the_rest_sent(client, session, spool):
    spool.append(epoch(1))
    with open(spool.path, "a") as handle:
        handle.write('{"op": "epoch", "bo')  # the process was killed mid-write
    result = sync_spool(client, spool)
    assert result.sent == 1
    assert len(result.rejected) == 1
    assert not spool.pending()


def test_a_refused_report_is_kept_to_retry_unless_asked_to_drop_it(client, session, spool):
    spool.append(epoch(1))
    spool.append(epoch(2))
    session.route("POST", "/epochs/upload", refused(400, "Results are required"))
    kept = sync_spool(client, spool)
    assert kept.sent == 1 and len(kept.rejected) == 1 and kept.remaining == 0
    assert spool.pending()
    session.route("POST", "/epochs/upload", refused(400, "Results are required"))
    dropped = sync_spool(client, spool, drop_rejected=True)
    assert len(dropped.rejected) == 1
    assert not spool.pending()


def test_a_refused_run_stops_its_sync(client, session, spool):
    spool.append({"op": "create_run", "body": {"uuid": "run-1", "name": "x"}})
    spool.append(epoch(1))
    session.route("POST", "/trainings", refused(403, "Access denied to project"))
    result = sync_spool(client, spool)
    assert result.sent == 0
    assert "Access denied" in result.rejected[0]
    assert session.bodies("/epochs/upload") == []


def test_a_transport_failure_leaves_everything_waiting(client, session, spool):
    client.retries = 0
    spool.append(epoch(1))
    session.route("POST", "/epochs/upload", TransportError("no route"))
    result = sync_spool(client, spool)
    assert (result.sent, result.remaining, result.rejected) == (0, 1, [])


def test_a_staged_file_is_removed_once_delivered(client, session, uploads, spool, tmp_path):
    source = tmp_path / "frame.png"
    source.write_bytes(b"png")
    staged = spool.stage(source)
    spool.append(
        {
            "op": "visualization",
            "staged": staged,
            "body": {"epoch_uuid": "e1", "filename": "frame.png", "type": "overlay", "mimetype": "image/png"},
        }
    )
    session.route(
        "POST", "/visualizations/upload-url", ok({"uploadUrl": "u", "visualization_uuid": "v", "fileId": "f"})
    )
    result = sync_spool(client, spool, context=DeliveryContext())
    assert result.complete
    assert uploads.puts[0]["bytes"] == b"png"
    assert not spool.files.exists()


def test_an_unknown_kind_is_rejected_not_sent(client, session, spool):
    spool.append({"op": "from-the-future", "body": {}})
    result = sync_spool(client, spool)
    assert result.rejected and session.calls == []


def test_the_server_answering_500_is_a_refusal_not_an_outage(client, session, spool):
    session.route("POST", "/epochs/upload", FakeResponse(500, None, text="boom"))
    spool.append(epoch(1))
    result = sync_spool(client, spool, drop_rejected=True)
    assert result.rejected and result.remaining == 0


# ---------------------------------------------------------------- damage and races


def test_a_report_after_a_torn_line_survives(client, session, spool):
    spool.append(epoch(1))
    with open(spool.path, "a") as handle:
        handle.write('{"op": "epoch", "bo')  # killed mid-write...
    spool.append(epoch(2))  # ...and the run, restarted, carries on
    result = sync_spool(client, spool)
    assert delivered(session, "/epochs/upload") == [1, 2]
    assert len(result.rejected) == 1


def test_a_run_does_not_take_the_batches_of_a_run_whose_name_extends_it(tmp_path):
    short, dotted = Spool(tmp_path, "run"), Spool(tmp_path, "run.b")
    dotted.append(epoch(1, "run.b"))
    dotted.claim()
    assert short.batches() == []
    assert len(dotted.batches()) == 1


def test_a_file_that_cannot_be_claimed_waits_for_the_next_sync(spool, monkeypatch):
    spool.append(epoch(1))

    def busy(*_args):
        raise PermissionError("in use by another process")

    monkeypatch.setattr("visin._internal.spool.os.replace", busy)
    assert spool.claim() is None
    assert spool.path.exists()


def test_a_line_that_lands_in_a_batch_during_its_sync_is_sent_too(client, session, spool, monkeypatch):
    import visin._internal.spool as module

    spool.append(epoch(1))
    real_deliver = module.deliver
    late = []

    def deliver_and_race(client_, op, context):
        if not late:
            # A writer that opened the file just before the claim finishes its line now.
            (batch,) = spool.batches()
            with open(batch, "a") as handle:
                handle.write(json.dumps(epoch(2)) + "\n")
            late.append(True)
        return real_deliver(client_, op, context)

    monkeypatch.setattr(module, "deliver", deliver_and_race)
    result = sync_spool(client, spool)
    assert result.sent == 2
    assert delivered(session, "/epochs/upload") == [1, 2]
    assert not spool.pending()
