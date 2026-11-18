"""Inference sample reports use the same ordered offline spool as runs."""

import json

import pytest

import visin
from visin._internal.reports import DeliveryContext, deliver
from visin.errors import ApiError


def test_inference_spools_batches_and_progress(tmp_path):
    run = visin.init(
        "judge",
        mode="offline",
        directory=tmp_path,
        kind="inference",
        suite="judge@1",
        total=3,
        unit="frame",
        uuid="judge-run",
        provenance=False,
    )
    run.log_samples({"id": str(i), "attrs": {}, "fields": {"decision": "yes"}} for i in range(2001))
    run.log_progress(2, accuracy=0.5)
    run.finish()
    ops = [json.loads(line) for line in (tmp_path / "runs" / "judge-run.jsonl").read_text().splitlines()]
    assert [op["op"] for op in ops] == ["create_run", "samples", "samples", "update", "epoch", "update"]
    assert ops[0]["body"]["suite"] == {"slug": "judge", "version": 1}
    assert [len(op["body"]["samples"]) for op in ops[1:3]] == [2000, 1]
    assert ops[3]["body"]["progress"] == {"done": 2, "total": 3, "unit": "frame"}


def test_sample_conflict_is_not_treated_as_success():
    class Client:
        def request(self, *args, **kwargs):
            raise ApiError("conflict", status=409)

    context = DeliveryContext(training_ids={"run": "id"})
    with pytest.raises(ApiError, match="conflict"):
        deliver(Client(), {"op": "samples", "training_uuid": "run", "body": {"samples": []}}, context)


def test_human_checkpoint_identifies_reference():
    assert visin.human_checkpoint("annotator_1") == {"kind": "human", "label": "annotator_1"}
    with pytest.raises(ValueError):
        visin.human_checkpoint(" ")


def test_vocabulary_is_fetched_once_and_bad_rows_warn(server, session, caplog):
    from fakes import ok

    session.route(
        "GET",
        "/suites/judge/1",
        ok(
            {
                "protocol": {
                    "samples": {
                        "attrs": [{"key": "pixels"}],
                        "fields": [{"key": "decision", "values": ["keep", "delete"], "required": True}],
                        "weight": "pixels",
                    }
                }
            }
        ),
    )
    run = visin.init("judge", kind="inference", suite="judge@1", provenance=False)
    run.log_samples([{"id": "a", "attrs": {"pixels": -1}, "fields": {"decision": "typo"}}])
    run.log_samples([{"id": "b", "attrs": {"pixels": 2}, "fields": {"decision": "keep"}}])
    run.finish()
    assert "invalid field decision" in caplog.text
    assert "invalid weight pixels" in caplog.text
    assert len([call for call in session.calls if "/suites/judge/1" in call["url"]]) == 1


def test_rejected_samples_block_completion_during_sync(tmp_path, server, session, monkeypatch):
    from fakes import ok, refused

    monkeypatch.setenv("VISIN_DIR", str(tmp_path))
    run = visin.init(
        "judge", mode="offline", directory=tmp_path, kind="inference", suite="judge@1", provenance=False
    )
    run.log_samples([{"id": "a", "attrs": {}, "fields": {"decision": "keep"}}])
    run.finish()
    session.route("POST", "/trainings", ok({"_id": "t1"}))
    session.route("POST", "/trainings/t1/samples", refused(409, "different sample"))
    results = visin.sync(directory=tmp_path)
    assert results[0].rejected
    assert not session.bodies("/trainings/t1", "PUT")
    assert visin.pending(tmp_path)[run.training_uuid] == 2


def test_sample_evaluation_waits_after_run_completion(tmp_path, server, session, monkeypatch):
    from fakes import ok

    monkeypatch.setenv("VISIN_DIR", str(tmp_path))
    run = visin.init(
        "judge", mode="offline", directory=tmp_path, kind="inference", suite="judge@1", provenance=False
    )
    run.log_samples([{"id": "a", "attrs": {}, "fields": {"decision": "keep"}}])
    evaluation = visin.evaluate(
        suite="judge@1", run=run, checkpoint=visin.human_checkpoint("annotator_1"), uuid="eval"
    )
    assert evaluation.queued
    ops = [
        json.loads(line)
        for line in (tmp_path / "runs" / f"{run.training_uuid}.jsonl").read_text().splitlines()
    ]
    assert [op["op"] for op in ops] == ["create_run", "samples", "update", "evaluation"]
    session.route("POST", "/trainings", ok({"_id": "t1"}))
    results = visin.sync(directory=tmp_path)
    assert results[0].complete
    paths = [call["url"].split("/api")[-1] for call in session.calls]
    assert paths.index("/trainings/t1/samples") < paths.index("/trainings/t1") < paths.index("/evaluations")


def test_samples_push_sends_and_reports_conflict(tmp_path, server, session, capsys):
    from fakes import ok, refused

    from visin.cli import main

    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"id":"a","attrs":{},"fields":{"decision":"keep"}}\n')
    session.route("GET", "/trainings/uuid/", ok({"_id": "t1"}), ok({"_id": "t1"}))
    session.route("POST", "/trainings/t1/samples", ok(), refused(409, "different row"))
    args = ["samples", "push", str(rows), "--run", "judge", "--dir", str(tmp_path)]
    assert main(args) == 0
    assert "Sent 1 samples" in capsys.readouterr().out
    assert main(args) == 1
    assert "different row" in capsys.readouterr().err
    assert visin.pending(tmp_path)["judge"] == 1


def test_samples_push_offline_and_malformed_input(tmp_path, monkeypatch, capsys):
    from visin.cli import main

    monkeypatch.setenv("VISIN_MODE", "offline")
    rows = tmp_path / "rows.jsonl"
    rows.write_text('{"id":"a","attrs":{},"fields":{"decision":"keep"}}\n')
    args = ["samples", "push", str(rows), "--run", "judge", "--dir", str(tmp_path)]
    assert main(args) == 0
    assert visin.pending(tmp_path)["judge"] == 1
    rows.write_text("invalid json\n")
    assert main(args) == 1
    assert "line 1" in capsys.readouterr().err


def test_compare_judges_checks_suite_and_prints_table(server, session, capsys):
    from fakes import ok

    from visin.cli import main

    pair = {
        "suite": {"slug": "judge", "version": 1},
        "fields": [
            {
                "field": "decision",
                "groups": [
                    {"groupBy": "all", "count": 10, "agreement": 0.8, "kappa": 0.4, "divergence": 0.2}
                ],
            }
        ],
    }
    session.route("GET", "/evaluations/a/compare/b", ok(pair), ok(pair))
    assert main(["compare", "judge@1", "a", "b"]) == 0
    assert "0.400" in capsys.readouterr().out
    assert main(["compare", "other@1", "a", "b"]) == 1


def test_ollama_checkpoint_uses_exact_model_manifest(monkeypatch):
    from fakes import FakeResponse

    import visin.evaluation as evaluation_module

    class Response(FakeResponse):
        def raise_for_status(self):
            pass

    def get(url, timeout):
        assert url == "http://ollama.example:11434/api/tags"
        return Response(
            payload={
                "models": [
                    {"name": "qwen2.5vl:72b", "digest": "a" * 64, "details": {"quantization_level": "Q4_K_M"}}
                ]
            }
        )

    monkeypatch.setattr(evaluation_module.requests, "get", get)
    result = visin.ollama_checkpoint("qwen2.5vl:72b", host="ollama.example:11434")
    assert result == {"kind": "local", "sha256": "a" * 64, "label": "qwen2.5vl:72b (Q4_K_M)"}
    with pytest.raises(visin.ConfigurationError, match="not installed"):
        visin.ollama_checkpoint("qwen2.5vl:7b", host="ollama.example:11434")


def test_ollama_checkpoint_rejects_missing_identity(monkeypatch):
    import visin.evaluation as evaluation_module

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"models": [{"name": "model:latest", "digest": "bad", "details": {}}]}

    monkeypatch.setattr(evaluation_module.requests, "get", lambda *args, **kwargs: Response())
    with pytest.raises(visin.ConfigurationError, match="digest"):
        visin.ollama_checkpoint("model")


def test_resumed_inference_checks_suite_and_reopens_failed_run(server, session):
    from fakes import ok, refused

    existing = {
        "_id": "t1",
        "kind": "inference",
        "status": "failed",
        "suite": {"slug": "judge", "version": 1},
    }
    session.route("POST", "/trainings", refused(409, "exists"), refused(409, "exists"))
    session.route("GET", "/trainings/uuid/", ok(existing), ok({**existing, "status": "completed"}))
    run = visin.init("judge", uuid="same", kind="inference", suite="judge@1", provenance=False)
    assert session.bodies("/trainings/t1", "PUT") == [{"status": "running"}]
    run.finish(status="failed")
    with pytest.raises(ApiError, match="cannot be resumed"):
        visin.init("judge", uuid="same", kind="inference", suite="judge@1", provenance=False)


def test_offline_sync_combines_adjacent_samples_without_crossing_progress(tmp_path, server, session):
    from fakes import ok

    run = visin.init(
        "judge",
        uuid="batched",
        mode="offline",
        directory=tmp_path,
        kind="inference",
        suite="judge@1",
        total=3,
        unit="frame",
        provenance=False,
    )

    def row(name):
        return {"id": name, "attrs": {}, "fields": {"decision": "keep"}}

    run.log_samples([row("a")])
    run.log_samples([row("b")])
    run.log_progress(2)
    run.log_samples([row("c")])
    run.finish()
    session.route("POST", "/trainings", ok({"_id": "t1"}))
    result = visin.sync(directory=tmp_path)[0]
    assert result.complete
    batches = session.bodies("/trainings/t1/samples")
    assert [[row["id"] for row in batch["samples"]] for batch in batches] == [["a", "b"], ["c"]]
    assert result.sent == 6


def test_sync_never_combines_duplicate_ids_into_one_batch(tmp_path, server, session):
    from fakes import ok

    run = visin.init(
        "judge",
        uuid="repeat",
        mode="offline",
        directory=tmp_path,
        kind="inference",
        suite="judge@1",
        provenance=False,
    )
    row = {"id": "a", "attrs": {}, "fields": {"decision": "keep"}}
    run.log_samples([row])
    run.log_samples([row])
    run.finish()
    session.route("POST", "/trainings", ok({"_id": "t1"}))
    assert visin.sync(directory=tmp_path)[0].complete
    assert len(session.bodies("/trainings/t1/samples")) == 2


def test_sync_caps_combined_batches_and_retains_conflicts(tmp_path, server, session):
    from fakes import ok, refused

    run = visin.init(
        "judge",
        uuid="cap",
        mode="offline",
        directory=tmp_path,
        kind="inference",
        suite="judge@1",
        provenance=False,
    )
    for group in range(3):
        run.log_samples(
            {"id": f"{group}-{i}", "attrs": {}, "fields": {"decision": "keep"}} for i in range(1000)
        )
    run.finish()
    session.route("POST", "/trainings", ok({"_id": "t1"}))
    session.route("POST", "/trainings/t1/samples", ok(), refused(409, "different sample"))
    result = visin.sync(directory=tmp_path)[0]
    assert result.rejected
    assert [len(batch["samples"]) for batch in session.bodies("/trainings/t1/samples")] == [2000, 1000]
    assert visin.pending(tmp_path)["cap"] == 2
    assert not session.bodies("/trainings/t1", "PUT")


def test_live_samples_are_durable_and_batched_across_calls(tmp_path, server, session):
    from fakes import ok

    session.route("POST", "/trainings", ok({"_id": "t1"}))
    run = visin.init(
        "judge", uuid="live-batch", directory=tmp_path, kind="inference", suite="judge@1", provenance=False
    )
    for name in ("a", "b", "c"):
        run.log_samples([{"id": name, "attrs": {}, "fields": {"decision": "keep"}}])
    assert run.flush(timeout=5)
    batches = session.bodies("/trainings/t1/samples")
    assert [[row["id"] for row in batch["samples"]] for batch in batches] == [["a", "b", "c"]]
    assert visin.pending(tmp_path) == {}
    run.finish()


def test_live_samples_survive_a_sender_timeout(tmp_path, server, session, monkeypatch):
    import threading

    from fakes import ok

    session.route("POST", "/trainings", ok({"_id": "t1"}))
    run = visin.init(
        "judge", uuid="live-timeout", directory=tmp_path, kind="inference", suite="judge@1", provenance=False
    )
    gate = threading.Event()
    entered = threading.Event()

    def blocked():
        entered.set()
        gate.wait(5)

    monkeypatch.setattr(run._delivery, "_send_durable", blocked)
    run.log_samples([{"id": "a", "attrs": {}, "fields": {"decision": "keep"}}])
    assert entered.wait(2)
    run.finish(timeout=0)
    assert visin.pending(tmp_path)["live-timeout"] == 2
    gate.set()


def test_live_sample_conflict_retains_later_completion_until_retry(tmp_path, server, session):
    from fakes import ok, refused

    session.route("POST", "/trainings", ok({"_id": "t1"}))
    session.route("POST", "/trainings/t1/samples", *(refused(409, "conflict") for _ in range(10)))
    run = visin.init(
        "judge", uuid="live-conflict", directory=tmp_path, kind="inference", suite="judge@1", provenance=False
    )
    run.log_samples([{"id": "a", "attrs": {}, "fields": {"decision": "keep"}}])
    run.finish()
    assert visin.pending(tmp_path)["live-conflict"] == 2
    assert not session.bodies("/trainings/t1", "PUT")
    session.routes[("POST", "/trainings/t1/samples")] = []
    session.route("GET", "/trainings/uuid/", ok({"_id": "t1"}))
    assert visin.sync(directory=tmp_path)[0].complete
    assert session.bodies("/trainings/t1", "PUT")[0]["status"] == "completed"


def test_bad_sample_never_raises_or_completes_the_run(tmp_path):
    run = visin.init(
        "judge",
        uuid="bad-row",
        mode="offline",
        directory=tmp_path,
        kind="inference",
        suite="judge@1",
        strict=True,
        provenance=False,
    )
    run.log_samples([{"id": "bad", "attrs": {}, "fields": {}}])
    with pytest.raises(visin.ConfigurationError, match="sample reporting failed"):
        visin.evaluate(suite="judge@1", run=run, checkpoint=visin.human_checkpoint("annotator_1"))
    ops = [json.loads(line) for line in (tmp_path / "runs" / "bad-row.jsonl").read_text().splitlines()]
    assert ops[-1]["body"]["status"] == "failed"


def test_inference_rejects_uuid_that_cannot_be_spooled(server):
    with pytest.raises(visin.ConfigurationError, match="durable spool"):
        visin.init("judge", uuid="bad/name", kind="inference", suite="judge@1", provenance=False)
