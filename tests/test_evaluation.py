"""Recording how a checkpoint scored on a suite: what is sent, what is kept, what is refused."""

import hashlib
import json
import logging

import pytest
import requests
from fakes import ok, refused

import visin
from visin.errors import ApiError, ConfigurationError, TransportError

COMMIT = "3f2a1c9d8e7b6a5f4e3d2c1b0a99887766554433"
RESULTS = {"day": {"overall": {"mIoU": 0.8}}, "night": {"overall": {"mIoU": 0.6}}}
COUNTS = {"day": 1200, "night": 800}
ELIGIBLE = {"state": "eligible", "reasons": [], "warnings": [], "scores": {"overall": {"mIoU": 0.7}}}


def stored(**overrides):
    return {
        "_id": "e1",
        "uuid": "u1",
        "projectId": "p1",
        "checkpoint": {"kind": "local", "sha256": "a" * 64, "label": "x"},
        "checkpointKey": "sha256:" + "a" * 64,
        "suite": {"slug": "road-test", "version": 1, "digest": "d" * 64},
        "status": "completed",
        "receivedAt": "2026-10-01T09:00:00.000Z",
        "validation": ELIGIBLE,
        **overrides,
    }


@pytest.fixture
def weights(tmp_path):
    path = tmp_path / "epoch_40.pth"
    path.write_bytes(b"weights")
    return path


def record(weights, **kwargs):
    options = {"suite": "road-test@1", "checkpoint": visin.local_checkpoint(weights), "sample_counts": COUNTS}
    options.update(kwargs)
    return visin.evaluate(RESULTS, **options)


class TestCheckpoints:
    def test_a_local_checkpoint_is_the_digest_of_the_weights_and_a_label_never_the_path(self, weights):
        checkpoint = visin.local_checkpoint(weights)
        assert checkpoint == {
            "kind": "local",
            "sha256": hashlib.sha256(b"weights").hexdigest(),
            "label": "epoch_40",
        }
        assert str(weights.parent) not in json.dumps(checkpoint)
        assert visin.local_checkpoint(weights, "clftv2-epoch-40")["label"] == "clftv2-epoch-40"

    def test_two_files_with_the_same_bytes_are_the_same_checkpoint(self, weights, tmp_path):
        copy = tmp_path / "renamed.bin"
        copy.write_bytes(b"weights")
        assert visin.local_checkpoint(copy)["sha256"] == visin.local_checkpoint(weights)["sha256"]

    def test_a_large_file_is_hashed_in_blocks(self, tmp_path, monkeypatch):
        monkeypatch.setattr("visin.evaluation._BLOCK", 4)
        path = tmp_path / "big.bin"
        path.write_bytes(b"0123456789")
        assert visin.sha256_of(path) == hashlib.sha256(b"0123456789").hexdigest()

    def test_a_hub_checkpoint_is_a_lowercase_commit_and_a_path_only_when_given(self):
        assert visin.hub_checkpoint("acme/clft", COMMIT.upper()) == {
            "kind": "hf",
            "repo": "acme/clft",
            "commit": COMMIT,
        }
        assert visin.hub_checkpoint("acme/clft", COMMIT, "best.safetensors")["path"] == "best.safetensors"


class TestRecording:
    def test_sends_the_result_with_the_checkpoint_the_suite_and_the_counts_and_returns_the_verdict(
        self, server, session, weights
    ):
        session.route("POST", "/evaluations", ok(stored(), 201))
        evaluation = record(weights, project="road-seg", uuid="u1")
        body = session.bodies("/evaluations")[0]
        assert body["projectId"] == "road-seg" and body["uuid"] == "u1" and body["suite"] == "road-test@1"
        assert body["checkpoint"]["kind"] == "local" and body["sampleCounts"] == COUNTS
        assert body["results"] == RESULTS and body["status"] == "completed"
        assert (evaluation.id, evaluation.suite, evaluation.ranked, evaluation.stored) == (
            "e1",
            "road-test@1",
            True,
            True,
        )
        assert evaluation.verdict.scores == {"overall": {"mIoU": 0.7}}
        assert evaluation.sample_counts == {} and not evaluation.queued and not evaluation.published

    def test_a_result_that_cannot_be_ranked_is_returned_with_its_reasons_and_warned_about(
        self, server, session, weights, caplog
    ):
        verdict = {
            "state": "incomplete",
            "reasons": [{"code": "missing-condition", "detail": "night"}],
            "warnings": [{"code": "submitted-overall-differs", "detail": "mIoU"}],
        }
        session.route("POST", "/evaluations", ok(stored(validation=verdict), 201))
        with caplog.at_level(logging.INFO, logger="visin"):
            evaluation = record(weights, project="road-seg")
        assert not evaluation.ranked and evaluation.verdict.state == "incomplete"
        assert [str(reason) for reason in evaluation.verdict.reasons] == ["missing-condition(night)"]
        assert [str(reason) for reason in evaluation.verdict.warnings] == ["submitted-overall-differs(mIoU)"]
        assert str(evaluation.verdict) == "incomplete: missing-condition(night)"
        assert "not ranked on road-test@1: incomplete: missing-condition(night)" in caplog.text

    def test_a_ranked_result_says_so_in_the_log(self, server, session, weights, caplog):
        session.route("POST", "/evaluations", ok(stored(), 201))
        with caplog.at_level(logging.INFO, logger="visin"):
            record(weights, project="road-seg")
        assert "evaluation recorded and ranked on road-test@1" in caplog.text

    def test_a_uuid_is_generated_and_kept_so_a_retry_is_the_same_request(self, server, session, weights):
        session.route("POST", "/evaluations", requests.exceptions.ReadTimeout("dropped"), ok(stored(), 201))
        record(weights, project="road-seg")
        first, second = session.bodies("/evaluations")
        assert first["uuid"] == second["uuid"] and first == second

    def test_the_run_the_checkpoint_came_from_is_provenance_only(self, server, session, weights):
        session.route("POST", "/evaluations", ok(stored(), 201))
        record(weights, project="road-seg", run="run-uuid", epoch=40, epoch_uuid="e-40")
        assert session.bodies("/evaluations")[0]["source"] == {
            "trainingUuid": "run-uuid",
            "epochUuid": "e-40",
            "epoch": 40,
        }

    def test_a_run_object_names_its_uuid_and_supplies_its_project(self, server, session, weights):
        session.route("POST", "/evaluations", ok(stored(), 201))
        run = visin.Training(id="t1", uuid="run-uuid", name="r", project_id="p9")
        record(weights, run=run)
        body = session.bodies("/evaluations")[0]
        assert body["source"] == {"trainingUuid": "run-uuid"}
        assert body["projectId"] == "p9"

    def test_the_evaluator_and_other_provenance_are_sent_and_what_was_given_wins(
        self, server, session, weights
    ):
        session.route("POST", "/evaluations", ok(stored(), 201))
        record(
            weights,
            project="road-seg",
            evaluator={"package": "visin-fusion", "version": "1.4.2", "commit": "9d1c2ab"},
            provenance={"seeds": [42], "host": "given"},
        )
        provenance = session.bodies("/evaluations")[0]["provenance"]
        assert provenance["evaluator"] == {"package": "visin-fusion", "version": "1.4.2", "commit": "9d1c2ab"}
        assert provenance["seeds"] == [42] and provenance["host"] == "given"

    def test_the_machine_and_code_are_collected_like_a_runs_unless_that_is_switched_off(
        self, server, session, weights, monkeypatch
    ):
        session.route("POST", "/evaluations", ok(stored(), 201))
        record(weights, project="road-seg")
        assert "provenance" not in session.bodies("/evaluations")[0]
        monkeypatch.setenv("VISIN_PROVENANCE", "1")
        record(weights, project="road-seg")
        assert "host" in session.bodies("/evaluations")[1]["provenance"]

    @pytest.mark.parametrize(
        ("name", "refused"),
        [
            ("tokenizers", False),
            ("torch", False),
            ("keyboard", False),
            ("auth-token-helper", True),
            ("HF_TOKEN", True),
            ("apiKey", True),
            ("python-dotenv", False),
        ],
    )
    def test_a_package_that_looks_like_a_credential_is_left_out_so_visin_does_not_refuse_the_evaluation(
        self, server, session, weights, monkeypatch, name, refused
    ):
        session.route("POST", "/evaluations", ok(stored(), 201))
        monkeypatch.setenv("VISIN_PROVENANCE", "1")
        monkeypatch.setattr("visin._internal.provenance.packages", lambda: {name: "1.0", "numpy": "2.0"})
        record(weights, project="road-seg")
        packages = session.bodies("/evaluations")[0]["provenance"]["packages"]
        assert ("numpy" in packages) and ((name in packages) is not refused)

    def test_numpy_and_non_finite_numbers_in_results_are_made_json(self, server, session, weights):
        session.route("POST", "/evaluations", ok(stored(), 201))
        visin.evaluate(
            {"day": {"overall": {"mIoU": float("nan")}}},
            suite="road-test@1",
            checkpoint=visin.local_checkpoint(weights),
            project="road-seg",
        )
        assert session.bodies("/evaluations")[0]["results"] == {"day": {"overall": {"mIoU": None}}}

    def test_optional_fields_are_sent_only_when_given(self, server, session, weights):
        session.route("POST", "/evaluations", ok(stored(), 201))
        visin.evaluate(RESULTS, suite=None, checkpoint=None, project="road-seg")
        body = session.bodies("/evaluations")[0]
        assert set(body) == {"status", "results", "uuid", "projectId"}
        record(
            weights,
            project="road-seg",
            protocol_digest="f" * 64,
            executed_at="2026-09-30T08:00:00Z",
            supersedes=visin.Evaluation(id="old"),
            status="failed",
        )
        body = session.bodies("/evaluations")[1]
        assert body["evidence"] == {"protocolDigest": "f" * 64}
        assert "suiteDigest" not in body and body["executedAt"] == "2026-09-30T08:00:00Z"
        assert body["supersedesId"] == "old" and body["status"] == "failed"

    def test_a_datetime_is_sent_as_iso_text(self, server, session, weights):
        import datetime

        session.route("POST", "/evaluations", ok(stored(), 201))
        record(weights, project="road-seg", executed_at=datetime.datetime(2026, 9, 30, 8, 0))
        assert session.bodies("/evaluations")[0]["executedAt"] == "2026-09-30T08:00:00"

    def test_an_evaluation_with_no_id_cannot_be_superseded(self, server, weights):
        with pytest.raises(ValueError, match="no id"):
            record(weights, project="road-seg", supersedes=visin.Evaluation(uuid="waiting"))


PROTOCOL = {
    "task": "seg",
    "data": {"kind": "external", "label": "x", "manifestSha256": "a" * 64},
    "split": "test",
    "conditions": [{"name": "day", "sampleCount": 1200}],
    "metrics": [{"key": "mIoU", "direction": "max", "headline": True}],
    "aggregation": "pooled",
    "evaluator": {"package": "visin-fusion", "minVersion": "1.0.0"},
}
DATA = {"kind": "external", "manifestSha256": "a" * 64}


class TestEvidence:
    def test_what_the_evaluator_ran_is_sent_as_evidence(self, server, session, weights):
        session.route("POST", "/evaluations", ok(stored(), 201))
        record(
            weights,
            project="road-seg",
            data=DATA,
            protocol_digest="d" * 64,
            classes={"scored": ["car", "tree"], "ignored": ["void"]},
            evaluator={"package": "visin-fusion", "version": "1.4.2", "commit": "9d1c2ab"},
        )
        body = session.bodies("/evaluations")[0]
        assert body["evidence"] == {
            "data": DATA,
            "protocolDigest": "d" * 64,
            "classes": {"scored": ["car", "tree"], "ignored": ["void"]},
            "evaluator": {"package": "visin-fusion", "version": "1.4.2"},
        }
        assert body["provenance"]["evaluator"]["commit"] == "9d1c2ab"

    def test_the_named_arguments_are_merged_over_a_whole_evidence_object(self, server, session, weights):
        session.route("POST", "/evaluations", ok(stored(), 201))
        record(
            weights,
            project="road-seg",
            evidence={
                "data": {"kind": "external", "manifestSha256": "b" * 64},
                "evaluator": {"package": "p", "version": "1"},
            },
            data=DATA,
            evaluator={"package": "ignored", "version": "9"},
        )
        evidence = session.bodies("/evaluations")[0]["evidence"]
        assert evidence["data"] == DATA and evidence["evaluator"] == {"package": "p", "version": "1"}

    def test_an_evaluator_without_a_version_is_not_reported_as_one(self, server, session, weights):
        session.route("POST", "/evaluations", ok(stored(), 201))
        record(weights, project="road-seg", evaluator={"package": "visin-fusion"})
        assert "evidence" not in session.bodies("/evaluations")[0]

    def test_a_protocol_file_has_its_digest_asked_of_visin_and_sent(self, server, session, weights, tmp_path):
        suite_file = tmp_path / "suite.json"
        suite_file.write_text(
            json.dumps({"slug": "road-test", "version": 1, "name": "x", "protocol": PROTOCOL})
        )
        session.route("POST", "/suites/check", ok({"digest": "c" * 64, "protocol": PROTOCOL}))
        session.route("POST", "/evaluations", ok(stored(), 201))
        record(weights, project="road-seg", protocol=suite_file, data=DATA)
        assert session.bodies("/suites/check") == [{"protocol": PROTOCOL}]
        assert session.bodies("/evaluations")[0]["evidence"]["protocolDigest"] == "c" * 64

    def test_a_given_digest_is_not_asked_for_again(self, server, session, weights):
        session.route("POST", "/evaluations", ok(stored(), 201))
        record(weights, project="road-seg", protocol=PROTOCOL, protocol_digest="e" * 64)
        assert session.bodies("/suites/check") == []
        assert session.bodies("/evaluations")[0]["evidence"]["protocolDigest"] == "e" * 64

    def test_a_dry_run_asks_for_the_digest_too_and_stores_nothing(self, server, session, weights):
        session.route("POST", "/suites/check", ok({"digest": "c" * 64}))
        session.route("POST", "/evaluations/check", ok({"validation": ELIGIBLE}))
        record(weights, project="road-seg", protocol=PROTOCOL, dry_run=True)
        assert session.bodies("/evaluations/check")[0]["evidence"]["protocolDigest"] == "c" * 64
        assert session.bodies("/evaluations") == []

    def test_offline_the_protocol_waits_with_the_result_and_sync_asks_for_its_digest(
        self, server, session, weights, monkeypatch
    ):
        monkeypatch.setenv("VISIN_MODE", "offline")
        evaluation = record(weights, project="road-seg", uuid="u9", protocol=PROTOCOL, data=DATA)
        assert evaluation.queued and session.calls == []
        monkeypatch.delenv("VISIN_MODE")
        session.route("POST", "/suites/check", ok({"digest": "c" * 64}))
        session.route("POST", "/evaluations", ok(stored(), 201))
        [result] = visin.sync()
        assert result.sent == 1 and result.complete
        sent = session.bodies("/evaluations")[-1]
        assert sent["evidence"] == {"data": DATA, "protocolDigest": "c" * 64}
        assert visin.pending() == {}

    def test_when_visin_cannot_answer_the_digest_the_result_waits_for_sync(self, server, session, weights):
        server.retries = 0
        session.route("POST", "/suites/check", TransportError("no route to host"))
        evaluation = record(weights, project="road-seg", uuid="u8", protocol=PROTOCOL)
        assert evaluation.queued and session.bodies("/evaluations") == []
        session.route("POST", "/suites/check", ok({"digest": "c" * 64}))
        session.route("POST", "/evaluations", ok(stored(), 201))
        visin.sync()
        assert session.bodies("/evaluations")[-1]["evidence"]["protocolDigest"] == "c" * 64

    def test_a_refused_protocol_is_an_error_not_a_result_kept_for_later(self, server, session, weights):
        session.route("POST", "/suites/check", refused(400, "Invalid request"))
        with pytest.raises(ApiError):
            record(weights, project="road-seg", protocol=PROTOCOL)
        assert visin.pending() == {}

    def test_check_protocol_returns_visins_digest_for_a_suite_file_a_suite_or_a_protocol(
        self, server, session, tmp_path
    ):
        suite_file = tmp_path / "suite.json"
        suite_file.write_text(json.dumps({"slug": "road-test", "protocol": PROTOCOL}))
        session.route("POST", "/suites/check", *[ok({"digest": "c" * 64}) for _ in range(4)])
        assert visin.check_protocol(suite_file) == "c" * 64
        assert visin.check_protocol({"slug": "road-test", "protocol": PROTOCOL}) == "c" * 64
        assert visin.check_protocol(PROTOCOL) == "c" * 64
        assert visin.Api().check_protocol(PROTOCOL) == "c" * 64
        assert set(map(json.dumps, session.bodies("/suites/check"))) == {json.dumps({"protocol": PROTOCOL})}

    def test_an_old_server_that_returns_no_digest_is_named(self, server, session):
        session.route("POST", "/suites/check", ok({}))
        with pytest.raises(ConfigurationError, match="up to date"):
            visin.check_protocol(PROTOCOL)

    def test_the_verdict_says_how_much_was_reported(self):
        verdict = visin.models.Verdict.from_json({"state": "eligible", "evidence": "observed"})
        assert (
            verdict.evidence == "observed" and visin.models.Verdict.from_json({"state": "x"}).evidence is None
        )


class TestProject:
    def test_the_argument_beats_the_environment_beats_the_run(self, server, session, weights, monkeypatch):
        session.route("POST", "/evaluations", ok(stored(), 201))
        run = visin.Training(id="t1", uuid="r", name="r", project_id="from-run")
        monkeypatch.setenv("VISIN_PROJECT", "from-env")
        record(weights, project="from-arg", run=run)
        record(weights, run=run)
        assert [b["projectId"] for b in session.bodies("/evaluations")] == ["from-arg", "from-env"]

    def test_a_pipeline_key_supplies_its_own_project(self, server, session, weights):
        session.route(
            "GET",
            "/.well-known/visin",
            ok({"credential": {"kind": "pipeline-key", "project": {"id": "p7", "name": "Road"}}}),
        )
        session.route("POST", "/evaluations", ok(stored(), 201))
        record(weights)
        assert session.bodies("/evaluations")[0]["projectId"] == "p7"

    def test_no_project_anywhere_is_an_error_that_says_how_to_give_one(self, server, session, weights):
        session.route("GET", "/.well-known/visin", refused(404))
        with pytest.raises(ConfigurationError, match="needs a project"):
            record(weights)
        assert session.bodies("/evaluations") == []


class TestDryRun:
    def test_judges_without_storing_and_sends_no_uuid(self, server, session, weights):
        session.route(
            "POST",
            "/evaluations/check",
            ok(
                {
                    "validation": {
                        "state": "incomplete",
                        "reasons": [{"code": "missing-sample-count", "detail": "day"}],
                        "warnings": [],
                    },
                    "checkpointKey": "sha256:abc",
                    "suite": {"slug": "road-test", "version": 1, "digest": "d" * 64},
                }
            ),
        )
        evaluation = record(weights, project="road-seg", dry_run=True)
        assert session.paths("POST") == ["/evaluations/check"]
        assert "uuid" not in session.bodies("/evaluations/check")[0]
        assert (evaluation.stored, evaluation.id, evaluation.ranked) == (False, None, False)
        assert evaluation.checkpoint_key == "sha256:abc" and evaluation.suite == "road-test@1"
        assert (
            evaluation.suite_digest == "d" * 64
            and str(evaluation.verdict.reasons[0]) == "missing-sample-count(day)"
        )

    def test_a_dry_run_without_a_server_is_an_error_not_a_silent_pass(self, weights):
        with pytest.raises(ConfigurationError, match="VISIN_URL"):
            record(weights, project="road-seg", dry_run=True)

    def test_a_dry_run_that_names_no_suite_says_so(self, server, session, weights):
        session.route(
            "POST",
            "/evaluations/check",
            ok({"validation": {"state": "exploratory", "reasons": [{"code": "no-suite"}], "warnings": []}}),
        )
        evaluation = visin.evaluate(RESULTS, suite=None, checkpoint=None, project="p", dry_run=True)
        assert evaluation.suite is None and evaluation.checkpoint_key is None and not evaluation.ranked


class TestRefusals:
    @pytest.mark.parametrize("status", [400, 403, 404, 409])
    def test_a_refusal_raises_and_is_not_kept_for_later(self, server, session, weights, tmp_path, status):
        session.route("POST", "/evaluations", refused(status, "no such suite"))
        with pytest.raises(ApiError) as caught:
            record(weights, project="road-seg", uuid="u1")
        assert caught.value.status == status and "no such suite" in str(caught.value)
        assert visin.pending() == {}

    def test_a_409_is_a_real_refusal_here_because_the_server_only_says_it_for_a_different_result(
        self, server, session, weights
    ):
        session.route("POST", "/evaluations", refused(409, "already exists with different content"))
        with pytest.raises(ApiError, match="different content"):
            record(weights, project="road-seg", uuid="u1")


class TestWhenVisinCannotBeReached:
    def test_the_result_waits_on_disk_and_sync_sends_it_later(self, server, session, weights, caplog):
        server.retries = 0
        session.route("POST", "/evaluations", TransportError("no route to host"))
        with caplog.at_level(logging.WARNING, logger="visin"):
            evaluation = record(weights, project="road-seg", uuid="u1")
        assert (
            evaluation.queued
            and not evaluation.stored
            and evaluation.id is None
            and evaluation.verdict is None
        )
        assert evaluation.uuid == "u1" and evaluation.suite == "road-test@1" and not evaluation.ranked
        assert "waits for `visin sync`" in caplog.text
        assert sum(visin.pending().values()) == 1

        session.route("POST", "/evaluations", ok(stored(), 201))
        results = visin.sync()
        assert [r.sent for r in results] == [1] and results[0].complete
        assert session.bodies("/evaluations")[-1]["uuid"] == "u1" and visin.pending() == {}

    def test_a_server_that_is_overloaded_is_waited_out_the_same_way(self, server, session, weights):
        server.retries = 0
        session.route("POST", "/evaluations", refused(503, "unavailable"))
        assert record(weights, project="road-seg").queued

    def test_offline_mode_keeps_it_without_calling_anyone(self, session, weights, monkeypatch):
        monkeypatch.setenv("VISIN_MODE", "offline")
        evaluation = record(weights, project="road-seg", uuid="u2")
        assert evaluation.queued and session.calls == []
        assert sum(visin.pending().values()) == 1

    def test_offline_needs_a_project_since_nothing_can_look_it_up(self, weights, monkeypatch):
        monkeypatch.setenv("VISIN_MODE", "offline")
        with pytest.raises(ConfigurationError, match="needs a project"):
            record(weights)

    def test_a_refusal_found_by_sync_is_reported_and_kept_not_taken_for_delivered(
        self, server, session, weights, monkeypatch
    ):
        monkeypatch.setenv("VISIN_MODE", "offline")
        record(weights, project="road-seg", uuid="u3")
        monkeypatch.delenv("VISIN_MODE")
        session.route("POST", "/evaluations", refused(409, "already exists with different content"))
        [result] = visin.sync()
        assert result.sent == 0 and len(result.rejected) == 1 and "different content" in result.rejected[0]
        assert sum(visin.pending().values()) == 1

    def test_with_nothing_configured_nothing_is_sent_and_the_caller_is_told(self, weights, caplog):
        with caplog.at_level(logging.WARNING, logger="visin"):
            evaluation = record(weights, project="road-seg")
        assert not evaluation.stored and not evaluation.queued and evaluation.verdict is None
        assert "nothing is configured" in caplog.text


class TestPromote:
    def test_copies_a_result_onto_a_suite_with_the_checkpoint_and_counts_it_never_had(
        self, server, session, weights
    ):
        session.route("POST", "/evaluations/promote", ok(stored(), 201))
        result = visin.TestResult(id="tr1", test_uuid="t")
        evaluation = visin.promote(
            result, suite="road-test@1", checkpoint=visin.local_checkpoint(weights), sample_counts=COUNTS
        )
        body = session.bodies("/evaluations/promote")[0]
        assert (
            body["evaluationId"] == "tr1"
            and body["suite"] == "road-test@1"
            and body["sampleCounts"] == COUNTS
        )
        assert body["checkpoint"]["kind"] == "local" and evaluation.ranked

    def test_takes_a_bare_id_and_refuses_a_result_without_one(self, server, session, weights):
        session.route("POST", "/evaluations/promote", ok(stored(), 201))
        visin.promote(
            "tr1", suite="road-test@1", checkpoint=visin.local_checkpoint(weights), sample_counts=COUNTS
        )
        assert session.bodies("/evaluations/promote")[0]["evaluationId"] == "tr1"
        with pytest.raises(ValueError, match="no id"):
            visin.promote(visin.TestResult(), suite="road-test@1", checkpoint={}, sample_counts={})

    def test_needs_a_server(self, weights):
        with pytest.raises(ConfigurationError, match="promote a result"):
            visin.promote(
                "tr1", suite="road-test@1", checkpoint=visin.local_checkpoint(weights), sample_counts=COUNTS
            )

    def test_a_refusal_raises(self, server, session, weights):
        session.route(
            "POST", "/evaluations/promote", refused(403, "Manage access to the project is required")
        )
        with pytest.raises(ApiError, match="Manage access"):
            visin.promote(
                "tr1", suite="road-test@1", checkpoint=visin.local_checkpoint(weights), sample_counts=COUNTS
            )


class TestPublication:
    def test_publishes_and_withdraws_by_id_or_by_evaluation(self, server, session):
        session.route("POST", "/evaluations/e1/publish", ok(stored(publishedAt="2026-10-02T09:00:00Z")))
        session.route("POST", "/evaluations/e1/withdraw", ok(stored()))
        assert visin.publish("e1").published
        assert not visin.withdraw(visin.Evaluation(id="e1")).published

    def test_a_dry_run_or_a_queued_evaluation_has_nothing_to_publish(self, server):
        with pytest.raises(ValueError, match="no id"):
            visin.publish(visin.Evaluation(stored=False))

    def test_a_refusal_says_why(self, server, session):
        session.route(
            "POST",
            "/evaluations/e1/publish",
            refused(409, "A result can only be published from a public project."),
        )
        with pytest.raises(ApiError, match="public project"):
            visin.publish("e1")

    def test_needs_a_server(self):
        with pytest.raises(ConfigurationError, match="publish an evaluation"):
            visin.publish("e1")


class TestWhatTheSuiteDecides:
    def test_a_published_result_is_public_unless_it_waits_for_approval_or_was_hidden(self):
        base = {"_id": "e1", "publishedAt": "2026-10-02T09:00:00Z"}
        assert visin.models.Evaluation.from_json(base).public
        waiting = visin.models.Evaluation.from_json({**base, "pendingApproval": True})
        assert waiting.published and waiting.pending_approval and not waiting.public
        hidden = visin.models.Evaluation.from_json(
            {**base, "hidden": {"at": "2026-10-03", "reason": "Not our protocol"}}
        )
        assert (
            hidden.published
            and hidden.hidden == {"at": "2026-10-03", "reason": "Not our protocol"}
            and not hidden.public
        )
        assert not visin.models.Evaluation.from_json({"_id": "e1"}).public

    def test_a_result_replaced_by_a_correction_says_which(self):
        assert visin.models.Evaluation.from_json({"_id": "e1", "supersededById": "e2"}).superseded_by == "e2"
        assert visin.models.Evaluation.from_json({"_id": "e1"}).superseded_by is None
