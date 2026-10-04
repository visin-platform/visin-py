import argparse
import json
import logging
import math
from dataclasses import dataclass

import pytest
import requests
from fakes import BASE, FakeResponse, ok, refused

import visin
from visin import Run, epoch_uuid_for
from visin.errors import ApiError, ConfigurationError, TransportError


def make_run(client, uuid="run-1", **kwargs):
    return Run(training_uuid=uuid, client=client, **kwargs)


def sent(session, fragment, method="POST"):
    return session.bodies(fragment, method)


# ---------------------------------------------------------------- identity


def test_epoch_uuid_is_stable_across_processes():
    assert epoch_uuid_for("run-1", 3) == epoch_uuid_for("run-1", 3)


def test_epoch_uuid_differs_per_run_and_per_epoch():
    assert epoch_uuid_for("run-1", 3) != epoch_uuid_for("run-2", 3)
    assert epoch_uuid_for("run-1", 3) != epoch_uuid_for("run-1", 4)


def test_three_and_three_point_oh_are_one_epoch(client):
    run = make_run(client)
    assert run.log_epoch(3.0, {"loss": 1}) == run.log_epoch(3, {"loss": 1}) == run.epoch_uuid(3)


def test_log_epoch_returns_the_uuid_before_it_is_sent(client):
    run = make_run(client)
    assert run.log_epoch(1, {"loss": 0.5}) == epoch_uuid_for("run-1", 1)


# ---------------------------------------------------------------- disabled mode


def test_unconfigured_env_yields_a_run_that_reports_nothing():
    run = visin.init("laptop run")
    assert run.mode == "disabled"
    assert run.enabled is False
    assert run.log_epoch(1, {"loss": 1.0}) is None
    assert run.log_test_results(1, {"x": 1}) is None
    run.log_benchmark({"fps": 1})
    run.finish()


def test_attaching_without_a_run_uuid_is_an_error(monkeypatch):
    monkeypatch.setenv("VISIN_URL", BASE)
    monkeypatch.setenv("VISIN_TOKEN", "t")
    with pytest.raises(ConfigurationError, match="VISIN_TRAINING_UUID"):
        Run.attach()


def test_only_rank_zero_reports(server, session, monkeypatch):
    monkeypatch.setenv("RANK", "1")
    run = visin.init("ddp run")
    assert run.mode == "disabled"
    assert session.calls == []


def test_a_run_needs_a_name(server):
    with pytest.raises(ConfigurationError):
        Run.create("  ")


# ---------------------------------------------------------------- epochs


def test_epoch_payload_carries_uuid_and_results(client, session):
    run = make_run(client)
    run.log_epoch(2, {"val_loss": 0.21, "mIoU": 0.74}, learning_rate=1e-4, epoch_time=812)
    run.flush()
    body = sent(session, "/epochs/upload")[0]
    assert body["training_uuid"] == "run-1"
    assert body["epoch_uuid"] == epoch_uuid_for("run-1", 2)
    assert body["results"] == {"val_loss": 0.21, "mIoU": 0.74}
    assert body["learning_rate"] == 1e-4
    assert body["epoch_time"] == 812


def test_results_are_passed_through_whatever_their_shape(client, session):
    run = make_run(client)
    nested = {"day_clear": {"overall": {"iou": 0.3}}, "arbitrary": [1, {"x": None}]}
    run.log_epoch(0, nested)
    run.flush()
    assert sent(session, "/epochs/upload")[0]["results"] == nested


def test_train_and_val_are_the_curves_the_charts_read(client, session):
    run = make_run(client)
    run.log_epoch(1, {"val": {"car": {"iou": 0.6}}}, train={"loss": 0.4}, val={"loss": 0.5})
    run.flush()
    assert sent(session, "/epochs/upload")[0]["results"] == {
        "train": {"loss": 0.4},
        "val": {"car": {"iou": 0.6}, "loss": 0.5},
    }


def test_array_like_metrics_are_converted(client, session):
    class Scalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    run = make_run(client)
    run.log_epoch(1, train={"loss": Scalar(0.4)}, learning_rate=Scalar(0.01))
    run.flush()
    body = sent(session, "/epochs/upload")[0]
    assert body["results"] == {"train": {"loss": 0.4}}
    assert body["learning_rate"] == 0.01


def test_nan_is_sent_as_a_gap_and_said_once(client, session, caplog):
    run = make_run(client)
    with caplog.at_level(logging.WARNING, logger="visin"):
        run.log_epoch(1, train={"loss": math.nan})
        run.log_epoch(2, train={"loss": math.nan})
    run.flush()
    assert sent(session, "/epochs/upload")[0]["results"] == {"train": {"loss": None}}
    assert sum("NaN" in record.message for record in caplog.records) == 1


def test_a_bad_call_is_reported_not_raised(client, session, caplog):
    run = make_run(client)
    with caplog.at_level(logging.WARNING, logger="visin"):
        assert run.log_epoch(1) is None
        assert run.log_epoch(1, {"train": 3}, train={"loss": 1}) is None
        assert run.log_epoch(1, {"loss": object()}) is None
    run.flush()
    assert sent(session, "/epochs/upload") == []
    assert len(caplog.records) == 3


def test_strict_mode_raises_bad_calls(client):
    run = make_run(client, strict=True)
    with pytest.raises(ValueError):
        run.log_epoch(1)


def test_system_metrics_fill_the_system_tab(client, session, monkeypatch):
    monkeypatch.setattr("visin.system.system_metrics", lambda: {"memory_used_gb": 3.0})
    run = make_run(client, system_metrics=True)
    run.log_epoch(1, train={"loss": 1})
    run.log_epoch(2, train={"loss": 1}, system=False)
    run.flush()
    first, second = sent(session, "/epochs/upload")
    assert first["results"]["system_info"] == {"memory_used_gb": 3.0}
    assert "system_info" not in second["results"]


def test_a_repeated_epoch_is_accepted_not_an_error(client, session):
    # What a retried POST actually gets back from the server.
    session.route("POST", "/epochs/upload", refused(409, "Epoch with uuid x already exists"))
    run = make_run(client)
    run.log_epoch(1, {"loss": 0.5})
    run.flush()
    assert run._delivery.sender.failed == 0
    assert run._delivery.sent == 1


def test_a_server_error_is_counted_but_never_raised(client, session):
    session.route("POST", "/epochs/upload", refused(500, "boom"))
    run = make_run(client)
    run.log_epoch(1, {"loss": 0.5})
    run.flush()
    assert run._delivery.sender.failed == 1


# ---------------------------------------------------------------- test results, benchmarks, configs


def test_test_results_are_recorded_as_evaluations_with_their_own_uuid_so_a_retry_is_harmless(client, session):
    run = make_run(client)
    test_uuid = run.log_test_results(4, {"day": {"overall": {"iou": 0.7}}})
    run.flush()
    body = sent(session, "/evaluations")[0]
    assert body["uuid"] == test_uuid
    assert body["results"] == {"day": {"overall": {"iou": 0.7}}}
    assert body["source"] == {"epochUuid": epoch_uuid_for("run-1", 4), "epoch": 4, "trainingUuid": "run-1"}
    assert "suite" not in body and "executedAt" in body


def test_test_results_are_sent_after_their_epoch(client, session):
    run = make_run(client)
    run.log_epoch(4, {"loss": 0.1})
    run.log_test_results(4, {"day_clear": {"iou": 0.7}})
    run.flush()
    paths = session.paths("POST")
    assert paths.index("/epochs/upload") < paths.index("/evaluations")


def test_benchmark_fills_in_the_system_info_visin_requires(client, session, monkeypatch):
    monkeypatch.setattr(
        "visin.system.system_info",
        lambda: {"cpu_count": 8, "cpu_count_logical": 16, "memory_total_gb": 31.2, "gpu_name": "auto"},
    )
    run = make_run(client)
    run.log_benchmark({"device": "cuda", "fps": 119.0}, {"gpu_name": "RTX 4090"}, epoch=12)
    run.flush()
    body = sent(session, "/benchmarks/upload")[0]
    assert body["results"] == [{"device": "cuda", "fps": 119.0}]
    assert body["system_info"]["cpu_count"] == 8
    assert body["system_info"]["gpu_name"] == "RTX 4090"
    assert body["epoch_uuid"] == epoch_uuid_for("run-1", 12)
    assert body["training_uuid"] == "run-1"


def test_config_is_stored_and_linked_to_the_run(client, session):
    session.route("GET", "/trainings/uuid/", ok({"_id": "t1", "projectId": "actual-project"}))
    session.route("POST", "/configs/upload", ok({"_id": "c1"}, 201))
    run = make_run(client)
    run.log_config({"lr": 1e-4, "batch": 8}, name="window16")
    run.flush()
    config = sent(session, "/configs/upload")[0]
    assert config.pop("config_uuid")
    assert config == {
        "config_data": {"lr": 1e-4, "batch": 8},
        "summary": "window16",
        "config_name": "window16",
        "projectId": "actual-project",
    }
    assert sent(session, "/trainings/t1", "PUT") == [{"configId": "c1"}]


def test_config_accepts_the_shapes_configs_come_in(client, session):
    @dataclass
    class Cfg:
        lr: float = 0.1

    class Loss:
        def __repr__(self):
            return "DiceLoss()"

    run = make_run(client)
    run.log_config(argparse.Namespace(lr=0.2, loss=Loss()))
    run.log_config(Cfg())
    run.flush()
    first, second = sent(session, "/configs/upload")
    assert first["config_data"] == {"lr": 0.2, "loss": "DiceLoss()"}
    assert second["config_data"] == {"lr": 0.1}


# ---------------------------------------------------------------- visualizations


def grant():
    return ok({"uploadUrl": f"{BASE}/files/signed?sig=1", "visualization_uuid": "v1", "fileId": "f1"})


def test_visualization_reserves_uploads_and_records(client, session, uploads, tmp_path):
    session.route("POST", "/visualizations/upload-url", grant())
    frame = tmp_path / "overlay_0007.png"
    frame.write_bytes(b"\x89PNG epoch 3")
    run = make_run(client, spool=visin._internal.spool.Spool(tmp_path / "d", "run-1"))
    run.upload_visualization(3, frame, "overlay", metadata={"sample": 7})
    run.flush()
    reserve = sent(session, "/visualizations/upload-url")[0]
    assert reserve.pop("visualization_uuid")
    assert reserve == {
        "epoch_uuid": epoch_uuid_for("run-1", 3),
        "filename": "overlay_0007.png",
        "type": "overlay",
        "mimetype": "image/png",
    }
    assert uploads.puts[0]["url"] == f"{BASE}/files/signed?sig=1"
    record = sent(session, "/visualizations")[0]
    assert record["size"] == len(b"\x89PNG epoch 3")
    assert record["fileId"] == "f1" and record["visualization_uuid"] == "v1"
    assert record["metadata"] == {"sample": 7}


def test_a_frame_overwritten_after_the_call_still_uploads_what_was_logged(client, session, uploads, tmp_path):
    session.route("POST", "/visualizations/upload-url", grant(), grant())
    frame = tmp_path / "overlay.png"
    run = make_run(client, spool=visin._internal.spool.Spool(tmp_path / "d", "run-1"))
    blocker = __import__("threading").Event()
    run._delivery.sender.submit(blocker.wait)  # hold the sender, as a slow network would
    frame.write_bytes(b"epoch 1")
    run.upload_visualization(1, frame)
    frame.write_bytes(b"epoch 2")
    run.upload_visualization(2, frame)
    blocker.set()
    run.flush()
    assert [put["bytes"] for put in uploads.puts] == [b"epoch 1", b"epoch 2"]
    assert list((tmp_path / "d" / "runs" / "run-1.files").glob("*")) == []


def test_unsupported_or_missing_files_are_refused_at_the_call(client, session, tmp_path, caplog):
    notes = tmp_path / "notes.txt"
    notes.write_text("x")
    run = make_run(client)
    with caplog.at_level(logging.WARNING, logger="visin"):
        run.upload_visualization(1, notes)
        run.upload_visualization(1, tmp_path / "missing.png")
    run.flush()
    assert session.calls == []
    assert "not .txt" in caplog.records[0].message
    assert "no such file" in caplog.records[1].message


# ---------------------------------------------------------------- creation


def test_create_puts_model_in_metadata_not_at_the_top_level(server, session, monkeypatch):
    # The bug this package exists to fix: `model` and `dataset` were sent as
    # top-level fields, which the training endpoint does not define and Zod
    # strips, so those runs recorded no model and reported success.
    monkeypatch.setenv("VISIN_PROJECT", "road-seg")
    session.route("POST", "/trainings", ok({"_id": "t1", "uuid": "u"}, 201))
    run = Run.create("w16", model="clft", dataset="zod", tags="ablation")
    body = sent(session, "/trainings")[0]
    assert "model" not in body
    assert body["metadata"] == {"model": "clft", "dataset": "zod"}
    assert body["datasetId"] == "zod"
    assert body["projectId"] == "road-seg"
    assert body["tags"] == ["ablation"]
    assert body["status"] == "running"
    assert run.training_id == "t1"
    run.finish()


def test_an_over_long_name_is_truncated_and_kept_in_the_description(server, session):
    long_name = "x" * 300
    Run.create(long_name)
    body = sent(session, "/trainings")[0]
    assert len(body["name"]) == 200
    assert body["description"] == long_name


def test_creating_a_run_that_exists_reports_into_it(server, session):
    # A restarted job, or one whose orchestrator already registered it.
    session.route("POST", "/trainings", refused(409, "exists"))
    session.route("GET", "/trainings/uuid/", ok({"_id": "t9", "uuid": "known"}))
    run = Run.create("resume", training_uuid="known")
    assert run.mode == "online"
    assert run.training_id == "t9"


def test_a_resumed_run_is_marked_running_again(server, session, caplog):
    # The first attempt finished it; while the resumed job trains, it is running.
    session.route("POST", "/trainings", refused(409, "exists"))
    session.route("GET", "/trainings/uuid/", ok({"_id": "t9", "uuid": "known"}))
    with caplog.at_level(logging.INFO, logger="visin"):
        run = Run.create("resume", training_uuid="known")
        run.flush()
    assert sent(session, "/trainings/t9", "PUT") == [{"status": "running"}]
    assert any("resumed run" in record.message for record in caplog.records)


def test_a_refused_run_raises_at_startup(server, session):
    session.route("POST", "/trainings", refused(403, "Access denied to project"))
    with pytest.raises(ApiError, match="Access denied to project") as caught:
        Run.create("wrong project")
    assert caught.value.status == 403
    assert session.paths() == ["/trainings"]


def test_a_refused_run_closes_its_connection(server, session, monkeypatch):
    closed = []
    monkeypatch.setattr(session, "close", lambda: closed.append(True))
    session.route("POST", "/trainings", refused(403, "Access denied to project"))
    with pytest.raises(ApiError):
        Run.create("wrong project")
    assert closed


def test_create_with_a_config_logs_it(server, session):
    Run.create("with config", config={"lr": 1}).flush()
    assert sent(session, "/configs/upload")[0]["config_data"] == {"lr": 1}


def test_attach_reports_into_the_run_an_orchestrator_named(server, session, monkeypatch):
    monkeypatch.setenv("VISIN_TRAINING_UUID", "from-orchestrator")
    run = Run.attach()
    run.log_epoch(1, {"loss": 1})
    run.flush()
    assert "/trainings" not in session.paths("POST")
    assert sent(session, "/epochs/upload")[0]["training_uuid"] == "from-orchestrator"


def test_init_names_an_unnamed_run_after_the_script(server, session):
    visin.init()
    assert sent(session, "/trainings")[0]["name"]


def test_init_with_a_name_uses_the_orchestrators_uuid(server, session, monkeypatch):
    monkeypatch.setenv("VISIN_TRAINING_UUID", "from-orchestrator")
    visin.init("named")
    assert sent(session, "/trainings")[0]["uuid"] == "from-orchestrator"


# ---------------------------------------------------------------- update and lifecycle


def test_update_changes_the_run(client, session):
    session.route("GET", "/trainings/uuid/", ok({"_id": "t1"}))
    run = make_run(client)
    run.update(tags=["best"], metadata={"best_epoch": 12}, name="renamed")
    run.update()  # nothing to change, nothing sent
    run.flush()
    assert sent(session, "/trainings/t1", "PUT") == [
        {"name": "renamed", "tags": ["best"], "metadata": {"best_epoch": 12}}
    ]


def test_notes_are_the_researchers_own_text_and_an_empty_string_removes_them(client, session, caplog):
    session.route("GET", "/trainings/uuid/", ok({"_id": "t1"}))
    run = make_run(client)
    run.update(notes="used the relabelled night set")
    run.update(notes="")
    run.update(notes="x" * 6000)
    run.flush()
    first, second, third = sent(session, "/trainings/t1", "PUT")
    assert first == {"notes": "used the relabelled night set"} and second == {"notes": ""}
    assert len(third["notes"]) == 5000 and "notes truncated to 5000" in caplog.text


def test_finish_sets_the_terminal_status(client, session):
    session.route("GET", "/trainings/uuid/", ok({"_id": "t1"}))
    run = make_run(client)
    run.log_epoch(1, {"loss": 1.0})
    run.finish()
    status = sent(session, "/trainings/t1", "PUT")[0]
    assert status["status"] == "completed"
    assert "endTime" in status


def test_the_training_id_is_looked_up_once(client, session):
    session.route("GET", "/trainings/uuid/", ok({"_id": "t1"}))
    run = make_run(client)
    run.update(tags=["a"])
    run.finish()
    assert session.paths("GET").count("/trainings/uuid/run-1") == 1


def test_finish_is_idempotent(client, session):
    run = make_run(client)
    run.finish()
    run.finish()
    assert len(session.paths("PUT")) <= 1


def test_nothing_is_sent_after_finish(client, session, caplog):
    run = make_run(client)
    run.finish()
    calls = len(session.calls)
    with caplog.at_level(logging.WARNING, logger="visin"):
        assert run.log_epoch(9, {"loss": 1}) is None
    assert len(session.calls) == calls
    assert "has finished" in caplog.records[-1].message


def test_an_unknown_status_marks_the_run_failed(client, session):
    session.route("GET", "/trainings/uuid/", ok({"_id": "t1"}))
    make_run(client).finish("done")
    assert sent(session, "/trainings/t1", "PUT")[0]["status"] == "failed"


def test_context_manager_marks_a_crash_as_failed(client, session):
    session.route("GET", "/trainings/uuid/", ok({"_id": "t1"}))

    def train():
        with make_run(client) as run:
            run.log_epoch(1, {"loss": 1.0})
            raise RuntimeError("diverged")

    with pytest.raises(RuntimeError):
        train()
    assert sent(session, "/trainings/t1", "PUT")[0]["status"] == "failed"


def test_a_clean_sys_exit_is_not_a_failure(client, session):
    session.route("GET", "/trainings/uuid/", ok({"_id": "t1"}))
    with pytest.raises(SystemExit), make_run(client):
        raise SystemExit(0)
    assert sent(session, "/trainings/t1", "PUT")[0]["status"] == "completed"


def test_epochs_logged_before_a_crash_are_still_sent(client, session):
    def train():
        with make_run(client) as run:
            for epoch in range(3):
                run.log_epoch(epoch, {"loss": 1.0 / (epoch + 1)})
            raise RuntimeError("OOM")

    with pytest.raises(RuntimeError):
        train()
    assert len(sent(session, "/epochs/upload")) == 3


def test_a_process_that_ends_without_finish_is_finished_at_exit(client, session, isolated):
    session.route("GET", "/trainings/uuid/", ok({"_id": "t1"}))
    run = make_run(client)
    run.log_epoch(1, {"loss": 1})
    for callback in list(isolated.callbacks):
        callback()
    assert run._finished
    assert sent(session, "/trainings/t1", "PUT")[0]["status"] == "completed"


def test_a_process_that_crashed_is_marked_failed_at_exit(client, session, isolated, monkeypatch):
    monkeypatch.setattr(visin._internal.process.ProcessHooks, "crashed", True)
    session.route("GET", "/trainings/uuid/", ok({"_id": "t1"}))
    make_run(client)
    for callback in list(isolated.callbacks):
        callback()
    assert sent(session, "/trainings/t1", "PUT")[0]["status"] == "failed"


def test_sigterm_becomes_an_ordinary_exit():
    with pytest.raises(SystemExit) as caught:
        visin._internal.process.ProcessHooks._on_sigterm(15, None)
    assert caught.value.code == 143


# ---------------------------------------------------------------- losing the server


def test_a_run_that_cannot_reach_visin_at_start_keeps_everything_and_catches_up(server, session, monkeypatch):
    session.route("POST", "/trainings", *[TransportError("no route to host")])
    monkeypatch.setattr(server, "retries", 0)
    run = Run.create("cluster job")
    assert run.mode == "online"
    assert run._delivery.spooling
    run.log_epoch(1, {"loss": 1})
    run.flush()
    # The server is back: the kept run and epoch go first, in order.
    assert session.paths("POST")[-2:] == ["/trainings", "/epochs/upload"]
    assert not run._delivery.spooling
    run.finish()


def test_losing_visin_mid_run_keeps_reports_until_it_answers(client, session, tmp_path):
    session.route("POST", "/epochs/upload", TransportError("down"), TransportError("down"))
    run = make_run(client, spool=visin._internal.spool.Spool(tmp_path, "run-1"))
    client.retries = 0
    run.log_epoch(1, {"loss": 1})  # lost; kept on disk
    run.flush()
    assert run._delivery.spool.count() == 1
    run.log_epoch(2, {"loss": 0.5})  # catch-up fails again: both kept
    run.flush()
    assert run._delivery.spool.count() == 2
    run.log_epoch(3, {"loss": 0.25})  # back: everything, in order
    run.flush()
    epochs = [body["epoch"] for body in sent(session, "/epochs/upload")]
    assert epochs[-3:] == [1, 2, 3]
    assert run._delivery.spool.count() == 0
    assert run._delivery.sender.failed == 0


def test_a_refusal_is_not_kept_for_later(client, session, tmp_path):
    session.route("POST", "/epochs/upload", refused(400, "Results are required"))
    run = make_run(client, spool=visin._internal.spool.Spool(tmp_path, "run-1"))
    run.log_epoch(1, {"loss": 1})
    run.flush()
    assert run._delivery.spool.count() == 0
    assert run._delivery.sender.failed == 1


def test_finish_reports_what_is_still_waiting(client, session, tmp_path, caplog):
    session.default = FakeResponse(503, None, text="down")
    client.retries = 0
    run = make_run(client, spool=visin._internal.spool.Spool(tmp_path, "run-1"))
    run.log_epoch(1, {"loss": 1})
    with caplog.at_level(logging.WARNING, logger="visin"):
        run.finish()
    assert run._delivery.spool.count() == 2  # the epoch and the final status
    assert "visin sync" in caplog.records[-1].message


def test_api_error_class_is_exported():
    assert issubclass(ApiError, visin.VisinError)


# ---------------------------------------------------------------- argument checking


@pytest.mark.parametrize(
    "call",
    [
        lambda run: run.log_epoch(True, {"loss": 1}),
        lambda run: run.log_epoch(math.inf, {"loss": 1}),
        lambda run: run.log_epoch(1, [0.5]),
        lambda run: run.log_test_results(1, None),
        lambda run: run.log_test_results(1, {"x": object()}),
        lambda run: run.log_benchmark([1, 2]),
        lambda run: run.log_config(object()),
        lambda run: run.update(metadata={"x": object()}),
    ],
    ids=[
        "bool-epoch",
        "infinite-epoch",
        "list-results",
        "no-test-results",
        "bad-test-results",
        "bad-benchmark-rows",
        "unreadable-config",
        "bad-update",
    ],
)
def test_bad_arguments_are_reported_not_raised(client, session, call, caplog):
    run = make_run(client)
    with caplog.at_level(logging.WARNING, logger="visin"):
        call(run)
    run.flush()
    assert session.calls == []
    assert "could not" in caplog.records[-1].message


def test_numpy_style_integers_are_epochs(client):
    class Int64:
        def __index__(self):
            return 7

    assert make_run(client).epoch_uuid(Int64()) == epoch_uuid_for("run-1", 7)


def test_pydantic_style_configs_are_read(client, session):
    class Model:
        def model_dump(self):
            return {"lr": 3}

    run = make_run(client)
    run.log_config(Model(), summary="from pydantic")
    run.flush()
    body = sent(session, "/configs/upload")[0]
    assert body.pop("config_uuid")
    assert body == {"config_data": {"lr": 3}, "summary": "from pydantic"}


def test_hydra_configs_are_resolved(client, session, monkeypatch):
    import sys
    import types

    omegaconf = types.ModuleType("omegaconf")
    omegaconf.OmegaConf = types.SimpleNamespace(to_container=lambda cfg, resolve: {"lr": cfg.lr})
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf)
    DictConfig = type("DictConfig", (), {"__module__": "omegaconf.dictconfig", "lr": 5})
    run = make_run(client)
    run.log_config(DictConfig())
    run.flush()
    assert sent(session, "/configs/upload")[0]["config_data"] == {"lr": 5}


def test_a_known_training_id_saves_the_lookup(client, session):
    run = make_run(client, training_id="t7")
    run.finish()
    assert session.paths("GET") == []
    assert session.paths("PUT") == ["/trainings/t7"]


# ---------------------------------------------------------------- constructors


def test_attach_defaults_to_the_named_run(server, session, monkeypatch):
    monkeypatch.setenv("VISIN_TRAINING_UUID", "named")
    run = Run.attach()
    assert run.training_uuid == "named" and run.mode == "online"
    assert repr(run) == "<visin.Run named online>"
    run.finish()
    assert repr(run) == "<visin.Run named finished>"


def test_constructors_without_a_server_are_disabled(monkeypatch):
    assert Run.attach("x").mode == "disabled"
    assert Run.attach().mode == "disabled"
    assert repr(Run.disabled()) == "<visin.Run - disabled>"


def test_other_ranks_are_disabled_whichever_constructor(server, monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("VISIN_TRAINING_UUID", "x")
    assert Run.attach("x").mode == "disabled"
    assert Run.attach().mode == "disabled"


def test_a_run_uuid_that_cannot_name_a_file_still_reports_online(server, session, tmp_path):
    frame = tmp_path / "f.png"
    frame.write_bytes(b"x")
    session.route("POST", "/visualizations/upload-url", grant())
    run = Run.attach("odd/uuid")
    assert run.mode == "online" and run._delivery.spool is None
    run.upload_visualization(1, frame)
    run.flush()
    assert sent(session, "/visualizations")[0]["size"] == 1


def test_a_run_uuid_that_cannot_name_a_file_cannot_be_offline(monkeypatch):
    monkeypatch.setenv("VISIN_MODE", "offline")
    with pytest.raises(ConfigurationError):
        Run.attach("odd/uuid")


def test_create_passes_a_config_id(server, session):
    Run.create("x", config_id="c1")
    assert sent(session, "/trainings")[0]["configId"] == "c1"


def test_create_with_metadata_json_cannot_carry_is_disabled(server, session):
    run = Run.create("x", metadata={"model": object()})
    assert run.mode == "disabled" and session.calls == []


def test_strict_create_raises_a_refusal(server, session):
    session.route("POST", "/trainings", refused(403, "Access denied"))
    with pytest.raises(ApiError):
        Run.create("x", strict=True)


def test_init_resuming_still_logs_its_config(server, session, monkeypatch):
    monkeypatch.setenv("VISIN_TRAINING_UUID", "from-orchestrator")
    visin.init(config={"lr": 1}).flush()
    assert sent(session, "/configs/upload")[0]["config_data"] == {"lr": 1}


def test_an_offline_disk_that_cannot_be_written_is_reported(monkeypatch, caplog, tmp_path):
    monkeypatch.setenv("VISIN_MODE", "offline")
    run = visin.init("x")

    def full(_op):
        raise OSError("No space left on device")

    monkeypatch.setattr(run._delivery.spool, "append", full)
    with caplog.at_level(logging.WARNING, logger="visin"):
        run.log_epoch(1, {"loss": 1})
    assert "No space left" in caplog.records[-1].message


def test_kept_reports_the_server_refuses_on_catch_up_are_logged(client, session, tmp_path, caplog):
    run = make_run(client, spool=visin._internal.spool.Spool(tmp_path, "run-1"))
    run._delivery.spool.append({"op": "epoch", "body": {}})
    run._delivery.start_spooling()
    session.route("POST", "/epochs/upload", refused(400, "Results are required"))
    with caplog.at_level(logging.WARNING, logger="visin"):
        run.log_epoch(2, {"loss": 1})
        run.flush()
    assert any("refused a kept report" in record.message for record in caplog.records)
    assert not run._delivery.spooling


# ---------------------------------------------------------------- a real process


SCRIPT = """
import os, signal, sys
import visin
run = visin.init("subprocess run")
run.log_epoch(1, {{"loss": 1.0}})
{ending}
"""


def run_script(tmp_path, ending):
    import json
    import os
    import subprocess
    import sys

    env = {key: value for key, value in os.environ.items() if not key.startswith("VISIN_")}
    env.update(VISIN_MODE="offline", VISIN_DIR=str(tmp_path))
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT.format(ending=ending)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    (spool,) = (tmp_path / "runs").glob("*.jsonl")
    ops = [json.loads(line) for line in spool.read_text().splitlines()]
    return done, ops


def test_a_script_that_ends_without_finish_is_completed(tmp_path):
    done, ops = run_script(tmp_path, "")
    assert done.returncode == 0, done.stderr
    assert [op["op"] for op in ops] == ["create_run", "epoch", "update"]
    assert ops[-1]["body"]["status"] == "completed"


def test_a_script_that_crashes_is_marked_failed_and_keeps_its_epochs(tmp_path):
    done, ops = run_script(tmp_path, "raise RuntimeError('diverged')")
    assert done.returncode == 1 and "diverged" in done.stderr
    assert [op["op"] for op in ops] == ["create_run", "epoch", "update"]
    assert ops[-1]["body"]["status"] == "failed"


@pytest.mark.skipif(not hasattr(__import__("signal"), "SIGKILL"), reason="POSIX signals")
def test_a_script_the_scheduler_terminates_is_marked_failed(tmp_path):
    done, ops = run_script(tmp_path, "os.kill(os.getpid(), signal.SIGTERM)\nimport time; time.sleep(10)")
    assert done.returncode == 143
    assert ops[-1]["body"]["status"] == "failed"


# ---------------------------------------------------------------- bounded waits


def test_creating_a_run_spends_a_short_retry_budget(server, session, monkeypatch):
    # Training waits on this call: with the client's full budget, a server that
    # drops packets held init() for close to a minute.
    session.route("POST", "/trainings", *[requests.exceptions.ConnectTimeout("timed out")] * 10)
    run = Run.create("x")
    assert run._delivery.spooling
    assert session.paths("POST").count("/trainings") == visin._internal.delivery.CREATE_RETRIES + 1


def test_a_catch_up_tries_once(client, session, tmp_path):
    session.route("POST", "/epochs/upload", *[requests.exceptions.ConnectTimeout("down")] * 20)
    run = make_run(client, spool=visin._internal.spool.Spool(tmp_path, "run-1"))
    run.log_epoch(1, {"loss": 1})
    run.flush()
    attempts = len(session.calls)
    assert attempts == client.retries + 1  # the first loss spends the full budget, in the background
    run.log_epoch(2, {"loss": 1})
    run.flush()
    assert len(session.calls) == attempts + visin._internal.delivery.CATCH_UP_RETRIES + 1


def test_reports_a_timed_out_finish_could_not_send_are_kept_on_disk(client, session, tmp_path):
    import threading

    run = make_run(client, spool=visin._internal.spool.Spool(tmp_path, "run-1"))
    blocked = threading.Event()
    run._delivery.sender.submit(blocked.wait)  # the network, stuck
    run.log_epoch(1, {"loss": 1})
    run.log_epoch(2, {"loss": 1})
    run.finish(timeout=0.1)
    blocked.set()
    kinds = [json.loads(line)["op"] for line in run._delivery.spool.path.read_text().splitlines()]
    assert kinds == ["epoch", "epoch", "update"]


def test_a_dropped_visualization_leaves_no_staged_file(client, session, tmp_path):
    import threading

    frame = tmp_path / "f.png"
    frame.write_bytes(b"x")
    run = make_run(client, spool=visin._internal.spool.Spool(tmp_path / "d", "run-1"))
    run._delivery.sender = visin._internal.sender.Sender(max_queue=1)
    blocked, running = threading.Event(), threading.Event()
    run._delivery.sender.submit(lambda: (running.set(), blocked.wait()))
    assert running.wait(5)  # the sender is busy...
    run._delivery.sender.submit(lambda: None)  # ...and the queue is full
    run.upload_visualization(1, frame)
    assert list(run._delivery.spool.files.glob("*")) == []
    blocked.set()


def test_an_explicitly_disabled_run_says_why(monkeypatch, caplog):
    monkeypatch.setenv("VISIN_MODE", "disabled")
    with caplog.at_level(logging.INFO, logger="visin"):
        assert visin.init("x").mode == "disabled"
    assert "VISIN_MODE=disabled" in caplog.records[-1].message


# ---------------------------------------------------------------- reporting into a run from another script


def test_a_script_that_does_not_own_the_run_leaves_its_status_alone(server, session):
    run = Run.attach("trained-earlier", mark_status=False)
    run.log_test_results(3, {"overall": {"iou": 0.5}})
    run.finish()
    assert session.paths("PUT") == []
    assert session.paths("GET") == []


def test_a_crashing_script_that_does_not_own_the_run_does_not_fail_it(server, session):
    with pytest.raises(RuntimeError), Run.attach("trained-earlier", mark_status=False):
        raise RuntimeError("test set missing")
    assert session.paths("PUT") == []


def test_results_can_name_an_epoch_this_package_did_not_log(client, session, tmp_path):
    legacy = "19f7fbfc-88ac-4944-b8c1-7cc2b96ed1c3"  # from a checkpoint's file name
    frame = tmp_path / "overlay.png"
    frame.write_bytes(b"png")
    session.route("POST", "/visualizations/upload-url", grant())
    run = make_run(client)
    run.log_test_results(9, {"overall": {"iou": 0.5}}, epoch_uuid=legacy)
    run.log_benchmark(
        {"fps": 1.0},
        {"cpu_count": 1, "cpu_count_logical": 1, "memory_total_gb": 1},
        epoch=9,
        epoch_uuid=legacy,
    )
    run.log_benchmark(
        {"fps": 2.0}, {"cpu_count": 1, "cpu_count_logical": 1, "memory_total_gb": 1}, epoch_uuid=legacy
    )
    run.upload_visualization(9, frame, "overlay", epoch_uuid=legacy)
    run.flush()
    assert sent(session, "/evaluations")[0]["source"]["epochUuid"] == legacy
    assert [b["epoch_uuid"] for b in sent(session, "/benchmarks/upload")] == [legacy, legacy]
    assert all("training_uuid" not in b for b in sent(session, "/benchmarks/upload"))
    assert sent(session, "/visualizations/upload-url")[0]["epoch_uuid"] == legacy


# ---------------------------------------------------------------- integration conveniences


def test_resumed_says_whether_visin_already_had_the_run(server, session):
    fresh = Run.create("fresh")
    assert fresh.resumed is False
    session.route("POST", "/trainings", refused(409, "exists"))
    session.route("GET", "/trainings/uuid/", ok({"_id": "t9", "uuid": "known"}))
    assert Run.create("again", training_uuid="known").resumed is True
    assert Run.attach("other").resumed is None


def test_resumed_is_unknown_offline(monkeypatch):
    monkeypatch.setenv("VISIN_MODE", "offline")
    assert visin.init("x").resumed is None


def test_a_relogged_epoch_that_visin_keeps_the_first_of_is_said_once(client, session, caplog):
    # A restarted job whose epoch numbering started over: its epochs are
    # answered 409 and the new values are not stored.
    session.route("POST", "/epochs/upload", refused(409, "exists"), refused(409, "exists"))
    run = make_run(client)
    with caplog.at_level(logging.WARNING, logger="visin"):
        run.log_epoch(0, {"loss": 1})
        run.log_epoch(1, {"loss": 1})
        run.flush()
    warnings = [r.message for r in caplog.records if "already recorded" in r.message]
    assert len(warnings) == 1 and "start a new run" in warnings[0]


def test_repeating_an_epoch_in_the_same_process_is_not_a_relog(client, session, caplog):
    session.route("POST", "/epochs/upload", ok(), refused(409, "exists"))
    run = make_run(client)
    with caplog.at_level(logging.WARNING, logger="visin"):
        run.log_epoch(2, {"loss": 1})
        run.log_epoch(2, {"loss": 99})
        run.flush()
    assert not [r for r in caplog.records if "already recorded" in r.message]


def test_a_benchmark_named_by_its_epoch_leaves_the_run_to_the_server(client, session):
    # Naming the attached run as well could only disagree with the epoch's own run.
    run = make_run(client)
    info = {"cpu_count": 1, "cpu_count_logical": 1, "memory_total_gb": 1}
    run.log_benchmark({"fps": 1}, info, epoch=3, epoch_uuid="e-from-a-checkpoint")
    run.log_benchmark({"fps": 1}, info, epoch=3)
    run.flush()
    named, derived = sent(session, "/benchmarks/upload")
    assert "training_uuid" not in named and named["epoch_uuid"] == "e-from-a-checkpoint"
    assert derived["training_uuid"] == "run-1" and derived["epoch_uuid"] == epoch_uuid_for("run-1", 3)


def test_console_logging_prints_visins_lines_once(capsys):
    import io
    import logging as std_logging

    stream = io.StringIO()
    visin.enable_console_logging(stream=stream)
    visin.enable_console_logging(stream=stream)  # again: replaced, not doubled
    std_logging.getLogger("visin").info("visin: created run x")
    assert stream.getvalue() == "visin: created run x\n"
    visin.enable_console_logging(level="WARNING", stream=stream)
    std_logging.getLogger("visin").info("visin: quiet now")
    assert "quiet now" not in stream.getvalue()
    logger = std_logging.getLogger("visin")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.propagate = True
    logger.setLevel(std_logging.NOTSET)


def test_a_terminated_run_says_so(client, session, caplog, monkeypatch):
    monkeypatch.setattr(visin._internal.process.ProcessHooks, "terminated", True)
    with caplog.at_level(logging.WARNING, logger="visin"), pytest.raises(SystemExit), make_run(client):
        raise SystemExit(143)
    assert any("terminated by SIGTERM" in record.message for record in caplog.records)


def test_a_nonzero_exit_names_its_status(client, session, caplog):
    with caplog.at_level(logging.WARNING, logger="visin"), pytest.raises(SystemExit), make_run(client):
        raise SystemExit(2)
    assert any("exited with status 2" in record.message for record in caplog.records)


def test_a_token_without_a_url_is_warned_about(monkeypatch, caplog):
    monkeypatch.setenv("VISIN_TOKEN", "t")
    with caplog.at_level(logging.WARNING, logger="visin"):
        assert not Run.create("lonely token").enabled
    assert "but VISIN_URL is not" in caplog.text and "vision-api.visin.eu" in caplog.text


def test_nothing_set_at_all_stays_quiet(caplog):
    with caplog.at_level(logging.WARNING, logger="visin"):
        assert not Run.create("laptop").enabled
    assert caplog.text == ""


def test_a_run_knows_its_page_in_the_web_app(server, session, monkeypatch):
    monkeypatch.setenv("VISIN_URL", "https://vision-api.visin.eu")
    session.route("POST", "/trainings", ok({"_id": "t77"}))
    run = Run.create("linked")
    assert run.url == "https://app.visin.eu/trainings/t77"


def test_a_self_hosted_run_has_a_page_only_when_told_the_app_address(server, session, monkeypatch):
    session.route("POST", "/trainings", ok({"_id": "t77"}))
    assert Run.create("unlinked").url is None
    monkeypatch.setenv("VISIN_APP_URL", "https://app.example.test/")
    session.route("POST", "/trainings", ok({"_id": "t78"}))
    assert Run.create("linked").url == "https://app.example.test/trainings/t78"


def test_an_array_is_uploaded_as_a_png_under_the_given_name(client, session, uploads, tmp_path):
    import numpy as np

    session.route("POST", "/visualizations/upload-url", grant())
    run = make_run(client, spool=visin._internal.spool.Spool(tmp_path / "d", "run-1"))
    run.upload_visualization(3, np.zeros((4, 6, 3), dtype=np.uint8), "overlay", name="sample_7")
    run.flush()
    reserve = sent(session, "/visualizations/upload-url")[0]
    assert (reserve["filename"], reserve["mimetype"], reserve["type"]) == (
        "sample_7.png",
        "image/png",
        "overlay",
    )
    assert uploads.puts[0]["bytes"].startswith(b"\x89PNG")
    assert list((tmp_path / "d" / "runs" / "run-1.files").glob("*")) == []


def test_an_array_without_a_name_is_called_after_its_kind(client, session, tmp_path):
    import numpy as np

    session.route("POST", "/visualizations/upload-url", grant())
    run = make_run(client, spool=visin._internal.spool.Spool(tmp_path / "d", "run-1"))
    run.upload_visualization(1, np.zeros((2, 2)), "mask")
    run.flush()
    assert sent(session, "/visualizations/upload-url")[0]["filename"] == "mask.png"


def test_an_image_that_cannot_be_read_is_reported_not_raised(client, tmp_path, caplog):
    import numpy as np

    run = make_run(client, spool=visin._internal.spool.Spool(tmp_path / "d", "run-1"))
    with caplog.at_level(logging.WARNING, logger="visin"):
        run.upload_visualization(1, np.zeros((2, 2, 7)))
    assert "cannot make an image" in caplog.text


def test_an_image_in_memory_needs_somewhere_to_be_kept(client, caplog):
    import numpy as np

    run = make_run(client)
    with caplog.at_level(logging.WARNING, logger="visin"):
        run.upload_visualization(1, np.zeros((2, 2)))
    assert "needs disk" in caplog.text


def test_a_report_the_queue_could_not_take_returns_no_uuid(client, tmp_path):
    run = make_run(client, spool=visin._internal.spool.Spool(tmp_path / "d", "run-1"))
    run.flush()
    run._delivery.sender.stop()
    assert run.log_epoch(1, train={"loss": 1.0}) is None
    assert run.log_test_results(1, {"overall": {"iou": 0.5}}) is None


def test_a_frame_whose_metadata_cannot_be_sent_leaves_no_staged_copy(client, tmp_path, caplog):
    import numpy as np

    run = make_run(client, spool=visin._internal.spool.Spool(tmp_path / "d", "run-1"))
    with caplog.at_level(logging.WARNING, logger="visin"):
        run.upload_visualization(1, np.zeros((2, 2)), metadata={"bad": object()})
    assert "could not upload visualization" in caplog.text
    assert list((tmp_path / "d" / "runs" / "run-1.files").glob("*")) == []


def test_attach_with_nothing_configured_replaces_the_current_run(server, monkeypatch):
    first = visin.init("first")
    monkeypatch.delenv("VISIN_URL")
    attached = Run.attach("other")
    assert attached.mode == "disabled" and visin.get_run() is attached
    first.finish()


def test_a_catch_up_waits_while_a_sync_holds_the_lock(client, session, tmp_path):
    from visin._internal.lock import exclusive

    spool = visin._internal.spool.Spool(tmp_path / "d", "run-1")
    run = make_run(client, spool=spool)
    spool.append(
        {"op": "epoch", "body": {"training_uuid": "run-1", "epoch_uuid": "e1", "epoch": 1, "results": {}}}
    )
    run._delivery.start_spooling()
    with exclusive(spool.root / ".sync.lock"):
        assert run._delivery.catch_up(force=True) is False
    assert spool.count() == 1
    assert run._delivery.catch_up(force=True) is True
