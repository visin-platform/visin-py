"""Uploading a checkpoint to the Hub and linking it to the run."""

import sys
import types
from pathlib import Path

import pytest
from fakes import FakeHub, ok, refused

import visin
from visin import Run
from visin.errors import VisinError

COMMIT = FakeHub.COMMIT


def make_run(client, uuid="run-1"):
    return Run(training_uuid=uuid, client=client)


@pytest.fixture
def run(client, session):
    session.route("GET", "/trainings/uuid/", ok({"_id": "t1", "projectId": "p1"}))
    return make_run(client)


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "best.pt"
    path.write_bytes(b"weights")
    return path


def links(session):
    return session.bodies("/trainings/t1/models")


def test_a_checkpoint_is_uploaded_and_linked_to_the_run_at_the_commit_it_made(run, session, hf, checkpoint):
    assert run.log_model(checkpoint, "acme/clftv2-zod", epoch=12) == COMMIT
    run.flush()
    assert hf.created == [
        {"repo_id": "acme/clftv2-zod", "repo_type": "model", "private": True, "exist_ok": True}
    ]
    assert hf.uploads == [
        {
            "kind": "file",
            "repo_id": "acme/clftv2-zod",
            "repo_type": "model",
            "path_or_fileobj": str(checkpoint),
            "path_in_repo": "best.pt",
        }
    ]
    assert links(session) == [
        {
            "provider": "hf",
            "kind": "model",
            "repo": "acme/clftv2-zod",
            "revision": COMMIT,
            "path": "best.pt",
            "epoch": 12,
        }
    ]


def test_a_folder_is_uploaded_whole_and_a_public_repo_must_be_asked_for(run, session, hf, tmp_path):
    folder = tmp_path / "export"
    folder.mkdir()
    (folder / "model.onnx").write_bytes(b"x")
    run.log_model(folder, "acme/clftv2-zod", private=False)
    run.flush()
    assert hf.created[0]["private"] is False
    assert hf.uploads[0]["kind"] == "folder" and hf.uploads[0]["path_in_repo"] is None
    assert "path" not in links(session)[0] and "epoch" not in links(session)[0]


def test_where_in_the_repo_it_goes_is_recorded(run, session, hf, checkpoint):
    run.log_model(checkpoint, "acme/m", path_in_repo="runs/1/best.pt")
    run.flush()
    assert hf.uploads[0]["path_in_repo"] == "runs/1/best.pt"
    assert links(session)[0]["path"] == "runs/1/best.pt"


def test_a_missing_extra_is_named_and_sends_nothing(run, session, monkeypatch, checkpoint, caplog):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    assert run.log_model(checkpoint, "acme/m") is None
    run.flush()
    assert "visin[hf]" in caplog.text
    assert not links(session)


def test_a_failed_upload_is_logged_not_raised_and_nothing_is_linked(run, session, hf, checkpoint, caplog):
    hf.failure = RuntimeError("403 forbidden")
    assert run.log_model(checkpoint, "acme/m") is None
    run.flush()
    assert "could not upload" in caplog.text and "403 forbidden" in caplog.text
    assert not links(session)


def test_a_missing_checkpoint_or_a_fractional_epoch_uploads_nothing(run, session, hf, checkpoint, tmp_path):
    assert run.log_model(tmp_path / "gone.pt", "acme/m") is None
    assert run.log_model(checkpoint, "acme/m", epoch=1.5) is None
    assert run.log_model(checkpoint, "acme/m", epoch=True) is None
    assert not hf.uploads and not hf.created


def test_strict_runs_raise_what_a_lenient_run_logs(client, hf, checkpoint):
    strict = Run(training_uuid="run-1", client=client, strict=True)
    hf.failure = RuntimeError("down")
    with pytest.raises(VisinError, match="could not upload"):
        strict.log_model(checkpoint, "acme/m")


def test_a_project_that_keeps_files_on_visin_refuses_the_link_without_stopping_training(
    run, session, hf, checkpoint, caplog
):
    session.route(
        "POST",
        "/trainings/t1/models",
        refused(409, "This project keeps its files on Visin.", "ConflictError"),
    )
    assert run.log_model(checkpoint, "acme/m") == COMMIT
    run.flush()
    assert len(links(session)) == 1
    assert run.log_epoch(1, {"loss": 1.0})
    run.flush()
    assert session.bodies("/epochs/upload")


def test_a_disabled_run_uploads_nothing(hf, checkpoint):
    run = visin.init("laptop run")
    assert run.log_model(checkpoint, "acme/m") is None
    assert visin.log_model(checkpoint, "acme/m") is None
    assert not hf.uploads


def test_the_shortcut_uses_the_current_run(server, session, hf, checkpoint):
    session.route("POST", "/trainings", ok({"_id": "t1", "uuid": "u", "projectId": "p1"}, 201))
    run = visin.init("a run", project="p1")
    assert visin.log_model(checkpoint, "acme/m", epoch=3) == COMMIT
    run.flush()
    assert links(session)[0]["epoch"] == 3


CARD = "---\nlibrary_name: visin-trained\n---\n\n# acme/m\n"


def card_route(session, answer=None):
    session.route("GET", "/trainings/t1/model-card", answer or ok({"readme": CARD}))


def test_visins_model_card_is_added_as_the_repos_readme_in_a_second_commit(run, session, hf, checkpoint):
    card_route(session)
    assert run.log_model(checkpoint, "acme/m", epoch=12) == COMMIT
    run.flush()
    (request,) = [call for call in session.calls if call["url"].endswith("/model-card")]
    assert request["params"] == {"repo": "acme/m", "epoch": 12}
    weights, readme = hf.uploads
    assert weights["path_in_repo"] == "best.pt"
    assert readme["path_in_repo"] == "README.md" and readme["path_or_fileobj"] == CARD.encode()
    assert links(session)[0]["revision"] == COMMIT


def test_a_readme_someone_wrote_is_never_replaced(run, session, hf, checkpoint):
    card_route(session)
    hf.existing.add("README.md")
    run.log_model(checkpoint, "acme/m")
    assert [upload["path_in_repo"] for upload in hf.uploads] == ["best.pt"]


def test_the_card_waits_for_the_epochs_reported_before_it(run, session, hf, checkpoint):
    card_route(session)
    run.log_epoch(1, {"loss": 1.0})
    run.log_model(checkpoint, "acme/m")
    paths = session.paths()
    assert paths.index("/epochs/upload") < next(
        i for i, path in enumerate(paths) if path.endswith("/model-card")
    )


def test_a_card_can_be_declined(run, session, hf, checkpoint):
    card_route(session)
    run.log_model(checkpoint, "acme/m", card=False)
    assert not [call for call in session.calls if call["url"].endswith("/model-card")]
    assert len(hf.uploads) == 1


@pytest.mark.parametrize("answer", [refused(404, "Not found"), ok({}), ok({"readme": ""})])
def test_a_card_that_cannot_be_had_leaves_the_checkpoint_uploaded_without_one(
    run, session, hf, checkpoint, answer
):
    card_route(session, answer)
    assert run.log_model(checkpoint, "acme/m") == COMMIT
    assert len(hf.uploads) == 1


def test_an_offline_run_uploads_without_a_card(monkeypatch, hf, checkpoint):
    monkeypatch.setenv("VISIN_MODE", "offline")
    run = visin.init("offline run")
    assert run.log_model(checkpoint, "acme/m") == COMMIT
    assert len(hf.uploads) == 1


class FakeTensor:
    def contiguous(self):
        return self


def install_torch(monkeypatch, state, saved):
    torch = types.ModuleType("torch")
    torch.Tensor = FakeTensor
    torch.load = lambda path, map_location=None, weights_only=None: state
    safetensors = types.ModuleType("safetensors")
    safetensors_torch = types.ModuleType("safetensors.torch")

    def save_file(tensors, filename):
        saved.append(sorted(tensors))
        Path(filename).write_bytes(b"safe")

    safetensors_torch.save_file = save_file
    safetensors.torch = safetensors_torch
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)


def uploaded(hf):
    return [upload["path_in_repo"] for upload in hf.uploads]


@pytest.mark.parametrize(
    "state",
    [
        {"w": FakeTensor(), "b": FakeTensor()},
        {"epoch": 3, "model_state_dict": {"w": FakeTensor(), "b": FakeTensor()}, "model_info": {"a": 1}},
        {"state_dict": {"w": FakeTensor(), "b": FakeTensor()}},
    ],
)
def test_a_plain_state_dict_is_also_uploaded_as_safetensors_beside_the_original(
    run, session, hf, checkpoint_file, monkeypatch, state
):
    saved = []
    install_torch(monkeypatch, state, saved)
    run.log_model(checkpoint_file, "acme/m", safetensors=True)
    assert uploaded(hf) == ["best.pt", "model.safetensors"]
    assert saved == [["b", "w"]]
    assert hf.uploads[1]["path_or_fileobj"].endswith("model.safetensors")


def test_the_safetensors_file_goes_into_the_same_folder_as_the_original(
    run, hf, checkpoint_file, monkeypatch
):
    install_torch(monkeypatch, {"w": FakeTensor()}, [])
    run.log_model(checkpoint_file, "acme/m", path_in_repo="runs/1/best.pt", safetensors=True)
    assert uploaded(hf) == ["runs/1/best.pt", "runs/1/model.safetensors"]


@pytest.mark.parametrize(
    "state", [{"epoch": 3}, {"model_state_dict": {"w": 1.0}}, {}, {"model_state_dict": {}}]
)
def test_a_checkpoint_that_is_not_a_plain_state_dict_is_uploaded_alone(
    run, hf, checkpoint_file, monkeypatch, state
):
    install_torch(monkeypatch, state, [])
    run.log_model(checkpoint_file, "acme/m", safetensors=True)
    assert uploaded(hf) == ["best.pt"]


def test_without_torch_or_safetensors_the_original_is_still_uploaded(run, hf, checkpoint_file, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    run.log_model(checkpoint_file, "acme/m", safetensors=True)
    assert uploaded(hf) == ["best.pt"]


def test_a_conversion_that_fails_leaves_the_upload_alone(run, hf, checkpoint_file, monkeypatch):
    install_torch(monkeypatch, {"w": FakeTensor()}, [])
    sys.modules["safetensors.torch"].save_file = lambda *_a: (_ for _ in ()).throw(
        ValueError("shared tensors")
    )
    assert run.log_model(checkpoint_file, "acme/m", safetensors=True) == COMMIT
    assert uploaded(hf) == ["best.pt"]


def test_safetensors_is_off_unless_asked_for_and_never_applies_to_a_folder(
    run, hf, checkpoint_file, tmp_path, monkeypatch
):
    install_torch(monkeypatch, {"w": FakeTensor()}, [])
    run.log_model(checkpoint_file, "acme/m")
    assert uploaded(hf) == ["best.pt"]
    folder = tmp_path / "export"
    folder.mkdir()
    (folder / "m.pt").write_bytes(b"x")
    hf.uploads.clear()
    run.log_model(folder, "acme/m", safetensors=True)
    assert [upload["kind"] for upload in hf.uploads] == ["folder"]


def test_safetensors_comes_before_the_model_card_so_the_last_commit_holds_both(
    run, session, hf, checkpoint_file, monkeypatch
):
    session.route("GET", "/trainings/t1/model-card", ok({"readme": CARD}))
    install_torch(monkeypatch, {"w": FakeTensor()}, [])
    run.log_model(checkpoint_file, "acme/m", safetensors=True)
    assert uploaded(hf) == ["best.pt", "model.safetensors", "README.md"]


@pytest.fixture
def checkpoint_file(checkpoint):
    return checkpoint


def test_a_failed_model_card_upload_still_links_the_uploaded_checkpoint(run, session, hf, checkpoint, caplog):
    card_route(session)
    hf.failure_on = "README.md"
    assert run.log_model(checkpoint, "acme/m") == COMMIT
    run.flush()
    assert uploaded(hf) == ["best.pt"]
    assert links(session)[0]["revision"] == COMMIT
    assert "model card could not be added" in caplog.text
