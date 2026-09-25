import pytest
from fakes import ok, refused

from visin._internal.reports import DeliveryContext, deliver, discard_staged, staged_path
from visin.errors import ApiError, ConfigurationError


@pytest.fixture
def context():
    return DeliveryContext()


def test_a_run_is_created_and_its_id_remembered(client, session, context):
    session.route("POST", "/trainings", ok({"_id": "t1"}, 201))
    deliver(client, {"op": "create_run", "body": {"uuid": "u1", "name": "x"}}, context)
    assert context.training_ids == {"u1": "t1"}


def test_a_run_that_exists_is_fetched_instead(client, session, context):
    session.route("POST", "/trainings", refused(409, "exists"))
    session.route("GET", "/trainings/uuid/u1", ok({"_id": "t1"}))
    assert deliver(client, {"op": "create_run", "body": {"uuid": "u1", "name": "x"}}, context)["_id"] == "t1"


def test_a_refused_run_is_an_error(client, session, context):
    session.route("POST", "/trainings", refused(403))
    with pytest.raises(ApiError):
        deliver(client, {"op": "create_run", "body": {"uuid": "u1", "name": "x"}}, context)


def test_a_repeated_test_result_counts_as_delivered(client, session, context):
    session.route("POST", "/test-results/upload", refused(409, "exists"))
    assert deliver(client, {"op": "test_result", "body": {}}, context) is None


def test_an_update_names_the_run_by_its_id(client, session, context):
    session.route("GET", "/trainings/uuid/u1", ok({"_id": "t1"}))
    deliver(client, {"op": "update", "training_uuid": "u1", "body": {"status": "failed"}}, context)
    assert session.bodies("/trainings/t1", "PUT") == [{"status": "failed"}]


def test_an_update_for_an_unknown_run_is_a_404(client, session, context):
    session.route("GET", "/trainings/uuid/u1", ok(None))
    session.default = ok(None)
    with pytest.raises(ApiError) as caught:
        deliver(client, {"op": "update", "training_uuid": "u1", "body": {}}, context)
    assert caught.value.status == 404


def test_a_config_without_a_run_is_only_stored(client, session, context):
    session.route("POST", "/configs/upload", ok({"_id": "c1"}))
    deliver(client, {"op": "config", "body": {"config_data": {}}}, context)
    assert session.paths("PUT") == []


def test_a_visualization_whose_file_is_gone_is_an_error(client, context, tmp_path):
    op = {"op": "visualization", "path": str(tmp_path / "gone.png"), "body": {"filename": "gone.png"}}
    with pytest.raises(ConfigurationError, match="gone"):
        deliver(client, op, context)


def test_an_incomplete_upload_grant_is_an_error(client, session, context, tmp_path):
    frame = tmp_path / "f.png"
    frame.write_bytes(b"x")
    session.route("POST", "/visualizations/upload-url", ok({"uploadUrl": "u"}))
    op = {
        "op": "visualization",
        "path": str(frame),
        "body": {"epoch_uuid": "e", "filename": "f.png", "type": "t", "mimetype": "image/png"},
    }
    with pytest.raises(ApiError, match="without"):
        deliver(client, op, context)


def test_an_unknown_kind_is_refused(client, context):
    with pytest.raises(ValueError, match="newer visin"):
        deliver(client, {"op": "teleport"}, context)


def test_staged_paths_resolve_inside_the_files_directory(tmp_path):
    context = DeliveryContext(files=tmp_path)
    assert staged_path({"staged": "a.png"}, context) == tmp_path / "a.png"
    assert staged_path({"path": "/abs/b.png"}, DeliveryContext()).name == "b.png"


def test_discarding_a_staged_file_that_is_already_gone_is_fine(tmp_path):
    discard_staged({"op": "visualization", "staged": "missing.png"}, DeliveryContext(files=tmp_path))
    discard_staged({"op": "epoch"}, DeliveryContext(files=tmp_path))
