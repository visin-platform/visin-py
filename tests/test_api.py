import pytest
from fakes import ok

from visin import Api, flatten
from visin.errors import ConfigurationError


@pytest.fixture
def api(server):
    return Api()


def page(key, items, page_number, pages):
    return ok({key: items, "pagination": {"page": page_number, "limit": 2, "total": 99, "pages": pages}})


def test_an_api_needs_a_url():
    with pytest.raises(ConfigurationError):
        Api()


def test_trainings_are_fetched_a_page_at_a_time(api, session):
    session.route(
        "GET",
        "/trainings",
        page("trainings", [{"name": "a"}, {"name": "b"}], 1, 2),
        page("trainings", [{"name": "c"}], 2, 2),
    )
    names = [t["name"] for t in api.trainings(project="road-seg", tags=["x", "y"], page_size=2)]
    assert names == ["a", "b", "c"]
    first = session.calls[0]["params"]
    assert first["projectId"] == "road-seg" and first["tags"] == "x,y" and first["page"] == 1


def test_limit_stops_early(api, session):
    session.route("GET", "/trainings", page("trainings", [{"name": "a"}, {"name": "b"}], 1, 5))
    assert [t["name"] for t in api.trainings(limit=1)] == ["a"]
    assert len(session.calls) == 1


def test_a_run_is_found_by_id_or_uuid(api, session):
    session.route("GET", "/trainings/0123456789abcdef01234567", ok({"_id": "0123456789abcdef01234567"}))
    session.route("GET", "/trainings/uuid/my-uuid", ok({"uuid": "my-uuid"}))
    assert api.training("0123456789abcdef01234567")["_id"] == "0123456789abcdef01234567"
    assert api.training("my-uuid")["uuid"] == "my-uuid"
    assert api.training({"_id": "x"}) == {"_id": "x"}


def test_epochs_of_a_run(api, session):
    session.route("GET", "/epochs/training/t1", ok({"epochs": [{"epoch": 1}, {"epoch": 2}]}))
    assert [e["epoch"] for e in api.epochs({"_id": "t1", "uuid": "u1"})] == [1, 2]


def test_test_results_and_benchmarks_filter_by_the_runs_uuid(api, session):
    session.route("GET", "/test-results", ok({"testResults": [{"test_uuid": "a"}]}))
    session.route("GET", "/benchmarks", ok({"benchmarks": [{"_id": "b"}]}))
    run = {"_id": "t1", "uuid": "u1"}
    assert api.test_results(run) == [{"test_uuid": "a"}]
    assert api.benchmarks(run, project="p") == [{"_id": "b"}]
    assert session.calls[0]["params"]["training_uuid"] == "u1"
    assert session.calls[1]["params"]["projectId"] == "p"


def test_projects(api, session):
    session.route("GET", "/projects/road-seg", ok({"name": "Road"}))
    session.route("GET", "/projects", ok([{"name": "Road"}]))
    assert api.project("road-seg") == {"name": "Road"}
    assert api.projects() == [{"name": "Road"}]


def test_the_api_closes_as_a_context_manager(server):
    with Api() as api:
        assert api.get("/anything") == {}


def test_flatten():
    nested = {"train": {"loss": 1}, "val": {"car": {"iou": 0.5}}, "tags": [1, 2], "empty": {}}
    assert flatten(nested) == {"train.loss": 1, "val.car.iou": 0.5, "tags": [1, 2], "empty": {}}
    assert flatten({"a": {"b": 1}}, sep="/") == {"a/b": 1}


def test_epochs_frame(api, session):
    pd = pytest.importorskip("pandas")
    session.route(
        "GET",
        "/epochs/training/t1",
        ok(
            {
                "epochs": [
                    {"epoch": 1, "learning_rate": 0.1, "results": {"val": {"loss": 1.0}}},
                    {"epoch": 2, "results": {"val": {"loss": 0.5, "iou": 0.6}}},
                ]
            }
        ),
    )
    frame = api.epochs_frame({"_id": "t1", "uuid": "u1"})
    assert list(frame.index) == [1, 2]
    assert frame.loc[2, "val.iou"] == 0.6
    assert pd.isna(frame.loc[1, "val.iou"])


def test_epochs_frame_without_pandas_says_how_to_get_it(api, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_pandas(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("no pandas")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pandas)
    with pytest.raises(ImportError, match=r"visin\[pandas\]"):
        api.epochs_frame({"_id": "t1", "uuid": "u1"})
