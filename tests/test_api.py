import pytest
from fakes import ok

from visin import Api, Training, flatten
from visin.errors import ConfigurationError


@pytest.fixture
def api(server):
    return Api()


def page(key, items, page_number, pages):
    return ok({key: items, "pagination": {"page": page_number, "limit": 2, "total": 99, "pages": pages}})


def test_an_api_needs_a_url():
    with pytest.raises(ConfigurationError, match=r"VISIN_URL.*vision-api\.visin\.eu"):
        Api()


def test_trainings_are_fetched_a_page_at_a_time(api, session):
    session.route(
        "GET",
        "/trainings",
        page("trainings", [{"name": "a"}, {"name": "b"}], 1, 2),
        page("trainings", [{"name": "c"}], 2, 2),
    )
    names = [t.name for t in api.trainings(project="road-seg", tags=["x", "y"], page_size=2)]
    assert names == ["a", "b", "c"]
    first = session.calls[0]["params"]
    assert first["projectId"] == "road-seg" and first["tags"] == "x,y" and first["page"] == 1


def test_limit_stops_early(api, session):
    session.route("GET", "/trainings", page("trainings", [{"name": "a"}, {"name": "b"}], 1, 5))
    assert [t.name for t in api.trainings(limit=1)] == ["a"]
    assert len(session.calls) == 1


def test_a_run_is_found_by_uuid(api, session):
    session.route("GET", "/trainings/uuid/my-uuid", ok({"_id": "t1", "uuid": "my-uuid", "projectId": "p"}))
    run = api.training("my-uuid")
    assert (run.id, run.uuid, run.project_id) == ("t1", "my-uuid", "p")
    assert api.training(run) is run


def test_a_run_reads_as_python_names(api, session):
    session.route(
        "GET",
        "/trainings/uuid/u",
        ok({"_id": "t", "uuid": "u", "name": "n", "tags": ["a"], "updatedAt": "2026-01-01", "x": 1}),
    )
    run = api.training("u")
    assert run.tags == ("a",) and run.updated_at == "2026-01-01" and run.raw["x"] == 1


def test_sort_keys_are_python_names(api, session):
    session.route("GET", "/trainings", page("trainings", [], 1, 1))
    list(api.trainings(sort_by="created_at"))
    assert session.calls[0]["params"]["sortBy"] == "createdAt"
    with pytest.raises(ValueError, match="sort_by"):
        list(api.trainings(sort_by="createdAt"))


RUN = Training(id="t1", uuid="u1", name="n")


def test_epochs_of_a_run(api, session):
    session.route("GET", "/epochs/training/t1", ok({"epochs": [{"epoch": 1}, {"epoch": 2}]}))
    assert [e.epoch for e in api.epochs(RUN)] == [1, 2]


def test_test_results_and_benchmarks_filter_by_the_runs_uuid(api, session):
    session.route("GET", "/test-results", ok({"testResults": [{"test_uuid": "a", "test_results": {"x": 1}}]}))
    session.route("GET", "/benchmarks", ok({"benchmarks": [{"_id": "b", "results": [1]}]}))
    [result] = api.test_results(RUN)
    assert (result.test_uuid, result.results) == ("a", {"x": 1})
    [bench] = api.benchmarks("u1", project="p")
    assert (bench.id, bench.results) == ("b", [1])
    assert session.calls[0]["params"]["training_uuid"] == "u1"
    assert session.calls[1]["params"]["projectId"] == "p"


def test_projects(api, session):
    session.route("GET", "/projects/road-seg", ok({"_id": "p", "name": "Road"}))
    session.route("GET", "/projects", ok([{"_id": "p", "name": "Road", "slug": "road-seg"}]))
    assert api.project("road-seg").name == "Road"
    assert [p.slug for p in api.projects()] == ["road-seg"]


def test_the_api_closes_as_a_context_manager(server):
    with Api() as api:
        assert api._get("/anything") == {}


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
    frame = api.epochs_frame(RUN)
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
        api.epochs_frame(RUN)


def test_a_model_survives_a_server_that_omits_fields():
    from visin.models import Benchmark, Dataset, Epoch, Project, TestResult

    assert Training.from_json({}) == Training(id="", uuid="", name="")
    assert Epoch.from_json({}).epoch == 0
    assert TestResult.from_json({}).results == {}
    assert Benchmark.from_json({"_id": "b"}).system_info == {}
    assert Project.from_json({"_id": "p", "name": "P"}).slug is None
    assert not Dataset.from_json({"_id": "d", "name": "D"}).downloadable


def test_a_model_reads_either_spelling_of_a_field():
    from visin.models import TestResult

    snake = TestResult.from_json({"epoch_uuid": "e", "training_uuid": "t", "test_uuid": "x"})
    camel = TestResult.from_json({"epochUuid": "e", "trainingUuid": "t", "testUuid": "x"})
    assert snake == camel


def test_find_returns_the_newest_run_with_exactly_that_name(api, session):
    session.route(
        "GET",
        "/trainings",
        page(
            "trainings",
            [{"name": "unet baseline v2", "uuid": "a"}, {"name": "unet baseline", "uuid": "b"}],
            1,
            1,
        ),
    )
    assert api.find("unet baseline", project="road-seg").uuid == "b"
    assert session.calls[0]["params"]["search"] == "unet baseline"
    assert session.calls[0]["params"]["projectId"] == "road-seg"


def test_find_is_none_when_nothing_matches(api, session):
    session.route("GET", "/trainings", page("trainings", [{"name": "other"}], 1, 1))
    assert api.find("missing") is None


def test_find_without_a_name_is_the_newest_run_that_matches_the_filters(api, session):
    session.route("GET", "/trainings", page("trainings", [{"name": "a", "uuid": "u"}], 1, 1))
    assert api.find(tags=["ablation"]).uuid == "u"


def test_tags(api, session):
    session.route("GET", "/trainings/tags", ok(["a", "b"]))
    assert api.tags() == ["a", "b"]


def test_config_is_what_the_run_logged(api, session):
    session.route(
        "GET", "/trainings/t1/configs", ok({"configs": [{"_id": "c1", "config": {"lr": 0.1}}], "total": 1})
    )
    config = api.config(RUN)
    assert config.config == {"lr": 0.1} and config.id == "c1"


def test_config_is_none_when_the_run_logged_none(api, session):
    session.route("GET", "/trainings/t1/configs", ok({"configs": [], "total": 0}))
    assert api.config(RUN) is None


def test_visualizations_come_with_signed_links_and_can_be_downloaded(api, session, uploads, tmp_path):
    items = [
        {
            "visualization_uuid": "v1",
            "type": "overlay",
            "filename": "a.png",
            "signedUrl": "https://files.example.test/a",
        }
    ]
    session.route("GET", "/visualizations/training/u1", page("visualizations", items, 1, 1))
    uploads.files["https://files.example.test/a"] = b"\x89PNG"
    (frame,) = api.visualizations(RUN, kind="overlay")
    assert (frame.kind, frame.filename) == ("overlay", "a.png")
    assert session.calls[0]["params"]["type"] == "overlay"
    assert session.calls[0]["params"]["includeUrls"] == "true"
    assert api.download_visualization(frame, tmp_path / "frames").read_bytes() == b"\x89PNG"


def test_a_visualization_without_a_link_cannot_be_downloaded(api, tmp_path):
    from visin import Visualization

    with pytest.raises(ConfigurationError, match="no link"):
        api.download_visualization(Visualization(uuid="v1", filename="a.png"), tmp_path)


def test_compare_frame_puts_runs_side_by_side(api, session):
    pytest.importorskip("pandas")
    session.route("GET", "/trainings/uuid/a", ok({"_id": "ta", "uuid": "a", "name": "one"}))
    session.route("GET", "/trainings/uuid/b", ok({"_id": "tb", "uuid": "b", "name": "two"}))
    session.route(
        "GET", "/epochs/training/ta", ok({"epochs": [{"epoch": 1, "results": {"val": {"iou": 0.1}}}]})
    )
    session.route(
        "GET",
        "/epochs/training/tb",
        ok(
            {
                "epochs": [
                    {"epoch": 1, "results": {"val": {"iou": 0.2}}},
                    {"epoch": 2, "results": {"val": {"iou": 0.4}}},
                ]
            }
        ),
    )
    frame = api.compare_frame(["a", "b"], "val.iou")
    assert list(frame.columns) == ["one", "two"]
    assert frame.loc[2, "two"] == 0.4
    assert frame["one"].isna().loc[2]
