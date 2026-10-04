import pytest
from fakes import ok

import visin
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
    session.route(
        "GET",
        "/evaluations",
        ok(
            {
                "evaluations": [
                    {"uuid": "a", "results": {"x": 1}, "source": {"epochUuid": "e1"}, "run": {"uuid": "u1"}}
                ]
            }
        ),
    )
    session.route("GET", "/benchmarks", ok({"benchmarks": [{"_id": "b", "results": [1]}]}))
    [result] = api.test_results(RUN)
    assert (result.test_uuid, result.results, result.epoch_uuid, result.training_uuid) == (
        "a",
        {"x": 1},
        "e1",
        "u1",
    )
    [bench] = api.benchmarks("u1", project="p")
    assert (bench.id, bench.results) == ("b", [1])
    assert session.calls[0]["params"]["trainingUuid"] == "u1"
    assert session.calls[0]["params"]["include"] == "results"
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


SUMMARY = {
    "training": {"_id": "t1", "uuid": "u1", "name": "n", "status": "completed", "datasetId": "zod"},
    "epochCount": 3,
    "lastEpoch": 3,
    "metrics": [
        {
            "path": "val.loss",
            "direction": "lower",
            "directionFrom": "default",
            "best": {"value": 0.2, "epoch": 3},
            "last": {"value": 0.2, "epoch": 3},
        },
        {
            "path": "val.mean_iou",
            "direction": "higher",
            "directionFrom": "taxonomy",
            "best": {"value": 0.7, "epoch": 2},
            "last": {"value": 0.6, "epoch": 3},
        },
    ],
    "models": [{"_id": "m1", "repo": "acme/m", "revision": "a" * 40}],
    "provenance": {"git": {"commit": "b" * 40, "dirty": False}},
}


def test_a_summary_gives_the_best_epoch_of_each_result_beside_the_last(api, session):
    session.route("GET", "/trainings/t1/summary", ok(SUMMARY))
    summary = api.summary(RUN)
    assert (summary.epoch_count, summary.last_epoch, summary.training.status) == (3, 3, "completed")
    iou = summary.metric("val.mean_iou")
    assert (iou.best_value, iou.best_epoch, iou.last_value, iou.last_epoch) == (0.7, 2, 0.6, 3)
    assert (iou.direction, iou.direction_from) == ("higher", "taxonomy")
    assert summary.metric("val.loss").direction == "lower"
    assert summary.metric("val.nothing") is None
    assert summary.models[0]["repo"] == "acme/m" and summary.provenance["git"]["commit"] == "b" * 40


def test_a_summary_is_found_by_uuid_and_of_a_run_that_reported_nothing(api, session):
    session.route("GET", "/trainings/uuid/u1", ok({"_id": "t1", "uuid": "u1", "name": "n"}))
    session.route(
        "GET", "/trainings/t1/summary", ok({"training": {"_id": "t1"}, "epochCount": 0, "lastEpoch": None})
    )
    summary = api.summary("u1")
    assert summary.last_epoch is None and summary.metrics == () and summary.provenance == {}


def test_a_run_carries_its_models_and_provenance(api, session):
    session.route(
        "GET",
        "/trainings/uuid/u1",
        ok(
            {
                "_id": "t1",
                "uuid": "u1",
                "name": "n",
                "models": [{"repo": "acme/m"}],
                "provenance": {"host": {}},
            }
        ),
    )
    run = api.training("u1")
    assert run.models == ({"repo": "acme/m"},) and run.provenance == {"host": {}}


def finding(number, created="2026-10-0%dT10:00:00.000Z"):
    return {
        "_id": f"{number:024x}",
        "projectId": "p1",
        "title": f"finding {number}",
        "body": "b",
        "createdAt": created % number,
        "trainingId": "t1",
        "trainingIds": ["t1", "t2"],
        "authorKind": "assistant",
        "authorLabel": "Claude",
        "recommendations": "next",
    }


def test_findings_are_read_newest_first_a_page_at_a_time_by_cursor(api, session, monkeypatch):
    monkeypatch.setattr("visin.api.FINDINGS_PAGE", 2)
    session.route(
        "GET", "/findings", ok([finding(5), finding(4)]), ok([finding(3), finding(2)]), ok([finding(1)])
    )
    found = list(api.findings(project="road-seg"))
    assert [f.title for f in found] == [f"finding {n}" for n in (5, 4, 3, 2, 1)]
    cursors = [call["params"].get("before") for call in session.calls]
    assert cursors == [None, f"2026-10-04T10:00:00.000Z_{4:024x}", f"2026-10-02T10:00:00.000Z_{2:024x}"]
    assert session.calls[0]["params"]["project"] == "road-seg"
    first = found[0]
    assert (first.author_kind, first.recommendations, first.training_ids) == (
        "assistant",
        "next",
        ("t1", "t2"),
    )


def test_findings_stop_at_the_limit_and_can_be_about_one_run(api, session):
    session.route("GET", "/trainings/uuid/u1", ok({"_id": "t1", "uuid": "u1", "name": "n"}))
    session.route("GET", "/findings", ok([finding(3), finding(2)]))
    assert [f.title for f in api.findings(run="u1", limit=1)] == ["finding 3"]
    assert session.calls[-1]["params"]["training"] == "t1" and session.calls[-1]["params"]["limit"] == 1


def test_comparisons_are_listed_a_page_at_a_time(api, session):
    item = {
        "_id": "c1",
        "uuid": "cu",
        "name": "A vs B",
        "type": "trainings",
        "itemIds": ["t1", "t2"],
        "projectId": "p1",
    }
    session.route(
        "GET",
        "/comparisons",
        page("comparisons", [item, {**item, "uuid": "cu2"}], 1, 2),
        page("comparisons", [{**item, "uuid": "cu3"}], 2, 2),
    )
    found = list(api.comparisons(project="road-seg", type="trainings", page_size=2))
    assert [c.uuid for c in found] == ["cu", "cu2", "cu3"]
    assert found[0].item_ids == ("t1", "t2") and found[0].type == "trainings"
    params = session.calls[0]["params"]
    assert params["projectId"] == "road-seg" and params["type"] == "trainings" and "search" not in params


def test_a_project_says_which_way_a_result_is_better_or_that_it_has_not():
    from visin.models import Project

    project = Project.from_json(
        {
            "_id": "p1",
            "name": "Road",
            "taxonomy": {
                "metrics": [
                    {"key": "mean_iou", "direction": "higher"},
                    {"key": "val.latency", "direction": "lower"},
                    {"key": "x"},
                ]
            },
        }
    )
    assert project.direction("val.mean_iou") == "higher" and project.direction("mean_iou") == "higher"
    assert project.direction("val.latency") == "lower" and project.direction("train.latency") is None
    assert project.direction("val.x") is None
    assert Project.from_json({"_id": "p2", "name": "Bare"}).direction("val.loss") is None


@pytest.mark.parametrize("limit", [0, -1])
def test_a_nonpositive_finding_limit_reads_nothing(api, session, limit):
    assert list(api.findings(run="u1", limit=limit)) == []
    assert session.calls == []


@pytest.mark.parametrize("full_first", [True, False])
def test_metric_direction_matches_platform_full_path_precedence(full_first):
    from visin.models import Project

    metrics = [{"key": "score", "direction": "higher"}, {"key": "val.score", "direction": "lower"}]
    project = Project.from_json(
        {"name": "Road", "taxonomy": {"metrics": metrics[::-1] if full_first else metrics}}
    )
    assert project.direction("val.score") == "lower"
    assert project.direction("train.score") == "higher"


SUITE_JSON = {
    "_id": "s1",
    "slug": "road-test",
    "version": 2,
    "name": "Road test",
    "projectId": "p1",
    "visibility": "public",
    "digest": "d" * 64,
    "protocol": {
        "metrics": [
            {"key": "mIoU", "direction": "max", "unit": "ratio", "headline": True},
            {"key": "ap", "direction": "max"},
        ]
    },
}


class TestSuitesAndEvaluations:
    def test_suites_are_paged_filtered_and_archived_ones_are_left_out_unless_asked(self, api, session):
        session.route("GET", "/suites", page("suites", [SUITE_JSON], 1, 1), page("suites", [], 1, 1))
        [suite] = api.suites(slug="road-test", project="road-seg")
        params = session.calls[0]["params"]
        assert (
            params["slug"] == "road-test"
            and params["projectId"] == "road-seg"
            and "includeArchived" not in params
        )
        assert (suite.ref, suite.visibility, suite.headline["key"], suite.archived_at) == (
            "road-test@2",
            "public",
            "mIoU",
            None,
        )
        api.suites(include_archived=True)
        assert session.calls[-1]["params"]["includeArchived"] == "true"

    @pytest.mark.parametrize(
        ("args", "path"),
        [
            (("road-test@2",), "/suites/road-test/2"),
            (("road-test",), "/suites/road-test/latest"),
            (("road-test", 3), "/suites/road-test/3"),
        ],
    )
    def test_a_suite_is_read_by_ref_by_slug_alone_or_by_slug_and_version(self, api, session, args, path):
        session.route("GET", path, ok(SUITE_JSON))
        assert api.suite(*args).slug == "road-test"
        assert session.paths("GET") == [path]

    def test_a_suite_without_a_headline_metric_has_an_empty_one(self):
        assert (
            visin.Suite.from_json(
                {"slug": "s", "version": 1, "name": "n", "protocol": {"metrics": [{"key": "x"}]}}
            ).headline
            == {}
        )

    def test_evaluations_are_paged_and_filtered_and_read_as_python_names(self, api, session):
        item = {
            "_id": "e1",
            "uuid": "u1",
            "projectId": "p1",
            "checkpoint": {"kind": "local", "label": "x"},
            "suite": {"slug": "road-test", "version": 1, "digest": "d"},
            "sampleCounts": {"day": 3},
            "publishedAt": "2026-10-02T09:00:00Z",
            "validation": {"state": "eligible", "reasons": [], "warnings": []},
        }
        session.route("GET", "/evaluations", page("evaluations", [item], 1, 1))
        [evaluation] = api.evaluations(
            project="road-seg", suite="road-test@1", state="eligible", status="completed", checkpoint_key="k"
        )
        params = session.calls[0]["params"]
        assert (
            params["projectId"],
            params["suite"],
            params["state"],
            params["status"],
            params["checkpointKey"],
        ) == ("road-seg", "road-test@1", "eligible", "completed", "k")
        assert (
            evaluation.id,
            evaluation.suite,
            evaluation.sample_counts,
            evaluation.published,
            evaluation.ranked,
        ) == ("e1", "road-test@1", {"day": 3}, True, True)

    def test_one_evaluation_by_id_by_object_or_by_the_uuid_the_writer_gave_it(self, api, session):
        session.route("GET", "/evaluations/e1", ok({"_id": "e1", "results": {"day": {}}}), ok({"_id": "e1"}))
        assert api.evaluation("e1").results == {"day": {}}
        assert api.evaluation(visin.Evaluation(id="e1")).id == "e1"
        session.route("GET", "/evaluations/uuid/mine", ok({"_id": "e2", "uuid": "mine"}))
        assert api.evaluation("mine", project="road-seg").uuid == "mine"
        assert session.calls[-1]["params"] == {"projectId": "road-seg"}

    def test_the_ranking_reads_with_its_scope_and_what_could_not_be_ranked(self, api, session):
        board = {
            "suite": {"slug": "road-test", "version": 1, "headline": {"key": "mIoU", "direction": "max"}},
            "scope": {"candidates": 4},
            "entries": [
                {
                    "rank": 1,
                    "evaluationId": "e1",
                    "attempts": 2,
                    "checkpoint": {"kind": "hf", "repo": "acme/clft", "commit": "3f2a1c9d8e"},
                    "project": {"name": "Road"},
                    "summary": {
                        "headline": {"value": 0.74},
                        "worst": {"condition": "night", "value": 0.7},
                        "gap": 0.04,
                    },
                }
            ],
            "unranked": [
                {
                    "checkpointKey": "k",
                    "state": "incomplete",
                    "reasons": [{"code": "missing-condition", "detail": "rain"}],
                }
            ],
        }
        session.route("GET", "/suites/road-test/1/leaderboard", ok(board))
        result = api.leaderboard("road-test@1")
        [entry] = result.entries
        assert (result.suite, result.headline_key, result.direction, result.candidates, result.public) == (
            "road-test@1",
            "mIoU",
            "max",
            4,
            False,
        )
        assert (
            entry.rank,
            entry.name,
            entry.headline,
            entry.worst_condition,
            entry.worst_value,
            entry.gap,
            entry.attempts,
            entry.project,
        ) == (1, "acme/clft @ 3f2a1c9", 0.74, "night", 0.7, 0.04, 2, "Road")
        assert result.unranked[0]["state"] == "incomplete"

    def test_the_public_ranking_has_the_flat_shape_and_asks_without_a_credential(self, api, session):
        board = {
            "suite": {"slug": "road-test", "version": 1, "headline": {"key": "mIoU", "direction": "min"}},
            "scope": {"candidates": 1},
            "entries": [
                {
                    "rank": 1,
                    "evaluationId": "e1",
                    "checkpoint": {"kind": "local", "label": "Model B"},
                    "headline": 0.5,
                    "worst": {"condition": "day", "value": 0.6},
                    "gap": 0.1,
                }
            ],
        }
        session.route("GET", "/public/leaderboards/road-test/1", ok(board))
        result = api.public_leaderboard("road-test@1")
        assert result.public and result.entries[0].name == "Model B" and result.entries[0].headline == 0.5
        assert result.entries[0].attempts == 0 and result.entries[0].project is None and result.unranked == ()

    def test_a_public_leaderboard_needs_a_version_because_a_link_should_keep_meaning_one_protocol(self, api):
        with pytest.raises(ValueError, match="version"):
            api.public_leaderboard("road-test")

    def test_the_public_list_is_read_anonymously(self, api, session):
        session.route(
            "GET", "/public/leaderboards", ok({"leaderboards": [{"slug": "road-test", "version": 1}]})
        )
        assert api.public_leaderboards() == [{"slug": "road-test", "version": 1}]

    @staticmethod
    def ranking_page(number, *, selected=230, unranked=130, size=100, public=False):
        start = (number - 1) * size
        entries = [
            {
                "rank": index // 2 + 1,
                "evaluationId": f"e{index}",
                "attempts": 3,
                "checkpoint": {"kind": "local", "label": f"m{index}"},
                "summary": {
                    "headline": {"value": 1 - index / 1000},
                    "worst": {"condition": "d", "value": 0.1},
                    "gap": 0.1,
                },
            }
            for index in range(start, min(start + size, selected))
        ]
        body = {
            "suite": {"slug": "road-test", "version": 1, "headline": {"key": "mIoU", "direction": "max"}},
            "scope": {"candidates": selected * 3},
            "entries": entries,
            "pagination": {"page": number, "limit": size, "total": selected, "pages": -(-selected // size)},
        }
        if not public:
            body["unranked"] = [
                {"evaluationId": f"u{i}", "checkpointKey": f"k{i}", "state": "incomplete"}
                for i in range(unranked)
            ][:size]
            body["unrankedPagination"] = {
                "page": 1,
                "limit": size,
                "total": unranked,
                "pages": -(-unranked // size),
            }
        return body

    def test_a_page_of_the_ranking_keeps_global_ranks_and_says_where_it_sits(self, api, session):
        session.route("GET", "/suites/road-test/1/leaderboard", ok(self.ranking_page(3)))
        result = api.leaderboard("road-test@1", page=3, limit=100, unranked_page=2)
        assert session.calls[-1]["params"] == {"page": 3, "limit": 100, "unrankedPage": 2}
        assert result.pagination == visin.Pagination(page=3, limit=100, total=230, pages=3)
        assert result.unranked_pagination == visin.Pagination(page=1, limit=100, total=130, pages=2)
        assert (result.candidates, len(result.entries), result.entries[0].rank, result.complete) == (
            690,
            30,
            101,
            False,
        )

    def test_observed_asks_for_the_observed_results_alone_on_every_way_of_reading_a_board(self, api, session):
        session.route(
            "GET", "/suites/road-test/1/leaderboard", ok(self.ranking_page(1, selected=3, unranked=0))
        )
        api.leaderboard("road-test@1", observed=True)
        assert session.calls[-1]["params"] == {"evidence": "observed"}
        session.route(
            "GET", "/suites/road-test/1/leaderboard", ok(self.ranking_page(1, selected=3, unranked=0))
        )
        api.leaderboard("road-test@1", observed=True, all_pages=True)
        assert session.calls[-1]["params"]["evidence"] == "observed"
        session.route(
            "GET", "/public/leaderboards/road-test/1", ok(self.ranking_page(1, selected=3, public=True))
        )
        api.public_leaderboard("road-test@1", observed=True)
        assert session.calls[-1]["params"] == {"evidence": "observed"}

    def test_every_page_of_an_observed_board_keeps_asking_for_the_observed_results(self, api, session):
        session.route(
            "GET",
            "/suites/road-test/1/leaderboard",
            *[ok(self.ranking_page(n)) for n in (1, 2, 3)],
            ok(self.ranking_page(4)),
        )
        api.leaderboard("road-test@1", observed=True, all_pages=True)
        assert [call["params"].get("evidence") for call in session.calls] == ["observed"] * 4

    def test_asking_for_no_page_sends_no_paging_parameters(self, api, session):
        session.route("GET", "/suites/road-test/1/leaderboard", ok(self.ranking_page(1)))
        api.leaderboard("road-test@1")
        assert session.calls[-1]["params"] == {}

    def test_every_page_of_more_than_a_hundred_checkpoints_comes_back_together(self, api, session):
        session.route(
            "GET",
            "/suites/road-test/1/leaderboard",
            *[ok(self.ranking_page(n)) for n in (1, 2, 3)],
            ok(self.ranking_page(4, selected=230, unranked=130)),
        )
        result = api.leaderboard("road-test@1", all_pages=True)
        assert [call["params"] for call in session.calls] == [
            {"page": 1, "limit": 100},
            {"page": 2, "limit": 100},
            {"page": 3, "limit": 100},
            {"page": 4, "limit": 100, "unrankedPage": 2},
        ]
        assert result.complete and len(result.entries) == 230 and len(result.unranked) == 100
        assert [entry.rank for entry in result.entries[:4]] == [1, 1, 2, 2] and result.entries[-1].rank == 115
        assert len({entry.evaluation_id for entry in result.entries}) == 230
        assert (
            len(result.raw["entries"]) == 230 and result.candidates == 690 and result.pagination.total == 230
        )

    def test_a_row_that_moved_across_a_page_boundary_is_not_counted_twice(self, api, session):
        first = self.ranking_page(1, selected=150, unranked=0)
        second = self.ranking_page(2, selected=150, unranked=0)
        second["entries"].insert(0, first["entries"][-1])
        session.route("GET", "/suites/road-test/1/leaderboard", ok(first), ok(second))
        assert len(api.leaderboard("road-test@1", all_pages=True).entries) == 100 + 50

    def test_all_pages_refuses_to_be_mixed_with_one_page(self, api):
        with pytest.raises(ValueError, match="every page"):
            api.leaderboard("road-test@1", all_pages=True, page=2)
        with pytest.raises(ValueError, match="every page"):
            api.leaderboard("road-test@1", all_pages=True, unranked_page=2)
        with pytest.raises(ValueError, match="every page"):
            api.public_leaderboard("road-test@1", all_pages=True, page=1)

    def test_the_public_ranking_pages_anonymously_and_reads_all_pages(self, api, session):
        session.route("GET", "/public/leaderboards/road-test/1", ok(self.ranking_page(2, public=True)))
        one = api.public_leaderboard("road-test@1", page=2, limit=50)
        assert session.calls[-1]["params"] == {"page": 2, "limit": 50}
        assert one.public and one.unranked_pagination is None
        session.route(
            "GET",
            "/public/leaderboards/road-test/1",
            *[ok(self.ranking_page(n, public=True)) for n in (1, 2, 3)],
        )
        result = api.public_leaderboard("road-test@1", all_pages=True)
        assert result.public and result.complete and len(result.entries) == 230 and result.unranked == ()

    def test_public_discovery_pages_and_iterates_every_page(self, api, session):
        def listing(number, count):
            items = [{"slug": f"s{number}-{i}", "version": 1} for i in range(count)]
            return ok(
                {
                    "leaderboards": items,
                    "pagination": {"page": number, "limit": 100, "total": 250, "pages": 3},
                }
            )

        session.route("GET", "/public/leaderboards", listing(2, 3))
        assert len(api.public_leaderboards(page=2, limit=3)) == 3
        assert session.calls[-1]["params"] == {"page": 2, "limit": 3}
        session.route("GET", "/public/leaderboards", listing(1, 100), listing(2, 100), listing(3, 50))
        everything = list(api.iter_public_leaderboards())
        assert len(everything) == 250 and everything[-1]["slug"] == "s3-49"
        assert [call["params"] for call in session.calls[-3:]] == [
            {"page": 1, "limit": 100},
            {"page": 2, "limit": 100},
            {"page": 3, "limit": 100},
        ]

    def test_an_unnamed_checkpoint_reads_as_unknown(self):
        entry = visin.LeaderboardEntry.from_json(
            {"rank": 1, "evaluationId": "e", "headline": 1, "worst": {}, "gap": 0}
        )
        assert entry.name == "unknown"
