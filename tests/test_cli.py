import json
import os
from typing import ClassVar

import pytest
from fakes import BASE, ok, refused

import visin
from visin._internal.config import HOSTED_DATASET_URL, HOSTED_URL, config_path, read_settings
from visin.cli import main
from visin.errors import TransportError


def run_cli(capsys, *argv):
    code = main(list(argv))
    return code, capsys.readouterr().out


def test_version(capsys):
    assert run_cli(capsys, "version") == (0, f"visin {visin.__version__}\n")


def test_check_without_a_url_fails(capsys):
    code, out = run_cli(capsys, "check")
    assert code == 1 and "VISIN_URL is not set" in out


def test_check_passes_with_a_good_key(server, capsys, monkeypatch):
    monkeypatch.setenv("VISIN_TOKEN", "vsn_live_abcdefghijklmnop")
    code, out = run_cli(capsys, "check")
    assert code == 0, out
    assert "API key is accepted" in out
    assert "if it is a pipeline key" in out
    assert "vsn_…mnop" in out


def test_check_catches_a_token_the_public_reads_would_let_through(server, session, capsys):
    session.route("GET", "/trainings/deleted", refused(401, "Invalid token"))
    code, out = run_cli(capsys, "check")
    assert code == 1 and "Invalid token" in out


def test_check_catches_an_unreachable_server(server, session, capsys):
    server.retries = 0
    session.route("GET", "/trainings", TransportError("no route to host"))
    code, out = run_cli(capsys, "check")
    assert code == 1 and "could not reach" in out


def test_check_points_a_token_that_is_not_a_key_at_pipeline_keys(server, capsys):
    code, out = run_cli(capsys, "check")
    assert code == 0 and "unrecognised token" in out
    assert "create a pipeline key" in out


def test_check_catches_a_project_the_token_cannot_see(server, session, capsys):
    session.route("GET", "/projects/other", refused(403, "Forbidden"))
    code, out = run_cli(capsys, "check", "--project", "other")
    assert code == 1 and "sees only its own project" in out


def test_check_write_catches_a_run_that_lands_in_another_project(server, session, capsys):
    session.route("GET", "/projects/other", ok({"_id": "p-other", "name": "Other"}))
    session.route("POST", "/trainings", ok({"_id": "t1", "projectId": "p-token"}, 201))
    code, out = run_cli(capsys, "check", "--project", "other", "--write")
    assert code == 1 and "went to project p-token" in out
    assert "/trainings/t1" in session.paths("DELETE")


def test_check_write_creates_and_deletes_a_run(server, session, capsys):
    session.route("GET", "/projects/road-seg", ok({"name": "Road"}))
    session.route("POST", "/trainings", ok({"_id": "t1"}, 201))
    code, out = run_cli(capsys, "check", "--project", "road-seg", "--write")
    assert code == 0, out
    assert session.bodies("/trainings")[0]["projectId"] == "road-seg"
    assert "/trainings/t1" in session.paths("DELETE")


def test_check_write_reports_a_refused_run(server, session, capsys):
    session.route("POST", "/trainings", refused(403, "Access denied to project"))
    code, out = run_cli(capsys, "check", "--project", "p", "--write")
    assert code == 1 and "Access denied" in out


def test_sync_with_nothing_waiting(server, capsys):
    code, out = run_cli(capsys, "sync")
    assert code == 0 and "nothing waiting" in out


def test_sync_lists_and_sends(server, session, capsys, monkeypatch):
    session.route("GET", "/trainings/uuid/", ok({"_id": "t1"}))
    monkeypatch.setenv("VISIN_MODE", "offline")
    run = visin.init("offline")
    run.finish()
    monkeypatch.delenv("VISIN_MODE")
    code, out = run_cli(capsys, "sync", "--list")
    assert code == 0 and f"{run.training_uuid}  2 reports" in out
    code, out = run_cli(capsys, "sync")
    assert code == 0 and "sent 2" in out


def test_sync_reports_refusals(server, session, capsys, monkeypatch):
    monkeypatch.setenv("VISIN_MODE", "offline")
    visin.init("offline").finish()
    monkeypatch.delenv("VISIN_MODE")
    session.route("POST", "/trainings", refused(403, "Access denied"))
    code, out = run_cli(capsys, "sync")
    assert code == 1 and "refused: create_run" in out


def test_runs_lists_recent_runs(server, session, capsys):
    session.route(
        "GET",
        "/trainings",
        ok(
            {
                "trainings": [
                    {
                        "uuid": "u1",
                        "status": "completed",
                        "name": "baseline",
                        "updatedAt": "2026-09-01T10:00:00Z",
                    }
                ]
            }
        ),
    )
    code, out = run_cli(capsys, "runs", "--limit", "5")
    assert code == 0 and "baseline" in out and "2026-09-01 10:00" in out


def test_a_command_is_required(capsys):
    with pytest.raises(SystemExit):
        main([])


def test_check_write_requires_a_project_for_an_unlimited_key(server, session, capsys, monkeypatch):
    monkeypatch.setenv("VISIN_TOKEN", "vsn_live_unlimited")
    session.route("POST", "/trainings", refused(400, "A training needs a project: pass projectId"))
    code, out = run_cli(capsys, "check", "--write")
    assert code == 1 and "--project" in out and "VISIN_PROJECT" in out
    assert session.paths("DELETE") == []


def test_check_write_accepts_a_pipeline_key_without_a_project(server, session, capsys, monkeypatch):
    monkeypatch.setenv("VISIN_TOKEN", "vsn_live_pipeline")
    session.route("POST", "/trainings", ok({"_id": "t1", "projectId": "p1"}, 201))
    code, out = run_cli(capsys, "check", "--write")
    assert code == 0, out
    assert "/trainings/t1" in session.paths("DELETE")


def test_login_saves_a_checked_token_and_later_runs_need_no_exports(server, capsys, monkeypatch):
    monkeypatch.delenv("VISIN_URL")
    monkeypatch.delenv("VISIN_TOKEN")
    code, out = run_cli(capsys, "login", "--url", "https://v.test", "--token", "vsn_live_abcdefghijklmnop")
    assert code == 0, out
    settings = read_settings()
    assert (settings.url, settings.token) == ("https://v.test", "vsn_live_abcdefghijklmnop")
    if os.name == "posix":
        assert oct(config_path().stat().st_mode & 0o777) == "0o600"


def test_login_does_not_save_a_token_the_server_refuses(server, session, capsys):
    session.route("GET", "/trainings/deleted", refused(401, "Invalid token"))
    code, out = run_cli(capsys, "login", "--url", "https://v.test", "--token", "bad")
    assert code == 1 and "not saved" in out
    assert not config_path().exists()


def test_login_for_the_hosted_visin_also_saves_the_dataset_address(server, capsys):
    run_cli(capsys, "login", "--url", HOSTED_URL, "--token", "vsn_live_abcdefghijklmnop", "--no-check")
    assert read_settings().dataset_url == HOSTED_DATASET_URL


def test_login_keeps_other_lines_in_the_file(capsys):
    config_path().parent.mkdir(parents=True)
    config_path().write_text("# mine\nVISIN_MODE=offline\nVISIN_URL=https://old.test\n")
    run_cli(capsys, "login", "--url", "https://new.test", "--token", "t", "--no-check")
    assert (
        config_path().read_text() == "# mine\nVISIN_MODE=offline\nVISIN_URL=https://new.test\nVISIN_TOKEN=t\n"
    )


def test_login_without_a_token_outside_a_terminal_says_what_to_pass(capsys):
    code = main(["login"])
    assert code == 1 and "--token" in capsys.readouterr().err


def test_logout_removes_the_saved_file(capsys):
    run_cli(capsys, "login", "--url", "https://v.test", "--token", "t", "--no-check")
    code, _ = run_cli(capsys, "logout")
    assert code == 0 and not config_path().exists()
    assert run_cli(capsys, "logout")[0] == 0


def test_check_says_when_a_value_came_from_the_config_file(server, capsys, monkeypatch):
    monkeypatch.delenv("VISIN_URL")
    config_path().parent.mkdir(parents=True)
    config_path().write_text(f"VISIN_URL={BASE}\n")
    _, out = run_cli(capsys, "check")
    assert f"[{config_path()}]" in out


def test_runs_can_be_printed_as_json_with_a_link_to_each(server, session, capsys, monkeypatch):
    monkeypatch.setenv("VISIN_URL", HOSTED_URL)
    session.route("GET", "/trainings", ok({"trainings": [{"_id": "t1", "uuid": "u1", "name": "baseline"}]}))
    code, out = run_cli(capsys, "runs", "--json")
    (run,) = json.loads(out)
    assert code == 0
    assert (run["uuid"], run["name"], run["url"]) == ("u1", "baseline", "https://app.visin.eu/trainings/t1")
    assert "raw" not in run


def test_sync_list_can_be_printed_as_json(server, session, capsys, monkeypatch):
    assert json.loads(run_cli(capsys, "sync", "--list", "--json")[1]) == {}
    monkeypatch.setenv("VISIN_MODE", "offline")
    run = visin.init("offline")
    run.finish()
    monkeypatch.delenv("VISIN_MODE")
    assert json.loads(run_cli(capsys, "sync", "--list", "--json")[1]) == {run.training_uuid: 2}


def test_datasets_can_be_listed_as_json(client, session, capsys, monkeypatch):
    monkeypatch.setenv("VISIN_DATASET_URL", BASE)
    monkeypatch.setattr("visin.datasets.HttpClient", lambda *_args, **_kwargs: client)
    session.route(
        "GET", "/datasets", ok([{"_id": "d1", "name": "ZOD", "archive": {"size": 5, "filename": "z.zip"}}])
    )
    code, out = run_cli(capsys, "datasets", "--json")
    (listed,) = json.loads(out)
    assert code == 0 and listed["name"] == "ZOD" and listed["downloadable"] is True


def test_cache_lists_and_removes_downloads(tmp_path, capsys):
    data = tmp_path / "data"
    folder = data / "zod-6aad"
    folder.mkdir(parents=True)
    (folder / ".visin-dataset.json").write_text(json.dumps({"id": "6aad", "name": "ZOD", "size": 3}))
    (folder / "file.bin").write_bytes(b"x" * 2048)
    code, out = run_cli(capsys, "cache", "--dir", str(data))
    assert code == 0 and "ZOD" in out and "dataset" in out
    listed = json.loads(run_cli(capsys, "cache", "--dir", str(data), "--json")[1])
    assert listed[0]["id"] == "6aad"
    code, out = run_cli(capsys, "cache", "rm", "zod", "--dir", str(data))
    assert code == 0 and "freed" in out and not folder.exists()
    code, _ = run_cli(capsys, "cache", "--dir", str(data), "rm", "zod")
    assert code == 1
    assert "nothing downloaded" in run_cli(capsys, "cache", "--dir", str(data))[1]


def test_login_adds_https_to_a_bare_host_and_knows_it_is_the_hosted_visin(capsys):
    run_cli(capsys, "login", "--url", "vision-api.visin.eu/", "--token", "t", "--no-check")
    settings = read_settings()
    assert settings.url == HOSTED_URL and settings.dataset_url == HOSTED_DATASET_URL


def test_login_says_when_an_exported_variable_will_win(capsys, monkeypatch):
    monkeypatch.setenv("VISIN_TOKEN", "exported")
    _, out = run_cli(capsys, "login", "--url", "https://v.test", "--token", "t", "--no-check")
    assert "VISIN_TOKEN set in this environment win" in out


def test_login_with_a_broken_env_file_says_so(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("VISIN_ENV_FILE", str(tmp_path / "missing.env"))
    assert main(["login", "--token", "t", "--no-check"]) == 1
    assert "VISIN_ENV_FILE" in capsys.readouterr().err


PIPELINE = {
    "kind": "pipeline-key",
    "scopes": ["vision:read", "vision:write"],
    "label": "nightly",
    "project": {"id": "p1", "name": "Road"},
}


def discovery(credential=None, **addresses):
    return ok({**addresses, "credential": credential or {"kind": "anonymous"}})


def test_check_says_what_the_server_says_the_key_is_and_what_it_can_do(server, session, capsys, monkeypatch):
    monkeypatch.setenv("VISIN_TOKEN", "vsn_live_abcdefghijklmnop")
    session.route("GET", "/.well-known/visin", discovery(PIPELINE))
    code, out = run_cli(capsys, "check")
    assert code == 0, out
    assert "the pipeline key for project 'Road' (vision:read, vision:write) is accepted" in out
    assert "if it is a pipeline key" not in out


def test_check_warns_that_a_read_only_key_cannot_report(server, session, capsys, monkeypatch):
    monkeypatch.setenv("VISIN_TOKEN", "vsn_live_abcdefghijklmnop")
    session.route("GET", "/.well-known/visin", discovery({"kind": "api-key", "scopes": ["vision:read"]}))
    code, out = run_cli(capsys, "check")
    assert code == 0 and "the API key (vision:read) is accepted" in out
    assert "this key can only read: it cannot report runs" in out
    assert "otherwise set VISIN_PROJECT" in out


def test_check_falls_back_to_the_look_of_the_token_when_the_server_cannot_say(
    server, session, capsys, monkeypatch
):
    monkeypatch.setenv("VISIN_TOKEN", "vsn_live_abcdefghijklmnop")
    session.route("GET", "/.well-known/visin", refused(404, "Not found"))
    code, out = run_cli(capsys, "check")
    assert code == 0 and "the API key is accepted" in out


def test_login_takes_the_dataset_and_app_addresses_from_the_server(server, session, capsys, monkeypatch):
    monkeypatch.delenv("VISIN_URL")
    monkeypatch.delenv("VISIN_TOKEN")
    session.route(
        "GET",
        "/.well-known/visin",
        discovery(PIPELINE, datasetApiUrl="https://datasets.lab.test", appUrl="https://app.lab.test"),
        discovery(PIPELINE, datasetApiUrl="https://datasets.lab.test", appUrl="https://app.lab.test"),
    )
    code, out = run_cli(capsys, "login", "--url", "https://v.test", "--token", "vsn_live_abcdefghijklmnop")
    assert code == 0, out
    settings = read_settings()
    assert (settings.dataset_url, settings.app_url) == ("https://datasets.lab.test", "https://app.lab.test")


def test_login_keeps_addresses_the_user_already_chose(server, session, capsys, monkeypatch):
    monkeypatch.delenv("VISIN_URL")
    monkeypatch.delenv("VISIN_TOKEN")
    config_path().parent.mkdir(parents=True, exist_ok=True)
    config_path().write_text("VISIN_DATASET_URL=https://mine.test\nVISIN_APP_URL=https://mine-app.test\n")
    session.route(
        "GET",
        "/.well-known/visin",
        *[discovery(PIPELINE, datasetApiUrl="https://datasets.lab.test", appUrl="https://app.lab.test")] * 2,
    )
    run_cli(capsys, "login", "--url", "https://v.test", "--token", "vsn_live_abcdefghijklmnop")
    settings = read_settings()
    assert (settings.dataset_url, settings.app_url) == ("https://mine.test", "https://mine-app.test")


def test_login_with_no_check_asks_the_server_nothing(server, session, capsys, monkeypatch):
    monkeypatch.delenv("VISIN_URL")
    monkeypatch.delenv("VISIN_TOKEN")
    run_cli(capsys, "login", "--url", "https://v.test", "--token", "x", "--no-check")
    assert not [call for call in session.calls if "well-known" in call["url"]]
    assert read_settings().dataset_url is None


SUITE_FILE = {
    "slug": "road-test",
    "version": 1,
    "name": "Road test",
    "protocol": {
        "task": "segmentation",
        "data": {"kind": "external", "label": "x", "manifestSha256": "a" * 64},
        "split": "test",
        "conditions": [{"name": "day", "sampleCount": 10}],
        "metrics": [{"key": "mIoU", "direction": "max", "headline": True}],
        "aggregation": "equal-mean-of-conditions",
        "evaluator": {"package": "p"},
    },
}


def suite_json(**overrides):
    return {
        "_id": "s1",
        "slug": "road-test",
        "version": 1,
        "name": "Road test",
        "visibility": "private",
        "digest": "d" * 64,
        "protocol": SUITE_FILE["protocol"],
        **overrides,
    }


def evaluation_json(state="eligible", reasons=(), **overrides):
    return {
        "_id": "e1",
        "uuid": "u1",
        "projectId": "p1",
        "checkpoint": {"kind": "local", "label": "best"},
        "suite": {"slug": "road-test", "version": 1, "digest": "d"},
        "status": "completed",
        "validation": {
            "state": state,
            "reasons": list(reasons),
            "warnings": [{"code": "submitted-overall-differs", "detail": "mIoU"}],
        },
        **overrides,
    }


@pytest.fixture
def files(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"day": {"overall": {"mIoU": 0.8}}}))
    weights = tmp_path / "best.pth"
    weights.write_bytes(b"weights")
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps(SUITE_FILE))
    return tmp_path


class TestSuitesCommand:
    def test_push_publishes_a_file_and_prints_the_ref_and_protocol(self, server, session, files, capsys):
        session.route("POST", "/suites", ok(suite_json(), 201))
        code, out = run_cli(
            capsys, "suites", "push", str(files / "suite.json"), "--project", "road-seg", "--public"
        )
        assert code == 0 and out.startswith("road-test@1  private  protocol dddddddddddd")
        body = session.bodies("/suites")[0]
        assert body["projectId"] == "road-seg" and body["visibility"] == "public"

    def test_push_can_say_private_and_reports_a_refusal_with_the_servers_words(
        self, server, session, files, capsys
    ):
        session.route("POST", "/suites", refused(409, "publish it as version 2"))
        code = main(["suites", "push", str(files / "suite.json"), "--project", "p", "--private"])
        assert code == 1 and "publish it as version 2" in capsys.readouterr().err
        assert session.bodies("/suites")[0]["visibility"] == "private"

    def test_push_names_a_file_it_cannot_read(self, server, capsys):
        assert main(["suites", "push", "nope.json", "--project", "p"]) == 1
        assert "cannot read the suite file" in capsys.readouterr().err

    def test_digest_prints_what_visin_gives_a_protocol_and_stores_nothing(
        self, server, session, files, capsys
    ):
        session.route("POST", "/suites/check", ok({"digest": "c" * 64, "protocol": {}}))
        code, out = run_cli(capsys, "suites", "digest", str(files / "suite.json"))
        assert code == 0 and out.strip() == "c" * 64
        assert "protocol" in session.bodies("/suites/check")[0] and session.bodies("/suites") == []

    def test_digest_names_a_file_it_cannot_read_and_a_server_that_does_not_know_the_route(
        self, server, session, files, capsys
    ):
        assert main(["suites", "digest", "nope.json"]) == 1
        assert "cannot read the suite file" in capsys.readouterr().err
        session.route("POST", "/suites/check", ok({}))
        assert main(["suites", "digest", str(files / "suite.json")]) == 1
        assert "up to date" in capsys.readouterr().err

    def test_list_shows_each_suite_with_its_visibility_and_marks_archived_ones(self, server, session, capsys):
        session.route(
            "GET",
            "/suites",
            ok(
                {
                    "suites": [
                        suite_json(),
                        suite_json(version=2, archivedAt="2026-10-02", visibility="public"),
                    ],
                    "pagination": {"pages": 1},
                }
            ),
        )
        code, out = run_cli(capsys, "suites", "list", "--all", "--project", "road-seg")
        assert code == 0 and "road-test@1" in out and "road-test@2" in out and "archived" in out
        assert session.calls[0]["params"]["includeArchived"] == "true"

    def test_list_as_json(self, server, session, capsys):
        session.route("GET", "/suites", ok({"suites": [suite_json()], "pagination": {"pages": 1}}))
        code, out = run_cli(capsys, "suites", "list", "--json")
        assert code == 0 and json.loads(out)[0]["slug"] == "road-test"

    def test_show_prints_what_the_suite_measures(self, server, session, capsys):
        session.route("GET", "/suites/road-test/1", ok(suite_json()))
        code, out = run_cli(capsys, "suites", "show", "road-test@1")
        assert code == 0
        for line in (
            "road-test@1  Road test  (private)",
            "headline    mIoU, higher is better",
            "condition   day  10 samples",
            "equal-mean-of-conditions",
        ):
            assert line in out

    def test_show_as_json_and_a_missing_suite(self, server, session, capsys):
        session.route("GET", "/suites/road-test/latest", ok(suite_json()), refused(404, "Suite not found"))
        code, out = run_cli(capsys, "suites", "show", "road-test", "--json")
        assert code == 0 and json.loads(out)["digest"] == "d" * 64
        assert main(["suites", "show", "road-test"]) == 1
        assert "Suite not found" in capsys.readouterr().err


class TestManifestCommand:
    def splits(self, tmp_path):
        (tmp_path / "day.txt").write_text("a.png\nb.png\n")
        (tmp_path / "night.txt").write_text("c.png\n")
        return [f"day={tmp_path / 'day.txt'}", f"night={tmp_path / 'night.txt'}"]

    def test_prints_the_digest_and_how_many_samples_each_condition_has(self, tmp_path, capsys):
        code, out = run_cli(capsys, "suites", "manifest", *self.splits(tmp_path))
        expected = visin.manifest_digest({"day": ["a.png", "b.png"], "night": ["c.png"]})
        assert code == 0 and f"manifestSha256  {expected}" in out
        assert "day" in out and "2 samples" in out and "1 samples" in out

    def test_json_is_what_a_suite_file_takes_and_needs_no_server(self, tmp_path, capsys):
        code, out = run_cli(capsys, "suites", "manifest", "--json", *self.splits(tmp_path))
        data = json.loads(out)
        assert code == 0 and data["conditions"] == [
            {"name": "day", "sampleCount": 2},
            {"name": "night", "sampleCount": 1},
        ]
        assert data["manifestSha256"] == visin.manifest_digest(
            {"day": ["a.png", "b.png"], "night": ["c.png"]}
        )

    def test_a_bad_argument_or_a_repeated_condition_or_a_missing_file_is_explained(self, tmp_path, capsys):
        assert main(["suites", "manifest", "day"]) == 1
        assert "expected NAME=FILE" in capsys.readouterr().err
        listed = self.splits(tmp_path)
        assert main(["suites", "manifest", listed[0], listed[0]]) == 1
        assert "given twice" in capsys.readouterr().err
        assert main(["suites", "manifest", "day=missing.txt"]) == 1
        assert "cannot read the split file" in capsys.readouterr().err


class TestEvaluateCommand:
    def args(self, files, *extra):
        return [
            "evaluate",
            str(files / "results.json"),
            "--suite",
            "road-test@1",
            "--project",
            "road-seg",
            "--checkpoint",
            str(files / "best.pth"),
            "--sample-count",
            "day=10",
            *extra,
        ]

    def test_records_and_prints_the_verdict(self, server, session, files, capsys):
        session.route("POST", "/evaluations", ok(evaluation_json(), 201))
        code, out = run_cli(
            capsys,
            *self.args(
                files,
                "--evaluator-package",
                "visin-fusion",
                "--evaluator-version",
                "1.4",
                "--evaluator-commit",
                "9d1",
                "--run",
                "r",
                "--epoch",
                "40",
                "--uuid",
                "u1",
            ),
        )
        assert code == 0 and "recorded evaluation e1 on road-test@1" in out and "verdict   eligible" in out
        assert "! submitted-overall-differs(mIoU)" in out
        body = session.bodies("/evaluations")[0]
        assert (
            body["uuid"] == "u1"
            and body["source"] == {"trainingUuid": "r", "epoch": 40}
            and body["sampleCounts"] == {"day": 10}
        )
        assert body["provenance"]["evaluator"] == {
            "package": "visin-fusion",
            "version": "1.4",
            "commit": "9d1",
        }
        assert body["checkpoint"]["label"] == "best"

    def test_sends_what_it_ran_as_evidence_and_prints_how_much_was_reported(
        self, server, session, files, capsys
    ):
        session.route("POST", "/suites/check", ok({"digest": "c" * 64}))
        session.route(
            "POST",
            "/evaluations",
            ok(
                evaluation_json()
                | {
                    "validation": {"state": "eligible", "reasons": [], "warnings": [], "evidence": "observed"}
                },
                201,
            ),
        )
        code, out = run_cli(
            capsys,
            *self.args(
                files,
                "--data",
                "external=" + "a" * 64,
                "--protocol",
                str(files / "suite.json"),
                "--evaluator-package",
                "visin-fusion",
                "--evaluator-version",
                "1.4.2",
                "--classes-scored",
                "car, tree",
                "--classes-ignored",
                "void",
            ),
        )
        assert code == 0 and "evidence  observed" in out
        assert session.bodies("/evaluations")[0]["evidence"] == {
            "data": {"kind": "external", "manifestSha256": "a" * 64},
            "protocolDigest": "c" * 64,
            "evaluator": {"package": "visin-fusion", "version": "1.4.2"},
            "classes": {"scored": ["car", "tree"], "ignored": ["void"]},
        }

    @pytest.mark.parametrize(
        ("flag", "expected"),
        [
            ("visin=" + "b" * 64, {"kind": "visin", "archiveSha256": "b" * 64}),
            ("hf=acme/frames@" + "c" * 40, {"kind": "hf", "repo": "acme/frames", "commit": "c" * 40}),
        ],
    )
    def test_the_data_flag_takes_each_kind_a_suite_can_pin(self, server, session, files, flag, expected):
        session.route("POST", "/evaluations", ok(evaluation_json(), 201))
        assert main(self.args(files, "--data", flag, "--protocol-digest", "d" * 64)) == 0
        evidence = session.bodies("/evaluations")[0]["evidence"]
        assert evidence == {"data": expected, "protocolDigest": "d" * 64}

    @pytest.mark.parametrize(
        ("extra", "message"),
        [
            (("--data", "nonsense"), "expected KIND=VALUE"),
            (("--data", "ftp=x"), "the kind is external, visin, hf"),
            (("--data", "hf=acme/frames"), "hf=org/name@<commit>"),
            (("--classes-ignored", "void"), "needs --classes-scored"),
        ],
    )
    def test_a_malformed_evidence_flag_is_explained(self, server, files, capsys, extra, message):
        assert main(self.args(files, *extra)) == 1
        assert message in capsys.readouterr().err

    def test_a_result_that_is_not_ranked_lists_the_reasons_and_points_to_the_docs(
        self, server, session, files, capsys
    ):
        session.route(
            "POST",
            "/evaluations",
            ok(evaluation_json("incomplete", [{"code": "missing-condition", "detail": "night"}]), 201),
        )
        code, out = run_cli(capsys, *self.args(files))
        assert code == 0 and "- missing-condition(night)" in out and "Why is my result unranked?" in out

    def test_require_ranked_turns_an_unranked_result_into_a_failing_exit_for_ci(
        self, server, session, files, capsys
    ):
        session.route(
            "POST",
            "/evaluations",
            ok(evaluation_json("incomplete", [{"code": "no-checkpoint"}]), 201),
            ok(evaluation_json(), 201),
        )
        assert main(self.args(files, "--require-ranked")) == 3
        capsys.readouterr()
        assert main(self.args(files, "--require-ranked")) == 0

    def test_dry_run_checks_and_stores_nothing(self, server, session, files, capsys):
        session.route(
            "POST",
            "/evaluations/check",
            ok(
                {
                    "validation": {"state": "eligible", "reasons": [], "warnings": []},
                    "suite": {"slug": "road-test", "version": 1, "digest": "d"},
                }
            ),
        )
        code, out = run_cli(capsys, *self.args(files, "--dry-run"))
        assert code == 0 and "checked on road-test@1; nothing was stored" in out
        assert session.paths("POST") == ["/evaluations/check"]

    def test_json_output_has_the_verdict_and_the_flags_ci_reads(self, server, session, files, capsys):
        session.route(
            "POST", "/evaluations", ok(evaluation_json("incomplete", [{"code": "no-checkpoint"}]), 201)
        )
        code, out = run_cli(capsys, *self.args(files, "--json"))
        data = json.loads(out)
        assert code == 0 and data["ranked"] is False and data["stored"] is True and data["queued"] is False
        assert data["verdict"]["reasons"] == [{"code": "no-checkpoint", "detail": None}]

    def test_a_hub_checkpoint_needs_its_full_commit_and_the_two_kinds_are_not_mixed(
        self, server, files, capsys
    ):
        base = ["evaluate", str(files / "results.json"), "--suite", "road-test@1", "--project", "p"]
        assert main([*base, "--hub-repo", "acme/clft"]) == 1
        assert "--hub-commit" in capsys.readouterr().err
        assert (
            main(
                [
                    *base,
                    "--hub-repo",
                    "acme/clft",
                    "--hub-commit",
                    "a" * 40,
                    "--checkpoint",
                    str(files / "best.pth"),
                ]
            )
            == 1
        )
        assert "not both" in capsys.readouterr().err

    def test_a_hub_checkpoint_is_sent_as_a_pin(self, server, session, files, capsys):
        session.route("POST", "/evaluations", ok(evaluation_json(), 201))
        code = main(
            [
                "evaluate",
                str(files / "results.json"),
                "--suite",
                "road-test@1",
                "--project",
                "p",
                "--hub-repo",
                "acme/clft",
                "--hub-commit",
                "A" * 40,
                "--hub-path",
                "best.safetensors",
            ]
        )
        assert code == 0
        assert session.bodies("/evaluations")[0]["checkpoint"] == {
            "kind": "hf",
            "repo": "acme/clft",
            "commit": "a" * 40,
            "path": "best.safetensors",
        }

    def test_sample_counts_come_from_a_file_and_flags_and_a_bad_flag_is_explained(
        self, server, session, files, capsys
    ):
        session.route("POST", "/evaluations", ok(evaluation_json(), 201))
        counts = files / "counts.json"
        counts.write_text(json.dumps({"day": 10, "night": 5}))
        args = [
            "evaluate",
            str(files / "results.json"),
            "--suite",
            "s@1",
            "--project",
            "p",
            "--sample-counts",
            str(counts),
            "--sample-count",
            "rain=3",
        ]
        assert main(args) == 0
        assert session.bodies("/evaluations")[0]["sampleCounts"] == {"day": 10, "night": 5, "rain": 3}
        assert main([*args[:-2], "--sample-count", "rain"]) == 1
        assert "expected NAME=NUMBER" in capsys.readouterr().err
        assert main([*args[:-2], "--sample-counts", str(files / "missing.json")]) == 1
        assert "cannot read --sample-counts" in capsys.readouterr().err
        counts.write_text("[1]")
        assert main(args[:-2]) == 1
        assert "should hold an object" in capsys.readouterr().err

    def test_a_results_file_that_is_not_an_object_or_not_json_is_named(self, server, files, capsys):
        bad = files / "bad.json"
        bad.write_text("[1]")
        assert main(["evaluate", str(bad), "--suite", "s@1", "--project", "p"]) == 1
        assert "should hold the results object" in capsys.readouterr().err
        bad.write_text("{ nope")
        assert main(["evaluate", str(bad), "--suite", "s@1", "--project", "p"]) == 1
        assert main(["evaluate", str(files / "missing.json"), "--suite", "s@1", "--project", "p"]) == 1

    def test_a_refusal_prints_the_servers_words_and_fails(self, server, session, files, capsys):
        session.route("POST", "/evaluations", refused(404, "Suite not found"))
        assert main(self.args(files)) == 1
        assert "Suite not found" in capsys.readouterr().err

    def test_a_result_kept_for_sync_says_so(self, files, capsys, monkeypatch):
        monkeypatch.setenv("VISIN_MODE", "offline")
        code, out = run_cli(capsys, *self.args(files, "--uuid", "u9"))
        assert code == 0 and "kept on disk, to send with `visin sync` (uuid u9)" in out

    def test_with_nothing_configured_it_says_nothing_was_recorded(self, files, capsys):
        code, out = run_cli(capsys, *self.args(files))
        assert code == 0 and "not recorded: nothing is configured" in out


class TestRankingCommands:
    BOARD: ClassVar[dict] = {
        "suite": {"slug": "road-test", "version": 1, "headline": {"key": "mIoU", "direction": "max"}},
        "scope": {"candidates": 3},
        "entries": [
            {
                "rank": 1,
                "evaluationId": "e1",
                "attempts": 1,
                "checkpoint": {"kind": "local", "label": "Model B"},
                "summary": {
                    "headline": {"value": 0.74},
                    "worst": {"condition": "night", "value": 0.7},
                    "gap": 0.04,
                },
            }
        ],
        "unranked": [
            {
                "checkpointKey": "sha256:x",
                "state": "incomplete",
                "reasons": [{"code": "missing-condition", "detail": "rain"}, {"code": "no-checkpoint"}],
            }
        ],
    }

    def test_leaderboard_prints_the_ranking_and_what_is_not_ranked(self, server, session, capsys):
        session.route("GET", "/suites/road-test/1/leaderboard", ok(self.BOARD))
        code, out = run_cli(capsys, "leaderboard", "road-test@1")
        assert (
            code == 0
            and "ranked among 3 visible to you" in out
            and "Model B" in out
            and "weakest night 0.7" in out
        )
        assert "not ranked  sha256:x  incomplete: missing-condition(rain), no-checkpoint" in out

    def test_leaderboard_public_and_json(self, server, session, capsys):
        public = {
            **self.BOARD,
            "entries": [
                {
                    "rank": 1,
                    "evaluationId": "e1",
                    "attempts": 2,
                    "checkpoint": {"kind": "local", "label": "A"},
                    "headline": 0.5,
                    "worst": {"condition": "d", "value": 0.4},
                    "gap": 0.1,
                }
            ],
            "unranked": [],
        }
        session.route("GET", "/public/leaderboards/road-test/1", ok(public), ok(public))
        code, out = run_cli(capsys, "leaderboard", "road-test@1", "--public")
        assert code == 0 and "ranked among 3 public" in out and "2 attempts" in out
        code, out = run_cli(capsys, "leaderboard", "road-test@1", "--public", "--json")
        assert code == 0 and json.loads(out)["scope"]["candidates"] == 3

    @staticmethod
    def paged(number, pages=3, total=250):
        return {
            **TestRankingCommands.BOARD,
            "pagination": {"page": number, "limit": 100, "total": total, "pages": pages},
            "unrankedPagination": {"page": 1, "limit": 100, "total": 1, "pages": 1},
        }

    def test_leaderboard_passes_the_page_controls_and_says_which_page_it_shows(self, server, session, capsys):
        session.route("GET", "/suites/road-test/1/leaderboard", ok(self.paged(2)))
        code, out = run_cli(
            capsys, "leaderboard", "road-test@1", "--page", "2", "--limit", "50", "--unranked-page", "3"
        )
        assert code == 0 and "(checkpoints: page 2 of 3 (1 of 250 shown))" in out and "(unranked" not in out
        assert session.calls[-1]["params"] == {"page": 2, "limit": 50, "unrankedPage": 3}

    def test_leaderboard_all_reads_every_page_and_json_carries_them_all(self, server, session, capsys):
        session.route(
            "GET", "/suites/road-test/1/leaderboard", ok(self.paged(1, pages=2)), ok(self.paged(2, pages=2))
        )
        code, out = run_cli(capsys, "leaderboard", "road-test@1", "--all")
        assert code == 0 and "(all 250 checkpoints)" in out and "(all 1 unranked)" in out
        assert [c["params"] for c in session.calls[-2:]] == [
            {"page": 1, "limit": 100},
            {"page": 2, "limit": 100},
        ]
        session.route("GET", "/suites/road-test/1/leaderboard", ok(self.paged(1, pages=1)))
        code, out = run_cli(capsys, "leaderboard", "road-test@1", "--all", "--json")
        assert code == 0 and len(json.loads(out)["entries"]) == 1

    def test_leaderboard_refuses_contradictory_or_impossible_paging(self, server, capsys):
        assert main(["leaderboard", "road-test@1", "--all", "--page", "2"]) == 1
        assert "every page" in capsys.readouterr().err
        assert main(["leaderboard", "road-test@1", "--public", "--unranked-page", "2"]) == 1
        assert "no unranked list" in capsys.readouterr().err

    def test_public_leaderboard_pages_anonymously(self, server, session, capsys):
        public = {
            **self.BOARD,
            "unranked": [],
            "pagination": {"page": 2, "limit": 10, "total": 25, "pages": 3},
        }
        session.route("GET", "/public/leaderboards/road-test/1", ok(public))
        code, out = run_cli(capsys, "leaderboard", "road-test@1", "--public", "--page", "2", "--limit", "10")
        assert code == 0 and "(checkpoints: page 2 of 3 (1 of 25 shown))" in out
        assert session.calls[-1]["params"] == {"page": 2, "limit": 10}

    def test_leaderboard_observed_asks_for_the_observed_results_alone(self, server, session, capsys):
        session.route("GET", "/suites/road-test/1/leaderboard", ok(self.BOARD))
        assert run_cli(capsys, "leaderboard", "road-test@1", "--observed")[0] == 0
        assert session.calls[-1]["params"] == {"evidence": "observed"}
        session.route("GET", "/public/leaderboards/road-test/1", ok({**self.BOARD, "unranked": []}))
        assert run_cli(capsys, "leaderboard", "road-test@1", "--public", "--observed")[0] == 0
        assert session.calls[-1]["params"] == {"evidence": "observed"}

    def test_leaderboard_public_needs_a_version(self, server, capsys):
        assert main(["leaderboard", "road-test", "--public"]) == 1
        assert "version" in capsys.readouterr().err

    def test_evaluations_lists_each_with_its_verdict_and_whether_it_is_public(self, server, session, capsys):
        items = [
            evaluation_json(),
            evaluation_json(
                "incomplete",
                [{"code": "no-checkpoint"}],
                _id="e2",
                checkpoint={"kind": "hf", "repo": "acme/clft"},
                publishedAt="2026-10-02",
            ),
        ]
        session.route("GET", "/evaluations", ok({"evaluations": items, "pagination": {"pages": 1}}))
        code, out = run_cli(
            capsys, "evaluations", "--suite", "road-test@1", "--state", "eligible", "--limit", "5"
        )
        assert (
            code == 0
            and "e1  road-test@1" in out
            and "best" in out
            and "acme/clft" in out
            and "public" in out
        )
        assert session.calls[0]["params"]["state"] == "eligible"

    def test_evaluations_as_json_and_a_failure(self, server, session, capsys):
        session.route(
            "GET",
            "/evaluations",
            ok({"evaluations": [evaluation_json()], "pagination": {"pages": 1}}),
            refused(403, "Access denied"),
        )
        code, out = run_cli(capsys, "evaluations", "--json")
        assert code == 0 and json.loads(out)[0]["checkpoint"]["label"] == "best"
        assert main(["evaluations"]) == 1
        assert "Access denied" in capsys.readouterr().err

    @pytest.mark.parametrize(("command", "shown"), [("publish", "public"), ("withdraw", "not public")])
    def test_publish_and_withdraw_print_the_new_state(self, server, session, capsys, command, shown):
        published = {"publishedAt": "2026-10-02"} if command == "publish" else {}
        session.route("POST", f"/evaluations/e1/{command}", ok(evaluation_json(**published)))
        code, out = run_cli(capsys, command, "e1")
        assert code == 0 and out.strip() == f"e1  {shown}"

    @pytest.mark.parametrize(
        ("extra", "shown"),
        [
            ({"pendingApproval": True}, "waiting for approval"),
            ({"hidden": {"at": "2026-10-03", "reason": "No"}}, "hidden by the suite's managers"),
        ],
    )
    def test_publish_says_when_the_result_is_not_yet_shown(self, server, session, capsys, extra, shown):
        session.route(
            "POST", "/evaluations/e1/publish", ok(evaluation_json(publishedAt="2026-10-02", **extra))
        )
        code, out = run_cli(capsys, "publish", "e1")
        assert code == 0 and out.strip() == f"e1  {shown}"

    def test_publish_says_why_it_was_refused(self, server, session, capsys):
        session.route(
            "POST", "/evaluations/e1/publish", refused(409, "Only a ranked result can be published.")
        )
        assert main(["publish", "e1"]) == 1
        assert "Only a ranked result" in capsys.readouterr().err

    def test_promote_copies_a_result_onto_a_suite(self, server, session, files, capsys):
        session.route("POST", "/evaluations/promote", ok(evaluation_json(), 201))
        code, out = run_cli(
            capsys,
            "promote",
            "tr1",
            "--suite",
            "road-test@1",
            "--checkpoint",
            str(files / "best.pth"),
            "--sample-count",
            "day=10",
        )
        assert code == 0 and "recorded evaluation e1" in out
        body = session.bodies("/evaluations/promote")[0]
        assert body["evaluationId"] == "tr1" and body["sampleCounts"] == {"day": 10}

    def test_promote_needs_the_checkpoint_and_the_counts_the_old_result_never_had(
        self, server, files, capsys
    ):
        assert main(["promote", "tr1", "--suite", "s@1", "--sample-count", "day=1"]) == 1
        assert "say which checkpoint" in capsys.readouterr().err
        assert main(["promote", "tr1", "--suite", "s@1", "--checkpoint", str(files / "best.pth")]) == 1
        assert "how many samples" in capsys.readouterr().err

    def test_promote_reports_a_refusal(self, server, session, files, capsys):
        session.route(
            "POST", "/evaluations/promote", refused(403, "Manage access to the project is required")
        )
        assert (
            main(
                [
                    "promote",
                    "tr1",
                    "--suite",
                    "s@1",
                    "--checkpoint",
                    str(files / "best.pth"),
                    "--sample-count",
                    "day=1",
                ]
            )
            == 1
        )
        assert "Manage access" in capsys.readouterr().err
