import json

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
