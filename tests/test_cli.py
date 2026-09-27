import pytest
from fakes import ok, refused

import visin
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
