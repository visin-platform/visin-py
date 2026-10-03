"""What a run records about where it came from, and what it must never record."""

import shutil
import subprocess
import sys
import types

import pytest
from fakes import ok

import visin
from visin._internal import provenance

COMMIT = "a" * 40


@pytest.fixture
def git_repo(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git is not installed")

    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)  # noqa: S607

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@example.test")
    git("config", "user.name", "T")
    (tmp_path / "train.py").write_text("print()\n")
    git("add", ".")
    git("commit", "-q", "-m", "first")
    git("remote", "add", "origin", "https://ghp_abcdefghijklmnopqrstuvwxyz0123@github.com/acme/fusion.git")
    return tmp_path


class TestCommandLine:
    @pytest.mark.parametrize(
        ("argv", "shown"),
        [
            (["train.py", "--epochs", "5"], "train.py --epochs 5"),
            (["train.py", "--token", "abc123"], "train.py --token '***'"),
            (["train.py", "--token=abc123"], "train.py '--token=***'"),
            (["train.py", "--hf-token", "x", "--lr", "0.1"], "train.py --hf-token '***' --lr 0.1"),
            (["train.py", "--api_key=k", "--password", "p"], "train.py '--api_key=***' --password '***'"),
            (["env", "HF_TOKEN=hf_x", "python"], "env 'HF_TOKEN=***' python"),
            (["train.py", "--auth-file", "creds.json"], "train.py --auth-file '***'"),
        ],
    )
    def test_credential_values_are_replaced(self, argv, shown):
        assert provenance.redact_command(argv) == shown

    @pytest.mark.parametrize(
        "secret",
        ["hf_" + "A1b2C3d4E5f6G7h8I9j0K1", "vsn_live_abcdefgh12345678", "ghp_" + "a" * 26, "sk-" + "b" * 24],
    )
    def test_a_token_pasted_anywhere_is_replaced(self, secret):
        assert secret not in provenance.redact_command(["curl", f"https://x.test/?k={secret}", secret])

    def test_a_flag_that_ends_the_line_is_left_alone(self):
        assert provenance.redact_command(["train.py", "--token"]) == "train.py --token"


class TestGit:
    def test_the_commit_branch_remote_and_dirtiness_are_read_and_the_remote_loses_its_credentials(
        self, git_repo
    ):
        info = provenance.git_info(git_repo)
        assert info["branch"] == "main" and info["dirty"] is False
        assert len(info["commit"]) == 40
        assert info["remote"] == "https://github.com/acme/fusion.git"

    def test_uncommitted_changes_make_it_dirty(self, git_repo):
        (git_repo / "train.py").write_text("print(1)\n")
        assert provenance.git_info(git_repo)["dirty"] is True

    def test_outside_a_repo_or_without_git_there_is_nothing(self, tmp_path, monkeypatch):
        assert provenance.git_info(tmp_path) is None
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        assert provenance.git_info(tmp_path) is None

    def test_a_git_that_hangs_or_fails_is_not_fatal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _name: "/bin/git")
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired("git", 3))
        )
        assert provenance.git_info(tmp_path) is None


class TestEnvironment:
    def test_packages_are_named_by_version_and_capped(self, monkeypatch):
        found = provenance.packages()
        assert found["requests"] and list(found) == sorted(found)
        monkeypatch.setattr(provenance, "MAX_PACKAGES", 3)
        assert len(provenance.packages()) == 3

    def test_the_host_names_cuda_only_when_torch_is_already_loaded(self, monkeypatch):
        assert "cuda" not in provenance.host()
        torch = types.ModuleType("torch")
        torch.version = types.SimpleNamespace(cuda="12.4")
        monkeypatch.setitem(sys.modules, "torch", torch)
        host = provenance.host()
        assert host["cuda"] == "12.4" and host["hostname"] and host["python"]

    def test_everything_is_collected_from_the_running_script(self, git_repo, monkeypatch):
        monkeypatch.setattr(sys, "argv", [str(git_repo / "train.py"), "--token", "s3cret"])
        collected = provenance.collect()
        assert collected["git"]["branch"] == "main"
        assert (
            collected["command"].endswith("train.py --token '***'") and "s3cret" not in collected["command"]
        )
        assert collected["packages"] and collected["host"]["python"]

    def test_a_script_outside_any_repo_falls_back_to_the_working_directory(
        self, git_repo, tmp_path_factory, monkeypatch
    ):
        elsewhere = tmp_path_factory.mktemp("elsewhere")
        (elsewhere / "run.py").write_text("")
        monkeypatch.setattr(sys, "argv", [str(elsewhere / "run.py")])
        monkeypatch.chdir(git_repo)
        assert provenance.collect()["git"]["branch"] == "main"

    def test_with_no_script_it_still_reports_the_machine(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "argv", [""])
        monkeypatch.chdir(tmp_path)
        collected = provenance.collect()
        assert "command" not in collected and "git" not in collected and collected["host"]["hostname"]


class TestSwitch:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(None, True), ("1", True), ("0", False), ("false", False), ("OFF", False), (" no ", False)],
    )
    def test_it_is_on_unless_the_environment_says_off(self, monkeypatch, value, expected):
        if value is None:
            monkeypatch.delenv("VISIN_PROVENANCE", raising=False)
        else:
            monkeypatch.setenv("VISIN_PROVENANCE", value)
        assert provenance.enabled() is expected

    def test_the_argument_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("VISIN_PROVENANCE", "0")
        assert provenance.enabled(True) is True
        monkeypatch.setenv("VISIN_PROVENANCE", "1")
        assert provenance.enabled(False) is False


class TestRun:
    def test_a_run_is_registered_with_its_provenance_unless_it_was_declined(
        self, server, session, monkeypatch
    ):
        monkeypatch.setattr(
            provenance, "collect", lambda: {"git": {"commit": COMMIT}, "host": {"python": "3.12"}}
        )
        session.route(
            "POST", "/trainings", ok({"_id": "t1", "uuid": "u"}, 201), ok({"_id": "t2", "uuid": "v"}, 201)
        )
        visin.init("with", project="p", provenance=True).finish()
        visin.init("without", project="p").finish()
        monkeypatch.setenv("VISIN_PROVENANCE", "1")
        visin.init("by environment", project="p").finish()
        with_, without, by_environment = session.bodies("/trainings")
        assert with_["provenance"] == {"git": {"commit": COMMIT}, "host": {"python": "3.12"}}
        assert "provenance" not in without
        assert by_environment["provenance"]["git"]["commit"] == COMMIT
