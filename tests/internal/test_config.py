from pathlib import Path

import pytest

from visin._internal.config import read_settings
from visin.errors import ConfigurationError


def test_the_documented_names_are_read(monkeypatch):
    monkeypatch.setenv("VISIN_URL", "https://v.test")
    monkeypatch.setenv("VISIN_TOKEN", "t")
    monkeypatch.setenv("VISIN_PROJECT", "road-seg")
    settings = read_settings()
    assert (settings.url, settings.token, settings.project) == ("https://v.test", "t", "road-seg")
    assert settings.configured
    assert settings.effective_mode == "online"


def test_the_older_spellings_are_gone(monkeypatch):
    monkeypatch.setenv("VISIN_API_URL", "https://old.test")
    monkeypatch.setenv("VISIN_API_TOKEN", "t")
    monkeypatch.setenv("VISIN_PROJECT_ID", "p")
    settings = read_settings()
    assert (settings.url, settings.token, settings.project) == (None, None, None)


def test_blank_values_count_as_unset(monkeypatch):
    monkeypatch.setenv("VISIN_URL", "  ")
    assert read_settings().url is None


def test_online_without_a_server_is_disabled():
    assert read_settings().effective_mode == "disabled"


def test_offline_needs_no_server(monkeypatch):
    monkeypatch.setenv("VISIN_MODE", "offline")
    assert read_settings().effective_mode == "offline"


def test_an_unknown_mode_is_refused(monkeypatch):
    monkeypatch.setenv("VISIN_MODE", "sometimes")
    with pytest.raises(ConfigurationError, match="sometimes"):
        read_settings()


def test_tls_verification_is_on_unless_turned_off(monkeypatch):
    assert read_settings().verify_ssl is True
    monkeypatch.setenv("VISIN_VERIFY_SSL", "0")
    assert read_settings().verify_ssl is False


def test_arguments_win_over_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIN_URL", "https://env.test")
    settings = read_settings(url="https://arg.test", token=None, directory=str(tmp_path), mode="OFFLINE")
    assert settings.url == "https://arg.test"
    assert settings.directory == Path(tmp_path)
    assert settings.mode == "offline"


def test_the_directory_defaults_to_home(monkeypatch):
    monkeypatch.delenv("VISIN_DIR")
    assert read_settings().directory == Path.home() / ".visin"


def write_config(tmp_path, text, mode=0o600):
    path = tmp_path / "config-dir" / "config"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text)
    path.chmod(mode)
    return path


def test_the_user_config_file_is_read(tmp_path):
    write_config(tmp_path, "# mine\nVISIN_URL=https://file.test\nexport VISIN_TOKEN='t o k'\nOTHER=1\n")
    settings = read_settings()
    assert (settings.url, settings.token) == ("https://file.test", "t o k")


def test_the_environment_beats_the_config_file(tmp_path, monkeypatch):
    write_config(tmp_path, "VISIN_URL=https://file.test\n")
    monkeypatch.setenv("VISIN_URL", "https://env.test")
    assert read_settings().url == "https://env.test"


def test_an_env_file_beats_the_config_file_and_must_exist(tmp_path, monkeypatch):
    write_config(tmp_path, "VISIN_URL=https://file.test\nVISIN_PROJECT=a\n")
    env_file = tmp_path / "job.env"
    env_file.write_text("VISIN_URL=https://job.test # the job's\n")
    monkeypatch.setenv("VISIN_ENV_FILE", str(env_file))
    settings = read_settings()
    assert (settings.url, settings.project) == ("https://job.test", "a")
    monkeypatch.setenv("VISIN_ENV_FILE", str(tmp_path / "missing.env"))
    with pytest.raises(ConfigurationError, match="VISIN_ENV_FILE"):
        read_settings()


def test_only_visin_names_are_read_from_a_file(tmp_path, monkeypatch):
    write_config(tmp_path, "AWS_SECRET=x\nVISIN_URL=https://file.test\n")
    read_settings()
    assert "AWS_SECRET" not in __import__("os").environ


def test_each_setting_says_where_it_came_from(tmp_path, monkeypatch):
    path = write_config(tmp_path, "VISIN_URL=https://file.test\n")
    monkeypatch.setenv("VISIN_TOKEN", "t")
    settings = read_settings(project="p")
    assert settings.sources == {"url": str(path), "token": "environment", "project": "argument"}


def test_a_token_in_a_file_others_can_read_is_warned_about(tmp_path, caplog):
    write_config(tmp_path, "VISIN_TOKEN=t\n", mode=0o644)
    read_settings()
    assert "chmod 600" in caplog.text


def test_the_config_file_lives_in_the_home_folder_not_in_visin_dir(monkeypatch, tmp_path):
    from visin._internal.config import config_path

    monkeypatch.delenv("VISIN_CONFIG")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VISIN_DIR", str(tmp_path / "scratch"))
    assert config_path() == tmp_path / "home" / ".visin" / "config"


def test_an_empty_variable_does_not_hide_the_config_file(tmp_path, monkeypatch):
    write_config(tmp_path, "VISIN_URL=https://file.test\n")
    monkeypatch.setenv("VISIN_URL", "")
    assert read_settings().url == "https://file.test"


def test_writing_a_name_set_twice_leaves_only_the_new_value(tmp_path):
    from visin._internal.config import write_config as save

    path = write_config(tmp_path, "VISIN_URL=https://a.test\n# keep\nVISIN_URL=https://b.test\n")
    save({"VISIN_URL": "https://new.test"}, path)
    assert path.read_text() == "VISIN_URL=https://new.test\n# keep\n"
    assert read_settings().url == "https://new.test"
