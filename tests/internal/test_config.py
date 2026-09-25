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


def test_the_older_spellings_still_work(monkeypatch):
    monkeypatch.setenv("VISIN_API_URL", "https://old.test")
    monkeypatch.setenv("VISIN_API_TOKEN", "t")
    monkeypatch.setenv("VISIN_PROJECT_ID", "p")
    settings = read_settings()
    assert (settings.url, settings.token, settings.project) == ("https://old.test", "t", "p")


def test_the_documented_name_wins(monkeypatch):
    monkeypatch.setenv("VISIN_URL", "https://new.test")
    monkeypatch.setenv("VISIN_API_URL", "https://old.test")
    assert read_settings().url == "https://new.test"


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
