"""Fixtures every test shares. The fakes themselves are in ``fakes.py``."""

import os

import pytest
from fakes import BASE, FakeAtexit, FakeSession, FakeUploads

import visin.run
from visin._internal.transport import HttpClient


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """No test reads the developer's environment, writes to ~/.visin, installs
    process-wide signal handlers or leaves atexit hooks behind."""
    for name in list(os.environ):
        if name.startswith("VISIN_") or name in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VISIN_DIR", str(tmp_path / "visin-dir"))
    monkeypatch.setattr(visin.run._ProcessHooks, "install", classmethod(lambda cls: None))
    # How the process ended is process-wide state; every test starts from a clean one.
    monkeypatch.setattr(visin.run._ProcessHooks, "crashed", False)
    monkeypatch.setattr(visin.run._ProcessHooks, "terminated", False)
    hooks = FakeAtexit()
    monkeypatch.setattr(visin.run, "atexit", hooks)
    monkeypatch.setattr(visin.run, "CATCH_UP_INTERVAL", 0.0)
    return hooks


@pytest.fixture
def session():
    return FakeSession()


@pytest.fixture
def uploads():
    return FakeUploads()


@pytest.fixture
def sleeps():
    return []


@pytest.fixture
def client(session, uploads, sleeps):
    return HttpClient(BASE, "token", session=session, upload_session=uploads, sleep=sleeps.append)


@pytest.fixture
def server(monkeypatch, client):
    """Configure the environment and route every HttpClient the package makes to the fake."""
    monkeypatch.setenv("VISIN_URL", BASE)
    monkeypatch.setenv("VISIN_TOKEN", "token")
    factory = lambda *_args, **_kwargs: client  # noqa: E731
    monkeypatch.setattr("visin.run.HttpClient", factory)
    monkeypatch.setattr("visin.offline.HttpClient", factory)
    monkeypatch.setattr("visin.cli.HttpClient", factory)
    monkeypatch.setattr("visin.api.HttpClient", factory)
    return client
