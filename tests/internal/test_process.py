import signal
import sys

import pytest

from visin._internal.process import ProcessHooks, rank

REAL_INSTALL = ProcessHooks.__dict__["install"]


@pytest.mark.parametrize("name", ["RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK"])
def test_every_launchers_rank_variable_is_read(monkeypatch, name):
    monkeypatch.setenv(name, "3")
    assert rank() == 3


def test_a_process_outside_a_distributed_job_is_rank_zero(monkeypatch):
    for name in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK", "PMI_RANK"):
        monkeypatch.delenv(name, raising=False)
    assert rank() == 0


def test_a_garbled_rank_is_ignored(monkeypatch):
    monkeypatch.setenv("RANK", "worker-1")
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    assert rank() == 0


def test_install_hooks_a_crash_and_sigterm_once(monkeypatch):
    monkeypatch.setattr(ProcessHooks, "install", REAL_INSTALL)
    monkeypatch.setattr(ProcessHooks, "installed", False)
    monkeypatch.setattr(ProcessHooks, "crashed", False)
    monkeypatch.setattr(ProcessHooks, "terminated", False)
    monkeypatch.setattr(sys, "excepthook", lambda *_: None)
    monkeypatch.setattr(signal, "getsignal", lambda _sig: signal.SIG_DFL)
    installed = []
    monkeypatch.setattr(signal, "signal", lambda sig, handler: installed.append((sig, handler)))

    ProcessHooks.install()
    ProcessHooks.install()

    assert [sig for sig, _ in installed] == [signal.SIGTERM]
    sys.excepthook(RuntimeError, RuntimeError("x"), None)
    assert ProcessHooks.crashed


def test_another_handler_for_sigterm_is_left_alone(monkeypatch):
    monkeypatch.setattr(ProcessHooks, "install", REAL_INSTALL)
    monkeypatch.setattr(ProcessHooks, "installed", False)
    monkeypatch.setattr(sys, "excepthook", lambda *_: None)
    monkeypatch.setattr(signal, "getsignal", lambda _sig: lambda *_: None)
    installed = []
    monkeypatch.setattr(signal, "signal", lambda sig, handler: installed.append(sig))
    ProcessHooks.install()
    assert installed == []
