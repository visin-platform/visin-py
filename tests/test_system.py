import sys
import types

import pytest

from visin import system

SMI_ROW = "NVIDIA RTX 4090, 550.54, 24564, 1830, 61, 285.5, 450.0, 38, 97, 40, 2520, 10501"


@pytest.fixture
def smi(monkeypatch):
    """A machine with one GPU and nvidia-smi on the path."""
    monkeypatch.setattr(system.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(system, "_run", lambda command, timeout=5.0: SMI_ROW + "\n")


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setattr(system.shutil, "which", lambda name: None)
    monkeypatch.delitem(sys.modules, "torch", raising=False)


@pytest.fixture
def fake_torch(monkeypatch):
    gib = 1024**3
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        memory_allocated=lambda i: 2 * gib,
        memory_reserved=lambda i: 3 * gib,
        max_memory_allocated=lambda i: 4 * gib,
        get_device_name=lambda i: "Fake GPU",
        get_device_properties=lambda i: types.SimpleNamespace(total_memory=16 * gib),
    )
    torch = types.SimpleNamespace(cuda=cuda, __version__="2.9.0", version=types.SimpleNamespace(cuda="12.8"))
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_system_info_carries_what_the_benchmark_api_requires(no_gpu):
    info = system.system_info()
    assert isinstance(info["cpu_count"], int) and info["cpu_count"] >= 1
    assert isinstance(info["cpu_count_logical"], int) and info["cpu_count_logical"] >= 1
    assert isinstance(info["memory_total_gb"], float)
    assert info["python_version"]
    assert "gpu_name" not in info


def test_system_info_reads_the_gpu_from_nvidia_smi(smi):
    info = system.system_info()
    assert info["gpu_name"] == "NVIDIA RTX 4090"
    assert info["gpu_driver"] == "550.54"
    assert info["gpu_memory_total_gb"] == 24.0
    assert info["gpu_count"] == 1


def test_system_info_falls_back_to_torch_for_the_gpu(no_gpu, fake_torch):
    info = system.system_info()
    assert info["gpu_name"] == "Fake GPU"
    assert info["gpu_memory_total_gb"] == 16.0
    assert info["cuda_version"] == "12.8"


def test_system_metrics_fill_the_system_tab_fields(smi):
    metrics = system.system_metrics()
    gpu = metrics["gpu"]["gpu_0"]
    assert gpu["temperature_celsius"] == 61
    assert gpu["power_watts"] == 285.5
    assert gpu["power_limit_watts"] == 450.0
    assert gpu["fan_speed_percent"] == 38
    assert gpu["power_percent"] == pytest.approx(63.4, abs=0.1)
    assert gpu["memory_used_gb"] == pytest.approx(1830 / 1024, abs=0.01)
    assert "memory_used_gb" in metrics or sys.platform != "linux"


def test_pytorchs_own_memory_numbers_win(smi, fake_torch):
    gpu = system.system_metrics()["gpu"]["gpu_0"]
    assert (gpu["memory_used_gb"], gpu["memory_reserved_gb"], gpu["memory_max_gb"]) == (2.0, 3.0, 4.0)


def test_unsupported_readings_are_left_out(monkeypatch):
    monkeypatch.setattr(system.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    row = "GPU, 1.0, [N/A], [N/A], [N/A], [N/A], [N/A], [N/A], [N/A], [N/A], [N/A], [N/A]"
    monkeypatch.setattr(system, "_run", lambda command, timeout=5.0: row)
    assert system.system_metrics()["gpu"]["gpu_0"] == {}


def test_a_failing_nvidia_smi_means_no_gpu(monkeypatch, no_gpu):
    monkeypatch.setattr(system.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(system, "_run", lambda command, timeout=5.0: None)
    assert "gpu" not in system.system_metrics()


def test_run_swallows_a_missing_command():
    assert system._run(["definitely-not-a-command-visin"]) is None


def test_physical_cores_are_read_from_cpuinfo(monkeypatch):
    cpuinfo = (
        "physical id\t: 0\ncore id\t: 0\n\nphysical id\t: 0\ncore id\t: 1\n\nphysical id\t: 0\ncore id\t: 1\n"
    )
    monkeypatch.setattr(system, "_psutil", lambda: None)
    monkeypatch.setattr(system, "_read", lambda path: cpuinfo if path == "/proc/cpuinfo" else "")
    assert system._physical_cores() == 2


def test_memory_is_read_from_meminfo_without_psutil(monkeypatch):
    meminfo = "MemTotal: 1000 kB\nMemAvailable: 250 kB\n"
    monkeypatch.setattr(system, "_psutil", lambda: None)
    monkeypatch.setattr(system, "_read", lambda path: meminfo if path == "/proc/meminfo" else "")
    used, percent = system._memory_used_bytes()
    assert used == 750 * 1024 and percent == 75.0


def test_psutil_is_used_when_installed(monkeypatch, no_gpu):
    process = types.SimpleNamespace(
        cpu_percent=lambda interval=None: 50.0,
        memory_info=lambda: types.SimpleNamespace(rss=2 * 1024**3),
        num_threads=lambda: 8,
    )
    psutil = types.SimpleNamespace(
        cpu_count=lambda logical=True: 6,
        virtual_memory=lambda: types.SimpleNamespace(total=8 * 1024**3, available=2 * 1024**3, percent=75.0),
        cpu_percent=lambda interval=None: 12.5,
        Process=lambda: process,
    )
    monkeypatch.setattr(system, "_psutil", lambda: psutil)
    assert system.system_info()["cpu_count"] == 6
    metrics = system.system_metrics()
    assert metrics["memory_used_gb"] == 6.0
    assert metrics["cpu_percent"] == 12.5
    assert metrics["process"] == {"cpu_percent": 50.0, "memory_gb": 2.0, "threads": 8}
