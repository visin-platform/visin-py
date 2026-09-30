"""What the machine is, and how much of it a run is using.

Two different questions, answered for two different places in Visin:

* :func:`system_info` describes the host, once. A benchmark must carry it:
  the API refuses a benchmark whose ``system_info`` lacks ``cpu_count``,
  ``cpu_count_logical`` and ``memory_total_gb``, and the benchmark page shows
  those plus the GPU's name, memory and driver.
* :func:`system_metrics` is a snapshot of use, per epoch. Sent as
  ``results["system_info"]``, it fills the run's System tab: memory used and
  peak, and per GPU its memory, temperature, power and fan.

Everything here is best effort and standard-library first. ``psutil`` makes
the numbers better when it is installed (``pip install 'visin[system]'``);
PyTorch is consulted only if the script already imported it, and
``nvidia-smi`` only if it is on the path. Nothing here raises.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from typing import Any

logger = logging.getLogger("visin")

_GB = 1024**3


def _psutil() -> Any:
    try:
        import psutil  # type: ignore[import-untyped]

        return psutil
    except ImportError:
        return None


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return ""


def _physical_cores() -> int:
    psutil = _psutil()
    if psutil:
        count = psutil.cpu_count(logical=False)
        if count:
            return int(count)
    cores = set()
    physical = core = None
    for line in _read("/proc/cpuinfo").splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        if key == "physical id":
            physical = value.strip()
        elif key == "core id":
            core = value.strip()
        elif not line.strip() and core is not None:
            cores.add((physical, core))
            physical = core = None
    if core is not None:
        cores.add((physical, core))
    return len(cores) or (os.cpu_count() or 1)


def _memory_total_bytes() -> int:
    psutil = _psutil()
    if psutil:
        return int(psutil.virtual_memory().total)
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return 0


def _cpu_model() -> str | None:
    for line in _read("/proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            return line.partition(":")[2].strip()
    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out:
            return out.strip()
    return platform.processor() or None


def _run(command: list[str], timeout: float = 5.0) -> str | None:
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def _number(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:  # "[N/A]", "[Not Supported]"
        return None


_SMI_FIELDS = (
    "name",
    "driver_version",
    "memory.total",
    "memory.used",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "fan.speed",
    "utilization.gpu",
    "utilization.memory",
    "clocks.sm",
    "clocks.mem",
)


def _nvidia_smi() -> list[dict[str, str]]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(["nvidia-smi", f"--query-gpu={','.join(_SMI_FIELDS)}", "--format=csv,noheader,nounits"])
    if not out:
        return []
    rows = []
    for line in out.strip().splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(_SMI_FIELDS):
            rows.append(dict(zip(_SMI_FIELDS, values)))
    return rows


def _torch_cuda() -> Any:
    """``torch.cuda``, if the script already imported torch and has a GPU.

    Never imports torch itself: that would cost a second or more and a
    gigabyte of address space in a script that had no use for it.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        return torch.cuda if torch.cuda.is_available() else None
    except Exception:
        return None


def system_info() -> dict[str, Any]:
    """Describe this machine, in the fields Visin's benchmark page reads."""
    info: dict[str, Any] = {
        "cpu_count": _physical_cores(),
        "cpu_count_logical": os.cpu_count() or 1,
        "memory_total_gb": round(_memory_total_bytes() / _GB, 1),
        "os": platform.platform(),
        "python_version": platform.python_version(),
    }
    info["cpu_count_physical"] = info["cpu_count"]
    model = _cpu_model()
    if model:
        info["cpu"] = model
    gpus = _nvidia_smi()
    if gpus:
        first = gpus[0]
        info["gpu_name"] = first["name"]
        total = _number(first["memory.total"])
        if total is not None:
            info["gpu_memory_total_gb"] = round(total / 1024, 1)
        info["gpu_driver"] = first["driver_version"]
        info["gpu_count"] = len(gpus)
        if len(gpus) > 1:
            info["gpu_names"] = [gpu["name"] for gpu in gpus]
    cuda = _torch_cuda()
    if cuda is not None:
        torch = sys.modules["torch"]
        info["torch_version"] = getattr(torch, "__version__", None)
        version = getattr(getattr(torch, "version", None), "cuda", None)
        if version:
            info["cuda_version"] = version
        if "gpu_name" not in info:
            try:
                info["gpu_name"] = cuda.get_device_name(0)
                info["gpu_memory_total_gb"] = round(cuda.get_device_properties(0).total_memory / _GB, 1)
                info["gpu_count"] = cuda.device_count()
            except Exception:
                pass
    return info


def _process_peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Kilobytes on Linux, bytes on macOS.
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _memory_used_bytes() -> tuple[int | None, float | None]:
    psutil = _psutil()
    if psutil:
        memory = psutil.virtual_memory()
        return int(memory.total - memory.available), float(memory.percent)
    meminfo = _read("/proc/meminfo")
    values = {}
    for line in meminfo.splitlines():
        match = re.match(r"(\w+):\s+(\d+)", line)
        if match:
            values[match.group(1)] = int(match.group(2)) * 1024
    if "MemTotal" in values and "MemAvailable" in values:
        used = values["MemTotal"] - values["MemAvailable"]
        return used, round(100.0 * used / values["MemTotal"], 1)
    return None, None


def system_metrics() -> dict[str, Any]:
    """A snapshot of what this process and machine are using right now.

    Shaped for the run's System tab, which reads it from each epoch's
    ``results["system_info"]``: ``memory_used_gb`` and ``memory_max_gb``, and
    ``gpu.gpu_0`` onwards with memory, temperature, power and fan speed.
    """
    metrics: dict[str, Any] = {}
    used, percent = _memory_used_bytes()
    if used is not None:
        metrics["memory_used_gb"] = round(used / _GB, 2)
    if percent is not None:
        metrics["memory_percent"] = percent
    total = _memory_total_bytes()
    if total:
        metrics["memory_total_gb"] = round(total / _GB, 1)
    peak = _process_peak_rss_bytes()
    if peak is not None:
        metrics["memory_max_gb"] = round(peak / _GB, 2)

    psutil = _psutil()
    if psutil:
        try:
            metrics["cpu_percent"] = psutil.cpu_percent(interval=None)
            process = psutil.Process()
            metrics["process"] = {
                "cpu_percent": process.cpu_percent(interval=None),
                "memory_gb": round(process.memory_info().rss / _GB, 2),
                "threads": process.num_threads(),
            }
        except Exception:
            pass

    gpus: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(_nvidia_smi()):
        gpu: dict[str, Any] = {}
        for field, key, scale in (
            ("memory.used", "memory_used_gb", 1 / 1024),
            ("temperature.gpu", "temperature_celsius", 1),
            ("power.draw", "power_watts", 1),
            ("power.limit", "power_limit_watts", 1),
            ("fan.speed", "fan_speed_percent", 1),
            ("utilization.gpu", "gpu_utilization_percent", 1),
            ("utilization.memory", "memory_utilization_percent", 1),
            ("clocks.sm", "clock_sm_mhz", 1),
            ("clocks.mem", "clock_memory_mhz", 1),
        ):
            value = _number(row[field])
            if value is not None:
                gpu[key] = round(value * scale, 2)
        if "power_watts" in gpu and gpu.get("power_limit_watts"):
            gpu["power_percent"] = round(100 * gpu["power_watts"] / gpu["power_limit_watts"], 1)
        gpus[f"gpu_{index}"] = gpu

    cuda = _torch_cuda()
    if cuda is not None:
        # What PyTorch itself holds is the number a training script can act
        # on, so it wins over nvidia-smi's, which counts every process.
        try:
            for index in range(cuda.device_count()):
                gpu = gpus.setdefault(f"gpu_{index}", {})
                gpu["memory_used_gb"] = round(cuda.memory_allocated(index) / _GB, 2)
                gpu["memory_reserved_gb"] = round(cuda.memory_reserved(index) / _GB, 2)
                gpu["memory_max_gb"] = round(cuda.max_memory_allocated(index) / _GB, 2)
        except Exception:
            pass

    if gpus:
        metrics["gpu"] = gpus
    return metrics
