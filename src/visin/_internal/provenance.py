"""What a run was started from: the code, the command, the packages, the machine.

Reported with the run so it can be reproduced without anyone remembering to write it down. Everything is
best effort: a missing ``git``, an unreadable repo or an odd environment leaves that part out and never
stops the run.

A command line can carry secrets, so it is redacted before it leaves the process: the value of any flag
or ``NAME=value`` whose name suggests a credential, and anything that looks like a known token.
``VISIN_PROVENANCE=0`` (or ``init(provenance=False)``) sends none of this.
"""

from __future__ import annotations

import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from .config import ENV_PROVENANCE

SECRET_NAME = re.compile(
    r"(token|secret|passw(or)?d|passwd|api[-_]?key|apikey|credential|auth|private[-_]?key)", re.IGNORECASE
)
SECRET_VALUE = re.compile(
    r"\b(hf_[A-Za-z0-9]{20,}|vsn_(?:live|test)_[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})\b"
)
MAX_PACKAGES = 300
GIT_TIMEOUT = 3.0
REDACTED = "***"


def enabled(requested: bool | None = None) -> bool:
    """Whether provenance is collected: the argument, else ``VISIN_PROVENANCE`` (on unless off)."""
    if requested is not None:
        return requested
    return os.environ.get(ENV_PROVENANCE, "1").strip().lower() not in {"0", "false", "off", "no"}


def redact_command(argv: list[str]) -> str:
    """The command line with credential values replaced by ``***``."""
    out: list[str] = []
    hide_next = False
    for argument in argv:
        if hide_next:
            out.append(REDACTED)
            hide_next = False
            continue
        name, sep, _ = argument.partition("=")
        if sep and SECRET_NAME.search(name.lstrip("-")):
            out.append(f"{name}={REDACTED}")
        elif argument.startswith("-") and SECRET_NAME.search(argument) and not sep:
            out.append(argument)
            hide_next = True
        else:
            out.append(SECRET_VALUE.sub(REDACTED, argument))
    return shlex.join(out)


def _git(directory: Path, *args: str) -> str | None:
    executable = shutil.which("git")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "-C", str(directory), *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_info(directory: Path) -> dict[str, Any] | None:
    """The commit, branch, dirtiness and remote of the repo holding ``directory``, or ``None`` outside one."""
    commit = _git(directory, "rev-parse", "HEAD")
    if not commit:
        return None
    info: dict[str, Any] = {"commit": commit}
    branch = _git(directory, "rev-parse", "--abbrev-ref", "HEAD")
    if branch and branch != "HEAD":
        info["branch"] = branch[:250]
    status = _git(directory, "status", "--porcelain")
    if status is not None:
        info["dirty"] = bool(status)
    remote = _git(directory, "config", "--get", "remote.origin.url")
    if remote:
        info["remote"] = re.sub(r"^([a-z][a-z0-9+.-]*://)[^/@\s]+@", r"\1", remote, flags=re.IGNORECASE)[:500]
    return info


def packages() -> dict[str, str]:
    """Installed distributions and their versions, by name, at most ``MAX_PACKAGES`` of them."""
    found: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        if name and len(name) <= 100 and distribution.version and len(distribution.version) <= 100:
            found.setdefault(name.lower(), distribution.version)
    return dict(sorted(found.items())[:MAX_PACKAGES])


def host() -> dict[str, str]:
    """The machine: hostname, platform, Python, and CUDA when ``torch`` is already imported."""
    info = {
        "hostname": socket.gethostname()[:255],
        "platform": platform.platform()[:255],
        "python": platform.python_version()[:100],
    }
    torch = sys.modules.get("torch")
    cuda = getattr(getattr(torch, "version", None), "cuda", None)
    if isinstance(cuda, str):
        info["cuda"] = cuda[:100]
    return info


def collect() -> dict[str, Any]:
    """Everything above, from the script that is running; parts that cannot be read are left out."""
    script = (
        Path(sys.argv[0]).resolve().parent
        if sys.argv and sys.argv[0] and Path(sys.argv[0]).exists()
        else Path.cwd()
    )
    collected: dict[str, Any] = {}
    git = git_info(script) or (git_info(Path.cwd()) if script != Path.cwd() else None)
    if git:
        collected["git"] = git
    if sys.argv and sys.argv[0]:
        collected["command"] = redact_command([sys.executable, *sys.argv] if sys.argv[0] else sys.argv)[:4000]
    found = packages()
    if found:
        collected["packages"] = found
    collected["host"] = host()
    return collected
