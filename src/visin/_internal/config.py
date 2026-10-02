"""Where configuration comes from.

There is no default API URL, on purpose. The integrations this package replaces
each hard-coded one instance's hostname, so a copy taken into a new project
reported into somebody else's Visin until the line was noticed. Unset means
disabled, which is loud in the log and harmless, rather than wrong and silent.

A setting is looked up in the process environment first, then in the file named
by ``VISIN_ENV_FILE``, then in the user's config file (``visin login`` writes
it). All use ``NAME=value`` lines, with the variable names the environment uses.
A file is read only for ``VISIN_`` names, so a shared ``.env`` cannot leak
anything else into the package. The working directory's ``.env`` is not read:
a library that picks up whichever file sits beside the script would send runs
wherever that file says.

An empty variable counts as unset, so a Compose file's
``VISIN_URL: ${VISIN_URL}`` with nothing exported does not hide the config
file. An argument beats all of these. ``Settings.sources`` says where each
value came from, for ``visin check``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError

HOSTED_URL = "https://vision-api.visin.eu"
HOSTED_DATASET_URL = "https://dataset-api.visin.eu"
HOSTED_APP_URL = "https://app.visin.eu"

ENV_URL = ("VISIN_URL",)
ENV_TOKEN = ("VISIN_TOKEN",)
ENV_TRAINING_UUID = ("VISIN_TRAINING_UUID",)
ENV_PROJECT = ("VISIN_PROJECT",)
ENV_VERIFY = "VISIN_VERIFY_SSL"
ENV_MODE = "VISIN_MODE"
ENV_DIR = "VISIN_DIR"
ENV_DATASET_URL = ("VISIN_DATASET_URL",)
ENV_APP_URL = ("VISIN_APP_URL",)
ENV_DATA_DIR = "VISIN_DATA_DIR"
ENV_ENV_FILE = "VISIN_ENV_FILE"
ENV_CONFIG = "VISIN_CONFIG"

logger = logging.getLogger("visin")

MODES = ("online", "offline", "disabled")


def config_path() -> Path:
    """The user's config file: ``$VISIN_CONFIG``, else ``~/.visin/config``.

    In the default home folder, beside the reports, but not in ``VISIN_DIR``:
    that one may be moved to scratch space for the reports, and the token should
    stay where every job on the machine finds it.
    """
    explicit = os.getenv(ENV_CONFIG)
    if explicit:
        return Path(explicit).expanduser()
    return default_directory() / "config"


def parse_env_file(text: str) -> dict[str, str]:
    """``NAME=value`` lines as a dict: ``export`` and quotes allowed, ``#`` comments ignored.

    Only ``VISIN_`` names are kept. A quoted value is taken whole; an unquoted one
    ends at `` #``.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not name.startswith("VISIN_"):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        values[name] = value
    return values


def _read_file(path: Path, *, required: bool) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        if required:
            raise ConfigurationError(f"{ENV_ENV_FILE} does not point to a file: {path}") from None
        return {}
    except OSError as exc:
        raise ConfigurationError(f"cannot read {path}: {exc}") from exc
    if not required and os.name == "posix":
        mode = path.stat().st_mode & 0o077
        if mode and "VISIN_TOKEN" in text:
            logger.warning("visin: %s holds a token but is readable by others; run: chmod 600 %s", path, path)
    return parse_env_file(text)


def write_config(values: Mapping[str, str], path: Path | None = None) -> Path:
    """Merge ``values`` into the config file and return its path.

    Created with mode 0600 before anything is written, since it holds a token.
    Lines it does not set, comments included, are kept; a name set twice keeps
    only its new line, since the last line of a name is the one read.
    """
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    kept: list[str] = []
    remaining = dict(values)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            name = line.strip().removeprefix("export ").partition("=")[0].strip()
            if name in remaining:
                kept.append(f"{name}={remaining.pop(name)}")
            elif name not in values:
                kept.append(line)
    kept.extend(f"{name}={value}" for name, value in remaining.items())
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write("\n".join(kept) + "\n")
    path.chmod(0o600)
    return path


class _Source:
    """The environment, then the env file, then the config file: one lookup for every setting."""

    def __init__(self) -> None:
        self.files: list[tuple[Path, dict[str, str]]] = []
        explicit = os.getenv(ENV_ENV_FILE)
        if explicit:
            path = Path(explicit).expanduser()
            self.files.append((path, _read_file(path, required=True)))
        path = config_path()
        self.files.append((path, _read_file(path, required=False)))
        self.found: dict[str, str] = {}

    def get(self, name: str) -> str | None:
        value = os.getenv(name)
        if value is not None and value.strip():
            self.found[name] = "environment"
            return value
        for path, values in self.files:
            if name in values:
                self.found[name] = str(path)
                return values[name]
        return None


def _first(source: _Source, names: tuple[str, ...]) -> str | None:
    for name in names:
        value = source.get(name)
        if value and value.strip():
            return value.strip()
    return None


def _flag(source: _Source, name: str, default: bool) -> bool:
    raw = source.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def default_directory() -> Path:
    """Where reports wait when they cannot be sent: ``~/.visin``.

    Home rather than the working directory because on a cluster the login node,
    where ``visin sync`` runs, shares home with the compute nodes and usually
    shares nothing else.
    """
    try:
        return Path.home() / ".visin"
    except RuntimeError:  # no HOME, as in some minimal containers
        return Path(tempfile.gettempdir()) / "visin"


def default_data_directory() -> Path:
    """Where downloaded datasets go: ``$XDG_CACHE_HOME/visin/datasets``, else ``~/.cache/visin/datasets``.

    A cache, not the report directory: a dataset can be downloaded again, and a
    cluster's home quota rarely fits one. Point ``VISIN_DATA_DIR`` at scratch space.
    """
    cache = os.getenv("XDG_CACHE_HOME")
    try:
        base = Path(cache) if cache else Path.home() / ".cache"
    except RuntimeError:  # no HOME, as in some minimal containers
        base = Path(tempfile.gettempdir())
    return base / "visin" / "datasets"


@dataclass(frozen=True)
class Settings:
    url: str | None
    token: str | None
    training_uuid: str | None
    project: str | None
    verify_ssl: bool
    mode: str
    directory: Path
    dataset_url: str | None = None
    data_directory: Path = field(default_factory=default_data_directory)
    app_url: str | None = None
    sources: Mapping[str, str] = field(default_factory=dict, compare=False)

    def run_link(self, training_id: str | None) -> str | None:
        """The web page of a run, when this deployment's app address is known."""
        if not (self.app_url and training_id):
            return None
        return f"{self.app_url.rstrip('/')}/trainings/{training_id}"

    @property
    def configured(self) -> bool:
        """Enough to talk to a server at all.

        The training UUID is not part of this: a run that registers itself has
        none yet, and that is the normal case when a script is launched by hand.
        """
        return bool(self.url and self.token)

    @property
    def effective_mode(self) -> str:
        """What a run will actually do: online without a server is disabled."""
        if self.mode == "online" and not self.configured:
            return "disabled"
        return self.mode


def read_settings(**overrides: Any) -> Settings:
    """Read the environment and config files. Keyword arguments that are not ``None`` win."""
    source = _Source()
    directory = source.get(ENV_DIR)
    data_directory = source.get(ENV_DATA_DIR)
    settings = Settings(
        url=_first(source, ENV_URL),
        token=_first(source, ENV_TOKEN),
        training_uuid=_first(source, ENV_TRAINING_UUID),
        project=_first(source, ENV_PROJECT),
        # Verification stays on unless someone deliberately turns it off for a
        # self-signed dev instance. The scripts this replaces disabled it
        # globally and silenced the warning, which also disabled it in
        # production.
        verify_ssl=_flag(source, ENV_VERIFY, True),
        mode=(source.get(ENV_MODE) or "online").strip().lower(),
        directory=Path(directory).expanduser() if directory else default_directory(),
        dataset_url=_first(source, ENV_DATASET_URL),
        app_url=_first(source, ENV_APP_URL),
        data_directory=Path(data_directory).expanduser() if data_directory else default_data_directory(),
    )
    given = {key: value for key, value in overrides.items() if value is not None}
    for key in ("directory", "data_directory"):
        if key in given:
            given[key] = Path(given[key]).expanduser()
    if "mode" in given:
        given["mode"] = str(given["mode"]).strip().lower()
    names = {
        "url": ENV_URL[0],
        "token": ENV_TOKEN[0],
        "project": ENV_PROJECT[0],
        "dataset_url": ENV_DATASET_URL[0],
    }
    sources = {key: source.found[name] for key, name in names.items() if name in source.found}
    sources.update({key: "argument" for key in given if key in names})
    if given:
        settings = replace(settings, **given)
    if not settings.app_url and settings.url and settings.url.rstrip("/").removesuffix("/api") == HOSTED_URL:
        settings = replace(settings, app_url=HOSTED_APP_URL)
    settings = replace(
        settings, sources={key: value for key, value in sources.items() if getattr(settings, key)}
    )
    if settings.mode not in MODES:
        raise ConfigurationError(f"{ENV_MODE}={settings.mode!r}; expected one of: {', '.join(MODES)}")
    return settings
