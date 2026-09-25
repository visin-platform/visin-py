"""Where configuration comes from.

There is no default API URL, on purpose. The integrations this package replaces
each hard-coded one instance's hostname, so a copy taken into a new project
reported into somebody else's Visin until the line was noticed. Unset means
disabled, which is loud in the log and harmless, rather than wrong and silent.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError

# The first name of each is the one Visin's documentation uses. The others are
# what earlier drafts of this package read, and scripts already export them.
ENV_URL = ("VISIN_URL", "VISIN_API_URL")
ENV_TOKEN = ("VISIN_TOKEN", "VISIN_API_TOKEN")
ENV_TRAINING_UUID = ("VISIN_TRAINING_UUID",)
ENV_PROJECT = ("VISIN_PROJECT", "VISIN_PROJECT_ID")
ENV_VERIFY = "VISIN_VERIFY_SSL"
ENV_MODE = "VISIN_MODE"
ENV_DIR = "VISIN_DIR"

MODES = ("online", "offline", "disabled")


def _first(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
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


@dataclass(frozen=True)
class Settings:
    url: str | None
    token: str | None
    training_uuid: str | None
    project: str | None
    verify_ssl: bool
    mode: str
    directory: Path

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
    """Read the environment. Keyword arguments that are not ``None`` win over it."""
    directory = os.getenv(ENV_DIR)
    settings = Settings(
        url=_first(ENV_URL),
        token=_first(ENV_TOKEN),
        training_uuid=_first(ENV_TRAINING_UUID),
        project=_first(ENV_PROJECT),
        # Verification stays on unless someone deliberately turns it off for a
        # self-signed dev instance. The scripts this replaces disabled it
        # globally and silenced the warning, which also disabled it in
        # production.
        verify_ssl=_flag(ENV_VERIFY, True),
        mode=(os.getenv(ENV_MODE) or "online").strip().lower(),
        directory=Path(directory).expanduser() if directory else default_directory(),
    )
    given = {key: value for key, value in overrides.items() if value is not None}
    if "directory" in given:
        given["directory"] = Path(given["directory"]).expanduser()
    if "mode" in given:
        given["mode"] = str(given["mode"]).strip().lower()
    if given:
        settings = replace(settings, **given)
    if settings.mode not in MODES:
        raise ConfigurationError(f"{ENV_MODE}={settings.mode!r}; expected one of: {', '.join(MODES)}")
    return settings
