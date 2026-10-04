"""Opening a client for the commands that write one thing and leave."""

from __future__ import annotations

from ..errors import ConfigurationError
from .config import HOSTED_URL, Settings
from .transport import HttpClient


def connect(settings: Settings, what: str) -> HttpClient:
    """A client for ``settings``, or a ConfigurationError saying what could not be done without a server."""
    if not (settings.url and settings.token):
        raise ConfigurationError(
            f"nowhere to {what}: set VISIN_URL and VISIN_TOKEN, or pass url= and token= "
            f"(hosted Visin: {HOSTED_URL})"
        )
    return HttpClient(settings.url, settings.token, verify=settings.verify_ssl)
