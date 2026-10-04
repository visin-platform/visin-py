"""Suites: the written-down way a model is scored.

A suite fixes what makes a score mean something: the data, the conditions and how many samples each has, the
metrics and which way is better, and how the overall figure is formed. Results on one suite version can be
compared; a published version never changes, and any change to what a score means is a new version.

A suite is a small JSON (or YAML, with ``pip install 'visin[yaml]'``) file kept beside your evaluation code:

    {
      "slug": "road-test", "version": 1, "name": "Road scenes, day and night", "visibility": "public",
      "protocol": {
        "task": "semantic-segmentation",
        "data": {"kind": "external", "label": "Road test frames", "manifestSha256": "9f2b…"},
        "split": "test",
        "conditions": [{"name": "day", "sampleCount": 1200}, {"name": "night", "sampleCount": 800}],
        "metrics": [{"key": "mIoU", "direction": "max", "headline": true}],
        "aggregation": "equal-mean-of-conditions",
        "evaluator": {"package": "visin-fusion"}
      }
    }

    suite = visin.push_suite("suites/road-test.json", project="road-seg")

Pushing is safe to repeat: the same protocol again returns the suite that exists, and a different one under
a version that exists is refused, which says to publish it as the next version.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ._internal.config import read_settings
from ._internal.connect import connect
from ._internal.reports import digest_of, discover_project
from .errors import ConfigurationError
from .models import Suite

__all__ = [
    "check_protocol",
    "load_suite",
    "manifest_digest",
    "parse_suite_ref",
    "protocol_of",
    "push_suite",
    "read_split",
]

_REF = re.compile(r"^([a-z0-9][a-z0-9-]{0,62}[a-z0-9]|[a-z0-9])(?:@([1-9][0-9]{0,5}|latest))?$")


def parse_suite_ref(text: str) -> tuple[str, str]:
    """``("road-test", "1")`` from ``road-test@1``; a bare name means ``latest``, the newest not archived."""
    match = _REF.match(text.strip())
    if not match:
        raise ValueError(f"not a suite name: {text!r}; expected slug@version, such as road-test@1")
    return match.group(1), match.group(2) or "latest"


def read_split(path: str | Path) -> list[str]:
    """The samples a split file lists: one per line, blank lines and surrounding spaces left out."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"cannot read the split file {path}: {exc}") from exc
    return [line.strip() for line in text.splitlines() if line.strip()]


def manifest_digest(conditions: Mapping[str, Iterable[str]]) -> str:
    """The identity of the samples a suite scores: the SHA-256 of one canonical manifest.

    ``conditions`` maps each condition's name to the samples scored in it (file names, frame ids: whatever
    names a sample on your side). The manifest is those lists sorted, under sorted condition names, as compact
    JSON, so the same data gives the same digest in any order and on any machine, and a changed, added or
    removed sample changes it. Put it in the suite as ``data.manifestSha256``; ``visin suites manifest``
    prints it from split files.
    """
    manifest = {
        "version": 1,
        "conditions": {name: sorted(samples) for name, samples in sorted(conditions.items())},
    }
    return hashlib.sha256(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_suite(path: str | Path) -> dict[str, Any]:
    """Read a suite file: JSON, or YAML (``.yml`` or ``.yaml``, needs PyYAML)."""
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"cannot read the suite file {file}: {exc}") from exc
    if file.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ConfigurationError(
                f"{file.name} is YAML, which needs PyYAML: pip install 'visin[yaml]' "
                "(or write the suite as JSON)"
            ) from exc
        loaded = yaml.safe_load(text)
    else:
        try:
            loaded = json.loads(text)
        except ValueError as exc:
            raise ConfigurationError(f"{file.name} is not valid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigurationError(
            f"{file.name} should hold one suite: an object with slug, version, name and protocol"
        )
    return loaded


def protocol_of(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """The protocol a suite file, a suite dict or a bare protocol dict holds.

    A path is read as a suite file (see :func:`load_suite`). Whatever is given, the result is the protocol
    itself: the ``protocol`` of a suite, or the dict as it came when it has no such key.
    """
    loaded = dict(source) if isinstance(source, Mapping) else load_suite(source)
    inner = loaded.get("protocol")
    return dict(inner) if isinstance(inner, Mapping) else loaded


def check_protocol(
    source: str | Path | Mapping[str, Any], *, url: str | None = None, token: str | None = None
) -> str:
    """The digest Visin gives a protocol, without publishing it.

    An evaluator sends this as ``protocol_digest`` to say which protocol it ran. The digest is computed by
    the server, which fills in defaults and sorts keys, so it is never reimplemented here and always equals
    the ``digest`` of the suite once published. Needs a server and a login, and stores nothing.
    """
    settings = read_settings(url=url, token=token)
    client = connect(settings, "check a protocol")
    try:
        return digest_of(client, protocol_of(source))
    finally:
        client.close()


def push_suite(
    source: str | Path | Mapping[str, Any],
    *,
    project: str | None = None,
    visibility: str | None = None,
    url: str | None = None,
    token: str | None = None,
) -> Suite:
    """Publish a suite version from a file or a dict, and return it.

    ``project`` (an id or slug) is the project the suite belongs to and whose access rules apply; it
    defaults to the file's own ``project``, then ``VISIN_PROJECT``, then the project a pipeline key is
    limited to. ``visibility`` is ``private`` (the project's readers) or ``public`` (anyone may read the
    protocol, which grants nothing in the project); left out, it is the file's, then the previous version's,
    then private. Needs contribute access.
    """
    suite = dict(source) if isinstance(source, Mapping) else load_suite(source)
    settings = read_settings(url=url, token=token, project=project)
    client = connect(settings, "publish a suite")
    try:
        named = suite.pop("project", None) or suite.get("projectId")
        chosen = project or named or settings.project or discover_project(client)
        if not chosen:
            raise ConfigurationError(
                'a suite needs a project: pass project=, give the file a "project", or set VISIN_PROJECT'
            )
        suite["projectId"] = chosen
        if visibility:
            suite["visibility"] = visibility
        data = client.request("POST", "/suites", json=suite, idempotent=True)
    finally:
        client.close()
    return Suite.from_json(data or {})
