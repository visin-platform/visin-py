"""Datasets on Visin: list them, and download one to disk once.

Datasets are served by Visin's dataset service, at its own address:
``VISIN_DATASET_URL`` (e.g. ``https://dataset-api.visin.eu``). Public datasets
need no token; ``VISIN_TOKEN`` is sent when set, for the private ones it may read.

    from visin import Datasets

    with Datasets() as datasets:
        for dataset in datasets.list():
            print(dataset["name"])
        root = datasets.download("zod")  # a folder; downloaded the first time only

A download resumes where an interrupted one stopped, is checked against the size
Visin reports, and is unpacked into ``VISIN_DATA_DIR`` (default
``~/.cache/visin/datasets``). Later calls find it there and download nothing.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ._internal.config import read_settings
from ._internal.transport import HttpClient
from .errors import ConfigurationError, VisinError

logger = logging.getLogger("visin")

_OBJECT_ID = re.compile(r"^[0-9a-fA-F]{24}$")
# Written last, after the zip is unpacked: a folder without it is incomplete.
MARKER = ".visin-dataset.json"


class Datasets:
    """A client for one Visin instance's datasets."""

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        *,
        directory: str | Path | None = None,
        verify_ssl: bool | None = None,
        client: HttpClient | None = None,
    ):
        settings = read_settings(
            dataset_url=url, token=token, data_directory=directory, verify_ssl=verify_ssl
        )
        if client is None:
            if not settings.dataset_url:
                raise ConfigurationError("no Visin dataset service: pass url= or set VISIN_DATASET_URL")
            client = HttpClient(settings.dataset_url, settings.token, verify=settings.verify_ssl)
        self._client = client
        self.directory = settings.data_directory

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Datasets:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ---------------------------------------------------------------- reading

    def list(self, search: str | None = None) -> list[dict[str, Any]]:
        """The datasets this credential can read (all public ones without a token)."""
        params = {"limit": 100, **({"search": search} if search else {})}
        data = self._client.request("GET", "/datasets", params=params)
        if isinstance(data, dict):
            data = data.get("datasets") or data.get("items") or []
        return [dict(item) for item in data or []]

    def get(self, ref: str | dict[str, Any]) -> dict[str, Any]:
        """A dataset by id, or by name (case-insensitive)."""
        if isinstance(ref, dict):
            return ref
        if _OBJECT_ID.match(ref):
            return dict(self._client.request("GET", f"/datasets/{quote(ref, safe='')}") or {})
        datasets = self.list()
        for dataset in datasets:
            if str(dataset.get("name", "")).lower() == ref.lower():
                return dataset
        names = sorted(str(d.get("name")) for d in datasets)
        raise VisinError(f"no dataset named {ref!r}; available: {', '.join(names) or 'none'}")

    # ---------------------------------------------------------------- downloading

    def download(
        self, ref: str | dict[str, Any], directory: str | Path | None = None, *, quiet: bool = False
    ) -> Path:
        """The dataset's folder on disk, downloaded and unpacked the first time.

        ``ref`` is a name, an id or a dataset from ``list()``. Returns the folder
        holding the dataset: the zip's single top-level folder when it has one.
        """
        dataset = self.get(ref)
        dataset_id = str(dataset.get("_id") or dataset.get("id") or "")
        archive = dataset.get("archive") or {}
        if not dataset_id or not archive:
            raise VisinError(f"dataset {dataset.get('name', ref)!r} has no zip to download")
        size = archive.get("size")
        base = Path(directory).expanduser() if directory else self.directory
        target = base / f"{_slug(str(dataset.get('name') or dataset_id))}-{dataset_id}"

        if _complete(target, size):
            return _dataset_root(target)

        base.mkdir(parents=True, exist_ok=True)
        # A fresh signed URL each time: they expire, and a resumed download needs a live one
        signed = self._client.request("GET", f"/datasets/{quote(dataset_id, safe='')}/download") or {}
        zip_path = base / f"{dataset_id}-{signed.get('filename') or 'dataset.zip'}"
        if not quiet:
            logger.info(
                "visin: downloading dataset %s (%s) to %s", dataset.get("name"), _gigabytes(size), base
            )
        self._client.download_file(
            signed["downloadUrl"], str(zip_path), size=size, progress=None if quiet else _Progress(size)
        )

        unpacking = target.with_name(target.name + ".unpacking")
        shutil.rmtree(unpacking, ignore_errors=True)
        _unzip(zip_path, unpacking)
        shutil.rmtree(target, ignore_errors=True)
        unpacking.rename(target)
        (target / MARKER).write_text(
            json.dumps({"id": dataset_id, "name": dataset.get("name"), "size": size, "file": zip_path.name})
        )
        zip_path.unlink()
        return _dataset_root(target)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "dataset"


def _gigabytes(size: int | None) -> str:
    return f"{size / 2**30:.1f} GB" if size else "unknown size"


def _complete(target: Path, size: int | None) -> bool:
    marker = target / MARKER
    if not marker.exists():
        return False
    try:
        return bool(json.loads(marker.read_text()).get("size") == size)
    except ValueError:
        return False


def _unzip(zip_path: Path, destination: Path) -> None:
    """Unpack, refusing any entry that would land outside ``destination``."""
    destination.mkdir(parents=True)
    root = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if not (root / member.filename).resolve().is_relative_to(root):
                raise VisinError(
                    f"{zip_path.name}: entry {member.filename!r} points outside the dataset folder"
                )
        archive.extractall(destination)


def _dataset_root(target: Path) -> Path:
    """The dataset inside an unpacked zip: its single top-level folder, or the folder itself."""
    entries = [e for e in target.iterdir() if e.name != MARKER and not e.name.startswith(("__MACOSX", "."))]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return target


class _Progress:
    """Logs every 5% of a download."""

    def __init__(self, total: int | None):
        self.total = total
        self.next = 0.0

    def __call__(self, done: int, total: int | None) -> None:
        total = total or self.total
        if not total:
            return
        if done / total >= self.next:
            logger.info("visin: %.1f / %.1f GB", done / 2**30, total / 2**30)
            self.next = done / total + 0.05
