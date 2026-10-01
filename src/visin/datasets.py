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
import sys
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ._internal.config import read_settings
from ._internal.transport import HttpClient
from .errors import ApiError, ConfigurationError, VisinError
from .models import Dataset

logger = logging.getLogger("visin")

_OBJECT_ID = re.compile(r"^[0-9a-fA-F]{24}$")
# Written last, after the zip is unpacked: a folder without it is incomplete.
MARKER = ".visin-dataset.json"
PAGE_SIZE = 100
MAX_PAGES = 100


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
        """Release the connection. ``with Datasets() as datasets`` does this for you."""
        self._client.close()

    def __enter__(self) -> Datasets:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ---------------------------------------------------------------- reading

    def list(self, search: str | None = None) -> list[Dataset]:
        """The datasets this credential can read (all public ones without a token)."""
        found: list[Dataset] = []
        seen: set[str] = set()
        for page in range(1, MAX_PAGES + 1):
            params = {"limit": PAGE_SIZE, "page": page, **({"search": search} if search else {})}
            data = self._client.request("GET", "/datasets", params=params)
            pagination = data.get("pagination") if isinstance(data, dict) else None
            if isinstance(data, dict):
                data = data.get("datasets") or data.get("items") or []
            items = [Dataset.from_json(item) for item in data or []]
            fresh = [item for item in items if item.id not in seen]
            # A server that ignores ``page`` answers with the first page again.
            if not fresh:
                break
            seen.update(item.id for item in fresh)
            found.extend(fresh)
            pages = (pagination or {}).get("pages")
            if (pages is not None and page >= pages) or (pages is None and len(items) < PAGE_SIZE):
                break
        return found

    def get(self, ref: str | Dataset) -> Dataset:
        """A dataset by id, or by name (case-insensitive)."""
        if isinstance(ref, Dataset):
            return ref
        if _OBJECT_ID.match(ref):
            try:
                found = self._client.request("GET", f"/datasets/{quote(ref, safe='')}")
                return Dataset.from_json(found or {})
            except ApiError as exc:
                # A name can look like an id; only a 404 means it was not one.
                if exc.status != 404:
                    raise
        datasets = self.list()
        for dataset in datasets:
            if dataset.name.lower() == ref.lower():
                return dataset
        names = sorted(d.name for d in datasets)
        raise VisinError(f"no dataset named {ref!r}; available: {', '.join(names) or 'none'}")

    # ---------------------------------------------------------------- downloading

    def download(
        self, ref: str | Dataset, directory: str | Path | None = None, *, quiet: bool = False
    ) -> Path:
        """The dataset's folder on disk, downloaded and unpacked the first time.

        ``ref`` is a name, an id or a :class:`~visin.models.Dataset` from ``list()``. Returns the folder
        holding the dataset: the zip's single top-level folder when it has one.
        """
        dataset = self.get(ref)
        dataset_id = dataset.id
        if not dataset.downloadable:
            raise VisinError(f"dataset {dataset.name or ref!r} has no zip to download")
        size = dataset.size
        base = Path(directory).expanduser() if directory else self.directory
        target = base / f"{_slug(dataset.name or dataset_id)}-{dataset_id}"

        if _complete(target, size):
            return _dataset_root(target)

        base.mkdir(parents=True, exist_ok=True)
        # A fresh signed URL each time: they expire, and a resumed download needs a live one
        signed = self._client.request("GET", f"/datasets/{quote(dataset_id, safe='')}/download") or {}
        zip_path = base / f"{dataset_id}-{signed.get('filename') or 'dataset.zip'}"
        if not quiet:
            logger.info("visin: downloading dataset %s (%s) to %s", dataset.name, _gigabytes(size), base)
        progress = None if quiet else _Progress(size)
        if progress is not None:
            part = Path(str(zip_path) + ".part")
            progress(part.stat().st_size if part.exists() else 0, size)
        try:
            self._client.download_file(signed["downloadUrl"], str(zip_path), size=size, progress=progress)
        finally:
            if progress is not None:
                progress.close()
        if not quiet:
            logger.info("visin: extracting %s...", dataset.name)

        unpacking = target.with_name(target.name + ".unpacking")
        shutil.rmtree(unpacking, ignore_errors=True)
        _unzip(zip_path, unpacking)
        shutil.rmtree(target, ignore_errors=True)
        unpacking.rename(target)
        (target / MARKER).write_text(
            json.dumps({"id": dataset_id, "name": dataset.name, "size": size, "file": zip_path.name})
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


def _bytes(value: int | float) -> str:
    for unit, scale in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if value >= scale:
            return f"{value / scale:.2f} {unit}"
    return f"{value:.0f} B"


class _Progress:
    """Refresh one terminal line; retain sparse log messages for redirected output."""

    def __init__(self, total: int | None):
        self.total = total
        self.next = 0.0
        self.stream = sys.stderr
        self.terminal = self.stream.isatty()
        self.started = time.monotonic()
        self.last_update: float | None = None
        self.initial: int | None = None
        self.width = 0

    def __call__(self, done: int, total: int | None) -> None:
        total = total or self.total
        now = time.monotonic()
        if self.initial is None or done < self.initial:
            self.initial = done
            self.started = now
        if not self.terminal:
            if total and done / total >= self.next:
                logger.info("visin: %.1f / %.1f GB", done / 2**30, total / 2**30)
                self.next = done / total + 0.05
            return
        if self.last_update is not None and now - self.last_update < 0.2 and not (total and done >= total):
            return
        self.last_update = now
        elapsed = now - self.started
        speed = max(0, done - self.initial) / elapsed if elapsed > 0 else 0
        if total:
            fraction = min(1.0, max(0.0, done / total))
            details = f" {fraction:6.1%}  {_bytes(done)} / {_bytes(total)}"
            if speed > 0:
                remaining = int(max(0, total - done) / speed)
                minutes, seconds = divmod(remaining, 60)
                details += f"  {_bytes(speed)}/s  ETA {minutes:02d}:{seconds:02d}"
            columns = shutil.get_terminal_size().columns - 1
            bar_width = max(4, min(24, columns - len(details) - len("visin: []")))
            filled = int(fraction * bar_width)
            bar = "#" * filled + "-" * (bar_width - filled)
            line = f"visin: [{bar}]{details}"
            if len(line) > columns:
                line = line.split("  ETA", 1)[0]
            if len(line) > columns:
                line = f"visin: {fraction:.1%} {_bytes(done)} / {_bytes(total)}"
        else:
            line = f"visin: downloaded {_bytes(done)}"
            if speed > 0:
                line += f"  {_bytes(speed)}/s"
        self.stream.write("\r" + line.ljust(self.width))
        self.stream.flush()
        self.width = len(line)

    def close(self) -> None:
        """End the terminal line before extraction or an error is printed."""
        if self.terminal and self.width:
            self.stream.write("\n")
            self.stream.flush()
            self.width = 0
