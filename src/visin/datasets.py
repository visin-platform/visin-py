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

A dataset kept on the Hugging Face Hub is fetched from there, at the exact commit
Visin names, into the same folder layout (``pip install 'visin[hf]'``; a private
repo needs your own ``HF_TOKEN``). When the Hub cannot be reached, or the extra
is not installed, the zip kept on Visin is used if the dataset has one.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ._internal import hub
from ._internal.config import HOSTED_DATASET_URL, read_settings
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
                raise ConfigurationError(
                    "no Visin dataset service: pass url= or set VISIN_DATASET_URL "
                    f"(hosted Visin: {HOSTED_DATASET_URL})"
                )
            client = HttpClient(settings.dataset_url, settings.token, verify=settings.verify_ssl)
        self._client = client
        self._anonymous_reads = False
        self.directory = settings.data_directory

    def close(self) -> None:
        """Release the connection. ``with Datasets() as datasets`` does this for you."""
        self._client.close()

    def __enter__(self) -> Datasets:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ---------------------------------------------------------------- reading

    def _read(self, path: str, **kwargs: Any) -> Any:
        """Older servers refuse pipeline keys before public visibility; retry only that refusal."""
        try:
            return self._client.request("GET", path, **kwargs, anonymous=self._anonymous_reads)
        except ApiError as exc:
            if exc.status != 403 or "This key is limited to one project and cannot reach this API." not in (
                exc.body or str(exc)
            ):
                raise
            self._anonymous_reads = True
            return self._client.request("GET", path, anonymous=True, **kwargs)

    def list(self, search: str | None = None) -> list[Dataset]:
        """The datasets this credential can read (all public ones without a token)."""
        found: list[Dataset] = []
        seen: set[str] = set()
        for page in range(1, MAX_PAGES + 1):
            params = {"limit": PAGE_SIZE, "page": page, **({"search": search} if search else {})}
            data = self._read("/datasets", params=params)
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
                found = self._read(f"/datasets/{quote(ref, safe='')}")
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
        self,
        ref: str | Dataset,
        directory: str | Path | None = None,
        *,
        quiet: bool = False,
        unzip: bool = True,
        keep_archive: bool = False,
    ) -> Path:
        """The dataset's folder on disk, downloaded and unpacked the first time.

        ``ref`` is a name, an id or a :class:`~visin.models.Dataset` from ``list()``. Returns the folder
        holding the dataset: the zip's single top-level folder when it has one.
        With ``unzip=False``, return the downloaded ZIP path instead and always retain it.
        With ``keep_archive=True``, retain the ZIP alongside the extracted dataset.
        """
        dataset = self.get(ref)
        dataset_id = dataset.id
        if not dataset.downloadable:
            raise VisinError(
                f"dataset {dataset.name or ref!r} has no zip to download, and no Hugging Face source"
            )
        size = dataset.size
        base = Path(directory).expanduser() if directory else self.directory
        target = base / f"{_slug(dataset.name or dataset_id)}-{dataset_id}"

        archived = bool(dataset.raw.get("archive"))
        revision = (dataset.raw.get("archive") or {}).get("uploadedAt")
        hub_source = dataset.source
        wants_folder = unzip and not keep_archive
        if wants_folder and hub_source and _complete(target, None, hub_source["revision"], hub_source):
            return _dataset_root(target)
        extracted = archived and _complete(target, size, revision)
        if extracted and wants_folder and not hub_source:
            return _dataset_root(target)

        base.mkdir(parents=True, exist_ok=True)
        signed = self._read(f"/datasets/{quote(dataset_id, safe='')}/download") or {}
        hub_source = signed.get("source") or None
        if hub_source and wants_folder:
            try:
                return self._from_hub(dataset, hub_source, target, quiet)
            except VisinError as exc:
                if not signed.get("downloadUrl"):
                    raise
                logger.warning("visin: %s; using the zip kept on Visin instead", exc)
        if not signed.get("downloadUrl"):
            raise VisinError(
                f"dataset {dataset.name or ref!r} is on Hugging Face only, so there is no zip to download"
            )
        revision = signed.get("archiveRevision") or signed.get("revision", revision)
        size = signed.get("size", size)
        extracted = _complete(target, size, revision)
        if extracted and wants_folder:
            return _dataset_root(target)
        archive_version = f"{_slug(revision)}-" if revision else ""
        filename = str(signed.get("filename") or "dataset.zip").replace("\\", "/").rsplit("/", 1)[-1]
        zip_path = base / f"{dataset_id}-{archive_version}{filename or 'dataset.zip'}"
        if not zip_path.is_file() or (size is not None and zip_path.stat().st_size != size):
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

        if not unzip:
            return zip_path
        if extracted:
            return _dataset_root(target)
        if not quiet:
            logger.info("visin: extracting %s...", dataset.name)

        unpacking = target.with_name(target.name + ".unpacking")
        shutil.rmtree(unpacking, ignore_errors=True)
        _unzip(zip_path, unpacking)
        shutil.rmtree(target, ignore_errors=True)
        unpacking.rename(target)
        (target / MARKER).write_text(
            json.dumps(
                {
                    "id": dataset_id,
                    "name": dataset.name,
                    "revision": revision,
                    "size": size,
                    "file": zip_path.name,
                }
            )
        )
        if not keep_archive:
            try:
                zip_path.unlink()
            except OSError as exc:
                logger.warning("visin: dataset extracted, but could not delete archive %s: %s", zip_path, exc)
        return _dataset_root(target)

    def push(
        self,
        ref: str | Dataset,
        repo: str,
        *,
        private: bool = True,
        directory: str | Path | None = None,
        quiet: bool = False,
    ) -> str:
        """Publish a dataset kept on Visin to the Hugging Face Hub and point Visin at it.

        Downloads the dataset if it is not on this machine, uploads its folder to
        the dataset repo ``repo`` (``org/name``, created if needed, private unless
        ``private=False``) with your own Hub token, then tells Visin the repo and
        the commit that upload made. The zip kept on Visin stays as the fallback.
        Needs ``pip install 'visin[hf]'`` and a token that may manage the dataset.
        Returns the commit hash.

        Do this only for data whose licence allows redistribution.
        """
        dataset = self.get(ref)
        if dataset.source:
            raise VisinError(
                f"dataset {dataset.name!r} is already on Hugging Face "
                f"({dataset.source['repo']}@{hub.short(dataset.source['revision'])})"
            )
        if not dataset.raw.get("archive"):
            raise VisinError(f"dataset {dataset.name!r} has no zip on Visin to publish")
        folder = self.download(dataset, directory, quiet=quiet)
        if not quiet:
            logger.info("visin: uploading dataset %s to Hugging Face (%s)", dataset.name, repo)
        revision = hub.upload_dataset(folder, repo, private=private, ignore=[MARKER])
        try:
            self._client.request(
                "PATCH",
                f"/datasets/{quote(dataset.id, safe='')}",
                json={"source": {"repo": repo, "revision": revision}},
            )
        except ApiError as exc:
            raise VisinError(
                f"uploaded to {repo}@{hub.short(revision)}, but Visin did not accept the link ({exc}). "
                f"Set it on the dataset's page with commit {revision}."
            ) from exc
        return revision

    def _from_hub(self, dataset: Dataset, source: dict[str, Any], target: Path, quiet: bool) -> Path:
        """Download the dataset's Hub repo at its pinned commit into ``target``, once."""
        repo, revision = str(source["repo"]), str(source["revision"])
        if _complete(target, None, revision, source):
            return _dataset_root(target)
        if not quiet:
            logger.info(
                "visin: downloading dataset %s from Hugging Face (%s@%s)",
                dataset.name,
                repo,
                hub.short(revision),
            )
        unpacking = target.with_name(target.name + ".unpacking")
        pending = unpacking / ".cache" / "visin-source.json"
        try:
            same_source = json.loads(pending.read_text()) == source
        except (OSError, ValueError):
            same_source = False
        if not same_source:
            shutil.rmtree(unpacking, ignore_errors=True)
        pending.parent.mkdir(parents=True, exist_ok=True)
        pending.write_text(json.dumps(source))
        hub.download_dataset(repo, revision, unpacking)
        shutil.rmtree(target, ignore_errors=True)
        unpacking.rename(target)
        (target / MARKER).write_text(
            json.dumps(
                {
                    "id": dataset.id,
                    "name": dataset.name,
                    "revision": revision,
                    "size": None,
                    "file": None,
                    "source": {"provider": "hf", "repo": repo, "revision": revision},
                }
            )
        )
        return _dataset_root(target)


@dataclass(frozen=True)
class CachedDataset:
    """Something downloaded datasets left on disk.

    ``kind`` is ``dataset`` (unpacked and complete), ``archive`` (a finished ZIP
    kept with ``keep_archive`` or ``unzip=False``), or ``partial`` (an
    interrupted download or unpack, which the next download resumes or redoes).
    """

    kind: str
    name: str
    id: str | None
    path: Path
    size: int


def _disk_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def cached_datasets(directory: str | Path | None = None) -> list[CachedDataset]:
    """What is in the data directory, largest first. No server is needed."""
    base = Path(directory).expanduser() if directory else read_settings().data_directory
    if not base.is_dir():
        return []
    found = []
    for entry in sorted(base.iterdir()):
        if entry.is_dir() and (entry / MARKER).exists():
            try:
                info = json.loads((entry / MARKER).read_text())
            except ValueError:
                info = {}
            found.append(
                CachedDataset(
                    "dataset", info.get("name") or entry.name, info.get("id"), entry, _disk_size(entry)
                )
            )
        elif entry.is_dir() and entry.name.endswith(".unpacking"):
            slug, _, dataset_id = entry.name.removesuffix(".unpacking").rpartition("-")
            found.append(CachedDataset("partial", slug, dataset_id or None, entry, _disk_size(entry)))
        elif entry.is_file() and entry.name.endswith((".zip", ".zip.part")):
            kind = "partial" if entry.name.endswith(".part") else "archive"
            dataset_id = entry.name.split("-", 1)[0]
            found.append(CachedDataset(kind, entry.name, dataset_id, entry, _disk_size(entry)))
    return sorted(found, key=lambda item: -item.size)


def remove_cached(ref: str, directory: str | Path | None = None) -> list[CachedDataset]:
    """Delete a downloaded dataset by name or id, with its ZIP and any partial download.

    Returns what was removed, and raises :class:`~visin.errors.VisinError` when nothing matches.
    """
    wanted = ref.strip().lower()
    items = cached_datasets(directory)
    named = {
        item.id
        for item in items
        if item.id
        and (
            item.id.lower() == wanted
            or (item.kind == "dataset" and item.name.lower() == wanted)
            or (item.kind == "partial" and item.path.is_dir() and item.name == _slug(ref))
        )
    }
    doomed = [item for item in items if item.id in named]
    if not doomed:
        raise VisinError(f"nothing downloaded matches {ref!r}")
    for item in doomed:
        if item.path.is_dir():
            shutil.rmtree(item.path)
        else:
            item.path.unlink()
    return doomed


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "dataset"


def _gigabytes(size: int | None) -> str:
    return f"{size / 2**30:.1f} GB" if size else "unknown size"


def _complete(
    target: Path, size: int | None, revision: str | None = None, source: dict[str, Any] | None = None
) -> bool:
    marker = target / MARKER
    if not marker.exists():
        return False
    try:
        saved = json.loads(marker.read_text())
        return bool(
            isinstance(saved, dict)
            and "size" in saved
            and saved["size"] == size
            and (revision is None or saved.get("revision") == revision)
            and (source is None or saved.get("source") == source)
        )
    except (OSError, ValueError):
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
