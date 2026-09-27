"""Reading what Visin holds: projects, runs, and their epochs and results.

For notebooks and analysis scripts. Visin's documentation recommends a user API
key with read scopes for this (``vsn_live_…``, made under **Account → API
keys**); a pipeline key works too, for its one project.

    from visin import Api

    api = Api()                        # VISIN_URL and VISIN_TOKEN
    for run in api.trainings(project="road-seg", status="completed"):
        print(run["name"])
    frame = api.epochs_frame(run)      # one row per epoch, needs pandas
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from typing import Any
from urllib.parse import quote

from ._internal.config import read_settings
from ._internal.transport import HttpClient
from .errors import ConfigurationError

_OBJECT_ID = re.compile(r"^[0-9a-fA-F]{24}$")

# The server's cap on one page.
MAX_PAGE_SIZE = 1000


def flatten(results: Mapping[str, Any], sep: str = ".", prefix: str = "") -> dict[str, Any]:
    """Nested results as one level: ``{"val": {"loss": 1}}`` → ``{"val.loss": 1}``.

    Lists are kept whole, as values: they are rarely metrics, and exploding
    them into numbered columns makes a frame nobody can read.
    """
    flat: dict[str, Any] = {}
    for key, value in results.items():
        name = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, Mapping) and value:
            flat.update(flatten(value, sep, name))
        else:
            flat[name] = value
    return flat


class Api:
    """A read client for one Visin instance."""

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        *,
        verify_ssl: bool | None = None,
        client: HttpClient | None = None,
    ):
        if client is None:
            settings = read_settings(url=url, token=token, verify_ssl=verify_ssl)
            if not settings.url:
                raise ConfigurationError("no Visin to read from: pass url= or set VISIN_URL")
            client = HttpClient(settings.url, settings.token, verify=settings.verify_ssl)
        self._client = client

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Api:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def get(self, path: str, **params: Any) -> Any:
        """Any GET under ``/api``, unwrapped: for endpoints this class has no method for."""
        return self._client.request("GET", path, params={k: v for k, v in params.items() if v is not None})

    def _pages(
        self, path: str, key: str, params: Mapping[str, Any], page_size: int
    ) -> Iterator[dict[str, Any]]:
        page = 1
        size = max(1, min(page_size, MAX_PAGE_SIZE))
        while True:
            data = self.get(path, **params, page=page, limit=size) or {}
            items = data.get(key) or []
            yield from items
            pages = (data.get("pagination") or {}).get("pages")
            if not items or (pages is not None and page >= pages) or (pages is None and len(items) < size):
                return
            page += 1

    # ---------------------------------------------------------------- projects

    def projects(self, search: str | None = None) -> list[dict[str, Any]]:
        """The projects this credential can see."""
        return list(self.get("/projects", search=search) or [])

    def project(self, id_or_slug: str) -> dict[str, Any]:
        return dict(self.get(f"/projects/{quote(id_or_slug, safe='')}") or {})

    # ---------------------------------------------------------------- runs

    def trainings(
        self,
        *,
        project: str | None = None,
        status: str | None = None,
        tags: Iterable[str] | str | None = None,
        search: str | None = None,
        dataset: str | None = None,
        sort_by: str = "updatedAt",
        order: str = "desc",
        limit: int | None = None,
        page_size: int = 100,
    ) -> Iterator[dict[str, Any]]:
        """Runs, newest first, fetched a page at a time as you iterate.

        ``limit`` stops after that many; without it, every matching run.
        """
        params = {
            "projectId": project,
            "status": status,
            "tags": tags if isinstance(tags, str) or tags is None else ",".join(tags),
            "search": search,
            "datasetId": dataset,
            "sortBy": sort_by,
            "order": order,
        }
        params = {key: value for key, value in params.items() if value is not None}
        if limit is not None:
            page_size = min(page_size, limit)
        for count, training in enumerate(self._pages("/trainings", "trainings", params, page_size)):
            if limit is not None and count >= limit:
                return
            yield training

    def training(self, ref: str | Mapping[str, Any]) -> dict[str, Any]:
        """One run, by its id, its UUID, or a run dict from :meth:`trainings`."""
        if isinstance(ref, Mapping):
            return dict(ref)
        if _OBJECT_ID.match(ref):
            return dict(self.get(f"/trainings/{ref}") or {})
        return dict(self.get(f"/trainings/uuid/{quote(ref, safe='')}") or {})

    def _training_ids(self, ref: str | Mapping[str, Any]) -> tuple[str, str]:
        training = (
            ref if isinstance(ref, Mapping) and ref.get("_id") and ref.get("uuid") else self.training(ref)
        )
        return str(training["_id"]), str(training["uuid"])

    def epochs(self, ref: str | Mapping[str, Any]) -> list[dict[str, Any]]:
        """Every epoch of a run, in epoch order."""
        ident, _ = self._training_ids(ref)
        return list(
            self._pages(f"/epochs/training/{ident}", "epochs", {"sortBy": "epoch", "order": "asc"}, 1000)
        )

    def test_results(self, ref: str | Mapping[str, Any]) -> list[dict[str, Any]]:
        """The run's test results, newest first."""
        _, uuid = self._training_ids(ref)
        return list(self._pages("/test-results", "testResults", {"training_uuid": uuid}, 500))

    def benchmarks(
        self,
        ref: str | Mapping[str, Any] | None = None,
        *,
        project: str | None = None,
    ) -> list[dict[str, Any]]:
        """Benchmarks of a run, or of a whole project, newest first."""
        params: dict[str, Any] = {}
        if ref is not None:
            params["training_uuid"] = self._training_ids(ref)[1]
        if project is not None:
            params["projectId"] = project
        return list(self._pages("/benchmarks", "benchmarks", params, 500))

    # ---------------------------------------------------------------- frames

    def epochs_frame(self, ref: str | Mapping[str, Any], *, sep: str = ".") -> Any:
        """A run's epochs as a pandas DataFrame, one row per epoch.

        Results are flattened into columns (``val.loss``, ``val.car.iou``), and
        ``learning_rate`` and ``epoch_time`` sit beside them. A metric an epoch
        did not report is missing (NaN), as it is in Visin's charts.
        """
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("epochs_frame needs pandas: pip install 'visin[pandas]'") from exc
        rows = []
        for epoch in self.epochs(ref):
            row: dict[str, Any] = {
                "epoch": epoch.get("epoch"),
                "timestamp": epoch.get("timestamp"),
                "learning_rate": epoch.get("learning_rate"),
                "epoch_time": epoch.get("epoch_time"),
            }
            results = epoch.get("results")
            if isinstance(results, Mapping):
                row.update(flatten(results, sep))
            rows.append(row)
        frame = pd.DataFrame(rows)
        if "timestamp" in frame:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        return frame.set_index("epoch") if "epoch" in frame else frame
